"""A sequential config-path workflow using lists of ordinary dataset records."""

from __future__ import annotations

import datetime as dt
import logging
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

from .benchmark import (
    preparation_settings,
    result_row,
    validate_originals,
    validate_selection,
)
from .config import (
    HTEBConfig,
    configure_runtime_environment,
    generation_job_groups,
    load_config,
    model_identity,
    model_settings,
    seed_pairs,
)
from .constants import MODELS_DIRECTORY, RUNTIME_CACHE_DIRECTORY
from .dataset_preparation import (
    filtering_summary,
    prepare_evaluation_data,
)
from .embedding import load_sentence_transformer
from .evaluation import evaluate_records, prepare_records
from .generation import GenerationResult, _release_cuda_memory, generate_jobs, translation_languages
from .load_datasets import load_source, validate_source_definition
from .parallel_evaluation import EmbeddingPool
from .parallel_generation import generate_transformations_on_gpus, resolve_gpu_devices
from .score_tables import print_score_tables
from .transformation_parquet import load_transformation, preserve_raw_originals
from .util import (
    check_directory,
    check_local_model_conflict,
    finish_run_directory,
    group_scores,
    hash_settings,
    run_logging,
    save_results,
)

logger = logging.getLogger(__name__)


def check_outputs() -> Path:
    output_root = Path("results").absolute()
    check_directory(output_root)
    check_directory(Path(MODELS_DIRECTORY).absolute())
    check_directory(Path(RUNTIME_CACHE_DIRECTORY).absolute())
    return output_root


def generate_selection(config: HTEBConfig, jobs: list[tuple[str, str, int]]) -> GenerationResult:
    """Load originals once and write each generated transformation to its final folder."""
    root = Path(config.transformations_directory).expanduser().absolute()
    check_directory(root)
    root.mkdir(parents=True, exist_ok=True)
    versions: list[dict[str, object]] = []
    paths: dict[tuple[str, str, int], Path] = {}
    original_paths: dict[str, Path] = {}
    for dataset in config.datasets:
        info = load_source(dataset)
        validate_originals(info, freshly_loaded=True)
        original_paths[dataset] = preserve_raw_originals(
            {**info, "input_scope": "source_records"}, info["original"], root=root
        )
        logger.info(
            "Loaded %s original records for %s; HF revision %s",
            len(info["original"]),
            dataset,
            info.get("source_revision", "unknown"),
        )
    for group_config, group_jobs in generation_job_groups(config, jobs):
        generate = (
            generate_jobs
            if group_config.generation["devices"] is None
            else generate_transformations_on_gpus
        )
        result = generate(group_config, group_jobs, original_paths)
        paths.update(result.paths)
        for version in result.versions:
            if version not in versions:
                versions.append(version)
    return GenerationResult(versions, paths)


def run_name(config: HTEBConfig, started_at: dt.datetime) -> str:
    selection = config.selection
    return (
        f"{started_at:%Y-%m-%d_%H-%M-%S}"
        f"_{len(config.emb_models)}-EMB-MODELS"
        f"_{len(selection['transformations'])}-TRANSFORMATIONS"
        f"_{len(selection['datasets'])}-DATASETS"
        f"_{len(seed_pairs(config))}-SEEDS"
    )


def run_hteb(config_path: str | Path | HTEBConfig) -> Path:
    started_at = dt.datetime.now().astimezone()
    config = config_path if isinstance(config_path, HTEBConfig) else load_config(config_path)
    supplied_settings = config.model_dump(mode="json", by_alias=True)
    model_versions: dict[str, list[dict[str, object]]] = {"embedding": [], "generation": []}
    name = run_name(config, started_at)
    reuse = config.load_existing_transformations
    metadata = validate_selection(config) if reuse else {}
    if not reuse:
        for dataset in config.datasets:
            validate_source_definition(dataset)
    jobs = (
        []
        if reuse
        else [
            (dataset, transformation, seed)
            for seed in config.selection["seed_transform"]
            for dataset in config.datasets
            for transformation in config.selection["transformations"]
        ]
    )
    for index, entry in enumerate(config.emb_models):
        check_local_model_conflict(entry["model_id"], f"emb_models.{index}.model_id")
    if jobs:
        check_local_model_conflict(config.generator_model, "generator_model")
        for group_config, _ in generation_job_groups(config, jobs):
            if group_config.generation["devices"] is not None:
                resolve_gpu_devices(group_config.generation["devices"])
    output_root = check_outputs()
    configure_runtime_environment()
    output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=".running-", dir=output_root))
    rows: list[dict[str, object]] = []
    evaluation_inputs: dict[tuple[str, str, int], dict[str, object]] = {}
    show_progress = sys.stderr.isatty()
    with run_logging(directory / "run_log.txt", progress=show_progress):
        try:
            logger.info("Run started: %s", name)
            logger.info(
                "Loading %s existing combinations; generating %s combinations",
                len(config.datasets)
                * len(config.selection["transformations"])
                * len(config.selection["seed_transform"])
                - len(jobs),
                len(jobs),
            )
            for dataset, source in metadata.items():
                logger.info("%s: %s", dataset, filtering_summary(source["data_filtering"]))
            generated = GenerationResult([], {})
            if jobs:
                generated = generate_selection(config, jobs)
                model_versions["generation"] = generated.versions

            for model_config in config.emb_models:
                for seed_transform, seed_evaluation in seed_pairs(config):
                    identity = model_identity(
                        model_config, seed_evaluation, evaluation=config.evaluation
                    )
                    settings = model_settings(
                        model_config, seed_evaluation, evaluation=config.evaluation
                    )
                    model = None
                    pool = None
                    try:
                        model, embedding_metadata = load_sentence_transformer(
                            identity["load_id"], settings
                        )
                        model_version = embedding_metadata["version"]
                        assert isinstance(model_version, dict)
                        if model_version not in model_versions["embedding"]:
                            model_versions["embedding"].append(model_version)
                        device = settings.get("device")
                        if isinstance(device, list) and len(device) > 1:
                            pool = EmbeddingPool(
                                model, device, settings, model_version=model_version
                            )
                        for dataset in config.datasets:
                            paths = (
                                metadata[dataset]["paths"]
                                if reuse
                                else {
                                    f"{transformation}_seed_{seed}": path
                                    for (
                                        name,
                                        transformation,
                                        seed,
                                    ), path in generated.paths.items()
                                    if name == dataset
                                }
                            )
                            first_column = (
                                f"{config.selection['transformations'][0]}_seed_{seed_transform}"
                            )
                            info = load_transformation(
                                paths[first_column], expected_dataset=dataset, originals_only=True
                            )
                            raw_originals = info["original"]
                            originals = prepare_evaluation_data(info).original
                            preparation = preparation_settings(info)
                            original_records = prepare_records(originals, **preparation)
                            metric_arguments = dict(
                                model_config=settings,
                                evaluation_config=config.evaluation,
                                task_type=info["task_type"],
                                dataset_metadata=info["dataset_metadata"],
                                train_split=preparation["train_split"],
                                eval_split=preparation["eval_split"],
                                seed_evaluation=seed_evaluation,
                                pool=pool,
                            )
                            with tqdm(
                                total=1 + len(config.selection["transformations"]),
                                desc=(
                                    f"{dataset} / {identity['display_id']} / "
                                    f"transform seed {seed_transform} / eval seed {seed_evaluation}"
                                ),
                                postfix="original",
                                unit="evaluation",
                                file=sys.stderr,
                                disable=not show_progress,
                                dynamic_ncols=True,
                                leave=True,
                            ) as progress:
                                original_score = evaluate_records(
                                    model, original_records, **metric_arguments
                                )
                                progress.update()
                                for transformation in config.selection["transformations"]:
                                    progress.set_postfix_str(transformation)
                                    column = f"{transformation}_seed_{seed_transform}"
                                    info = load_transformation(
                                        paths[column], expected_dataset=dataset
                                    )
                                    prepared = prepare_evaluation_data(info)
                                    current_originals = prepared.original
                                    current_retained = prepared.retained_indices
                                    report = prepared.data_filtering
                                    if current_originals != originals:
                                        raise ValueError(
                                            "transformation files have inconsistent canonical originals"
                                        )
                                    transformed = prepared.transformed
                                    assert transformed is not None
                                    if "evaluation_input" in info:
                                        input_key = (dataset, transformation, seed_transform)
                                        input_details = {
                                            **info["evaluation_input"],
                                            "evaluation_preparation_version": 1,
                                            "sha256": {
                                                **info["evaluation_input"]["sha256"],
                                                "original": hash_settings(current_originals),
                                                "transformed": hash_settings(transformed),
                                            },
                                        }
                                        if (
                                            input_key in evaluation_inputs
                                            and evaluation_inputs[input_key] != input_details
                                        ):
                                            raise ValueError(
                                                "transformation input changed during evaluation"
                                            )
                                        evaluation_inputs[input_key] = input_details
                                    transformed_records = prepare_records(
                                        transformed, **preparation
                                    )
                                    transformed_score = evaluate_records(
                                        model, transformed_records, **metric_arguments
                                    )
                                    row = result_row(
                                        identity["display_id"],
                                        dataset,
                                        transformation,
                                        seed_transform,
                                        seed_evaluation,
                                        original_score,
                                        transformed_score,
                                    )
                                    row["data_filtering"] = report
                                    if transformation in {"translation", "cross_translation"}:
                                        row["languages"] = translation_languages(
                                            info["original"],
                                            info["task_type"],
                                            transformation,
                                            seed_transform,
                                            dataset_metadata=info["dataset_metadata"],
                                            transform_corpus=info.get("transform_corpus", True),
                                            retained_indices=current_retained,
                                            dataset=dataset,
                                            translation_language_selection=info.get(
                                                "translation_language_selection"
                                            ),
                                        )
                                    rows.append(row)
                                    progress.update()
                                    del info, current_originals, transformed, transformed_records
                                    logger.info(
                                        "Evaluated %s / %s / %s",
                                        identity["display_id"],
                                        dataset,
                                        column,
                                        extra={"evaluation_progress": True},
                                    )
                            del raw_originals, originals, original_records
                        del metric_arguments
                    finally:
                        primary_error = sys.exc_info()[1]
                        try:
                            if pool is not None:
                                pool.close()
                        except Exception as cleanup_error:
                            if primary_error is not None:
                                raise primary_error from cleanup_error
                            raise
                        finally:
                            del pool, model
                            _release_cuda_memory()
            logger.info("Completed %s paired scores", len(rows))
        except BaseException as error:
            logger.error("Run failed (%s)", type(error).__name__)
            raise
    supplied_settings["model_versions"] = model_versions
    supplied_settings["evaluation_inputs"] = list(evaluation_inputs.values())
    groups = group_scores(rows)
    supplied_settings["data_filtering"] = [
        {
            "dataset": group["dataset"],
            "seed_transform": group["seed_transform"],
            "transformations": group["data_filtering"],
        }
        for group in groups
        if group["model"] == groups[0]["model"]
    ]
    save_results(directory, groups, supplied_settings)
    path = finish_run_directory(directory, name) / "scores.JSON"
    print_score_tables(groups)
    return path
