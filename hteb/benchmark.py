"""Dataset validation and paired score reporting."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import HTEBConfig
from .constants import (
    SUPPORTED_TASK_TYPES,
)
from .dataset_preparation import (
    filter_data,
    prepare_evaluation_data,
)
from .evaluation import prepare_records, validate_required_fields
from .transformation_parquet import (
    latest_transformation,
    load_transformation,
    transformation_metadata_path,
    transformation_path,
)
from .util import (
    reported_score,
)

logger = logging.getLogger(__name__)


def _validate_split_counts(info: Mapping[str, Any], originals: list[dict[str, Any]]) -> None:
    metadata = info["dataset_metadata"]
    if "split_counts" not in metadata:
        return
    observed_counts = Counter(
        str(record.get("_hteb_split", record.get("split", ""))) for record in originals
    )
    if dict(observed_counts) != metadata["split_counts"]:
        raise ValueError(
            f"{info['dataset']}: split counts are {dict(observed_counts)}, "
            f"expected {metadata['split_counts']}"
        )


def result_row(
    model: str,
    dataset: str,
    transformation: str,
    seed_transform: int,
    seed_evaluation: int,
    original: Mapping[str, Any],
    transformed: Mapping[str, Any],
) -> dict[str, object]:
    if original["main_metric"] != transformed["main_metric"]:
        raise ValueError(
            f"{dataset}: original and transformed evaluations returned different metrics"
        )
    return {
        "model": model,
        "dataset": dataset,
        "transformation": transformation,
        "seed_transform": seed_transform,
        "seed_evaluation": seed_evaluation,
        "score_original": reported_score(original["score"]),
        "score_transformed": reported_score(transformed["score"]),
    }


def variant_paths(config: HTEBConfig, dataset: str) -> dict[str, Path]:
    root = Path(config.transformations_directory).expanduser().absolute()
    return {
        f"{transformation}_seed_{seed}": latest_transformation(
            transformation_path(root, config.generator_model, dataset, transformation, seed)
        )
        for transformation in config.selection["transformations"]
        for seed in config.selection["seed_transform"]
    }


def preparation_settings(info: Mapping[str, Any]) -> dict[str, Any]:
    """Use the dataset's declared splits and grouping for evaluation."""
    definition = info["dataset_metadata"]
    return {
        "task_type": info["task_type"],
        "train_split": definition.get("train_split", ""),
        "eval_split": definition["eval_split"],
        "clustering_format": definition.get("clustering_format"),
    }


def validate_originals(info: Mapping[str, Any], *, freshly_loaded: bool = False) -> None:
    """Validate original records before any generation or embedding model loads."""
    if info["task_type"] not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"{info['dataset']}: unsupported task type {info['task_type']!r}")
    originals, _, _ = filter_data(info["original"], info["task_type"])
    if freshly_loaded:
        import pyarrow as pa  # type: ignore[import-untyped]

        _validate_split_counts(info, info["original"])
        # Check that the complete source can use one Parquet record structure
        # before workers load generators. Keep the original dictionaries intact.
        pa.array(info["original"])
    prepare_records(originals, **preparation_settings(info))


def _check_dataset(config: HTEBConfig, dataset: str) -> tuple[dict[str, Any], list[str]]:
    first: dict[str, Any] = {}
    reference: dict[str, Any] | None = None
    reference_originals: list[dict[str, Any]] = []
    problems: list[str] = []
    paths: dict[str, Path] = {}
    requested = {
        f"{transformation}_seed_{seed}": transformation_path(
            config.transformations_directory, config.generator_model, dataset, transformation, seed
        )
        for transformation in config.selection["transformations"]
        for seed in config.selection["seed_transform"]
    }
    for column, path in requested.items():
        context = f"{dataset} / {column}"
        try:
            path = latest_transformation(path)
            paths[column] = path
            info = load_transformation(path, expected_dataset=dataset)
            validate_originals(info)
            prepared = prepare_evaluation_data(info)
            originals = prepared.original
            report = prepared.data_filtering
            if reference is None:
                reference = info
                reference_originals = originals
                first = {
                    key: value
                    for key, value in info.items()
                    if key not in {"original", "transformed"}
                }
                first["excluded_original_count"] = len(info["original"]) - len(originals)
                first["data_filtering"] = report
            elif originals != reference_originals or any(
                info[key] != reference[key] for key in ("task_type", "dataset_metadata")
            ):
                raise ValueError(
                    f"inconsistent canonical originals or dataset settings compared with {reference['path']}"
                )
            assert prepared.transformed is not None
            validate_required_fields(
                prepared.transformed, info["task_type"], allow_missing_text=True
            )
            prepare_records(prepared.transformed, **preparation_settings(info))
            logger.info(
                "%s: valid; %s transformed texts require restoration",
                context,
                prepared.replacements,
            )
        except (OSError, ValueError, TypeError, OverflowError, KeyError) as error:
            problems.append(
                f"{context}\n  {path}\n  {transformation_metadata_path(path)}\n  {error}"
            )
    first["paths"] = paths
    return first, problems


def validate_transformations(config: HTEBConfig, dataset: str) -> dict[str, Any]:
    """Load every selected pair for a dataset before reporting all problems."""
    info, problems = _check_dataset(config, dataset)
    if problems:
        raise ValueError(_problem_overview(problems))
    return info


def _problem_overview(problems: list[str]) -> str:
    return (
        f"Cannot load the selected transformations: {len(problems)} combinations have problems.\n"
        + "\n".join(problems)
    )


def validate_selection(config: HTEBConfig) -> dict[str, dict[str, Any]]:
    """Finish checking every dataset before raising one combined error."""
    metadata: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for dataset in config.datasets:
        info, errors = _check_dataset(config, dataset)
        metadata[dataset] = info
        problems.extend(errors)
    if problems:
        raise ValueError(_problem_overview(problems))
    return metadata
