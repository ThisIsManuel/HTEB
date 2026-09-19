"""Parse YAML into one Pydantic configuration model before runtime preparation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import yaml
from dotenv import load_dotenv
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from typing_extensions import NotRequired, Self, TypedDict

from .constants import (
    DATASET_GROUPS,
    EMBEDDING_PROMPT_POLICY,
    FULL_BENCHMARK_DATASETS,
    MODELS_DIRECTORY,
    RUNTIME_CACHE_DIRECTORY,
    SUPPORTED_TRANSFORMATIONS,
)
from .util import hash_settings, hub_model_name

Nonempty = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Trimmed = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
HubModel = Annotated[Trimmed, AfterValidator(hub_model_name)]
PositiveInt = Annotated[int, Field(gt=0)]
NonnegativeInt = Annotated[int, Field(ge=0)]
Number = int | float
Fraction = Annotated[Number, Field(gt=0, le=1)]


def distinct_devices(devices: list[int]) -> list[int]:
    if len(devices) != len(set(devices)):
        raise ValueError("devices must contain distinct GPU indices")
    return devices


Devices = Annotated[list[NonnegativeInt], Field(min_length=1), AfterValidator(distinct_devices)]
Dtype = Literal["float16", "float32", "bfloat16", "auto"]
Task = Literal[
    "classification",
    "clustering",
    "pair_classification",
    "reranking",
    "retrieval",
    "str",
    "sts",
    "summarisation",
]
Scalar = str | int | float | bool | None


class Pooling(TypedDict):
    strategy: Annotated[
        Literal["model_default", "mean", "cls", "last_token"], Field(default="model_default")
    ]
    include_prompt: Annotated[bool | None, Field(default=None)]


class ModelSettings(TypedDict):
    batch_size: Annotated[PositiveInt | None, Field(default=None)]
    encoding_chunk_size: Annotated[PositiveInt | None, Field(default=None)]
    attn_implementation: Annotated[Trimmed | None, Field(default=None)]
    device_map: Annotated[str | dict[str, str | int] | None, Field(default=None)]
    device_map_min_gpus: Annotated[PositiveInt | None, Field(default=None)]
    device_map_memory_fraction: Annotated[Fraction | None, Field(default=None)]
    max_sequence_length: Annotated[PositiveInt | None, Field(default=None)]
    pooling: Annotated[Pooling, Field(default_factory=dict)]
    normalize_embeddings: Annotated[bool, Field(default=False)]
    deterministic_algorithms: Annotated[bool, Field(default=False)]
    allow_tf32: Annotated[bool, Field(default=True)]
    cudnn_benchmark: Annotated[bool, Field(default=True)]
    cache_folder: Annotated[str | None, Field(default=None)]
    encode_kwargs: Annotated[dict[Task, dict[str, Scalar]], Field(default_factory=dict)]
    prompt_names: NotRequired[
        dict[Task, dict[Literal["text", "query", "document"], Nonempty | None]]
    ]


class EmbeddingModel(TypedDict):
    model_id: HubModel
    settings: Annotated[ModelSettings, Field(default_factory=dict)]


class GenerationSettings(TypedDict):
    trust_remote_code: Annotated[bool | None, Field(default=None)]
    dtype: Annotated[Dtype, Field(default="auto")]
    quantization: Annotated[
        Literal["awq", "gptq", "marlin", "squeezellm"] | None, Field(default=None)
    ]
    tensor_parallel_size: Annotated[PositiveInt, Field(default=1)]
    devices: Annotated[Devices | None, Field(default=None)]
    gpu_memory_utilization: Annotated[Fraction, Field(default=0.85)]
    max_model_length: Annotated[PositiveInt, Field(default=8192, alias="max_model_length_default")]
    max_tokens: Annotated[PositiveInt, Field(default=512, alias="max_tokens_default")]
    max_tokens_expansion_default: Annotated[PositiveInt, Field(default=1024)]
    batch_size: Annotated[PositiveInt, Field(default=32)]
    chunk_size: Annotated[PositiveInt, Field(default=5000)]
    temperature: Annotated[Number, Field(default=1.0, ge=0, le=2)]
    top_p: Annotated[Fraction, Field(default=0.95)]
    top_k: Annotated[int, Field(default=64, ge=-1)]
    repetition_penalty: Annotated[Number, Field(default=1.0, gt=0)]
    structured_output: Annotated[bool, Field(default=True)]
    max_retries: Annotated[NonnegativeInt, Field(default=3)]
    transform_corpus: Annotated[bool, Field(default=True)]


class EvaluationSettings(TypedDict):
    trust_remote_code: bool
    seed_evaluation: Annotated[list[NonnegativeInt], Field(min_length=1)]
    batch_size: Annotated[PositiveInt, Field(default=4096)]
    encoding_chunk_size: Annotated[PositiveInt, Field(default=4096)]
    clustering_batch_size: Annotated[PositiveInt, Field(default=500)]
    ignore_identical_ids: Annotated[bool, Field(default=False)]
    devices: Annotated[Devices | Literal["cpu"] | None, Field(default=None)]
    dtype: Annotated[Dtype, Field(default="bfloat16")]


class DatasetGeneration(TypedDict, total=False):
    max_model_length: PositiveInt
    max_tokens: PositiveInt
    tensor_parallel_size: PositiveInt
    devices: Devices | None


class DatasetOptions(TypedDict, total=False):
    generation: DatasetGeneration


DatasetEntry = str | Annotated[dict[Nonempty, DatasetOptions], Field(min_length=1, max_length=1)]
DatasetSelection = (
    Nonempty
    | Annotated[list[DatasetEntry], Field(min_length=1)]
    | Annotated[dict[Nonempty, DatasetOptions], Field(min_length=1)]
)


class Selection(TypedDict):
    datasets: DatasetSelection
    transformations: Annotated[list[Trimmed], Field(min_length=1)]
    seed_transform: Annotated[list[NonnegativeInt], Field(min_length=1)]


class HTEBConfig(BaseModel):
    """The authoritative static configuration; nested sections are typed dictionaries."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )

    load_existing_transformations: bool
    transformations_directory: Nonempty = "data/transformations"
    generator_model: HubModel
    emb_models: Annotated[list[EmbeddingModel], Field(min_length=1)]
    selection: Selection
    evaluation: EvaluationSettings
    generation: Annotated[GenerationSettings, Field(default_factory=dict)]

    @field_validator("selection")
    @classmethod
    def resolve_selection(cls, selection: Selection) -> Selection:
        value = selection["datasets"]
        entries = (
            [{name: specification} for name, specification in value.items()]
            if isinstance(value, dict)
            else [value]
            if isinstance(value, str)
            else value
        )
        resolved: dict[str, DatasetOptions] = {}
        has_options = False
        for entry in entries:
            name, options = next(iter(entry.items())) if isinstance(entry, dict) else (entry, None)
            name = name.strip()
            if options is not None:
                if name not in FULL_BENCHMARK_DATASETS:
                    raise ValueError(
                        f"selection.datasets: unsupported dataset specification {name!r}"
                    )
                has_options = True
                resolved[name] = options
            else:
                for dataset in DATASET_GROUPS.get(name, [name]):
                    resolved.setdefault(dataset, {})
        unknown = set(resolved) - FULL_BENCHMARK_DATASETS
        if unknown:
            raise ValueError(f"selection.datasets: unsupported values {sorted(unknown)}")
        selection["datasets"] = resolved if has_options else list(resolved)
        unknown_transformations = set(selection["transformations"]) - SUPPORTED_TRANSFORMATIONS
        if unknown_transformations:
            raise ValueError(
                f"selection.transformations: unsupported values {sorted(unknown_transformations)}"
            )
        return selection

    @field_validator("generation")
    @classmethod
    def validate_generation(cls, generation: GenerationSettings) -> GenerationSettings:
        if generation["top_k"] == 0:
            raise ValueError("generation.top_k must be -1 or positive")
        return generation

    @field_validator("emb_models")
    @classmethod
    def validate_model_arguments(cls, models: list[EmbeddingModel]) -> list[EmbeddingModel]:
        reserved = {
            "sentences",
            "batch_size",
            "normalize_embeddings",
            "show_progress_bar",
            "pool",
            "prompt",
            "prompt_name",
            "device",
            "convert_to_numpy",
            "convert_to_tensor",
            "output_value",
        }
        for index, model in enumerate(models):
            settings = model["settings"]
            for task, names in settings.get("prompt_names", {}).items():
                roles = {"query", "document"} if task in {"retrieval", "reranking"} else {"text"}
                if names.keys() - roles:
                    raise ValueError(
                        f"emb_models.{index}.settings.prompt_names.{task}: allowed roles are {sorted(roles)}"
                    )
            for task, arguments in settings["encode_kwargs"].items():
                for key, value in arguments.items():
                    location = f"emb_models.{index}.settings.encode_kwargs.{task}.{key}"
                    if key in reserved:
                        raise ValueError(f"{location}: reserved encode argument")
                    if key == "precision" and value != "float32":
                        raise ValueError(
                            f"{location} must be 'float32' or omitted; quantized embedding output is unsupported by HTEB metrics"
                        )
        return models

    @model_validator(mode="after")
    def validate_relationships(self) -> Self:
        devices = self.evaluation["devices"]
        if isinstance(devices, list) and len(devices) > 1:
            for index, model in enumerate(self.emb_models):
                if model["settings"]["device_map"]:
                    raise ValueError(
                        f"emb_models.{index}.settings.device_map cannot be combined with multiple evaluation.devices entries"
                    )
        if len(self.selection["seed_transform"]) != len(self.evaluation["seed_evaluation"]):
            raise ValueError(
                "seed_transform and seed_evaluation must have the same length; entries are paired by position"
            )
        if not self.load_existing_transformations and self.generation["trust_remote_code"] is None:
            raise ValueError(
                "generation.trust_remote_code is required when load_existing_transformations is false; explicitly set it to true or false"
            )
        for dataset in self.datasets:
            for transformation in self.selection["transformations"]:
                for stage in transformation_stages(transformation):
                    effective = generation_settings(self, dataset, stage)
                    location = f"selection.datasets.{dataset}.generation.effective {stage}"
                    if effective["max_tokens"] >= effective["max_model_length"]:
                        raise ValueError(
                            f"{location}.max_tokens must be less than max_model_length"
                        )
                    if effective["devices"] is not None and effective["tensor_parallel_size"] != 1:
                        raise ValueError(f"{location}.devices requires tensor_parallel_size: 1")
        return self

    @property
    def datasets(self) -> list[str]:
        """Dataset names in their resolved execution order."""
        return list(cast("list[str] | dict[str, DatasetOptions]", self.selection["datasets"]))


def load_config(source: str | Path) -> HTEBConfig:
    return parse_config(Path(source).read_text(encoding="utf-8"))


def parse_config(text: str) -> HTEBConfig:
    return HTEBConfig.model_validate(yaml.safe_load(text))


def seed_pairs(config: HTEBConfig) -> list[tuple[int, int]]:
    return list(
        zip(
            config.selection["seed_transform"],
            config.evaluation["seed_evaluation"],
            strict=True,
        )
    )


def model_settings(
    model: Mapping[str, Any], seed: int, *, evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    settings = {key: value for key, value in model["settings"].items() if value is not None}
    settings.update({key: evaluation[key] for key in ("dtype", "trust_remote_code")})
    devices = evaluation["devices"]
    device = [f"cuda:{index}" for index in devices] if isinstance(devices, list) else devices
    # Keep the derived value in effective settings so existing single-device identities agree.
    settings["multi_gpu"] = isinstance(device, list) and len(device) > 1
    if device is not None:
        settings["device"] = device[0] if isinstance(device, list) and len(device) == 1 else device
    if settings.get("pooling") == {"strategy": "model_default", "include_prompt": None}:
        settings.pop("pooling")
    settings["seed_evaluation"] = seed
    return settings


def model_identity(
    model: Mapping[str, Any], seed: int, *, evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    load_id = hub_model_name(model["model_id"])
    return {
        "load_id": load_id,
        "display_id": load_id,
        "source": "remote",
        "identity_sha256": hash_settings(
            {
                "model_id": load_id,
                "source": "remote",
                "settings": model_settings(model, seed, evaluation=evaluation),
                "prompt_policy": EMBEDDING_PROMPT_POLICY,
            }
        ),
    }


def transformation_stages(transformation: str) -> tuple[str, ...]:
    if transformation == "backtranslation":
        return ("backtranslate_forward", "backtranslate_backward")
    if transformation == "summarised_expansion":
        return ("expansion", "summarise")
    return (transformation,)


def generation_settings(
    config: HTEBConfig, dataset: str | None, transformation: str
) -> dict[str, Any]:
    """Resolve output defaults before explicit dataset settings."""
    settings = dict(config.generation)
    if transformation == "expansion" and "max_tokens_expansion_default" in settings:
        settings["max_tokens"] = settings["max_tokens_expansion_default"]
    datasets = config.selection["datasets"]
    if isinstance(datasets, dict):
        settings.update(
            datasets.get(dataset, {}).get("generation", {}) if dataset is not None else {}
        )
    return settings


def generation_job_groups(
    config: HTEBConfig, jobs: list[tuple[str, str, int]]
) -> list[tuple[HTEBConfig, list[tuple[str, str, int]]]]:
    """Group engine settings without binding a job's sampling output allowance."""
    groups: list[tuple[HTEBConfig, list[tuple[str, str, int]]]] = []
    for job in jobs:
        settings = generation_settings(config, job[0], job[1])
        settings["max_tokens"] = config.generation["max_tokens"]
        for group_config, group_jobs in groups:
            if group_config.generation == settings:
                group_jobs.append(job)
                break
        else:
            groups.append((config.model_copy(update={"generation": settings}), [job]))
    return groups


def configure_runtime_environment() -> None:
    """Set cache locations before importing Hub and model libraries."""
    load_dotenv(Path.cwd() / ".env", override=False)
    root = Path(RUNTIME_CACHE_DIRECTORY).absolute()
    directories = {
        "HF_HOME": "huggingface",
        "HF_DATASETS_CACHE": "huggingface/datasets",
        "HF_MODULES_CACHE": "huggingface/modules",
        "TORCH_HOME": "torch",
        "TORCHINDUCTOR_CACHE_DIR": "torch/inductor",
        "XDG_CACHE_HOME": "xdg",
        "XDG_CONFIG_HOME": "config",
        "TRITON_CACHE_DIR": "triton",
        "VLLM_CACHE_ROOT": "vllm",
        "VLLM_CONFIG_ROOT": "vllm-config",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer",
        "CUDA_CACHE_PATH": "cuda",
    }
    for variable, directory in directories.items():
        os.environ[variable] = str(root / directory)
    os.environ["HF_HUB_CACHE"] = str(Path(MODELS_DIRECTORY).absolute())
