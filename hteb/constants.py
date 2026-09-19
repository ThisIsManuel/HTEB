"""Small public constants and metadata-derived dataset selections."""

import json
from pathlib import Path

EMBEDDING_PROMPT_POLICY = "sentence_transformers_builtin_v1"
MODELS_DIRECTORY = "models"
RUNTIME_CACHE_DIRECTORY = ".cache/hteb"

AXES = {
    "Lexical/Stylistic": ("paraphrasing", "backtranslation", "style_change"),
    "Length": ("expansion", "summarise", "summarised_expansion"),
    "Language": ("translation", "cross_translation"),
}

SUPPORTED_TRANSFORMATIONS = frozenset(
    transformation for transformations in AXES.values() for transformation in transformations
)

SUPPORTED_TASK_TYPES = frozenset(
    {
        "classification",
        "clustering",
        "pair_classification",
        "reranking",
        "retrieval",
        "str",
        "sts",
        "summarisation",
    }
)


def _read_dataset_groups() -> dict[str, list[str]]:
    path = Path(__file__).with_name("dataset_metadata") / "groups.json"
    try:
        groups = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Missing or invalid dataset group metadata") from error
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Dataset group metadata must be a nonempty object")
    result: dict[str, list[str]] = {}
    for group, datasets in groups.items():
        if not isinstance(group, str) or not group.strip() or group != group.strip():
            raise ValueError(
                "Dataset group names must be nonempty strings without surrounding spaces"
            )
        if not isinstance(datasets, list) or not datasets:
            raise ValueError(f"Dataset group {group!r} must be a nonempty list")
        if any(
            not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            for name in datasets
        ):
            raise ValueError(f"Dataset group {group!r} requires plain dataset names")
        if len(set(datasets)) != len(datasets):
            raise ValueError(f"Dataset group {group!r} contains duplicate datasets")
        result[group] = datasets
    if set(result) & {name for datasets in result.values() for name in datasets}:
        raise ValueError("Dataset group names must differ from dataset names")
    return result


DATASET_GROUPS = _read_dataset_groups()

FULL_BENCHMARK_DATASETS = frozenset(
    dataset for datasets in DATASET_GROUPS.values() for dataset in datasets
)
