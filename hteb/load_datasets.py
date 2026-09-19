"""Load the supported HF datasets using their checked-in loading instructions."""

from __future__ import annotations

import copy
import importlib
import logging
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .constants import RUNTIME_CACHE_DIRECTORY
from .dataset_preparation import CLUSTERING_FORMATS, retrieval_text
from .transformation_parquet import _read_document

logger = logging.getLogger(__name__)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without surrounding spaces")
    return value


def _strings(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    result = [_text(item, name) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def validate_source_definition(dataset: str) -> dict[str, Any]:
    """Check loading instructions without importing HF or accessing the network."""
    document = _read_document(dataset)
    metadata = document["dataset_metadata"]
    task = document["task_type"]
    if task not in {
        "classification",
        "pair_classification",
        "sts",
        "str",
        "clustering",
        "retrieval",
        "reranking",
        "summarisation",
    }:
        raise ValueError(f"{dataset}: unsupported source task {task!r}")
    allowed = {"hf_dataset_name", "revision", "languages", "eval_split", "split_counts"}
    for key in ("hf_dataset_name", "revision", "eval_split"):
        _text(metadata.get(key), f"{dataset}.{key}")
    if metadata["revision"] == "unknown":
        raise ValueError(f"{dataset}: set an explicit dataset revision before loading originals")
    repository = metadata["hf_dataset_name"]
    if re.fullmatch(r"[A-Za-z0-9_][\w.-]*/[A-Za-z0-9_][\w.-]*", repository) is None:
        raise ValueError(f"{dataset}: hf_dataset_name must be an HF dataset name")
    _strings(metadata.get("languages"), f"{dataset}.languages")
    if "split_counts" in metadata:
        counts = metadata["split_counts"]
        if not isinstance(counts, dict) or not counts:
            raise ValueError(f"{dataset}.split_counts must be a nonempty object")
        for split, count in counts.items():
            _text(split, f"{dataset}.split_counts split")
            if type(count) is not int or count < 1:
                raise ValueError(f"{dataset}.split_counts.{split} must be a positive integer")
    if task == "classification":
        allowed.add("train_split")
        train = _text(metadata.get("train_split"), f"{dataset}.train_split")
        if train == metadata["eval_split"]:
            raise ValueError(f"{dataset}: train_split and eval_split must differ")
    if task == "retrieval":
        allowed.add("ignore_identical_ids")
    components = metadata.get("components")
    if task == "retrieval" or components is not None:
        allowed.add("components")
        if task not in {"retrieval", "reranking"}:
            raise ValueError(f"{dataset}: components only apply to retrieval or reranking")
        expected = ["corpus", "queries", "qrels"]
        if task == "reranking":
            expected.append("top_ranked")
        if not isinstance(components, list) or len(components) != len(expected):
            raise ValueError(f"{dataset}: components must specify {', '.join(expected)}")
        for component, part in zip(components, expected, strict=True):
            if not isinstance(component, dict) or set(component) not in (
                {"part", "config", "split"},
                {"part", "file", "split"},
            ):
                raise ValueError(
                    f"{dataset}: each component needs part, split and exactly one of config or file"
                )
            if component["part"] != part:
                raise ValueError(f"{dataset}: components must follow {', '.join(expected)} order")
            for key in ("split", "file" if "file" in component else "config"):
                _text(component[key], f"{dataset}.{part}.{key}")
            if "file" in component:
                filename = component["file"]
                if (
                    re.fullmatch(r"(?:[\w.-]+/)*[\w.-]+\.parquet", filename) is None
                    or ".." in PurePosixPath(filename).parts
                ):
                    raise ValueError(f"{dataset}.{part}.file must be a relative Parquet file path")
        if components[2]["split"] != metadata["eval_split"]:
            raise ValueError(f"{dataset}: qrels split must equal eval_split")
    elif "language_configs" in metadata:
        allowed.add("language_configs")
        _strings(metadata["language_configs"], f"{dataset}.language_configs")
    else:
        allowed.add("hf_subset")
        if "hf_subset" in metadata:
            _text(metadata["hf_subset"], f"{dataset}.hf_subset")
    if task == "clustering":
        allowed.add("clustering_format")
        if metadata.get("clustering_format") not in CLUSTERING_FORMATS:
            raise ValueError(f"{dataset}: invalid clustering_format")
        if metadata["clustering_format"] == "per_language" and "language_configs" not in metadata:
            raise ValueError(f"{dataset}: per_language clustering requires language_configs")
    if task == "summarisation":
        fields = ("human_summaries_field", "machine_summaries_field", "score_field")
        allowed.update(fields)
        names = [_text(metadata.get(key), f"{dataset}.{key}") for key in fields]
        if len(set(names)) != len(names):
            raise ValueError(f"{dataset}: summary text and score fields must differ")
    unsupported = set(metadata) - allowed
    if unsupported:
        raise ValueError(
            f"{dataset}: unsupported loading settings: {', '.join(sorted(unsupported))}"
        )
    return document


def _offline() -> bool:
    return any(
        os.environ.get(key, "").upper() in {"1", "ON", "YES", "TRUE"}
        for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    )


def _resolve_revision(repository: str, revision: str) -> str:
    if revision == "unknown":
        raise ValueError(f"{repository}: set an explicit dataset revision before loading originals")
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import HfApi, snapshot_download

    resolved: str | None
    if _offline():
        cached = snapshot_download(
            repository,
            repo_type="dataset",
            revision=revision,
            cache_dir=os.environ.get("HF_HUB_CACHE"),
            local_files_only=True,
            allow_patterns=[],
        )
        resolved = Path(cached).name
    else:
        resolved = HfApi().dataset_info(repository, revision=revision).sha
    if not isinstance(resolved, str) or re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
        raise ValueError(f"{repository}: HF did not identify an exact dataset version")
    return resolved


def _load_split(
    repository: str,
    config: str | None,
    split: str,
    revision: str,
    *,
    file: str | None = None,
) -> list[dict[str, Any]]:
    datasets = importlib.import_module("datasets")

    cache = Path(
        os.environ.get(
            "HF_DATASETS_CACHE",
            str(Path(RUNTIME_CACHE_DIRECTORY) / "huggingface" / "datasets"),
        )
    ).absolute()
    loaded = datasets.load_dataset(
        repository,
        name=config,
        split=split,
        revision=revision,
        cache_dir=str(cache),
        download_config=datasets.DownloadConfig(local_files_only=_offline()),
        **({"data_files": {split: file}} if file is not None else {}),
    )
    return [dict(row) for row in loaded]


def _id(value: object, location: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value) == "":
        raise ValueError(f"{location}: invalid ID")
    return str(value)


def _index(rows: list[dict[str, Any]], part: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = _id(row.get("_id"), part)
        if identifier in result:
            raise ValueError(f"{part}: duplicate ID {identifier!r}")
        result[identifier] = row
    return result


def _candidate_text(row: Mapping[str, Any]) -> str:
    """Keep the complete upstream text, including its leading/trailing whitespace."""
    body, title = row.get("text"), row.get("title")
    body = "" if body is None else body
    title = "" if title is None else title
    if not isinstance(body, str) or not isinstance(title, str):
        raise ValueError("reranking corpus requires text and title strings")
    return f"{title} {body}" if title else body


def _reranking(parts: Mapping[str, list[dict[str, Any]]], split: str) -> list[dict[str, Any]]:
    corpus = _index(parts["corpus"], "corpus")
    queries = _index(parts["queries"], "queries")
    relevance: dict[str, dict[str, int]] = {}
    for row in parts["qrels"]:
        query = _id(row.get("query-id"), "qrels query")
        document = _id(row.get("corpus-id"), "qrels document")
        if query not in queries or document not in corpus:
            raise ValueError("reranking qrels refer to a missing query/document")
        score = row.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or int(score) != score
        ):
            raise ValueError("reranking qrels require integer relevance")
        labels = relevance.setdefault(query, {})
        if document in labels:
            raise ValueError("reranking qrels contain duplicate query/document pairs")
        labels[document] = int(score)
    ranked: dict[str, list[str]] = {}
    for row in parts["top_ranked"]:
        query = _id(row.get("query-id"), "top_ranked query")
        if query not in queries or query in ranked:
            raise ValueError("top_ranked contains a missing or duplicate query ID")
        candidates = row.get("corpus-ids")
        if not isinstance(candidates, list):
            raise ValueError("top_ranked corpus-ids must be a list")
        ids = [_id(value, "top_ranked document") for value in candidates]
        if len(ids) != len(set(ids)) or any(value not in corpus for value in ids):
            raise ValueError("top_ranked contains a missing or duplicate document ID")
        ranked[query] = ids
    if set(ranked) != set(queries):
        raise ValueError("top_ranked must supply candidates for every query")
    records = []
    for query, row in queries.items():
        positives: list[str] = []
        negatives: list[str] = []
        for document in ranked[query]:
            target = positives if relevance.get(query, {}).get(document, 0) > 0 else negatives
            target.append(_candidate_text(corpus[document]))
        records.append(
            {
                "_id": query,
                "query": retrieval_text(row),
                "positive": positives,
                "negative": negatives,
                "_hteb_split": split,
                **({"language": row["language"]} if "language" in row else {}),
            }
        )
    return records


def _summary_fields(records: list[dict[str, Any]], metadata: Mapping[str, Any]) -> None:
    for row in records:
        for setting, target in (
            ("human_summaries_field", "human_summaries"),
            ("machine_summaries_field", "machine_summaries"),
            ("score_field", "relevance"),
        ):
            source = metadata[setting]
            if source not in row:
                raise ValueError(f"summary source lacks configured field {source!r}")
            if target != source:
                if target in row:
                    raise ValueError(f"summary field mapping would overwrite {target!r}")
                row[target] = row.pop(source)
        # The scorer gives gold_scores/scores precedence; never silently ignore score_field.
        for unused in ("gold_scores", "scores"):
            if unused in row:
                raise ValueError(f"summary source contains conflicting score field {unused!r}")


def _nested_clustering(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records or isinstance(records[0].get("sentences"), list):
        return records
    if any(
        not isinstance(row.get("sentences"), str) or isinstance(row.get("labels"), list)
        for row in records
    ):
        raise ValueError("clustering source mixes scalar and grouped records")
    extra = {key for row in records for key in row} - {
        "sentences",
        "labels",
        "language",
        "_hteb_split",
    }
    if extra:
        raise ValueError(f"cannot group unrecognized clustering fields: {sorted(extra)}")
    return [
        {
            "sentences": [row["sentences"] for row in records],
            "labels": [row["labels"] for row in records],
            **{key: records[0][key] for key in ("language", "_hteb_split") if key in records[0]},
        }
    ]


def load_source(dataset: str) -> dict[str, Any]:
    """Read a dataset once; all selected transformations use these original records."""
    definition = validate_source_definition(dataset)
    metadata = copy.deepcopy(definition["dataset_metadata"])
    repository = metadata["hf_dataset_name"]
    revision = _resolve_revision(repository, metadata["revision"])
    logger.info("%s: loading HF dataset %s at version %s", dataset, repository, revision)
    task = definition["task_type"]
    languages = metadata["languages"]

    def read(
        config: str | None,
        split: str,
        part: str | None = None,
        language: str | None = None,
        file: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = _load_split(
            repository, config, split, revision, **({"file": file} if file is not None else {})
        )
        if not rows:
            raise ValueError(f"{dataset}: {config or 'default'}/{split} has no records")
        rows = copy.deepcopy(rows)
        for row in rows:
            if any(key.startswith("_hteb_") for key in row):
                raise ValueError(f"{dataset}: HF source uses reserved _hteb_ fields")
            row["_hteb_split"] = split
            if part:
                row["_hteb_part"] = part
            if language is not None:
                row.setdefault("language", language)
            elif len(languages) == 1:
                row.setdefault("language", languages[0])
        return rows

    if "components" in metadata:
        parts = {
            item["part"]: read(
                item.get("config"), item["split"], item["part"], file=item.get("file")
            )
            for item in metadata["components"]
        }
        records = (
            _reranking(parts, metadata["eval_split"])
            if task == "reranking"
            else [row for rows in parts.values() for row in rows]
        )
    else:
        records = []
        configs = metadata.get("language_configs", [metadata.get("hf_subset")])
        splits = (
            [metadata["train_split"], metadata["eval_split"]]
            if task == "classification"
            else [metadata["eval_split"]]
        )
        for config in configs:
            for split in splits:
                rows = read(
                    config,
                    split,
                    language=config if "language_configs" in metadata else None,
                )
                if task == "clustering" and metadata["clustering_format"] == "nested":
                    rows = _nested_clustering(rows)
                records.extend(rows)
    if task == "summarisation":
        _summary_fields(records, metadata)
    return {
        "dataset": dataset,
        "task_type": task,
        "dataset_metadata": metadata,
        "source_revision": revision,
        "original": records,
    }
