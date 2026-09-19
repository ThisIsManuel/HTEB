"""Load ordinary records and safely store one transformation per Parquet file."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .dataset_preparation import (
    CLUSTERING_FORMATS,
    clustering_groups,
    data_preprocessing,
    filter_data,
    filtering_report,
    retrieval_text_fields,
    stored_generation_statistics,
    validate_filtering_report,
)
from .util import check_directory, hash_settings, hub_model_name, safe_model_name

SCHEMA_VERSION = 3
IDENTITY_FIELDS = ("generator_model", "transformation", "dataset", "seed_transform")
GENERATION_FIELDS = ("max_model_length", "max_tokens", "max_retries")


def _plain_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or any(c in value for c in "/\\"):
        raise ValueError(f"{label} must be a plain name")
    return value


def transformation_path(
    root: str | Path,
    generator_model: str,
    dataset: str,
    transformation: str,
    seed: int,
    *,
    generation_datetime: str | None = None,
) -> Path:
    """Build a dated filename, or the historical undated path when no date is supplied."""
    identity = transformation_identity(generator_model, dataset, transformation, seed)
    filename = (
        f"data_{hash_settings({**identity, 'generation_datetime': generation_datetime})}.parquet"
        if generation_datetime is not None
        else "data.parquet"
    )
    return (
        Path(root).expanduser().absolute()
        / generator_directory(identity["generator_model"])
        / dataset
        / transformation
        / str(seed)
        / filename
    )


def generator_directory(generator_model: str) -> str:
    """Encode the complete checkpoint name for filesystem and browser paths."""
    if not isinstance(generator_model, str) or not generator_model.strip():
        raise ValueError("generator_model must be nonempty text")
    # Escape literal tildes before using them in place of URL percent escapes.
    name = quote(generator_model, safe="").replace("~", "%7E")
    # Encode characters that remain special on Windows as well.
    if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", name.split(".")[0], re.I):
        name = f"%{ord(name[0]):02X}" + name[1:]
    if name.endswith("."):
        name = name.rstrip(".") + "%2E" * (len(name) - len(name.rstrip(".")))
    name = name.replace("%", "~")
    if len(name.encode("utf-8")) > 255:
        raise ValueError("encoded generator_model exceeds the filesystem name limit")
    return name


def transformation_identity(
    generator_model: str, dataset: str, transformation: str, seed: int
) -> dict[str, Any]:
    _plain_component(dataset, "dataset")
    _plain_component(transformation, "transformation")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed_transform must be a nonnegative integer")
    return dict(
        zip(
            IDENTITY_FIELDS,
            (
                hub_model_name(generator_model, "generator_model"),
                transformation,
                dataset,
                seed,
            ),
            strict=True,
        )
    )


def transformation_metadata_path(path: Path) -> Path:
    if path.name == "data.parquet":
        return path.with_name("metadata.JSON")
    if re.fullmatch(r"data_[0-9a-f]{64}\.parquet", path.name) is None:
        raise ValueError(f"{path}: expected data.parquet or data_<hash>.parquet")
    return path.with_name("metadata_" + path.stem.removeprefix("data_") + ".JSON")


def transformation_pair_exists(path: Path) -> bool:
    """Only two absent files mean a missing combination."""
    companion = transformation_metadata_path(path)
    present = [p.exists() or p.is_symlink() for p in (path, companion)]
    if not any(present):
        return False
    if not all(present) or any(p.is_symlink() or not p.is_file() for p in (path, companion)):
        raise ValueError(f"{path}: incomplete transformation pair or non-regular file")
    return True


def _generation_datetime(metadata: Mapping[str, Any], path: Path) -> str:
    """Return an ordered UTC datetime; an explicitly unknown date sorts first."""
    if "generation_datetime" not in metadata:
        raise ValueError(f"{path}: missing generation_datetime")
    value = metadata["generation_datetime"]
    if value is None:
        return ""
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value) is None
    ):
        raise ValueError(f"{path}: generation_datetime must be UTC YYYY-MM-DDTHH:MM:SSZ or null")
    try:
        dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(f"{path}: invalid generation_datetime") from error
    return value


def latest_transformation(path: Path) -> Path:
    """Find the newest complete pair in this transformation/seed folder."""
    folder = path.parent
    if folder.is_symlink() or folder.absolute() != folder.resolve(strict=False):
        raise ValueError(f"transformation folder must not use symbolic links: {folder}")
    candidates = set(folder.glob("data*.parquet"))
    for companion in folder.glob("metadata*.JSON"):
        candidates.add(
            companion.with_name("data" + companion.stem.removeprefix("metadata") + ".parquet")
        )
    if not candidates:
        raise ValueError(f"{folder}: transformation pair is missing")
    dated: list[tuple[str, Path]] = []
    for candidate in sorted(candidates):
        metadata, _ = read_transformation_metadata(candidate)
        dated.append((_generation_datetime(metadata, candidate), candidate))
    newest = max(date for date, _ in dated)
    selected = [candidate for date, candidate in dated if date == newest]
    if len(selected) != 1:
        raise ValueError(
            f"{folder}: multiple newest transformation pairs ({newest or 'null'}): "
            + ", ".join(candidate.name for candidate in selected)
        )
    return selected[0]


def _validate_statistics(statistics: Any, max_retries: int) -> None:
    visible = {
        "records_total",
        "records_targeted",
        "text_fields_total",
        "records_with_fallback",
        "text_fields_with_fallback",
        "unchanged_outputs",
    }
    history = {
        "retried_text_fields",
        "recovered_after_retry_text_fields",
        "fallback_reasons",
    }
    if not isinstance(statistics, dict) or set(statistics) != visible | history:
        raise ValueError("invalid generation_statistics fields")
    if any(type(statistics[key]) is not int or statistics[key] < 0 for key in visible):
        raise ValueError("generation_statistics counts must be nonnegative integers")
    total, targeted, texts, rows, fallback, unchanged = (
        statistics[key]
        for key in (
            "records_total",
            "records_targeted",
            "text_fields_total",
            "records_with_fallback",
            "text_fields_with_fallback",
            "unchanged_outputs",
        )
    )
    if (
        targeted > min(total, texts)
        or rows > min(targeted, fallback)
        or fallback + unchanged > texts
        or bool(rows) != bool(fallback)
        or bool(targeted) != bool(texts)
    ):
        raise ValueError("inconsistent generation_statistics counts")
    if all(statistics[key] == "" for key in history):
        return
    if any(statistics[key] == "" for key in history):
        raise ValueError("generation history must be entirely known or entirely unknown")

    def check_outcomes(values: Any) -> None:
        counts = (
            "text_fields_total",
            "text_fields_with_fallback",
            "retried_text_fields",
            "recovered_after_retry_text_fields",
        )
        if any(type(values.get(key)) is not int or values[key] < 0 for key in counts):
            raise ValueError("generation outcome counts must be nonnegative integers")
        total, failed, retried, recovered = (values[key] for key in counts)
        if (
            failed > total
            or retried > total
            or recovered > min(retried, total - failed)
            or (max_retries == 0 and retried)
        ):
            raise ValueError("inconsistent generation outcome counts")
        reasons = values.get("fallback_reasons")
        allowed = {
            "prompt_too_long",
            "invalid_json_after_retries",
            "invalid_text_after_retries",
            "output_length_after_retries",
            "backend_failure",
        }
        if (
            not isinstance(reasons, dict)
            or not set(reasons) <= allowed
            or any(type(count) is not int or count < 1 for count in reasons.values())
            or sum(reasons.values()) != failed
        ):
            raise ValueError("fallback reasons must account for every failed text field")

    check_outcomes(statistics)


def _validate_metadata(
    metadata: Any,
    path: Path,
    document: dict[str, Any] | None = None,
    *,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: transformation metadata must be an object")
    required = {
        *IDENTITY_FIELDS,
        *GENERATION_FIELDS,
        "generation_statistics",
        "generation_datetime",
    }
    if not required <= metadata.keys():
        raise ValueError(f"{path}: missing transformation metadata fields")
    _generation_datetime(metadata, path)
    for name in IDENTITY_FIELDS[:-1]:
        if not isinstance(metadata[name], str) or not metadata[name].strip():
            raise ValueError(f"{path}: {name} must be nonempty text")
    for name in ("dataset", "transformation"):
        _plain_component(metadata[name], name)
    for name in ("seed_transform", *GENERATION_FIELDS):
        minimum = 1 if name in {"max_model_length", "max_tokens"} else 0
        if type(metadata[name]) is not int or metadata[name] < minimum:
            raise ValueError(f"{path}: invalid {name}")
    model = metadata["generator_model"]
    if Path(model).is_absolute() or model.startswith(("./", "../", "~")):
        raise ValueError(f"{path}: generator_model must not contain a local path")
    identity = {name: metadata[name] for name in IDENTITY_FIELDS}
    if path.parts[-5:-1] == (
        generator_directory(model),
        metadata["dataset"],
        metadata["transformation"],
        str(metadata["seed_transform"]),
    ):
        dated_identity = {**identity, "generation_datetime": metadata["generation_datetime"]}
        expected_hash = hash_settings(
            identity if metadata["generation_datetime"] is None else dated_identity
        )
        if path.name not in {"data.parquet", f"data_{expected_hash}.parquet"}:
            raise ValueError(
                f"{path}: metadata identity and datetime do not match its filename hash"
            )
    elif path.name == "data.parquet" or path.parent.name.isdigit():
        raise ValueError(f"{path}: transformation metadata does not match its folder")
    elif allow_legacy:
        if path.name != f"data_{hash_settings(identity)}.parquet":
            raise ValueError(f"{path}: metadata identity does not match its filename hash")
        if path.parts[-4:-1] != (
            safe_model_name(model),
            metadata["dataset"],
            metadata["transformation"],
        ):
            raise ValueError(f"{path}: historical metadata does not match its folder")
    else:
        raise ValueError(f"{path}: unsupported historical layout; expected a seed subfolder")
    if document is None:
        document = _read_document(metadata["dataset"])
    if "source_revision" in metadata:
        required.add("source_revision")
        _validate_source_revision(metadata["source_revision"])
    if "data_filtering" in metadata:
        required.add("data_filtering")
        validate_filtering_report(metadata["data_filtering"], document["task_type"])
    if "translation_language_selection" in metadata:
        required.add("translation_language_selection")
        if (
            metadata["transformation"] != "translation"
            or metadata["translation_language_selection"] != "dataset"
        ):
            raise ValueError(f"{path}: invalid translation_language_selection")
    if "max_tokens_expansion" in metadata:
        required.add("max_tokens_expansion")
        if (
            metadata["transformation"] != "summarised_expansion"
            or type(metadata["max_tokens_expansion"]) is not int
            or not 0 < metadata["max_tokens_expansion"] < metadata["max_model_length"]
        ):
            raise ValueError(f"{path}: invalid max_tokens_expansion")
    if document["task_type"] in {"retrieval", "reranking"}:
        required.add("transform_corpus")
        if type(metadata.get("transform_corpus")) is not bool:
            raise ValueError(f"{path}: transform_corpus must be a boolean")
    _validate_statistics(
        metadata["generation_statistics"],
        metadata["max_retries"],
    )
    if document["task_type"] == "clustering":
        required.add("clustering_format")
        if metadata.get("clustering_format") != document["dataset_metadata"]["clustering_format"]:
            raise ValueError(f"{path}: clustering_format differs from the dataset definition")
    if set(metadata) != required:
        raise ValueError(f"{path}: unexpected transformation metadata fields")
    return document


def _validate_source_revision(revision: Any) -> None:
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("source_revision must be the exact downloaded dataset commit")


def read_transformation_metadata(
    path: Path, *, allow_legacy: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return current-structure metadata without retired descriptive labels."""
    if not transformation_pair_exists(path):
        raise ValueError(f"{path}: transformation pair is missing")
    companion = transformation_metadata_path(path)
    try:
        metadata = json.loads(companion.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{companion}: invalid transformation metadata JSON") from error
    if isinstance(metadata, dict):
        metadata.pop("format_version", None)
        statistics = metadata.get("generation_statistics")
        if isinstance(statistics, dict):
            statistics.pop("source", None)
            statistics.pop("stages", None)
            for key in (
                "retried_text_fields",
                "recovered_after_retry_text_fields",
                "fallback_reasons",
            ):
                if key in statistics and statistics[key] is None:
                    statistics[key] = ""
    document = _validate_metadata(metadata, path, allow_legacy=allow_legacy)
    return metadata, document


def dataset_metadata_path(dataset: str) -> Path:
    """Use workspace JSON when present, otherwise the bundled dataset definition."""
    _plain_component(dataset, "dataset")
    for root in (
        Path.cwd() / "hteb" / "dataset_metadata",
        Path(__file__).with_name("dataset_metadata"),
    ):
        matches = [
            path
            for group in ("English", "Multilingual")
            if (path := root / group / f"{dataset}.JSON").exists() or path.is_symlink()
        ]
        if len(matches) > 1:
            raise ValueError(f"{dataset}: ambiguous dataset definitions in {root}")
        if matches:
            return matches[0]
    raise ValueError(f"{dataset}: missing dataset metadata JSON in English or Multilingual")


def _read_document(dataset: str) -> dict[str, Any]:
    path = dataset_metadata_path(dataset)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"{dataset}: missing or invalid dataset metadata JSON: {path}") from error
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported dataset metadata schema")
    if document.get("dataset") != dataset:
        raise ValueError(f"{path}: dataset identity does not match {dataset!r}")
    if not isinstance(document.get("task_type"), str) or not document["task_type"]:
        raise ValueError(f"{path}: missing task_type")
    if not isinstance(document.get("dataset_metadata"), dict):
        raise ValueError(f"{path}: dataset metadata must be an object")
    if "ignore_identical_ids" in document["dataset_metadata"]:
        if document["task_type"] != "retrieval":
            raise ValueError(f"{path}: ignore_identical_ids only applies to retrieval")
        if type(document["dataset_metadata"]["ignore_identical_ids"]) is not bool:
            raise ValueError(f"{path}: ignore_identical_ids must be a boolean")
    if document["task_type"] == "classification":
        for name in ("train_split", "eval_split"):
            split = document["dataset_metadata"].get(name)
            if not isinstance(split, str) or not split.strip():
                raise ValueError(f"{path}: classification {name} must be a nonempty string")
    if document["task_type"] == "clustering":
        format_name = document["dataset_metadata"].get("clustering_format")
        if not isinstance(format_name, str) or format_name not in CLUSTERING_FORMATS:
            raise ValueError(f"{path}: missing or invalid clustering_format metadata")
    if "variants" in document:
        raise ValueError(f"{path}: dataset definitions must not contain generated transformations")
    return document


def _write_document(path: Path, document: Mapping[str, Any], *, overwrite: bool = False) -> None:
    with path.open("w" if overwrite else "x", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_transformation(
    path: Path,
    *,
    expected_dataset: str | None = None,
    originals_only: bool = False,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Read once into lists of record dictionaries, with the dataset definition.

    Source discovery and generation can request only the original column. No open
    file handles or lazy accessors are retained in the returned dictionary.
    """
    path = path.expanduser().absolute()
    metadata, document = read_transformation_metadata(path, allow_legacy=allow_legacy)
    dataset = metadata["dataset"]
    if expected_dataset is not None and dataset != expected_dataset:
        raise ValueError(f"{path}: dataset does not match {expected_dataset!r}")
    column = f"{metadata['transformation']}_seed_{metadata['seed_transform']}"
    with pq.ParquetFile(path) as parquet:
        schema = parquet.schema_arrow
        if schema.metadata or schema.names != ["original", column]:
            raise ValueError(
                f"{path}: expected only original and {column} columns without metadata"
            )
        original_type = schema.field("original").type
        if not pa.types.is_struct(original_type) or schema.field(column).type != original_type:
            raise ValueError(
                f"{path}: original and transformed records must have the same struct type"
            )
        columns = ["original"] if originals_only else ["original", column]
        records = parquet.read(columns=columns).to_pydict()
    for name, rows in records.items():
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"{path}: {name} must contain record objects")
    if metadata["generation_statistics"]["records_total"] != len(records["original"]):
        raise ValueError(f"{path}: generation_statistics record count differs from Parquet")
    if "data_filtering" in metadata:
        _, _, observed = filter_data(records["original"], document["task_type"])
        filtering_report(
            {**metadata, "task_type": document["task_type"], "original": records["original"]},
            observed,
        )
    if document["task_type"] == "clustering":
        layout = document["dataset_metadata"]["clustering_format"]
        clustering_groups(records["original"], layout)
        if not originals_only:
            restored, _ = data_preprocessing(records["original"], records[column], "clustering")
            clustering_groups(restored, layout)
    return {
        "path": path,
        "dataset": dataset,
        "task_type": document["task_type"],
        "dataset_metadata": document["dataset_metadata"],
        "column": column,
        **metadata,
        "row_count": len(records["original"]),
        "original_type": original_type,
        "original": records["original"],
        **(
            {
                "evaluation_input": evaluation_input_details(
                    path, metadata, records["original"], document["task_type"]
                )
            }
            if not originals_only
            else {}
        ),
        **({"transformed": records[column]} if not originals_only else {}),
    }


def evaluation_input_details(
    path: Path, metadata: Mapping[str, Any], originals: list[dict[str, Any]], task_type: str
) -> dict[str, Any]:
    """Record portable saved-input identities without assigning unknown revisions."""

    def checksum(source: Path) -> str:
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    companion = transformation_metadata_path(path)
    depth = 5 if path.parent.name.isdigit() else 4
    details = {
        **{key: metadata[key] for key in IDENTITY_FIELDS},
        "generation_datetime": metadata["generation_datetime"],
        "source_revision": metadata.get("source_revision"),
        "parquet": Path(*path.parts[-depth:]).as_posix(),
        "metadata": Path(*companion.parts[-depth:]).as_posix(),
        "sha256": {"parquet": checksum(path), "metadata": checksum(companion)},
        "generation_settings": {key: metadata[key] for key in GENERATION_FIELDS},
    }
    if task_type == "retrieval":
        details["corpus_text_fields"] = [
            list(fields)
            for fields in sorted(
                {
                    retrieval_text_fields(record)
                    for record in originals
                    if record.get("_hteb_part", record.get("part")) == "corpus"
                }
            )
        ]
    return details


def load_original_records(path: Path, *, expected_dataset: str) -> dict[str, Any]:
    """Read originals from a transformation pair or a preserved original snapshot."""
    path = path.expanduser().absolute()
    if path.suffix != ".JSON":
        return load_transformation(path, expected_dataset=expected_dataset, originals_only=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError("original snapshot must be a regular file")
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset",
        "task_type",
        "dataset_metadata",
        "original",
    }
    if (
        not isinstance(snapshot, dict)
        or not required <= snapshot.keys()
        or (snapshot.keys() - required - {"source_revision", "input_scope"})
    ):
        raise ValueError("invalid original snapshot fields")
    if "source_revision" in snapshot:
        _validate_source_revision(snapshot["source_revision"])
    if "input_scope" in snapshot and snapshot["input_scope"] not in (
        "source_records",
        "stored_records",
    ):
        raise ValueError("invalid original snapshot input scope")
    if path.name != f"originals_{hash_settings(snapshot)}.JSON":
        raise ValueError("original snapshot content differs from its filename hash")
    if snapshot["dataset"] != expected_dataset:
        raise ValueError("original snapshot dataset differs from selection")
    definition = _read_document(expected_dataset)
    if any(snapshot[key] != definition[key] for key in ("task_type", "dataset_metadata")):
        raise ValueError("original snapshot differs from the dataset definition")
    records = snapshot["original"]
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(row, dict) for row in records)
    ):
        raise ValueError("original snapshot must contain nonempty records")
    return {
        **snapshot,
        "path": path,
        "row_count": len(records),
        "original_type": pa.array(records).type,
    }


def preserve_raw_originals(
    info: Mapping[str, Any],
    records: Sequence[Mapping[str, object]],
    *,
    root: str | Path,
) -> Path:
    """Save the unfiltered source when generation uses a filtered record selection."""
    document = {
        "dataset": info["dataset"],
        "task_type": info["task_type"],
        "dataset_metadata": info["dataset_metadata"],
        "original": list(records),
    }
    if info.get("source_revision") is not None:
        _validate_source_revision(info["source_revision"])
        document["source_revision"] = info["source_revision"]
    if "input_scope" in info:
        if info["input_scope"] not in ("source_records", "stored_records"):
            raise ValueError("invalid original snapshot input scope")
        document["input_scope"] = info["input_scope"]
    directory = Path(root).expanduser().absolute() / "raw_sources"
    check_directory(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"originals_{hash_settings(document)}.JSON"
    if path.is_symlink():
        raise ValueError("raw source snapshot must not be a symbolic link")
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A stopped direct write may leave incomplete JSON or a partial UTF-8 character.
            _write_document(path, document, overwrite=True)
            return path
        if previous != document:
            raise ValueError("raw source snapshot differs from its recorded content")
    else:
        _write_document(path, document)
    return path


def write_dataset_file(
    path: Path,
    *,
    dataset: str,
    task_type: str,
    original_records: Sequence[Mapping[str, object]],
    variant: Mapping[str, Any],
    original_type: Any = None,
    generator_model: str,
    generation_settings: Mapping[str, Any],
    source_revision: str | None = None,
    generation_datetime: str | None = None,
    data_filtering: Mapping[str, Any] | None = None,
) -> Path:
    """Write a completed transformation directly, without rereading or checking its data."""
    generated_at = generation_datetime or dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    column = f"{variant['transformation']}_seed_{variant['seed_transform']}"
    identity = transformation_identity(
        generator_model, dataset, variant["transformation"], variant["seed_transform"]
    )
    dated_identity = {**identity, "generation_datetime": generated_at}
    destination = (
        path.expanduser().absolute().with_name(f"data_{hash_settings(dated_identity)}.parquet")
    )
    document = _read_document(dataset)
    statistics = variant.get("generation_statistics")
    if statistics is None:
        statistics = stored_generation_statistics(
            original_records,
            variant["records"],
            task_type,
            transform_corpus=generation_settings.get("transform_corpus", True),
        )
    if isinstance(statistics, dict):
        statistics = {
            key: ""
            if key
            in {
                "retried_text_fields",
                "recovered_after_retry_text_fields",
                "fallback_reasons",
            }
            and value is None
            else value
            for key, value in statistics.items()
            if key not in {"source", "stages"}
        }
    metadata = {
        **dated_identity,
        **{name: generation_settings[name] for name in GENERATION_FIELDS},
        "generation_statistics": statistics,
    }
    if data_filtering is not None:
        validate_filtering_report(data_filtering, task_type)
        if data_filtering["counts_after"]["records"] != len(original_records):
            raise ValueError("filtering report differs from saved row count")
        metadata["data_filtering"] = dict(data_filtering)
    if source_revision is not None:
        metadata["source_revision"] = source_revision
    if "translation_language_selection" in variant:
        metadata["translation_language_selection"] = variant["translation_language_selection"]
    if "max_tokens_expansion" in variant:
        metadata["max_tokens_expansion"] = variant["max_tokens_expansion"]
    if task_type in {"retrieval", "reranking"}:
        metadata["transform_corpus"] = generation_settings["transform_corpus"]
    if task_type == "clustering":
        metadata["clustering_format"] = document["dataset_metadata"]["clustering_format"]
    companion = transformation_metadata_path(destination)
    check_directory(destination.parent)
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError("generated Parquet target must be a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    original_array = pa.array([dict(record) for record in original_records], type=original_type)
    transformed_array = pa.array(
        [dict(record) for record in variant["records"]], type=original_array.type
    )
    table = pa.Table.from_arrays([original_array, transformed_array], names=["original", column])
    # Exclusive creation prevents overwrites, including a same-second collision.
    # A failed write remains visible as incomplete; there is no rollback.
    if destination.exists() or companion.exists() or companion.is_symlink():
        raise FileExistsError(f"transformation filename already exists: {destination}")
    with destination.open("xb") as handle:
        pq.write_table(
            table,
            handle,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
            version="2.6",
        )
    _write_document(companion, metadata)
    return destination
