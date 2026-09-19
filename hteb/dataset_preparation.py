"""Shared text-field selection and task-record normalization.

Pair-classification records can contain parallel arrays or scalar rows.
Normalizing at the local-data boundary keeps evaluation on one row representation.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, TypeGuard

_LEFT_FIELDS = ("sentence1", "sent1", "sentence_1", "text_1")
_RIGHT_FIELDS = ("sentence2", "sent2", "sentence_2", "text_2")
_LABEL_FIELDS = ("label", "labels", "target")
_TEXT_FIELDS = {
    "classification": (("text", "sentence", "input", "title"),),
    "clustering": (("sentences", "text", "sentence"),),
    "sts": (_LEFT_FIELDS, _RIGHT_FIELDS),
    "str": (_LEFT_FIELDS, _RIGHT_FIELDS),
    "pair_classification": (_LEFT_FIELDS, _RIGHT_FIELDS),
    "pairclassification": (_LEFT_FIELDS, _RIGHT_FIELDS),
    "reranking": (("query",), ("positive",), ("negative",)),
    "summarisation": (("human_summaries", "human"), ("machine_summaries", "machine")),
}


def text_fields(record: Mapping[str, object], task_type: str) -> tuple[str, ...]:
    """Return actual field names in task-role order, preserving scoring precedence."""
    task = task_type.lower().replace("-", "_")
    if task == "retrieval":
        return retrieval_text_fields(record)
    if task == "summarization":
        task = "summarisation"
    if task not in _TEXT_FIELDS:
        raise ValueError(f"unsupported task type for text selection: {task_type!r}")
    return tuple(
        next(
            (
                key
                for key in aliases
                if (isinstance(record.get(key), str) if task == "classification" else key in record)
            ),
            aliases[0],
        )
        for aliases in _TEXT_FIELDS[task]
    )


def retrieval_text_fields(record: Mapping[str, object]) -> tuple[str, ...]:
    """Select the same retrieval fields for generation, validation and scoring."""
    part = str(record.get("_hteb_part", record.get("part", "")))
    if part == "corpus":
        if record.get("document") is not None:
            return ("document",)
        return tuple(key for key in ("title", "text") if record.get(key) is not None) or ("text",)
    if part == "queries":
        return ("query",) if record.get("query") is not None else ("text",)
    return ()


def retrieval_text(record: Mapping[str, object]) -> str:
    """Resolve usable text without converting nulls or non-text values to strings."""
    values = [record.get(key, "") for key in retrieval_text_fields(record)]
    if not all(isinstance(value, str) for value in values):
        return ""
    texts = [value for value in values if isinstance(value, str)]
    return texts[0] if len(texts) == 1 else " ".join(texts).strip()


def _text_slots(
    value: object, prefix: tuple[str | int, ...]
) -> list[tuple[tuple[str | int, ...], str]]:
    if isinstance(value, str):
        return [(prefix, value)]
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes, bytearray))):
        flattened: list[tuple[tuple[str | int, ...], str]] = []
        for index, item in enumerate(value):
            flattened.extend(_text_slots(item, (*prefix, index)))
        return flattened
    return []


def _transformation_fields(
    record: Mapping[str, object], task_type: str, *, transform_corpus: bool
) -> tuple[str, ...]:
    task = task_type.lower().replace("-", "_")
    fields = text_fields(record, task)
    if task == "retrieval":
        part = str(record.get("_hteb_part", record.get("part", "")))
        return fields if part == "queries" or transform_corpus else ()
    if task == "reranking" and (not transform_corpus):
        return fields[:1]
    return fields


def transformation_text_targets(
    records: Sequence[Mapping[str, object]],
    task_type: str,
    *,
    transform_corpus: bool,
) -> tuple[dict[str, Any], ...]:
    """Locate every text slot; nonempty ordinals retain their existing language selections."""
    targets: list[dict[str, Any]] = []
    nonempty_count = 0
    pair_task = task_type.lower().replace("-", "_") in {
        "sts",
        "str",
        "pair_classification",
        "pairclassification",
    }
    work = (
        (
            (record_index, record, text_fields(record, task_type)[role])
            for role in (0, 1)
            for record_index, record in enumerate(records)
        )
        if pair_task
        else (
            (record_index, record, field)
            for record_index, record in enumerate(records)
            for field in _transformation_fields(
                record, task_type, transform_corpus=transform_corpus
            )
        )
    )
    for record_index, record, field in work:
        if field not in record:
            continue
        for path, text in _text_slots(record[field], (field,)):
            nonempty = bool(text.strip())
            targets.append(
                {
                    "record_index": record_index,
                    "path": path,
                    "text": text,
                    "ordinal": nonempty_count if nonempty else None,
                }
            )
            nonempty_count += nonempty
    if not targets:
        raise ValueError(f"{task_type}: no nonempty transformable text was found")
    return tuple(targets)


def _first_field(record: Mapping[str, object], names: Sequence[str], description: str) -> str:
    key = next((name for name in names if name in record), None)
    if key is None:
        raise ValueError(f"pair record lacks a recognized {description} field")
    return key


def _is_array(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _text_structure(value: object) -> bool:
    return isinstance(value, str) or (_is_array(value) and all(_text_structure(v) for v in value))


def _missing_text(value: object) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    return _is_array(value) and (not value or any(_missing_text(item) for item in value))


def _restore_text(
    original: object, transformed: object, path: str, *, allow_empty_text: bool = False
) -> tuple[object, int]:
    if isinstance(original, str):
        if not original.strip():
            if allow_empty_text:
                if transformed is not None and not isinstance(transformed, str):
                    raise ValueError(f"{path}: transformed text must be a string")
                return original, 0
            raise ValueError(f"{path}: original text is empty")
        if transformed is None or (isinstance(transformed, str) and not transformed.strip()):
            return original, 1
        if not isinstance(transformed, str):
            raise ValueError(f"{path}: transformed text must be a string")
        return transformed, 0
    if not _is_array(original) or not original:
        raise ValueError(f"{path}: original text or text list is missing")
    missing_list = (
        transformed is None
        or (isinstance(transformed, str) and not transformed.strip())
        or (_is_array(transformed) and not transformed)
    )
    if missing_list:
        restored_items = [
            _restore_text(item, None, f"{path}[{index}]", allow_empty_text=allow_empty_text)
            for index, item in enumerate(original)
        ]
        return [item for item, _ in restored_items], sum(count for _, count in restored_items)
    if not _is_array(transformed):
        raise ValueError(f"{path}: transformed text must be a list")
    if not _missing_text(transformed) and not allow_empty_text:
        return transformed, 0
    if len(original) != len(transformed):
        raise ValueError(f"{path}: cannot restore text in unequally sized lists")
    values = []
    count = 0
    for index, (source, value) in enumerate(zip(original, transformed, strict=True)):
        restored, replaced = _restore_text(
            source, value, f"{path}[{index}]", allow_empty_text=allow_empty_text
        )
        values.append(restored)
        count += replaced
    return values, count


def retrieval_id(record: Mapping[str, object], *names: str) -> str:
    """Read a retrieval ID using the aliases shared by filtering and validation."""
    value = next((record[name] for name in names if name in record), None)
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"retrieval {names[0]}: invalid identifier")
    return str(value)


def retrieval_structure(
    records: Sequence[Mapping[str, object]],
) -> tuple[set[str], dict[str, int], list[tuple[int, str, str]]]:
    """Check IDs and judgments before exclusions; missing documents are returned."""
    corpus: set[str] = set()
    queries: dict[str, int] = {}
    judgments: list[tuple[int, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or not record:
            raise ValueError(f"retrieval record {index}: whole record is missing or invalid")
        part = record.get("_hteb_part", record.get("part"))
        if not isinstance(part, str):
            raise ValueError(f"retrieval record {index}.part: expected corpus, queries or qrels")
        if part in {"corpus", "queries"}:
            identifier = retrieval_id(record, "_id", "id")
            if identifier in (corpus if part == "corpus" else queries):
                raise ValueError(f"retrieval record {index}.id: duplicate identifier")
            if part == "corpus":
                corpus.add(identifier)
            else:
                queries[identifier] = index
        elif part == "qrels":
            query = retrieval_id(record, "query-id", "query_id", "qid")
            document = retrieval_id(record, "corpus-id", "doc_id", "pid")
            score = record.get("score", record.get("relevance"))
            try:
                if isinstance(score, bool) or not isinstance(score, (str, int, float)):
                    raise ValueError
                number = float(score)
                if not math.isfinite(number) or not number.is_integer():
                    raise ValueError
            except (ValueError, OverflowError) as error:
                raise ValueError(
                    f"retrieval record {index}.score: expected integer relevance"
                ) from error
            if (query, document) in seen:
                raise ValueError(f"retrieval record {index}: duplicate query/document judgment")
            seen.add((query, document))
            judgments.append((index, query, document))
        else:
            raise ValueError(f"retrieval record {index}.part: expected corpus, queries or qrels")
    for index, query, _ in judgments:
        if query not in queries:
            raise ValueError(
                f"retrieval record {index}: relevance refers to a missing query/document"
            )
    return corpus, queries, judgments


def _record_counts(records: Sequence[Mapping[str, object]], task_type: str) -> dict[str, int]:
    counts = {"records": len(records)}
    if task_type == "retrieval":
        parts = Counter(row.get("_hteb_part", row.get("part")) for row in records)
        counts.update({part: parts[part] for part in ("corpus", "queries", "qrels")})
    return counts


def filter_data(
    records: Sequence[Mapping[str, object]], task_type: str
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    """Select usable queries without changing corpus documents or candidate lists."""
    task = task_type.lower().replace("-", "_")
    excluded: list[dict[str, Any]] = []
    removed: set[int] = set()
    if task == "retrieval":
        corpus, queries, judgments = retrieval_structure(records)
        for index, record in enumerate(records):
            for field in retrieval_text_fields(record):
                if record.get(field) is None:
                    continue
                if not isinstance(record.get(field), str):
                    raise ValueError(f"retrieval record {index}.{field}: required text is invalid")
        documents: dict[str, list[str]] = {query: [] for query in queries}
        for _, query, document in judgments:
            documents[query].append(document)
        excluded_queries = set()
        for query, index in queries.items():
            missing = sorted(set(documents[query]) - corpus)
            blank = not retrieval_text(records[index]).strip()
            if blank or missing or not documents[query]:
                excluded_queries.add(query)
                removed.add(index)
                excluded.append(
                    {
                        "row_index": index,
                        "query_id": query,
                        "reason": (
                            "empty_text"
                            if blank
                            else "missing_documents"
                            if missing
                            else "no_judgments"
                        ),
                        "missing_document_ids": missing,
                    }
                )
        removed.update(index for index, query, _ in judgments if query in excluded_queries)
        if len(excluded_queries) == len(queries):
            raise ValueError("retrieval source has no usable queries after filtering")
    elif task == "reranking":
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"reranking record {index}: expected a record")
            if not _text_structure(record.get("query")):
                raise ValueError(f"reranking record {index}.query: required text is invalid")
            for key in ("positive", "negative"):
                candidates = record.get(key)
                if not _is_array(candidates):
                    raise ValueError(f"reranking record {index}.{key}: expected a candidate list")
                if any(not _text_structure(text) for text in candidates):
                    raise ValueError(f"reranking record {index}.{key}: candidate text is invalid")
            blank = not any(text.strip() for _, text in _text_slots(record["query"], ("query",)))
            if blank or not record["positive"] or not record["negative"]:
                removed.add(index)
                excluded.append(
                    {"row_index": index, "reason": "empty_text" if blank else "empty_candidates"}
                )
        if len(removed) == len(records):
            raise ValueError(
                "reranking source has no records with both positive and negative candidates"
            )
    kept = [index for index in range(len(records)) if index not in removed]
    if not kept:
        raise ValueError(f"{task}: no usable records after filtering")
    filtered: list[dict[str, Any]] = []
    for index in kept:
        row = records[index]
        filtered.append(row if isinstance(row, dict) else dict(row))
    report = {
        "version": 2,
        "task_type": task,
        "input_scope": "stored_records",
        "counts_before": _record_counts(records, task),
        "counts_after": _record_counts(filtered, task),
        "excluded": excluded,
        "excluded_documents": [],
        "excluded_candidates": [],
    }
    return filtered, kept, report


def select_filtered_transformation(
    originals: Sequence[Mapping[str, object]],
    transformed: Sequence[Mapping[str, object]],
    retained: Sequence[int],
    task_type: str,
) -> list[dict[str, Any]]:
    """Apply original row positions without inspecting untargeted transformed fields."""
    if len(originals) != len(transformed):
        raise ValueError("original and transformed record counts differ")
    return [dict(transformed[index]) for index in retained]


def filtering_changed(report: Mapping[str, Any]) -> bool:
    """Check whether the single filtering report records any exclusions."""
    return bool(
        report["excluded"] or report.get("excluded_documents") or report.get("excluded_candidates")
    )


def filtering_report(info: Mapping[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    """Retain saved source counts when the pair already contains filtered records."""
    saved = info.get("data_filtering")
    if saved is None:
        return observed
    validate_filtering_report(saved, info["task_type"])
    if saved["counts_after"] != observed["counts_before"] or filtering_changed(observed):
        raise ValueError("saved filtering report differs from stored originals")
    if info["task_type"] == "retrieval":
        corpus, queries, _ = retrieval_structure(info["original"])
        if any(item["query_id"] in queries for item in saved["excluded"]):
            raise ValueError("saved excluded query is still present in originals")
        if any(item["document_id"] in corpus for item in saved.get("excluded_documents", [])):
            raise ValueError("saved excluded document is still present in originals")
    elif info["task_type"] == "reranking" and saved["version"] == 2:
        removed = {item["row_index"] for item in saved["excluded"]}
        retained = [
            index for index in range(saved["counts_before"]["records"]) if index not in removed
        ]
        positions = {index: position for position, index in enumerate(retained)}
        for item in saved["excluded_candidates"]:
            if item["row_index"] in removed:
                continue
            record = info["original"][positions[item["row_index"]]]
            candidates = record.get(item["field"])
            if not _is_array(candidates) or len(candidates) != item["count"] - len(item["indices"]):
                raise ValueError("saved candidate exclusions differ from stored candidates")
    return dict(saved)


def validate_filtering_report(report: Any, task_type: str) -> None:
    """Validate a single filtering report and its recorded counts."""
    if isinstance(report, dict) and (
        type(report.get("version")) is not int or report["version"] not in {1, 2}
    ):
        raise ValueError("unsupported data filtering version or task")
    required = {"version", "task_type", "input_scope", "counts_before", "counts_after", "excluded"}
    if isinstance(report, dict) and report.get("version") == 2:
        required.update({"excluded_documents", "excluded_candidates"})
    if not isinstance(report, dict) or set(report) != required:
        raise ValueError("invalid data filtering report")
    if (
        type(report["version"]) is not int
        or report["version"] not in {1, 2}
        or report["task_type"] != task_type
    ):
        raise ValueError("unsupported data filtering version or task")
    if not isinstance(report["input_scope"], str) or report["input_scope"] not in {
        "source_records",
        "stored_records",
    }:
        raise ValueError("invalid filtering input scope")
    keys = {"records", "corpus", "queries", "qrels"} if task_type == "retrieval" else {"records"}
    for name in ("counts_before", "counts_after"):
        counts = report[name]
        if (
            not isinstance(counts, dict)
            or set(counts) != keys
            or any(type(n) is not int or n < 0 for n in counts.values())
        ):
            raise ValueError("invalid filtering counts")
        if task_type == "retrieval" and counts["records"] != sum(
            counts[k] for k in keys - {"records"}
        ):
            raise ValueError("inconsistent retrieval filtering counts")
    before, after = report["counts_before"], report["counts_after"]
    if any(after[k] > before[k] for k in keys) or after["records"] == 0:
        raise ValueError("inconsistent filtering counts")
    excluded = report["excluded"]
    if not isinstance(excluded, list):
        raise ValueError("invalid filtering exclusions")
    seen = set()
    query_ids = set()
    for item in excluded:
        fields = {"row_index", "reason"} | (
            {"query_id", "missing_document_ids"} if task_type == "retrieval" else set()
        )
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("invalid filtering exclusion")
        if not isinstance(item["reason"], str):
            raise ValueError("invalid exclusion reason")
        index = item["row_index"]
        if type(index) is not int or not 0 <= index < before["records"] or index in seen:
            raise ValueError("invalid excluded row index")
        seen.add(index)
        if task_type == "retrieval":
            missing = item["missing_document_ids"]
            if (
                not isinstance(item["query_id"], str)
                or not item["query_id"].strip()
                or not isinstance(missing, list)
                or any(not isinstance(i, str) or not i.strip() for i in missing)
            ):
                raise ValueError("invalid excluded retrieval IDs")
            if item["reason"] != ("missing_documents" if missing else "no_judgments") and not (
                report["version"] == 2 and item["reason"] == "empty_text"
            ):
                raise ValueError("invalid retrieval exclusion reason")
            if item["query_id"] in query_ids or len(set(missing)) != len(missing):
                raise ValueError("duplicate excluded retrieval IDs")
            query_ids.add(item["query_id"])
        elif task_type != "reranking" or item["reason"] not in (
            {"empty_candidates", "empty_text"} if report["version"] == 2 else {"empty_candidates"}
        ):
            raise ValueError("invalid exclusion reason")
    documents = report.get("excluded_documents", [])
    candidates = report.get("excluded_candidates", [])
    if not isinstance(documents, list) or not isinstance(candidates, list):
        raise ValueError("invalid document/candidate exclusions")
    document_ids = set()
    for item in documents:
        if (
            task_type != "retrieval"
            or not isinstance(item, dict)
            or set(item) != {"row_index", "document_id", "reason"}
        ):
            raise ValueError("invalid document exclusion")
        index, identifier = item["row_index"], item["document_id"]
        if (
            type(index) is not int
            or not 0 <= index < before["records"]
            or index in seen
            or not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in document_ids
            or item["reason"] != "empty_text"
        ):
            raise ValueError("invalid excluded document")
        seen.add(index)
        document_ids.add(identifier)
    candidate_fields = set()
    for item in candidates:
        if (
            task_type != "reranking"
            or not isinstance(item, dict)
            or set(item) != {"row_index", "field", "indices", "count"}
        ):
            raise ValueError("invalid candidate exclusion")
        index, field, positions, count = (
            item["row_index"],
            item["field"],
            item["indices"],
            item["count"],
        )
        if (
            type(index) is not int
            or not 0 <= index < before["records"]
            or not isinstance(field, str)
            or field not in {"positive", "negative"}
            or (index, field) in candidate_fields
            or type(count) is not int
            or count < 1
            or not isinstance(positions, list)
            or not positions
            or any(type(i) is not int or not 0 <= i < count for i in positions)
            or positions != sorted(set(positions))
        ):
            raise ValueError("invalid excluded candidate positions")
        candidate_fields.add((index, field))
    unit = "queries" if task_type == "retrieval" else "records"
    if before[unit] - after[unit] != len(excluded):
        raise ValueError("filtering exclusions differ from counts")
    if not excluded and not documents and before != after:
        raise ValueError("filtering counts changed without exclusions")
    if task_type == "retrieval" and before["corpus"] - after["corpus"] != len(documents):
        raise ValueError("filtering document exclusions differ from corpus counts")
    if task_type == "retrieval" and not excluded and before["qrels"] != after["qrels"]:
        raise ValueError("filtering judgments changed without query exclusions")


def filtering_summary(report: Mapping[str, Any]) -> str:
    """Describe the evaluated population, with an explicit stored/source boundary."""
    task = report["task_type"]
    unit = "queries" if task == "retrieval" else "records"
    before, after = report["counts_before"][unit], report["counts_after"][unit]
    summary = (
        f"{task}: {after}/{before} {unit}; {before - after} excluded ({report['input_scope']})"
    )
    if report.get("excluded_documents"):
        summary += f"; {len(report['excluded_documents'])} blank documents excluded"
    if report.get("excluded_candidates"):
        count = sum(len(item["indices"]) for item in report["excluded_candidates"])
        summary += f"; {count} blank candidates excluded"
    return summary


def validate_reranking_candidate_counts(
    originals: Sequence[Mapping[str, object]], transformed: Sequence[Mapping[str, object]]
) -> None:
    """Candidate membership must keep its size and relevance class after generation."""
    if len(originals) != len(transformed):
        raise ValueError("original and transformed record counts differ")
    for index, (source, record) in enumerate(zip(originals, transformed, strict=True)):
        for key in ("positive", "negative"):
            original, changed = source.get(key), record.get(key)
            if not _is_array(original) or not _is_array(changed):
                raise ValueError(f"reranking record {index}.{key}: expected a candidate list")
            if len(original) != len(changed):
                raise ValueError(
                    f"reranking record {index}.{key}: candidate count changed "
                    f"from {len(original)} to {len(changed)}"
                )


def stored_generation_statistics(
    original_records: Sequence[Mapping[str, object]],
    transformed_records: Sequence[Mapping[str, object]],
    task_type: str,
    *,
    transform_corpus: bool,
) -> dict[str, Any]:
    """Count visible targeted text outcomes without inventing generation history."""
    if len(original_records) != len(transformed_records):
        raise ValueError("original and transformed record counts differ")
    originals, retained, _ = filter_data(original_records, task_type)
    transformed = select_filtered_transformation(
        original_records, transformed_records, retained, task_type
    )
    targets = tuple(
        target
        for target in transformation_text_targets(
            originals, task_type, transform_corpus=transform_corpus
        )
        if target["ordinal"] is not None
    )
    fallback_records: set[int] = set()
    fallbacks = unchanged = 0
    for target in targets:
        value: Any = transformed[target["record_index"]]
        for part in target["path"]:
            if value is None or (isinstance(value, str) and not value.strip()):
                value = None
                break
            if isinstance(part, str) and isinstance(value, Mapping):
                value = value.get(part)
            elif isinstance(part, int) and _is_array(value):
                if not value:
                    value = None
                    break
                if part >= len(value):
                    raise ValueError("transformed text arrays do not match original text slots")
                value = value[part]
            else:
                raise ValueError("transformed text structure does not match original text slots")
        if value is not None and not isinstance(value, str):
            raise ValueError("transformed text slots must contain text or null")
        if value is None or not value.strip():
            fallbacks += 1
            fallback_records.add(target["record_index"])
        elif value == target["text"]:
            unchanged += 1
    return {
        "records_total": len(original_records),
        "records_targeted": len({target["record_index"] for target in targets}),
        "text_fields_total": len(targets),
        "records_with_fallback": len(fallback_records),
        "text_fields_with_fallback": fallbacks,
        "unchanged_outputs": unchanged,
        "retried_text_fields": "",
        "recovered_after_retry_text_fields": "",
        "fallback_reasons": "",
    }


def data_preprocessing(
    originals: Sequence[Mapping[str, object]],
    transformed: Sequence[Mapping[str, object]],
    task_type: str,
) -> tuple[list[dict[str, Any]], int]:
    """Restore missing transformed text in memory without changing either input."""
    if len(originals) != len(transformed):
        raise ValueError("original and transformed record counts differ")
    if task_type == "reranking":
        validate_reranking_candidate_counts(originals, transformed)
    records: list[dict[str, Any]] = []
    replacement_count = 0
    for index, (original, record) in enumerate(zip(originals, transformed, strict=True)):
        if not original or not record:
            raise ValueError(f"record {index}: a whole record is missing")
        prepared = record if isinstance(record, dict) else dict(record)
        for key in text_fields(original, task_type):
            source = original.get(key)
            corpus_field = (
                task_type == "retrieval"
                and original.get("_hteb_part", original.get("part")) == "corpus"
            )
            if corpus_field and source is None:
                continue
            allow_empty = corpus_field or (
                task_type == "reranking" and key in {"positive", "negative"}
            )
            value, count = _restore_text(
                source, record.get(key), f"record {index}.{key}", allow_empty_text=allow_empty
            )
            if count or (allow_empty and value != record.get(key)):
                if prepared is record:
                    prepared = dict(record)
                prepared[key] = value
                replacement_count += count
        records.append(prepared)
    return records, replacement_count


def _retrieval_value(record: Mapping[str, object], *names: str) -> object:
    values = [record[name] for name in names if record.get(name) is not None]
    if any(str(value) != str(values[0]) for value in values[1:]):
        raise ValueError(f"retrieval: conflicting aliases {names}")
    return values[0] if values else None


def _relevance_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("retrieval: expected integer relevance")
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError("retrieval: expected integer relevance")
        return int(number)
    except InvalidOperation as error:
        raise ValueError("retrieval: expected integer relevance") from error


def normalize_retrieval_records(
    records: Sequence[Mapping[str, object]], *, source: bool = True
) -> list[dict[str, Any]]:
    """Construct ordered evaluation records, checking source text aliases exactly."""
    corpus, _, judgments = retrieval_structure(records)
    if any(document not in corpus for _, _, document in judgments):
        raise ValueError("retrieval: relevance refers to a missing query/document")
    result: list[dict[str, Any]] = []
    for record in records:
        part = _retrieval_value(record, "_hteb_part", "part")
        normalized: dict[str, Any] = {"_hteb_part": part}
        split = _retrieval_value(record, "_hteb_split", "split")
        if split is not None:
            normalized["_hteb_split"] = split
        if record.get("language") is not None:
            normalized["language"] = record["language"]
        if part == "qrels":
            for target, aliases in (
                ("query-id", ("query-id", "query_id", "qid")),
                ("corpus-id", ("corpus-id", "doc_id", "pid")),
            ):
                normalized[target] = str(_retrieval_value(record, *aliases))
            scores = [
                _relevance_integer(record[key])
                for key in ("score", "relevance")
                if record.get(key) is not None
            ]
            if not scores or any(score != scores[0] for score in scores):
                raise ValueError("retrieval: conflicting or missing relevance")
            normalized["score"] = scores[0]
        else:
            normalized["_id"] = str(_retrieval_value(record, "_id", "id"))
            if part == "corpus":
                components = [
                    record[key] for key in ("title", "text") if record.get(key) is not None
                ]
                document = record.get("document")
                if document is None or source:
                    if any(not isinstance(value, str) for value in components):
                        raise ValueError("retrieval: corpus text must be a string")
                    joined = " ".join(
                        value.strip()
                        for value in components
                        if isinstance(value, str) and value.strip()
                    )
                    if len(components) == 1 and isinstance(components[0], str):
                        joined = components[0]
                    if source and document is not None and components and document != joined:
                        raise ValueError("retrieval: conflicting document and title/text aliases")
                    if document is None:
                        document = joined
                text = document
                field = "document"
            else:
                query = record.get("query")
                if (
                    source
                    and query is not None
                    and record.get("text") is not None
                    and query != record["text"]
                ):
                    raise ValueError("retrieval: conflicting query and text aliases")
                text = query if query is not None else record.get("text")
                field = "query"
            if not isinstance(text, str) or (part != "corpus" and not text.strip()):
                raise ValueError(f"retrieval: {field} must contain text")
            normalized[field] = text
        result.append(normalized)
    return result


@dataclass
class EvaluationData:
    original: list[dict[str, Any]]
    transformed: list[dict[str, Any]] | None
    retained_indices: list[int]
    data_filtering: dict[str, Any]
    replacements: int


def prepare_evaluation_data(info: Mapping[str, Any]) -> EvaluationData:
    """Filter and restore in each pair's native schema before evaluation conversion."""
    task = info["task_type"]
    original, retained, observed = filter_data(info["original"], task)
    report = filtering_report(info, observed)
    transformed = None
    replacements = 0
    if "transformed" in info:
        selected = select_filtered_transformation(
            info["original"], info["transformed"], retained, task
        )
        transformed, replacements = data_preprocessing(original, selected, task)
    if task == "retrieval":
        original = normalize_retrieval_records(original)
        if transformed is not None:
            transformed = normalize_retrieval_records(transformed, source=False)
            for before, after in zip(original, transformed, strict=True):
                identity = {
                    key: value for key, value in before.items() if key not in {"document", "query"}
                }
                candidate = {
                    key: value for key, value in after.items() if key not in {"document", "query"}
                }
                if identity != candidate:
                    raise ValueError(
                        "retrieval: transformed record identity or relevance differs from original"
                    )
    return EvaluationData(original, transformed, retained, report, replacements)


def normalize_pair_records(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return only the sentence pairs and labels consumed by the metric."""

    normalized: list[dict[str, object]] = []
    for record_index, record in enumerate(records):
        left_key, right_key = text_fields(record, "pair_classification")
        if left_key not in record or right_key not in record:
            raise ValueError("pair record lacks recognized sentence fields")
        label_key = _first_field(record, _LABEL_FIELDS, "label")
        left = record[left_key]
        right = record[right_key]
        labels = record[label_key]
        array_flags = (_is_array(left), _is_array(right), _is_array(labels))
        if any(array_flags) and not all(array_flags):
            raise ValueError(
                f"pair record {record_index} mixes scalar and array-valued pair columns"
            )
        if not any(array_flags):
            if not isinstance(left, str) or not isinstance(right, str):
                raise ValueError(f"pair record {record_index} sentence fields must be strings")
            normalized.append({"sentence1": left, "sentence2": right, "label": labels})
            continue

        if not (_is_array(left) and _is_array(right) and _is_array(labels)):
            raise AssertionError("array-valued pair columns were not narrowed consistently")
        left_values = list(left)
        right_values = list(right)
        label_values = list(labels)
        size = len(left_values)
        if len(right_values) != size or len(label_values) != size:
            raise ValueError(
                f"pair record {record_index} has unequal array lengths: "
                f"left={size}, right={len(right_values)}, labels={len(label_values)}"
            )
        for item_index, (left_item, right_item, label_item) in enumerate(
            zip(left_values, right_values, label_values, strict=True)
        ):
            if not isinstance(left_item, str) or not isinstance(right_item, str):
                raise ValueError(
                    f"pair record {record_index} item {item_index} sentences must be strings"
                )
            normalized.append(
                {"sentence1": left_item, "sentence2": right_item, "label": label_item}
            )
    return normalized


def plain_record(record: Mapping[str, object]) -> dict[str, Any]:
    """Drop historical sampling bookkeeping, retaining dataset fields."""
    return {
        key: value
        for key, value in record.items()
        if key not in {"_hteb_record_index", "_hteb_source_items"}
        and not key.startswith("_hteb_clustering_")
    }


CLUSTERING_FORMATS = {"flat", "nested", "per_language"}


def clustering_groups(
    records: Sequence[Mapping[str, object]],
    clustering_format: str | None,
) -> tuple[dict[str, object], ...]:
    """Normalize the declared clustering layout into independent problems.

    Flat monolingual datasets become one group, nested TwentyNewsgroups rows
    remain independent groups, and multilingual rows are flattened separately
    per language, matching the legacy HTEB evaluation protocol.
    """

    if not isinstance(clustering_format, str) or clustering_format not in CLUSTERING_FORMATS:
        raise ValueError(
            "clustering_format metadata must explicitly specify flat, nested, or per_language"
        )
    parsed: list[tuple[list[object], list[object], str | None]] = []
    for index, record in enumerate(records):
        (sentence_key,) = text_fields(record, "clustering")
        label_key = next((key for key in ("labels", "label") if key in record), None)
        if sentence_key not in record or label_key is None:
            raise ValueError(f"clustering record {index} lacks sentence/label fields")
        sentence_value = record[sentence_key]
        label_value = record[label_key]
        if _is_array(sentence_value):
            nested = True
            nested_sentences = list(sentence_value)
            if not _is_array(label_value):
                raise ValueError(
                    f"clustering record {index} has nested sentences but scalar labels"
                )
            nested_labels = list(label_value)
        else:
            nested = False
            if not isinstance(sentence_value, str):
                raise ValueError(f"clustering record {index} sentence must be text or a text array")
            nested_sentences = [sentence_value]
            nested_labels = [label_value]
        if len(nested_sentences) != len(nested_labels):
            raise ValueError(f"clustering record {index} sentences/labels have unequal lengths")
        language_value = record.get("language", record.get("lang"))
        language = str(language_value) if language_value is not None else None
        if (clustering_format == "flat" and nested) or (
            clustering_format == "nested" and not nested
        ):
            raise ValueError(
                f"clustering record {index} does not match clustering_format {clustering_format!r}"
            )
        if clustering_format == "per_language" and (not language or not language.strip()):
            raise ValueError(
                f"clustering record {index} requires a language for per_language format"
            )
        parsed.append((nested_sentences, nested_labels, language))

    language_order = tuple(
        dict.fromkeys(language for _, _, language in parsed if language is not None)
    )
    if clustering_format == "per_language":
        language_groups: list[dict[str, object]] = []
        for language in language_order:
            language_sentences: list[object] = []
            language_labels: list[object] = []
            for row_sentences, row_labels, row_language in parsed:
                if row_language == language:
                    language_sentences.extend(row_sentences)
                    language_labels.extend(row_labels)
            language_groups.append(
                {
                    "sentences": language_sentences,
                    "labels": language_labels,
                    "language": language,
                }
            )
        return tuple(language_groups)

    if clustering_format == "nested":
        return tuple(
            {
                "sentences": sentences,
                "labels": labels,
                **({"language": language} if language is not None else {}),
            }
            for sentences, labels, language in parsed
        )

    flat_sentences = [sentence for row, _, _ in parsed for sentence in row]
    flat_labels = [label for _, row, _ in parsed for label in row]
    group: dict[str, object] = {
        "sentences": flat_sentences,
        "labels": flat_labels,
    }
    if len(language_order) == 1:
        group["language"] = language_order[0]
    return (group,) if parsed else ()
