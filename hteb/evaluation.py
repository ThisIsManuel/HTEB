"""Task validation and metrics using ordinary records and library models."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeGuard, cast

from .dataset_preparation import (
    clustering_groups,
    normalize_pair_records,
    plain_record,
    retrieval_structure,
    retrieval_text,
    text_fields,
)
from .embedding import resolve_embedding_prompt
from .util import as_float, as_int

if TYPE_CHECKING:
    from .parallel_evaluation import EmbeddingPool


def encode_texts(
    model: Any,
    texts: Sequence[str],
    *,
    task_type: str,
    model_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    prompt_text: str,
    pool: EmbeddingPool | None = None,
) -> tuple[Any, int]:
    import numpy as np

    if not texts:
        return np.empty((0, 0), dtype=float), 0
    if pool is not None:
        return pool.encode(
            texts,
            task_type=task_type,
            model_config=model_config,
            evaluation_config=evaluation_config,
            prompt_text=prompt_text,
        )
    kwargs = dict(model_config.get("encode_kwargs", {}).get(task_type, {}))
    requested_batch = as_int(
        model_config.get("batch_size") or evaluation_config.get("batch_size", 4096)
    )
    kwargs.update(
        {
            "batch_size": requested_batch,
            "normalize_embeddings": bool(model_config.get("normalize_embeddings", False)),
            "show_progress_bar": False,
        }
    )
    kwargs["prompt"] = prompt_text
    chunk_size = as_int(
        model_config.get("encoding_chunk_size")
        or evaluation_config.get("encoding_chunk_size", 4096)
    )
    batch_size = requested_batch
    while True:
        kwargs["batch_size"] = batch_size
        try:
            with warnings.catch_warnings():
                # Remote model code still uses this deprecated Transformers helper.
                warnings.filterwarnings(
                    "ignore",
                    message=r"The attention mask API under `transformers\.modeling_attn_mask_utils` .* is deprecated",
                    category=FutureWarning,
                    module=r"transformers\.modeling_attn_mask_utils$",
                )
                chunks = [
                    model.encode(list(texts[start : start + chunk_size]), **kwargs)
                    for start in range(0, len(texts), chunk_size)
                ]
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover - GPU extra supplies torch
                pass
            continue
        return np.concatenate(chunks, axis=0), batch_size


def _cosine(left: Any, right: Any) -> Any:
    import numpy as np

    left_norm = left / np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-12)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-12)
    return np.sum(left_norm * right_norm, axis=-1)


def _classification(
    records: Sequence[Mapping[str, object]],
    encode: Callable[..., Any],
    *,
    seed_evaluation: int,
    train_split: str,
    eval_split: str,
) -> tuple[str, dict[str, float]]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    train = [record for record in records if _record_split(record) == train_split.strip()]
    test = [record for record in records if _record_split(record) == eval_split.strip()]
    train_texts = [_text(record) for record in train]
    test_texts = [_text(record) for record in test]
    train_labels = np.asarray([_label(record) for record in train])
    test_labels = np.asarray([_label(record) for record in test])
    classifier = LogisticRegression(random_state=seed_evaluation, n_jobs=1, max_iter=100)
    classifier.fit(encode(train_texts), train_labels)
    predictions = classifier.predict(encode(test_texts))
    accuracy = float(accuracy_score(test_labels, predictions))
    return "accuracy", {
        "accuracy": accuracy,
        "f1_macro": float(f1_score(test_labels, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(test_labels, predictions, average="weighted", zero_division=0)
        ),
    }


def _clustering(
    records: Sequence[Mapping[str, object]],
    encode: Callable[..., Any],
    *,
    seed_evaluation: int,
    evaluation_config: Mapping[str, Any],
) -> tuple[str, dict[str, float]]:
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import v_measure_score

    scores: list[float] = []
    for group in records:
        sentences = [str(value) for value in cast(Sequence[object], group["sentences"])]
        labels = [str(value) for value in cast(Sequence[object], group["labels"])]
        embeddings = encode(sentences)
        clusterer = MiniBatchKMeans(
            n_clusters=len(set(labels)),
            batch_size=as_int(evaluation_config.get("clustering_batch_size", 500)),
            n_init="auto",
            random_state=seed_evaluation,
        ).fit(embeddings)
        scores.append(float(v_measure_score(labels, clusterer.labels_)))
    return "v_measure", {
        "v_measure": float(np.mean(scores)),
        "v_measure_std": float(np.std(scores)),
        "v_measure_min": float(np.min(scores)),
        "v_measure_max": float(np.max(scores)),
        "n_groups": float(len(scores)),
    }


def _pair_classification(
    records: Sequence[Mapping[str, object]],
    encode: Callable[..., Any],
) -> tuple[str, dict[str, float]]:
    from sklearn.metrics import average_precision_score

    left, right = _pairs(records)
    labels = [_label(record) for record in records]
    unique = list(dict.fromkeys([*left, *right]))
    unique_embeddings = encode(unique)
    by_text = {text: unique_embeddings[index] for index, text in enumerate(unique)}
    import numpy as np

    left_embeddings = np.asarray([by_text[text] for text in left])
    right_embeddings = np.asarray([by_text[text] for text in right])
    similarity = _cosine(left_embeddings, right_embeddings)
    score = float(average_precision_score(labels, similarity))
    return "avg_precision", {"cosine_ap": score, "avg_precision": score}


def _sts(
    records: Sequence[Mapping[str, object]],
    encode: Callable[..., Any],
) -> tuple[str, dict[str, float]]:
    from scipy.stats import spearmanr

    left, right = _pairs(records)
    scores = [as_float(_score(record)) for record in records]
    if len(scores) < 2:
        raise ValueError("STS evaluation requires at least two scored pairs")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("STS gold scores must all be finite")
    if len(set(scores)) < 2:
        raise ValueError("STS evaluation requires variation in gold scores")
    similarity = _cosine(encode(left), encode(right))
    similarities = [as_float(value) for value in similarity]
    if len(similarities) != len(scores):
        raise ValueError("STS similarities and gold scores have unequal lengths")
    if not all(math.isfinite(value) for value in similarities):
        raise ValueError("STS similarity predictions must all be finite")
    if len(set(similarities)) < 2:
        raise ValueError("STS evaluation requires variation in similarity predictions")
    correlation = float(spearmanr(similarities, scores).statistic)
    if not math.isfinite(correlation):
        raise ValueError("STS evaluation produced a non-finite Spearman result")
    return "cosine_spearman", {"cosine_spearman": correlation}


def _reranking(
    model: Any,
    records: Sequence[Mapping[str, object]],
    encode: Callable[..., Any],
) -> tuple[str, dict[str, float]]:
    import numpy as np
    from sklearn.metrics import average_precision_score

    sample_queries = [
        [str(value) for value in record["query"]]
        if isinstance(record["query"], Sequence) and not isinstance(record["query"], str)
        else [str(record["query"])]
        for record in records
    ]
    all_queries = [value for values in sample_queries for value in values]
    all_documents = [
        str(text)
        for record in records
        for text in [
            *cast(Sequence[object], record["positive"]),
            *cast(Sequence[object], record["negative"]),
        ]
    ]
    unique_queries = list(dict.fromkeys(all_queries))
    unique_documents = list(dict.fromkeys(all_documents))
    query_vectors = encode(unique_queries, role="query")
    document_vectors = encode(unique_documents, role="document")
    query_by_text = {text: query_vectors[index] for index, text in enumerate(unique_queries)}
    document_by_text = {
        text: document_vectors[index] for index, text in enumerate(unique_documents)
    }
    ap_scores: list[float] = []
    for queries, record in zip(sample_queries, records, strict=True):
        positives = [str(text) for text in cast(Sequence[object], record["positive"])]
        negatives = [str(text) for text in cast(Sequence[object], record["negative"])]
        documents = positives + negatives
        query_embeddings = np.asarray([query_by_text[text] for text in queries])
        doc_embeddings = np.asarray([document_by_text[text] for text in documents])
        raw_similarities = model.similarity(query_embeddings, doc_embeddings)
        if hasattr(raw_similarities, "detach"):
            raw_similarities = raw_similarities.detach().cpu().numpy()
        similarity_matrix = np.asarray(raw_similarities)
        similarities = (
            np.max(similarity_matrix, axis=0) if similarity_matrix.ndim > 1 else similarity_matrix
        )
        labels = [1] * len(positives) + [0] * len(negatives)
        average_precision = float(average_precision_score(labels, similarities))
        if not math.isfinite(average_precision):
            raise ValueError("reranking evaluation produced a non-finite average precision")
        ap_scores.append(average_precision)
    score = float(np.mean(ap_scores))
    if not math.isfinite(score):
        raise ValueError("reranking evaluation produced a non-finite MAP result")
    return "map", {"map": score}


def _retrieval(
    records: Sequence[Mapping[str, object]],
    encode: Callable[..., Any],
    *,
    ignore_identical_ids: bool,
) -> tuple[str, dict[str, float]]:
    import numpy as np
    import pytrec_eval

    corpus: dict[str, str] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    for record in records:
        part = str(record.get("_hteb_part", record.get("part", "")))
        identifier = str(record.get("_id", record.get("id", "")))
        if part == "corpus":
            corpus[identifier] = retrieval_text(record)
        elif part == "queries":
            queries[identifier] = retrieval_text(record)
        elif part == "qrels":
            query_id = str(record.get("query-id", record.get("query_id", record.get("qid", ""))))
            doc_id = str(record.get("corpus-id", record.get("doc_id", record.get("pid", ""))))
            qrels.setdefault(query_id, {})[doc_id] = as_int(
                record.get("score", record.get("relevance", 1))
            )
    if not corpus or not queries or not qrels:
        raise ValueError(
            "retrieval records must contain normalized corpus, queries, and qrels parts"
        )
    corpus_ids = list(corpus)
    query_ids = list(queries)
    corpus_embeddings = encode(list(corpus.values()), role="document")
    query_embeddings = encode(list(queries.values()), role="query")
    corpus_norm = corpus_embeddings / np.maximum(
        np.linalg.norm(corpus_embeddings, axis=1, keepdims=True), 1e-12
    )
    query_norm = query_embeddings / np.maximum(
        np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12
    )
    similarities = query_norm @ corpus_norm.T
    results: dict[str, dict[str, float]] = {}
    for query_index, query_id in enumerate(query_ids):
        # Python's sort is stable, preserving corpus order for exact ties.
        ranked = list(zip(corpus_ids, similarities[query_index], strict=True))
        ranked.sort(key=lambda item: item[1], reverse=True)
        if ignore_identical_ids:
            ranked = [item for item in ranked if item[0] != query_id]
        selected = ranked[:10]
        results[query_id] = {doc_id: float(score) for doc_id, score in selected}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    per_query = evaluator.evaluate(results)
    if not per_query:
        raise ValueError("pytrec_eval returned no retrieval query scores")
    score = round(
        sum(float(metrics["ndcg_cut_10"]) for metrics in per_query.values()) / len(per_query),
        5,
    )
    return "NDCG@10", {"NDCG@10": score}


def _summarisation(
    records: Sequence[Mapping[str, Any]],
    encode: Callable[..., Any],
) -> tuple[str, dict[str, float]]:
    import numpy as np
    from scipy.stats import spearmanr

    predicted: list[float] = []
    gold: list[float] = []
    for row in records:
        human_embeddings = encode(row["human_summaries"])
        machine_embeddings = encode(row["machine_summaries"])
        if len(human_embeddings) != len(row["human_summaries"]):
            raise ValueError("summarisation encoding returned the wrong number of human rows")
        if len(machine_embeddings) != len(row["machine_summaries"]):
            raise ValueError("summarisation encoding returned the wrong number of machine rows")
        for embedding in machine_embeddings:
            repeated = np.repeat(embedding[None, :], len(human_embeddings), axis=0)
            predicted.append(float(np.max(_cosine(repeated, human_embeddings))))
        gold.extend(row["relevance_scores"])
    if len(predicted) != len(gold):
        raise ValueError("summarisation predictions and relevance scores have unequal lengths")
    if not all(math.isfinite(value) for value in predicted):
        raise ValueError("summarisation similarity predictions must all be finite")
    if len(set(predicted)) < 2:
        raise ValueError("summarisation evaluation requires variation in similarity predictions")
    correlation = float(spearmanr(gold, predicted).statistic)
    if not math.isfinite(correlation):
        raise ValueError("summarisation evaluation produced a non-finite Spearman result")
    return "spearman", {"spearman": correlation}


def _text(record: Mapping[str, object]) -> str:
    (key,) = text_fields(record, "classification")
    value = record.get(key)
    if isinstance(value, str):
        return value
    raise ValueError("record contains no recognized text field")


def _label(record: Mapping[str, object]) -> object:
    for key in ("label", "labels", "target"):
        if key in record:
            return record[key]
    raise ValueError("record contains no label")


def _validate_classification_records(
    records: Sequence[Mapping[str, object]],
    train_split: str,
    eval_split: str,
) -> None:
    """Require records from distinct configured training and evaluation splits."""

    if not isinstance(train_split, str) or not train_split.strip():
        raise ValueError("classification train_split must be a nonempty string")
    train_split = train_split.strip()
    if not isinstance(eval_split, str) or not eval_split.strip():
        raise ValueError("classification eval_split must be a nonempty string")
    eval_split = eval_split.strip()
    if train_split == eval_split:
        raise ValueError("classification train_split and eval_split must differ")
    train = tuple(record for record in records if _record_split(record) == train_split)
    evaluation = tuple(record for record in records if _record_split(record) == eval_split)
    if not train:
        raise ValueError(f"classification training requires records from split {train_split!r}")
    if not evaluation:
        raise ValueError(f"classification evaluation requires records from split {eval_split!r}")
    import numpy as np

    labels = [_label(record) for record in (*train, *evaluation)]
    if any(
        not isinstance(label, (str, int, float))
        or (isinstance(label, float) and not math.isfinite(label))
        for label in labels
    ):
        raise ValueError("classification labels must be non-null scalar class values")
    if all(isinstance(label, (int, float)) for label in labels) and any(
        isinstance(label, float) and not label.is_integer() for label in labels
    ):
        raise ValueError("classification numeric labels must be discrete class values")
    # Match the array conversion used by LogisticRegression, including mixed scalar inputs.
    if len(np.unique(np.asarray(labels[: len(train)]))) < 2:
        raise ValueError("classification training requires at least two distinct labels")


def _record_split(record: Mapping[str, object]) -> str:
    return str(record.get("_hteb_split", record.get("split", ""))).strip()


def _required_value(
    value: object, path: str, *, text: bool = False, allow_empty_text: bool = False
) -> None:
    if _is_nonstring_sequence(value):
        if not value:
            raise ValueError(f"{path}: required list is empty")
        for index, item in enumerate(value):
            _required_value(item, f"{path}[{index}]", text=text, allow_empty_text=allow_empty_text)
        return
    if (
        value is None
        or (isinstance(value, str) and not value.strip() and not (text and allow_empty_text))
        or (isinstance(value, float) and not math.isfinite(value))
        or not isinstance(value, (str, int, float))
        or (text and not isinstance(value, str))
    ):
        raise ValueError(f"{path}: required {'text' if text else 'value'} is missing or invalid")


def validate_required_fields(
    records: Sequence[Mapping[str, object]],
    task_type: str,
    *,
    allow_missing_text: bool = False,
    allow_empty_text: bool = False,
) -> None:
    """Reject missing original/non-text data before any restoration or normalization."""
    task = task_type.lower().replace("-", "_")
    for index, record in enumerate(records):
        location = f"{task} record {index}"
        if not record:
            raise ValueError(f"{location}: whole record is missing")
        if not allow_missing_text:
            if task == "retrieval":
                if record.get("_hteb_part", record.get("part")) in {"corpus", "queries"}:
                    is_corpus = record.get("_hteb_part", record.get("part")) == "corpus"
                    if allow_empty_text or is_corpus:
                        for key in text_fields(record, task):
                            if is_corpus and record.get(key) is None:
                                continue
                            _required_value(
                                record.get(key),
                                f"{location}.{key}",
                                text=True,
                                allow_empty_text=True,
                            )
                    else:
                        _required_value(retrieval_text(record), f"{location}.text", text=True)
            else:
                for key in text_fields(record, task):
                    _required_value(
                        record.get(key),
                        f"{location}.{key}",
                        text=True,
                        allow_empty_text=allow_empty_text
                        or (task == "reranking" and key in {"positive", "negative"}),
                    )
                if task == "classification" and not isinstance(
                    record.get(text_fields(record, task)[0]), str
                ):
                    raise ValueError(f"{location}: classification text must be a string")
        if task in {"classification", "clustering", "pair_classification", "pairclassification"}:
            labels = (
                _first_record_value(record, ("labels", "label"))
                if task == "clustering"
                else _label(record)
            )
            _required_value(labels, f"{location}.labels")
        if task == "classification":
            split = _first_record_value(record, ("_hteb_split", "split"))
            _required_value(split, f"{location}.split", text=True)
            if not isinstance(split, str):
                raise ValueError(f"{location}.split: expected a string")
        elif task == "reranking":
            for key in ("positive", "negative"):
                candidates = record.get(key)
                if not _is_nonstring_sequence(candidates) or not candidates:
                    raise ValueError(f"{location}.{key}: expected a nonempty candidate list")
        elif task in {"sts", "str"}:
            _required_value(_score(record), f"{location}.score")
        elif task in {"summarisation", "summarization"}:
            _required_value(
                _first_record_value(record, ("gold_scores", "scores", "relevance")),
                f"{location}.scores",
            )
    if task == "retrieval":
        corpus, _, judgments = retrieval_structure(records)
        for index, _, document in judgments:
            if document not in corpus:
                raise ValueError(
                    f"retrieval record {index}: relevance refers to a missing query/document"
                )


def prepare_records(
    source_records: Sequence[Mapping[str, object]],
    task_type: str,
    eval_split: str = "test",
    *,
    train_split: str = "",
    clustering_format: str | None = None,
    allow_empty_text: bool = False,
) -> list[dict[str, Any]]:
    """Check raw fields and normalize once before passing records to metrics."""

    validate_required_fields(source_records, task_type, allow_empty_text=allow_empty_text)
    records: list[dict[str, Any]] = [plain_record(record) for record in source_records]
    task = task_type.lower().replace("-", "_")
    if task == "classification":
        _validate_classification_records(records, train_split, eval_split)
        return records
    if task == "clustering":
        records = list(clustering_groups(records, clustering_format))
        if not records:
            raise ValueError("clustering evaluation received no groups")
        for group in records:
            sentences = list(cast(Sequence[object], group["sentences"]))
            labels = list(cast(Sequence[object], group["labels"]))
            if len(sentences) < 2 or len(set(str(label) for label in labels)) < 2:
                raise ValueError("each clustering group requires at least two samples and labels")
        return records
    if task in {"pair_classification", "pairclassification"}:
        records = normalize_pair_records(records)
        left, right = _pairs(records)
        if not left or len(left) != len(right):
            raise ValueError("pair classification evaluation requires paired records")
        for record in records:
            _label(record)
        return records
    if task in {"sts", "str"}:
        left, right = _pairs(records)
        if len(left) < 2 or len(left) != len(right):
            raise ValueError("STS evaluation requires at least two paired records")
        scores = [as_float(_score(record)) for record in records]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("STS gold scores must all be finite")
        if len(set(scores)) < 2:
            raise ValueError("STS evaluation requires variation in gold scores")
        return records
    if task == "reranking":
        if not records:
            raise ValueError("reranking evaluation requires at least one record")
        return records
    if task == "retrieval":
        parts = {str(record.get("_hteb_part", record.get("part", ""))) for record in records}
        if not {"corpus", "queries", "qrels"} <= parts:
            raise ValueError(
                "retrieval records must contain normalized corpus, queries, and qrels parts"
            )
        return records
    if task in {"summarisation", "summarization"}:
        records = list(_summarisation_inputs(records, allow_empty_text=allow_empty_text))
        return records
    raise ValueError(f"unsupported HTEB task type: {task_type}")


def _summarisation_inputs(
    records: Sequence[Mapping[str, object]],
    *,
    allow_empty_text: bool = False,
) -> tuple[dict[str, Any], ...]:
    if not records:
        raise ValueError("summarisation evaluation requires at least one record")
    prepared: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        human_key, machine_key = text_fields(record, "summarisation")
        human = _summary_texts(
            record, (human_key,), index, "human", allow_empty_text=allow_empty_text
        )
        machine = _summary_texts(
            record,
            (machine_key,),
            index,
            "machine",
            allow_empty_text=allow_empty_text,
        )
        raw_scores = _first_record_value(
            record,
            ("gold_scores", "scores", "relevance"),
        )
        if not _is_nonstring_sequence(raw_scores):
            raise ValueError(f"summarisation record {index} relevance scores must be a sequence")
        try:
            scores = tuple(as_float(value) for value in raw_scores)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"summarisation record {index} relevance scores must be numeric"
            ) from exc
        if len(machine) != len(scores):
            raise ValueError(
                f"summarisation record {index} has {len(machine)} machine summaries "
                f"but {len(scores)} relevance scores"
            )
        if not all(math.isfinite(value) for value in scores):
            raise ValueError(f"summarisation record {index} relevance scores must all be finite")
        prepared.append(
            dict(
                human_summaries=human,
                machine_summaries=machine,
                relevance_scores=scores,
            )
        )
    relevance_scores = [score for row in prepared for score in row["relevance_scores"]]
    if len(relevance_scores) < 2 or len(set(relevance_scores)) < 2:
        raise ValueError("summarisation evaluation requires at least two distinct relevance scores")
    return tuple(prepared)


def _summary_texts(
    record: Mapping[str, object],
    keys: Sequence[str],
    record_index: int,
    role: str,
    *,
    allow_empty_text: bool = False,
) -> tuple[str, ...]:
    raw = _first_record_value(record, keys)
    if not _is_nonstring_sequence(raw):
        raise ValueError(f"summarisation record {record_index} {role} summaries must be a sequence")
    values = tuple(raw)
    if not values:
        raise ValueError(f"summarisation record {record_index} requires nonempty {role} summaries")
    if any(
        not isinstance(value, str) or (not value.strip() and not allow_empty_text)
        for value in values
    ):
        raise ValueError(
            f"summarisation record {record_index} {role} summaries must be nonempty text"
        )
    return cast(tuple[str, ...], values)


def _first_record_value(record: Mapping[str, object], keys: Sequence[str]) -> object:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _score(record: Mapping[str, object]) -> object:
    for key in ("score", "scores", "similarity_score", "label"):
        if key in record:
            return record[key]
    raise ValueError("record contains no score")


def _pairs(records: Sequence[Mapping[str, object]]) -> tuple[list[str], list[str]]:
    left: list[str] = []
    right: list[str] = []
    for index, record in enumerate(records):
        left_key, right_key = text_fields(record, "sts")
        if left_key not in record or right_key not in record:
            raise ValueError(f"pair record {index} lacks recognized sentence fields")
        left_value = record[left_key]
        right_value = record[right_key]
        left_is_array = isinstance(left_value, Sequence) and not isinstance(
            left_value, (str, bytes)
        )
        right_is_array = isinstance(right_value, Sequence) and not isinstance(
            right_value, (str, bytes)
        )
        if left_is_array != right_is_array:
            raise ValueError(f"pair record {index} mixes scalar and array sentence fields")
        if left_is_array:
            left_values = cast(Sequence[object], left_value)
            right_values = cast(Sequence[object], right_value)
            if len(left_values) != len(right_values):
                raise ValueError(f"pair record {index} sentence arrays have unequal lengths")
            left.extend(str(value) for value in left_values)
            right.extend(str(value) for value in right_values)
        else:
            if not isinstance(left_value, str) or not isinstance(right_value, str):
                raise ValueError(f"pair record {index} sentence fields must be strings")
            left.append(left_value)
            right.append(right_value)
    return left, right


def _is_nonstring_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def evaluate_records(
    model: Any,
    records: list[dict[str, Any]],
    *,
    model_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    task_type: str,
    dataset_metadata: Mapping[str, Any],
    eval_split: str,
    seed_evaluation: int,
    train_split: str = "",
    pool: Any = None,
) -> dict[str, Any]:
    """Score records returned by prepare_records; metric outputs remain checked."""
    prompts: dict[str, str] = {}

    def encode(texts: Sequence[str], *, role: str = "text") -> Any:
        if role == "document" and task_type in {"retrieval", "reranking"}:
            texts = [text if text.strip() else "" for text in texts]
        if role not in prompts:
            prompts[role] = str(
                resolve_embedding_prompt(
                    model, task_type, role, prompt_names=model_config.get("prompt_names", {})
                )["text"]
            )
        embeddings, _ = encode_texts(
            model,
            texts,
            task_type=task_type,
            model_config=model_config,
            evaluation_config=evaluation_config,
            prompt_text=prompts[role],
            pool=pool,
        )
        return embeddings

    task = task_type.lower().replace("-", "_")
    if task == "classification":
        metric, metrics = _classification(
            records,
            encode,
            seed_evaluation=seed_evaluation,
            train_split=train_split,
            eval_split=eval_split,
        )
    elif task == "clustering":
        metric, metrics = _clustering(
            records, encode, seed_evaluation=seed_evaluation, evaluation_config=evaluation_config
        )
    elif task in {"pair_classification", "pairclassification"}:
        metric, metrics = _pair_classification(records, encode)
    elif task in {"sts", "str"}:
        metric, metrics = _sts(records, encode)
    elif task == "reranking":
        metric, metrics = _reranking(model, records, encode)
    elif task == "retrieval":
        metric, metrics = _retrieval(
            records,
            encode,
            ignore_identical_ids=bool(
                dataset_metadata.get("ignore_identical_ids", False)
                or evaluation_config.get("ignore_identical_ids", False)
            ),
        )
    elif task in {"summarisation", "summarization"}:
        metric, metrics = _summarisation(records, encode)
    else:
        raise ValueError(f"unsupported HTEB task type: {task_type}")
    return {"score": float(metrics[metric]), "main_metric": metric, "metrics": metrics}
