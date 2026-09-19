"""Generate selected text variants using functions and library model objects."""

from __future__ import annotations

import ast
import gc
import json
import logging
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any, NamedTuple

from tqdm import tqdm

from .config import (
    HTEBConfig,
    generation_job_groups,
    generation_settings,
    transformation_stages,
)
from .constants import MODELS_DIRECTORY
from .dataset_preparation import (
    filter_data,
    filtering_changed,
    filtering_report,
    filtering_summary,
    stored_generation_statistics,
    text_fields,
    transformation_text_targets,
    validate_reranking_candidate_counts,
)
from .prompts import generation_prompt
from .util import (
    capture_model_version,
    check_directory,
    check_local_model_conflict,
    hash_settings,
)

logger = logging.getLogger(__name__)
PathPart = str | int


class GenerationResult(NamedTuple):
    versions: list[dict[str, object]]
    paths: dict[tuple[str, str, int], Path]


class _GenerationDiagnostics:
    """Bounded file logging, independent of response validation and saved results."""

    def __init__(self, dataset: str, transformation: str, seed: int | None) -> None:
        self.dataset = dataset
        self.transformation = transformation
        self.seed = seed
        self.stage = "unknown"
        self.chunk_start = 0
        self.attempt = 0
        self.total = 0
        self.processed = 0
        self.token_limit_hits = 0
        self.started_at = time.monotonic()
        self.last_progress_at = self.started_at
        self.samples: Counter[str] = Counter()

    def context(self) -> str:
        return (
            f"dataset={self.dataset}, transformation={self.transformation}, seed={self.seed}, "
            f"stage={self.stage}, chunk_start={self.chunk_start}, attempt={self.attempt}"
        )

    def begin_attempt(self, attempt: int, total: int) -> None:
        self.attempt = attempt
        self.total = total
        self.processed = 0
        self.token_limit_hits = 0
        self.started_at = time.monotonic()

    def sample(self, reason: str, level: int, message: str, *args: object) -> None:
        if self.samples[reason] < 3:
            logger.log(level, message + "; %s", *args, self.context(), extra={"file_only": True})
            self.samples[reason] += 1

    def batch_completed(self, responses: Sequence[dict[str, Any]]) -> None:
        self.processed += len(responses)
        self.token_limit_hits += sum(
            response.get("finish_reason") == "length" for response in responses
        )
        now = time.monotonic()
        if now - self.last_progress_at >= 300 and self.processed < self.total:
            self.progress("running", now=now)
            self.last_progress_at = now

    def progress(self, status: str, *, now: float | None = None) -> None:
        elapsed = max(0.0, (time.monotonic() if now is None else now) - self.started_at)
        rate = self.processed / elapsed if elapsed else 0.0
        logger.info(
            "Generation progress %s: processed=%s/%s, elapsed_s=%.1f, requests_per_s=%.2f, "
            "token_limit_hits=%s, status=%s",
            self.context(),
            self.processed,
            self.total,
            elapsed,
            rate,
            self.token_limit_hits,
            status,
            extra={"file_only": True},
        )


# Scope batch observations to the current attempt without changing the backend callable.
# Each generation worker creates its own reporter; reset the context even on interruption.
_ACTIVE_DIAGNOSTICS: ContextVar[_GenerationDiagnostics | None] = ContextVar(
    "hteb_generation_diagnostics", default=None
)


_STRUCTURED_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"transformed_text": {"type": "string"}},
    "required": ["transformed_text"],
}
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_TEXT_FIELD = re.compile(r'"transformed_text"\s*:\s*("(?:\\.|[^"\\])*")')
_PUNCTUATION_ONLY = re.compile(r"^[\s.,;:!?\-_\'\"…·•*+()\[\]<>/\\|#@&^~`]+$")

_LANGUAGE_NAMES = {
    "af": "Afrikaans",
    "afr": "Afrikaans",
    "am": "Amharic",
    "amh": "Amharic",
    "ar": "Arabic",
    "ara": "Arabic",
    "arb": "Modern Standard Arabic",
    "arq": "Algerian Arabic",
    "ary": "Moroccan Arabic",
    "as": "Assamese",
    "asm": "Assamese",
    "az": "Azerbaijani",
    "aze": "Azerbaijani",
    "be": "Belarusian",
    "ben": "Bengali",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bul": "Bulgarian",
    "ca": "Catalan",
    "cat": "Catalan",
    "ces": "Czech",
    "cmn": "Mandarin Chinese",
    "cs": "Czech",
    "cy": "Welsh",
    "cym": "Welsh",
    "da": "Danish",
    "dan": "Danish",
    "de": "German",
    "deu": "German",
    "en": "English",
    "eng": "English",
    "es": "Spanish",
    "esp": "Spanish",
    "et": "Estonian",
    "est": "Estonian",
    "fa": "Persian",
    "fas": "Persian",
    "fi": "Finnish",
    "fin": "Finnish",
    "spa": "Spanish",
    "fr": "French",
    "fra": "French",
    "gu": "Gujarati",
    "guj": "Gujarati",
    "ha": "Hausa",
    "hau": "Hausa",
    "he": "Hebrew",
    "heb": "Hebrew",
    "hi": "Hindi",
    "hin": "Hindi",
    "hr": "Croatian",
    "hrv": "Croatian",
    "hu": "Hungarian",
    "hun": "Hungarian",
    "hye": "Armenian",
    "ibo": "Igbo",
    "ig": "Igbo",
    "id": "Indonesian",
    "ind": "Indonesian",
    "is": "Icelandic",
    "isl": "Icelandic",
    "it": "Italian",
    "ita": "Italian",
    "ja": "Japanese",
    "jav": "Javanese",
    "jpn": "Japanese",
    "ka": "Georgian",
    "kan": "Kannada",
    "kat": "Georgian",
    "khm": "Khmer",
    "kin": "Kinyarwanda",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "kor": "Korean",
    "lav": "Latvian",
    "lg": "Luganda",
    "lin": "Lingala",
    "lit": "Lithuanian",
    "ln": "Lingala",
    "lt": "Lithuanian",
    "lug": "Luganda",
    "lv": "Latvian",
    "mal": "Malayalam",
    "mar": "Marathi",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mon": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "msa": "Malay",
    "my": "Burmese",
    "mya": "Burmese",
    "nb": "Norwegian Bokmal",
    "ne": "Nepali",
    "nep": "Nepali",
    "nl": "Dutch",
    "nld": "Dutch",
    "no": "Norwegian",
    "nob": "Norwegian Bokmal",
    "nor": "Norwegian",
    "npi": "Nepali",
    "om": "Oromo",
    "or": "Odia",
    "orm": "Oromo",
    "ory": "Odia",
    "pa": "Punjabi",
    "pan": "Punjabi",
    "pcm": "Nigerian Pidgin",
    "pl": "Polish",
    "pol": "Polish",
    "por": "Portuguese",
    "pt": "Portuguese",
    "ro": "Romanian",
    "rom": "Romani",
    "ron": "Romanian",
    "ru": "Russian",
    "run": "Kirundi",
    "rus": "Russian",
    "rw": "Kinyarwanda",
    "sk": "Slovak",
    "sl": "Slovenian",
    "slk": "Slovak",
    "slv": "Slovenian",
    "sn": "Shona",
    "sna": "Shona",
    "som": "Somali",
    "sq": "Albanian",
    "sqi": "Albanian",
    "sr": "Serbian",
    "srp": "Serbian",
    "sv": "Swedish",
    "swa": "Swahili",
    "swe": "Swedish",
    "ta": "Tamil",
    "tam": "Tamil",
    "te": "Telugu",
    "tel": "Telugu",
    "tg": "Tagalog",
    "tgl": "Tagalog",
    "th": "Thai",
    "tha": "Thai",
    "ti": "Tigrinya",
    "tir": "Tigrinya",
    "tr": "Turkish",
    "tur": "Turkish",
    "uk": "Ukrainian",
    "ukr": "Ukrainian",
    "ur": "Urdu",
    "urd": "Urdu",
    "vi": "Vietnamese",
    "vie": "Vietnamese",
    "xh": "Xhosa",
    "xho": "Xhosa",
    "yo": "Yoruba",
    "yor": "Yoruba",
    "zh": "Chinese",
    "zho": "Chinese",
    "zu": "Zulu",
    "zul": "Zulu",
}
_TRANSLATION_LANGUAGES = ("Spanish", "French", "German", "Turkish", "Arabic")
_BACKTRANSLATION_LANGUAGES = (
    "English",
    "Spanish",
    "French",
    "German",
    "Turkish",
    "Arabic",
)


def _language_names(label: str | None) -> tuple[str, ...]:
    if not label:
        return ()
    names: list[str] = []
    for raw in re.split("[-_/]", label):
        token = raw.strip().lower()
        name = _LANGUAGE_NAMES.get(token)
        if name is not None and name not in names:
            names.append(name)
    return tuple(names)


def _source_language(label: str | None) -> str:
    names = _language_names(label)
    return " and ".join(names) if names else "the input language"


def _language_target(
    seed: int,
    ordinal: int,
    source_language: str | None,
    *,
    backtranslation: bool,
    dataset: str | None = None,
) -> str:
    choices = _BACKTRANSLATION_LANGUAGES if backtranslation else _TRANSLATION_LANGUAGES
    excluded = set(_language_names(source_language))
    options = tuple(name for name in choices if name not in excluded) or choices
    if dataset is not None:
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("translation requires a nonempty dataset name")
        choice = int(hash_settings({"dataset": dataset, "seed_transform": seed}), 16)
        return options[choice % len(options)]
    return random.Random(seed + ordinal).choice(options)


def _translation_target(
    target: Mapping[str, Any], transformation: str, seed: int, *, dataset: str | None = None
) -> str:
    shared = transformation == "translation"
    return _language_target(
        seed,
        0 if shared else target["ordinal"],
        target["translation_language" if shared else "language"],
        backtranslation=False,
        dataset=dataset if shared else None,
    )


def translation_languages(
    records: Sequence[Mapping[str, object]],
    task_type: str,
    transformation: str,
    seed: int,
    *,
    dataset_metadata: Mapping[str, object],
    transform_corpus: bool = True,
    retained_indices: Sequence[int] | None = None,
    dataset: str | None = None,
    translation_language_selection: str | None = None,
) -> list[str]:
    """List historical targets without renumbering source rows after filtering."""
    if translation_language_selection is not None:
        if translation_language_selection != "dataset" or transformation != "translation":
            raise ValueError("invalid translation_language_selection")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("translation requires a nonempty dataset name")
    else:
        dataset = None
    targets = extract_text_targets(
        records, task_type, transform_corpus=transform_corpus, dataset_metadata=dataset_metadata
    )
    targets = tuple(target for target in targets if target["ordinal"] is not None)
    if retained_indices is not None:
        retained = set(retained_indices)
        targets = tuple(target for target in targets if target["record_index"] in retained)
    return sorted(
        {_translation_target(target, transformation, seed, dataset=dataset) for target in targets}
    )


def _metadata_languages(metadata: Mapping[str, object] | None) -> tuple[str, ...]:
    if metadata is None:
        return ()
    raw = metadata.get("language_configs") or metadata.get("languages")
    if isinstance(raw, str):
        return (raw,) if raw.strip() else ()
    if isinstance(raw, Sequence) and (not isinstance(raw, (str, bytes, bytearray))):
        return tuple(
            str(value).strip() for value in raw if isinstance(value, str) and value.strip()
        )
    return ()


def _record_language(
    record: Mapping[str, object], dataset_languages: Sequence[str] = ()
) -> str | None:
    for key in ("language", "lang", "languages"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if len(dataset_languages) == 1:
        return dataset_languages[0]
    return None


def _field_language(
    record: Mapping[str, object], field: str, dataset_languages: Sequence[str] = ()
) -> str | None:
    label = _record_language(record, dataset_languages)
    left, right = text_fields(record, "sts")
    if field not in {left, right} or label is None:
        return label
    parts = tuple(part for part in re.split("[-_/]", label) if part)
    if len(parts) != 2:
        return label
    return parts[0] if field == left else parts[1]


def extract_text_targets(
    records: Sequence[Mapping[str, object]],
    task_type: str,
    *,
    transform_corpus: bool,
    dataset_metadata: Mapping[str, object] | None = None,
) -> tuple[dict[str, Any], ...]:
    targets = transformation_text_targets(records, task_type, transform_corpus=transform_corpus)
    dataset_languages = _metadata_languages(dataset_metadata)
    for target in targets:
        record = records[target["record_index"]]
        target["language"] = _field_language(record, target["path"][0], dataset_languages)
        target["translation_language"] = _record_language(record, dataset_languages)
    return targets


def _usable_text(value: object) -> str | None:
    cleaned = value.strip() if isinstance(value, str) else ""
    if not cleaned or cleaned.casefold() in {"[text]", "<text>", "placeholder", "n/a", "none"}:
        return None
    if _PUNCTUATION_ONLY.fullmatch(cleaned):
        return None
    return cleaned


def _recovered_structured_text(response: str) -> str | None:
    candidates: list[str] = []
    if response.startswith("```") and response.endswith("```"):
        lines = response.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]))
    object_start = response.find("{")
    object_end = response.rfind("}")
    if object_start >= 0 and object_end > object_start:
        candidates.append(response[object_start : object_end + 1])
    for candidate in dict.fromkeys(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
        if isinstance(value, Mapping):
            recovered = _usable_text(value.get("transformed_text"))
            if recovered is not None:
                return recovered
    match = _JSON_TEXT_FIELD.search(response)
    if match is not None:
        try:
            return _usable_text(json.loads(match.group(1)))
        except json.JSONDecodeError:
            return None
    return None


def _parsed_text(response: str, *, structured_output: bool) -> dict[str, Any]:
    cleaned = _THINK_PATTERN.sub("", response).strip()
    if not structured_output:
        return {"text": _usable_text(cleaned), "malformed_json": False, "recovered": False}
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as error:
        recovered = _recovered_structured_text(cleaned)
        return {
            "text": recovered,
            "malformed_json": True,
            "recovered": recovered is not None,
            "parse_error": str(error),
        }
    if isinstance(value, Mapping):
        return {
            "text": _usable_text(value.get("transformed_text")),
            "malformed_json": False,
            "recovered": False,
        }
    return {"text": None, "malformed_json": False, "recovered": False}


def _stage_requests(
    config: HTEBConfig,
    targets: Sequence[dict[str, Any]],
    texts: Sequence[str],
    stage: str,
    seed: int,
    *,
    dataset: str | None = None,
) -> tuple[dict[str, Any], ...]:
    requests: list[dict[str, Any]] = []
    settings = generation_settings(config, dataset, stage)
    for target, text in zip(targets, texts, strict=True):
        source = _source_language(target["language"])
        target_language = source
        if stage in {"translation", "cross_translation"}:
            target_language = _translation_target(target, stage, seed, dataset=dataset)
        elif stage == "backtranslate_forward":
            target_language = _language_target(
                seed, target["ordinal"], target["language"], backtranslation=True
            )
        elif stage == "backtranslate_backward":
            source = _language_target(
                seed, target["ordinal"], target["language"], backtranslation=True
            )
            target_language = _source_language(target["language"])
            if target_language == "the input language":
                target_language = "the original language"
        requests.append(
            {
                "text": text,
                "prompt": generation_prompt(
                    stage, source_language=source, target_language=target_language
                ),
                "seed": seed,
                **({"max_tokens": settings["max_tokens"]} if "max_tokens" in settings else {}),
            }
        )
    return tuple(requests)


def _generate_with_length_fallback(
    generate: Callable[[Sequence[dict[str, Any]]], Sequence[dict[str, Any]]],
    requests: Sequence[dict[str, Any]],
) -> Sequence[dict[str, Any]]:
    """Split only batches rejected for prompt length; other failures remain fatal."""
    diagnostics = _ACTIVE_DIAGNOSTICS.get()
    checkpoint = (diagnostics.processed, diagnostics.token_limit_hits) if diagnostics else (0, 0)
    try:
        responses = generate(requests)
    except ValueError as error:
        message = str(error).lower()
        length_limit = any(
            label in message
            for label in ("maximum model length", "max_model_len", "maximum context length")
        )
        too_long = any(label in message for label in ("exceed", "longer", "too long", "larger"))
        if (
            not length_limit
            or not too_long
            or (not any(label in message for label in ("prompt", "input", "context")))
        ):
            raise
        if diagnostics is not None:
            discarded = diagnostics.processed - checkpoint[0]
            diagnostics.processed, diagnostics.token_limit_hits = checkpoint
            if discarded:
                diagnostics.sample(
                    "discarded_before_split",
                    logging.WARNING,
                    "Context-length rejection: discarded %s processed responses before splitting; "
                    "progress reset to %s/%s",
                    discarded,
                    diagnostics.processed,
                    diagnostics.total,
                )
        if len(requests) <= 1:
            responses = ({"text": "", "failure_reason": "prompt_too_long"},)
            if diagnostics is not None:
                diagnostics.batch_completed(responses)
            return responses
        middle = len(requests) // 2
        return (
            *_generate_with_length_fallback(generate, requests[:middle]),
            *_generate_with_length_fallback(generate, requests[middle:]),
        )
    # Fake/custom backends may not report individual batches. Reconcile returned
    # responses without counting observations twice, including recursive splits.
    if diagnostics is not None:
        diagnostics.processed, diagnostics.token_limit_hits = checkpoint
        diagnostics.batch_completed(responses)
    return responses


def _generate_stage(
    generate: Callable[[Sequence[dict[str, Any]]], Sequence[dict[str, Any]]],
    requests: Sequence[dict[str, Any]],
    *,
    max_retries: int,
    structured_output: bool,
    diagnostics: _GenerationDiagnostics | None = None,
) -> tuple[dict[str, Any], ...]:
    if diagnostics is None:
        diagnostics = _GenerationDiagnostics(
            "unknown", "unknown", requests[0]["seed"] if requests else None
        )
    results: list[str | None] = [None] * len(requests)
    attempts = [0] * len(requests)
    malformed_attempts = [0] * len(requests)
    failure_reasons: list[str | None] = [None] * len(requests)
    pending = list(range(len(requests)))
    for attempt in range(max_retries + 1):
        if not pending:
            break
        diagnostics.begin_attempt(attempt + 1, len(pending))
        if attempt:
            logger.info(
                "Retrying %s invalid generation requests, attempt %s; %s",
                len(pending),
                attempt + 1,
                diagnostics.context(),
                extra={"file_only": True},
            )
        context_token = _ACTIVE_DIAGNOSTICS.set(diagnostics)
        try:
            responses = _generate_with_length_fallback(
                generate,
                tuple(
                    {**requests[index], "seed": requests[index]["seed"] + attempt}
                    for index in pending
                ),
            )
            if len(responses) != len(pending):
                raise RuntimeError("generation backend returned the wrong number of responses")
        except BaseException as error:
            diagnostics.progress(
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            )
            raise
        else:
            diagnostics.progress("complete")
        finally:
            _ACTIVE_DIAGNOSTICS.reset(context_token)
        next_pending: list[int] = []
        counts: Counter[str] = Counter()
        invalid_reasons: Counter[str] = Counter()
        for index, response in zip(pending, responses, strict=True):
            attempts[index] += 1
            if response["failure_reason"] is not None:
                failure_reasons[index] = (
                    "prompt_too_long"
                    if response["failure_reason"] == "prompt_too_long"
                    else "backend_failure"
                )
                counts["terminal_failures"] += 1
                invalid_reasons[str(failure_reasons[index])] += 1
                diagnostics.sample(
                    str(failure_reasons[index]),
                    logging.WARNING,
                    "Generation response %s: %s; leaving transformed field empty",
                    index,
                    failure_reasons[index],
                )
                continue
            parsed = _parsed_text(response["text"], structured_output=structured_output)
            reason = (
                "output_length_after_retries"
                if response.get("finish_reason") == "length"
                else "invalid_json_after_retries"
                if parsed["malformed_json"] and parsed["text"] is None
                else "invalid_text_after_retries"
                if parsed["text"] is None
                else "repaired_json"
            )
            if parsed["malformed_json"]:
                malformed_attempts[index] += 1
                counts["malformed_json"] += 1
                counts["repaired_json"] += bool(parsed["recovered"])
                diagnostics.sample(
                    reason,
                    logging.WARNING,
                    "Malformed JSON in generation response %s, attempt %s: %s; "
                    "%s; finish_reason=%s, output_tokens=%s, prompt_tokens=%s, response_chars=%s",
                    index,
                    attempt + 1,
                    "recovered" if parsed["recovered"] else "unrecovered",
                    parsed["parse_error"],
                    response.get("finish_reason", "unknown"),
                    response.get("output_tokens", "unknown"),
                    response.get("prompt_tokens", "unknown"),
                    len(response["text"]),
                )
            elif parsed["text"] is None or response.get("finish_reason") == "length":
                diagnostics.sample(
                    reason,
                    logging.WARNING,
                    "Generation response %s, attempt %s: %s; "
                    "finish_reason=%s, output_tokens=%s, prompt_tokens=%s, response_chars=%s",
                    index,
                    attempt + 1,
                    "no usable transformed text"
                    if parsed["text"] is None
                    else "output reached token limit",
                    response.get("finish_reason", "unknown"),
                    response.get("output_tokens", "unknown"),
                    response.get("prompt_tokens", "unknown"),
                    len(response["text"]),
                )
            if response.get("finish_reason") == "length":
                failure_reasons[index] = "output_length_after_retries"
                invalid_reasons["output_length_after_retries"] += 1
                next_pending.append(index)
            elif parsed["text"] is None:
                failure_reasons[index] = (
                    "invalid_json_after_retries"
                    if parsed["malformed_json"]
                    else "invalid_text_after_retries"
                )
                invalid_reasons[str(failure_reasons[index])] += 1
                next_pending.append(index)
            else:
                results[index] = parsed["text"]
                counts["valid"] += 1
                counts["recovered_after_retry"] += attempt > 0
                if malformed_attempts[index] and (not parsed["recovered"]):
                    diagnostics.sample(
                        "recovered_after_retry",
                        logging.INFO,
                        "Recovered malformed JSON by retry for response %s, attempt %s",
                        index,
                        attempt + 1,
                    )
                elif attempt:
                    diagnostics.sample(
                        "recovered_after_retry",
                        logging.INFO,
                        "Generation response %s recovered after retry, attempt %s",
                        index,
                        attempt + 1,
                    )
        retrying = len(next_pending) if attempt < max_retries else 0
        logger.log(
            logging.WARNING if invalid_reasons or counts["malformed_json"] else logging.INFO,
            "Generation validation %s: processed=%s, valid=%s, retrying=%s, failed=%s, "
            "token_limit_hits=%s, malformed_json=%s, repaired_json=%s, recovered_after_retry=%s, "
            "failure_reasons=%s",
            diagnostics.context(),
            len(pending),
            counts["valid"],
            retrying,
            counts["terminal_failures"] + len(next_pending) - retrying,
            diagnostics.token_limit_hits,
            counts["malformed_json"],
            counts["repaired_json"],
            counts["recovered_after_retry"],
            dict(invalid_reasons),
            extra={"file_only": True},
        )
        pending = next_pending
    return tuple(
        {
            "text": text,
            "attempt_count": attempts[index],
            "failure_reason": failure_reasons[index] if text is None else None,
        }
        for index, text in enumerate(results)
    )


def generate_transformation(
    config: HTEBConfig,
    generate: Callable[[Sequence[dict[str, Any]]], Sequence[dict[str, Any]]],
    targets: Sequence[dict[str, Any]],
    texts: Sequence[str],
    stage: str,
    seed: int,
    *,
    dataset: str | None = None,
    diagnostics: _GenerationDiagnostics | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return empty outputs directly; prompt and retry only nonempty inputs in batches."""
    if len(targets) != len(texts):
        raise ValueError("generation targets and input texts must have the same length")
    outputs = [{"text": "", "attempt_count": 0, "failure_reason": None} for _ in texts]
    pending = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"generation input {index}: text must be a string")
        if text.strip():
            pending.append(index)
    if len(pending) != len(texts):
        logger.info(
            "Skipped %s empty generation inputs without backend calls",
            len(texts) - len(pending),
            extra={"file_only": True},
        )
    if diagnostics is None:
        diagnostics = _GenerationDiagnostics(dataset or "unknown", stage, seed)
    diagnostics.stage = stage
    for start in range(0, len(pending), config.generation["chunk_size"]):
        positions = pending[start : start + config.generation["chunk_size"]]
        diagnostics.chunk_start = start
        requests = _stage_requests(
            config,
            tuple(targets[index] for index in positions),
            tuple(texts[index] for index in positions),
            stage,
            seed,
            dataset=dataset,
        )
        logger.info(
            "Generating %s seed %s: stage=%s, chunk_start=%s, text_fields=%s",
            diagnostics.transformation,
            seed,
            stage,
            start,
            len(requests),
            extra={"file_only": True},
        )
        outcomes = _generate_stage(
            generate,
            requests,
            max_retries=config.generation["max_retries"],
            structured_output=config.generation["structured_output"],
            diagnostics=diagnostics,
        )
        for index, outcome in zip(positions, outcomes, strict=True):
            outputs[index] = outcome
    return tuple(outputs)


def generate_dataset_transformation(
    config: HTEBConfig,
    generate: Callable[[Sequence[dict[str, Any]]], Sequence[dict[str, Any]]],
    records: Sequence[Mapping[str, object]],
    task_type: str,
    transformation: str,
    seed: int,
    dataset_metadata: Mapping[str, object] | None = None,
    *,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Generate one selected transformation and seed using a supplied backend."""
    if transformation == "translation" and (not isinstance(dataset, str) or not dataset.strip()):
        raise ValueError("translation requires a nonempty dataset name")
    if task_type == "reranking":
        from .evaluation import prepare_records

        prepare_records(records, task_type)
    targets = extract_text_targets(
        records,
        task_type,
        transform_corpus=config.generation["transform_corpus"],
        dataset_metadata=dataset_metadata,
    )
    # Preserve original blank slots exactly; only nonblank sources need replacements.
    active_targets = tuple(target for target in targets if target["ordinal"] is not None)
    active_texts = tuple(target["text"] for target in active_targets)
    failed_targets: list[dict[str, Any]] = []
    retried_targets: set[int] = set()
    final_reasons: Counter[str] = Counter()
    diagnostics = _GenerationDiagnostics(dataset or "unknown", transformation, seed)
    for stage in transformation_stages(transformation):
        stage_responses = generate_transformation(
            config,
            generate,
            active_targets,
            active_texts,
            stage,
            seed,
            dataset=dataset,
            diagnostics=diagnostics,
        )
        next_targets: list[dict[str, Any]] = []
        next_texts: list[str] = []
        reasons: Counter[str] = Counter()
        attempted = 0
        for target, outcome in zip(active_targets, stage_responses, strict=True):
            if outcome["attempt_count"]:
                diagnostics.chunk_start = (
                    attempted // config.generation["chunk_size"] * config.generation["chunk_size"]
                )
                attempted += 1
            if outcome["attempt_count"] > 1:
                retried_targets.add(target["ordinal"])
            if outcome["text"] is None:
                reasons[outcome["failure_reason"]] += 1
                diagnostics.attempt = outcome["attempt_count"]
                diagnostics.sample(
                    outcome["failure_reason"],
                    logging.WARNING,
                    "%s row %s: leaving transformed field empty after %s attempts; reason=%s",
                    stage,
                    target["record_index"],
                    outcome["attempt_count"],
                    outcome["failure_reason"],
                )
                failed_targets.append(target)
            else:
                next_targets.append(target)
                next_texts.append(outcome["text"])
        final_reasons.update(reasons)
        active_targets = tuple(next_targets)
        active_texts = tuple(next_texts)
        if not active_targets:
            break
    replacements: list[dict[tuple[PathPart, ...], str]] = [{} for _ in records]
    for target, text in zip(active_targets, active_texts, strict=True):
        replacements[target["record_index"]][target["path"]] = text
    for target in failed_targets:
        replacements[target["record_index"]][target["path"]] = ""
    transformed = [
        replace_text_values(record, changes)
        for record, changes in zip(records, replacements, strict=True)
    ]
    if task_type == "reranking":
        validate_reranking_candidate_counts(records, transformed)
    statistics = stored_generation_statistics(
        records, transformed, task_type, transform_corpus=config.generation["transform_corpus"]
    )
    statistics.update(
        retried_text_fields=len(retried_targets),
        recovered_after_retry_text_fields=len(
            retried_targets & {target["ordinal"] for target in active_targets}
        ),
        fallback_reasons=dict(final_reasons),
    )
    logger.log(
        logging.WARNING if failed_targets else logging.INFO,
        "Generation outcome %s seed %s: text_fields_total=%s, original_text_replacements=%s, "
        "failed_all_attempts=%s, recovered_after_retry=%s, fallback_reasons=%s",
        transformation,
        seed,
        statistics["text_fields_total"],
        statistics["text_fields_with_fallback"],
        final_reasons["invalid_json_after_retries"]
        + final_reasons["invalid_text_after_retries"]
        + final_reasons["output_length_after_retries"],
        statistics["recovered_after_retry_text_fields"],
        dict(final_reasons),
        extra={"file_only": True},
    )
    variant = {
        "transformation": transformation,
        "seed_transform": seed,
        "records": transformed,
        "generation_statistics": statistics,
    }
    if transformation == "translation":
        variant["translation_language_selection"] = "dataset"
    if transformation == "summarised_expansion" and "max_tokens" in config.generation:
        variant["max_tokens_expansion"] = generation_settings(config, dataset, "expansion")[
            "max_tokens"
        ]
    return variant


def _cleanup_vllm_state() -> None:
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory(shutdown_ray=False)


def _release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def replace_text_values(
    value: Any, replacements: Mapping[tuple[PathPart, ...], str], path: tuple[PathPart, ...] = ()
) -> Any:
    if path in replacements:
        return replacements[path]
    if isinstance(value, Mapping):
        return {
            key: replace_text_values(item, replacements, (*path, key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            replace_text_values(item, replacements, (*path, index))
            for index, item in enumerate(value)
        ]
    return value


def load_generator(config: HTEBConfig, seed_transform: int) -> Any:
    check_local_model_conflict(config.generator_model, "generator_model")
    from vllm import LLM

    generation = config.generation
    kwargs = {
        "model": config.generator_model,
        "download_dir": str(Path(MODELS_DIRECTORY).absolute()),
        "trust_remote_code": generation["trust_remote_code"],
        "dtype": generation["dtype"],
        "tensor_parallel_size": generation["tensor_parallel_size"],
        "gpu_memory_utilization": generation["gpu_memory_utilization"],
        "max_model_len": generation["max_model_length"],
        "seed": seed_transform,
    }
    if generation["quantization"] is not None:
        kwargs["quantization"] = generation["quantization"]
    try:
        return LLM(**kwargs)
    except BaseException as initialization_error:
        try:
            _cleanup_vllm_state()
        except Exception as cleanup_error:
            raise initialization_error from cleanup_error
        finally:
            _release_cuda_memory()
        raise


def generate_batch(
    model: Any,
    settings: Mapping[str, Any],
    requests: Sequence[dict[str, Any]],
    *,
    progress_description: str = "Generating transformations",
) -> list[dict[str, Any]]:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    responses: list[dict[str, Any]] = []
    for start in tqdm(
        range(0, len(requests), settings["batch_size"]),
        desc=progress_description,
        unit="batch",
        dynamic_ncols=True,
    ):
        batch = requests[start : start + settings["batch_size"]]
        if len({request["seed"] for request in batch}) != 1:
            raise ValueError("generation batch must have one sampling seed")
        output_limits = {request.get("max_tokens", settings["max_tokens"]) for request in batch}
        if len(output_limits) != 1:
            raise ValueError("generation batch must have one output allowance")
        sampling = {
            key: settings[key]
            for key in ("max_tokens", "temperature", "top_p", "top_k", "repetition_penalty")
        }
        sampling["seed"] = batch[0]["seed"]
        sampling["max_tokens"] = output_limits.pop()
        if settings["structured_output"]:
            sampling["structured_outputs"] = StructuredOutputsParams(json=_STRUCTURED_SCHEMA)
        messages = [
            [
                {"role": "system", "content": request["prompt"]},
                {"role": "user", "content": request["text"]},
            ]
            for request in batch
        ]
        outputs = model.chat(
            messages,
            sampling_params=SamplingParams(**sampling),
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        if len(outputs) != len(batch):
            raise RuntimeError("generator returned the wrong number of responses")
        for output in outputs:
            completion = output.outputs[0] if output.outputs else None
            output_tokens = getattr(completion, "token_ids", None)
            prompt_tokens = getattr(output, "prompt_token_ids", None)
            responses.append(
                {
                    "text": str(completion.text) if completion is not None else "",
                    "failure_reason": None,
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "output_tokens": len(output_tokens) if output_tokens is not None else None,
                    "prompt_tokens": len(prompt_tokens) if prompt_tokens is not None else None,
                }
            )
        diagnostics = _ACTIVE_DIAGNOSTICS.get()
        if diagnostics is not None:
            diagnostics.batch_completed(responses[-len(batch) :])
    return responses


def close_generator(model: Any) -> None:
    try:
        if model is not None:
            model.llm_engine.engine_core.shutdown()
    finally:
        try:
            _cleanup_vllm_state()
        finally:
            _release_cuda_memory()


def generate_jobs(
    config: HTEBConfig,
    jobs: list[tuple[str, str, int]],
    original_paths: Mapping[str, Any],
) -> GenerationResult:
    """Finish each settings group before loading the next generator."""
    versions: list[dict[str, object]] = []
    paths: dict[tuple[str, str, int], Path] = {}
    for group_config, group_jobs in generation_job_groups(config, jobs):
        result = _generate_jobs(group_config, group_jobs, original_paths)
        paths.update(result.paths)
        for version in result.versions:
            if version not in versions:
                versions.append(version)
    return GenerationResult(versions, paths)


def _generate_jobs(
    config: HTEBConfig,
    jobs: list[tuple[str, str, int]],
    original_paths: Mapping[str, Any],
) -> GenerationResult:
    """Reuse one seeded generator across its assigned datasets; save each variant."""
    from functools import partial

    from .benchmark import validate_originals
    from .transformation_parquet import (
        load_original_records,
        preserve_raw_originals,
        transformation_path,
        write_dataset_file,
    )

    versions: list[dict[str, object]] = []
    paths: dict[tuple[str, str, int], Path] = {}
    if jobs:
        check_local_model_conflict(config.generator_model, "generator_model")
    for dataset in dict.fromkeys(job[0] for job in jobs):
        info = load_original_records(original_paths[dataset], expected_dataset=dataset)
        validate_originals(info)
        del info
    for seed in dict.fromkeys(job[2] for job in jobs):
        model = load_generator(config, seed)
        primary_error: BaseException | None = None
        try:
            engine_config = getattr(getattr(model, "llm_engine", None), "model_config", None)
            generator_version = capture_model_version(
                config.generator_model,
                getattr(engine_config, "hf_config", None),
                model.get_tokenizer(),
            )
            if generator_version not in versions:
                versions.append(generator_version)
            for dataset in dict.fromkeys(job[0] for job in jobs if job[2] == seed):
                info = load_original_records(original_paths[dataset], expected_dataset=dataset)
                raw_records = info["original"]
                records, _, observed = filter_data(raw_records, info["task_type"])
                report = filtering_report(info, observed)
                report["input_scope"] = info.get("input_scope", report["input_scope"])
                if filtering_changed(observed):
                    preserve_raw_originals(info, raw_records, root=config.transformations_directory)
                logger.info("%s: %s", dataset, filtering_summary(report))
                for name, transformation, job_seed in jobs:
                    if name != dataset or job_seed != seed:
                        continue
                    settings = generation_settings(config, dataset, transformation)
                    job_config = config.model_copy(update={"generation": settings})
                    logger.info("Generating %s / %s seed %s", dataset, transformation, seed)
                    variant = generate_dataset_transformation(
                        job_config,
                        partial(
                            generate_batch,
                            model,
                            settings,
                            progress_description=f"{dataset} / {transformation} / seed {seed}",
                        ),
                        records,
                        info["task_type"],
                        transformation,
                        seed,
                        info["dataset_metadata"],
                        dataset=dataset,
                    )
                    target = transformation_path(
                        config.transformations_directory,
                        config.generator_model,
                        dataset,
                        transformation,
                        seed,
                    )
                    check_directory(target.parent)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    paths[dataset, transformation, seed] = write_dataset_file(
                        target,
                        dataset=dataset,
                        task_type=info["task_type"],
                        original_records=records,
                        variant=variant,
                        original_type=info["original_type"],
                        generator_model=config.generator_model,
                        generation_settings=settings,
                        source_revision=info.get("source_revision"),
                        data_filtering=report,
                    )
                    del variant
                    logger.info("Generated %s / %s seed %s", dataset, transformation, seed)
                del info, raw_records, records
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                try:
                    close_generator(model)
                except Exception as cleanup_error:
                    if primary_error is not None:
                        raise primary_error from cleanup_error
                    raise
            finally:
                del model
                _release_cuda_memory()
    return GenerationResult(versions, paths)
