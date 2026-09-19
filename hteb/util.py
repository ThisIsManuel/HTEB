"""Shared file, serialization, logging, version and reporting helpers."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from statistics import mean
from typing import Any, Final, cast

from tqdm import tqdm

from .constants import AXES

logger = logging.getLogger(__name__)

_SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|^token$|"
    r"auth(?:orization)?|credential|password|secret)",
    re.IGNORECASE,
)


def sanitize_path(path: str | os.PathLike[str]) -> str:
    """Remove machine-local absolute prefixes from a shareable record."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return candidate.as_posix()
    resolved = candidate.resolve(strict=False)
    return "${LOCAL}/" + resolved.name


def to_json_value(
    value: object,
    *,
    shareable: bool = False,
    redact_secrets: bool = True,
) -> object:
    """Convert supported Python values into a deterministic JSON value.

    Unsupported objects fail instead of falling back to ``repr`` because a
    repr commonly contains an address or another process-specific value.
    """

    if isinstance(value, str):
        if shareable and (
            Path(value).expanduser().is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value)
        ):
            return sanitize_path(value)
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NaN and infinity are not valid JSON values")
        return value
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        if isinstance(value, dt.datetime) and value.tzinfo is None:
            raise ValueError("metadata datetimes must include a timezone")
        return value.isoformat()
    if isinstance(value, Path):
        return sanitize_path(value) if shareable else value.as_posix()
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            if not isinstance(raw_key, str):
                raise ValueError("JSON object keys must be strings")
            key = raw_key
            if shareable and key.endswith("_sha256"):
                continue
            if redact_secrets and _SECRET_KEY_RE.search(key):
                result[key] = "<redacted>"
            else:
                result[key] = to_json_value(
                    value[raw_key],
                    shareable=shareable,
                    redact_secrets=redact_secrets,
                )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            to_json_value(
                item,
                shareable=shareable,
                redact_secrets=redact_secrets,
            )
            for item in value
        ]
    raise ValueError(f"unsupported JSON value: {type(value).__module__}.{type(value).__name__}")


def json_dumps(value: object) -> str:
    """Serialize metadata with stable key order and compact JSON formatting."""

    normalized = to_json_value(value, redact_secrets=False)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def hash_settings(value: object) -> str:
    """Hash model settings, including the seed and prompt policy."""
    return hashlib.sha256(json_dumps(value).encode("utf-8")).hexdigest()


def safe_model_name(model_name: str) -> str:
    """Make a portable path component without changing the model's identity."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model name must be nonempty text")
    name = re.sub(r"\s+", "_", model_name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip(".-_")
    name = name or "model"
    if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", name.split(".")[0], re.I):
        name = "model-" + name
    if len(name) > 200:
        name = name[:135].rstrip(".-_") + "-" + hash_settings(model_name)
    return name


def _progress_log_emit(console: logging.StreamHandler[Any]) -> Callable[[logging.LogRecord], None]:
    emit = console.emit

    def write(record: logging.LogRecord) -> None:
        if record.levelno < logging.WARNING and getattr(record, "evaluation_progress", False):
            return
        with tqdm.external_write_mode(file=console.stream):
            emit(record)

    return write


@contextmanager
def run_logging(path: Path, *, progress: bool = False) -> Iterator[dict[str, Any]]:
    logger = logging.getLogger("hteb")
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    status: dict[str, Any] = {"path": None, "error": None}
    console = None
    handler = None
    stream = None
    filtered: list[logging.StreamHandler[Any]] = []
    redirected: list[tuple[logging.StreamHandler[Any], Callable[[logging.LogRecord], None]]] = []

    def console_record(record: logging.LogRecord) -> bool:
        return not getattr(record, "file_only", False)

    def failed(error: BaseException | None) -> None:
        if status["error"] is None:
            status["error"] = f"Run log unavailable: {type(error).__name__}"
            if handler is not None:
                handler.setLevel(logging.CRITICAL + 1)
                logger.removeHandler(handler)
            logger.warning("%s; scores will still be saved", status["error"])

    try:
        if not logger.hasHandlers():
            console = logging.StreamHandler()
            logger.addHandler(console)
        current: logging.Logger | None = logger
        while current is not None:
            for existing in current.handlers:
                if isinstance(existing, logging.StreamHandler) and (
                    existing.stream is sys.stderr or existing.stream is sys.stdout
                ):
                    existing.addFilter(console_record)
                    filtered.append(existing)
                    if progress:
                        redirected.append((existing, existing.emit))
                        existing.emit = _progress_log_emit(existing)  # type: ignore[method-assign, assignment]
            current = current.parent if current.propagate else None
        try:
            stream = path.open("x", encoding="utf-8")
            status["path"] = path.name
            handler = logging.StreamHandler(stream)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s [%(processName)s] %(message)s")
            )
            handler.handleError = lambda record: failed(sys.exc_info()[1])  # type: ignore[method-assign]
            logger.addHandler(handler)
        except Exception as error:
            failed(error)
        yield status
    finally:
        if handler is not None:
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception as error:
                failed(error)
        if stream is not None:
            try:
                stream.close()
            except Exception as error:
                failed(error)
        for original, emit in reversed(redirected):
            original.emit = emit  # type: ignore[method-assign, assignment]
        for original in filtered:
            original.removeFilter(console_record)
        if console is not None:
            logger.removeHandler(console)
            console.close()
        logger.setLevel(previous_level)


def _commit_id(value: object) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value) else None


def _snapshot_revision(value: object) -> str | None:
    if not isinstance(value, (str, os.PathLike)):
        return None
    parts = Path(value).parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots":
            revision = _commit_id(parts[index + 1])
            if revision is not None:
                return revision
    return None


def hub_model_name(model_id: str, location: str = "model_id") -> str:
    """Reject explicit local selectors; leave Hub name validation to the loader."""
    if Path(model_id).is_absolute() or model_id.startswith(("./", "../", "~", "local=")):
        raise ValueError(
            f"{location} must be a Hugging Face Hub model ID; "
            "local checkpoint paths are no longer supported"
        )
    return model_id


def check_local_model_conflict(model_id: str, location: str = "model_id") -> None:
    """Prevent model libraries from interpreting a Hub ID as a local checkpoint."""
    hub_model_name(model_id, location)
    candidate = Path(model_id)
    if candidate.exists() or candidate.is_symlink():
        raise ValueError(
            f"{location}: local filesystem entry conflicts with Hub model ID {model_id!r}; "
            "rename or move that entry before loading the model"
        )


def capture_model_version(
    model_id: str, model_config: Any = None, tokenizer: Any = None
) -> dict[str, object]:
    """Record IDs from loaded objects and tokenizer cache paths without online lookup."""
    revision = _commit_id(getattr(model_config, "_commit_hash", None))
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {})
    if not isinstance(tokenizer_kwargs, Mapping):
        tokenizer_kwargs = {}
    tokenizer_revision = _commit_id(tokenizer_kwargs.get("_commit_hash"))
    for value in (
        getattr(tokenizer, "name_or_path", None),
        *(tokenizer_kwargs.get(key) for key in ("name_or_path", "tokenizer_file", "vocab_file")),
    ):
        tokenizer_revision = tokenizer_revision or _snapshot_revision(value)
    cleaned = to_json_value(
        {"model_id": model_id, "revision": revision, "tokenizer_revision": tokenizer_revision},
        shareable=True,
    )
    assert isinstance(cleaned, dict)
    return cleaned


SCORE_MULTIPLIER: Final = 100.0


def reported_score(raw_score: float) -> float:
    """Convert an internal decimal score to the public percentage scale."""

    return float(raw_score) * SCORE_MULTIPLIER


def check_directory(path: Path) -> None:
    if path.is_symlink() or path.absolute() != path.resolve(strict=False):
        raise ValueError(f"output directory must not use symbolic links: {path}")
    parent = path
    while not parent.exists() and parent.parent != parent:
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise ValueError(f"output directory cannot be used: {path}")


def group_scores(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report each original score once for its model, dataset and seed pair."""
    groups: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    checkpoints: dict[str, str] = {}
    for row in rows:
        model = str(row["model"])
        checkpoint = model.rstrip("/").rsplit("/", 1)[-1]
        if checkpoints.setdefault(checkpoint, model) != model:
            raise ValueError(f"models must have distinct checkpoint names: {checkpoint}")
        key = (model, row["dataset"], row["seed_transform"], row["seed_evaluation"])
        group = groups.setdefault(
            key,
            {
                "model": model,
                "dataset": row["dataset"],
                "seed_transform": row["seed_transform"],
                "seed_evaluation": row["seed_evaluation"],
                "original": row["score_original"],
                "transformations": {},
            },
        )
        if group["original"] != row["score_original"]:
            raise ValueError(
                "inconsistent original scores for the same model, dataset and seed pair"
            )
        transformation = row["transformation"]
        if transformation in group["transformations"]:
            raise ValueError(f"duplicate transformation score: {transformation}")
        group["transformations"][transformation] = row["score_transformed"]
        if "data_filtering" in row:
            group.setdefault("data_filtering", {})[transformation] = row["data_filtering"]
        if "languages" in row:
            group.setdefault("languages", {})[transformation] = row["languages"]
    for group in groups.values():
        group["axes"] = {}
        for axis, transformations in AXES.items():
            values = [
                group["transformations"][name]
                for name in transformations
                if name in group["transformations"]
            ]
            if values:
                group["axes"][axis] = mean(values)
    return list(groups.values())


def format_scores(groups: Sequence[Mapping[str, Any]]) -> str:
    """Use one line per transformation instead of expanding scalar fields."""

    def inline(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)

    lines = ["["]
    for index, group in enumerate(groups):
        identity = {
            key: group[key] for key in ("model", "dataset", "seed_transform", "seed_evaluation")
        }
        identity["model"] = str(identity["model"]).rstrip("/").rsplit("/", 1)[-1]
        lines.append("  " + inline(identity)[:-1] + ",")
        lines.append('   "original": ' + inline(group["original"]) + ",")
        lines.append('   "axes": ' + inline(group["axes"]) + ",")
        if "data_filtering" in group:
            lines.append('   "data_filtering": ' + inline(group["data_filtering"]) + ",")
        lines.append('   "transformations": {')
        items = list(group["transformations"].items())
        for offset, (name, score) in enumerate(items):
            comma = "," if offset + 1 < len(items) else ""
            languages = group.get("languages", {}).get(name)
            if languages is not None:
                score = {"score": score, "languages": languages}
            lines.append("     " + inline(name) + ": " + inline(score) + comma)
        lines.append("   }}" + ("," if index + 1 < len(groups) else ""))
    return "\n".join([*lines, "]", ""])


def save_results(
    directory: Path, groups: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]
) -> None:
    directory = directory.expanduser().absolute()
    check_directory(directory)
    for filename in ("scores.JSON", "settings.json"):
        target = directory / filename
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"output target already exists: {target}")
    cleaned = to_json_value(settings, shareable=True, redact_secrets=True)
    assert isinstance(cleaned, dict)
    lines = [
        "  " + json.dumps(key) + ": " + json.dumps(value, ensure_ascii=False, allow_nan=False)
        for key, value in cleaned.items()
        if key != "transformations_directory"
    ]
    documents = {
        "scores.JSON": format_scores(groups),
        "settings.json": "{\n" + ",\n".join(lines) + "\n}\n",
    }
    staging = Path(tempfile.mkdtemp(prefix=".results-", dir=directory))
    try:
        for filename, text in documents.items():
            descriptor = os.open(staging / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
    except BaseException:
        shutil.rmtree(staging)
        raise
    published: list[Path] = []
    try:
        for filename in documents:
            target = directory / filename
            os.link(staging / filename, target)
            published.append(target)
    except OSError as error:
        for target in published:
            # Keep replacements made by another writer and retain recovery files
            # even if a filesystem failure also prevents removing our own links.
            with suppress(OSError):
                if not target.is_symlink() and target.samefile(staging / target.name):
                    target.unlink()
        raise RuntimeError(
            f"Could not publish results; complete scores.JSON and settings.json remain in {staging}"
        ) from error
    shutil.rmtree(staging)


def finish_run_directory(directory: Path, name: str) -> Path:
    destination = directory.parent / name
    suffix = 2
    while destination.exists() or destination.is_symlink():
        destination = directory.parent / f"{name}_{suffix}"
        suffix += 1
    try:
        directory.rename(destination)
    except OSError as error:
        raise RuntimeError(
            f"Could not finalize results; saved files remain in {directory}"
        ) from error
    return destination


def as_int(value: object) -> int:
    """Preserve runtime conversion while making dynamically sourced config explicit."""

    return int(cast(Any, value))


def as_float(value: object) -> float:
    """Preserve runtime conversion while making dynamically sourced config explicit."""

    return float(cast(Any, value))
