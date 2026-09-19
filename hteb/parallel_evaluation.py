"""Supervised embedding workers with process-local runtime settings and retries."""

from __future__ import annotations

import math
import multiprocessing
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from typing import Any

from .embedding import _configure_torch_runtime, load_sentence_transformer
from .evaluation import encode_texts
from .util import as_int

ModelLoader = Callable[[str, Mapping[str, Any]], tuple[Any, dict[str, object]]]


def _model_description(model: Any, version: object) -> dict[str, Any]:
    return {
        "version": version,
        "prompts": getattr(model, "prompts", {}),
        "default_prompt_name": getattr(model, "default_prompt_name", None),
    }


def _encode_on_device(
    connection: Connection,
    model_name: str,
    device: str,
    settings: Mapping[str, Any],
    loader: ModelLoader,
) -> None:
    job_id = None
    try:
        # Dynamic Hub classes need not be pickleable. Construct them in the worker.
        model, metadata = loader(model_name, {**settings, "device": device})
        import torch

        if device.startswith("cuda:"):
            torch.cuda.set_device(device)
        # Loading can consume RNG state; encoding starts from the configured seed.
        _configure_torch_runtime(torch, settings)
        connection.send(("ready", _model_description(model, metadata["version"])))
        while True:
            job = connection.recv()
            if job is None:
                return
            job_id, texts, arguments = job
            vectors, batch_size = encode_texts(model, texts, **arguments)
            connection.send(("result", job_id, vectors, batch_size))
    except BaseException as error:
        # The parent may already be cleaning up after another failure.
        with suppress(OSError, EOFError):
            connection.send(("error", job_id, f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


@dataclass
class _Worker:
    process: BaseProcess
    connection: Connection
    device: str


class EmbeddingPool:
    """Keep one model replica per device and monitor replies and process exits."""

    def __init__(
        self,
        model: Any,
        devices: Sequence[str],
        settings: Mapping[str, Any],
        *,
        model_version: Mapping[str, Any],
        loader: ModelLoader = load_sentence_transformer,
    ):
        self._workers: list[_Worker] = []
        self._closed = False
        if not devices:
            raise ValueError("embedding workers require at least one device")
        try:
            expected = _model_description(model, dict(model_version))
            context = multiprocessing.get_context("spawn")
            for device in devices:
                parent, child = context.Pipe()
                try:
                    process = context.Process(
                        target=_encode_on_device,
                        args=(child, model.model_name, device, dict(settings), loader),
                        name=f"hteb-embedding-{device}",
                    )
                    self._workers.append(_Worker(process, parent, device))
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
            pending = set(range(len(self._workers)))
            while pending:
                index, message = self._receive()
                if index not in pending or message[0] != "ready":
                    raise RuntimeError("unexpected embedding worker startup reply")
                if message[1] != expected:
                    raise RuntimeError(
                        f"embedding worker on {self._workers[index].device} "
                        "loaded different model versions or prompts"
                    )
                pending.remove(index)
        except BaseException as error:
            self._close_after_error(error)
            raise

    def _receive(self) -> tuple[int, Any]:
        ready = wait(
            [worker.connection for worker in self._workers]
            + [worker.process.sentinel for worker in self._workers]
        )
        # Read an explicit error before reporting the process's exit, if both arrived.
        for index, worker in enumerate(self._workers):
            if worker.connection in ready:
                try:
                    message = worker.connection.recv()
                except (EOFError, OSError) as error:
                    raise RuntimeError(
                        f"embedding worker on {worker.device} disconnected without a reply"
                    ) from error
                if message[0] == "error":
                    raise RuntimeError(f"embedding worker on {worker.device} failed: {message[2]}")
                return index, message
        for worker in self._workers:
            if worker.process.sentinel in ready:
                worker.process.join()
                raise RuntimeError(
                    f"embedding worker on {worker.device} exited with code "
                    f"{worker.process.exitcode} without a reply"
                )
        raise RuntimeError("embedding worker wait returned without a reply")  # pragma: no cover

    def encode(
        self,
        texts: Sequence[str],
        *,
        task_type: str,
        model_config: Mapping[str, Any],
        evaluation_config: Mapping[str, Any],
        prompt_text: str,
    ) -> tuple[Any, int]:
        import numpy as np

        if self._closed:
            raise RuntimeError("embedding worker pool is closed")
        if not texts:
            return np.empty((0, 0), dtype=float), 0
        outer_chunk_size = as_int(
            model_config.get("encoding_chunk_size")
            or evaluation_config.get("encoding_chunk_size", 4096)
        )
        explicit_chunk_size = (
            model_config.get("encode_kwargs", {}).get(task_type, {}).get("chunk_size")
        )

        def chunks() -> Iterator[list[str]]:
            # Preserve outer boundaries and the library's automatic or explicit work units.
            for start in range(0, len(texts), outer_chunk_size):
                stop = min(start + outer_chunk_size, len(texts))
                chunk_size = (
                    explicit_chunk_size
                    if explicit_chunk_size is not None
                    else max(1, min(math.ceil((stop - start) / len(self._workers) / 10), 5000))
                )
                for offset in range(start, stop, chunk_size):
                    yield list(texts[offset : min(offset + chunk_size, stop)])

        arguments = {
            "task_type": task_type,
            "model_config": dict(model_config),
            "evaluation_config": dict(evaluation_config),
            "prompt_text": prompt_text,
        }
        jobs = enumerate(chunks())
        pending: dict[int, tuple[int, int]] = {}
        results: dict[int, Any] = {}
        batch_sizes = []

        def dispatch(index: int) -> None:
            job = next(jobs, None)
            if job is None:
                return
            job_id, chunk = job
            worker = self._workers[index]
            try:
                worker.connection.send((job_id, chunk, arguments))
            except (OSError, EOFError) as error:
                raise RuntimeError(
                    f"cannot send work to embedding worker on {worker.device}"
                ) from error
            pending[index] = (job_id, len(chunk))

        try:
            for index in range(len(self._workers)):
                dispatch(index)
            while pending:
                index, message = self._receive()
                if index not in pending or message[0] != "result":
                    raise RuntimeError("unexpected embedding worker result")
                job_id, row_count = pending.pop(index)
                if message[1] != job_id or len(message[2]) != row_count:
                    raise RuntimeError("embedding worker returned mismatched rows")
                results[job_id] = message[2]
                batch_sizes.append(message[3])
                dispatch(index)
            return np.concatenate([results[key] for key in sorted(results)], axis=0), min(
                batch_sizes
            )
        except BaseException as error:
            self._close_after_error(error)
            raise

    def _close_after_error(self, primary_error: BaseException) -> None:
        try:
            self.close()
        except Exception as cleanup_error:
            raise primary_error from cleanup_error

    def close(self) -> None:
        """Stop all workers, including idle or blocked ones, with bounded joins."""
        if self._closed:
            return
        self._closed = True
        errors = []
        for worker in self._workers:
            try:
                worker.connection.close()
                if worker.process.pid is not None and worker.process.is_alive():
                    worker.process.terminate()
            except Exception as error:
                errors.append(error)
        for worker in self._workers:
            try:
                if worker.process.pid is not None:
                    worker.process.join(timeout=2)
                    if worker.process.is_alive():
                        worker.process.kill()
                        worker.process.join(timeout=2)
                    if worker.process.is_alive():
                        raise RuntimeError(f"embedding worker on {worker.device} did not stop")
                worker.process.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError("could not clean up embedding workers") from errors[0]
