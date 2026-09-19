"""Run independent transformation jobs in separate GPU processes."""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
from collections.abc import Mapping
from contextlib import suppress
from logging.handlers import QueueHandler, QueueListener
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from typing import Any

from .config import HTEBConfig
from .generation import GenerationResult, generate_jobs

logger = logging.getLogger(__name__)


def _generate_on_gpu(
    connection: Connection,
    log_queue: Any,
    device: str,
    config: HTEBConfig,
    jobs: list[tuple[str, str, int]],
    original_paths: Mapping[str, Any],
) -> None:
    if os.name == "posix":
        os.setsid()
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    process_logger = logging.getLogger("hteb")
    handler = QueueHandler(log_queue)
    process_logger.handlers = [handler]
    process_logger.setLevel(logging.INFO)
    process_logger.propagate = False
    try:
        logger.info("Generation process started on GPU %s", device)
        versions = generate_jobs(config, jobs, original_paths)
        logger.info("Generation process finished on GPU %s", device)
        connection.send(("done", versions))
    except BaseException as error:
        description = f"{type(error).__name__}: {error}"
        logger.error("Generation process on GPU %s failed (%s)", device, description)
        connection.send(("error", description))
    finally:
        process_logger.removeHandler(handler)
        handler.close()
        connection.close()
        log_queue.close()
        log_queue.join_thread()


def _stop_process(process: BaseProcess) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if os.name == "posix" and process.pid is not None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, sig)
        if process.is_alive():
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        process.join(timeout=5)


def resolve_gpu_devices(devices: list[int]) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_devices = visible.split(",") if visible else None
    if visible is not None and (
        not visible.strip() or any(device >= len(visible_devices or []) for device in devices)
    ):
        raise ValueError("generation.devices selects a GPU outside CUDA_VISIBLE_DEVICES")
    selected = [
        visible_devices[device].strip() if visible_devices is not None else str(device)
        for device in devices
    ]
    if any(device in {"", "-1"} for device in selected) or len(set(selected)) != len(selected):
        raise ValueError("generation.devices requires distinct enabled CUDA_VISIBLE_DEVICES")
    return selected


def generate_transformations_on_gpus(
    config: HTEBConfig,
    jobs: list[tuple[str, str, int]],
    original_paths: Mapping[str, Any],
) -> GenerationResult:
    selected = config.generation["devices"]
    assert selected is not None
    devices = resolve_gpu_devices(selected)
    if not jobs:
        return GenerationResult([], {})
    context = multiprocessing.get_context("spawn")
    log_queue = context.Queue()
    listener = QueueListener(
        log_queue, *logging.getLogger("hteb").handlers, respect_handler_level=True
    )
    processes: dict[Connection, tuple[BaseProcess, str]] = {}
    successful = False
    versions: list[dict[str, object]] = []
    paths = {}
    listener.start()
    try:
        count = min(len(devices), len(jobs))
        for index, device in enumerate(devices[:count]):
            assigned = jobs[index::count]
            datasets = {job[0] for job in assigned}
            receiver, sender = context.Pipe(duplex=False)
            process: BaseProcess = context.Process(
                target=_generate_on_gpu,
                args=(
                    sender,
                    log_queue,
                    device,
                    config,
                    assigned,
                    {name: original_paths[name] for name in datasets},
                ),
                name=f"hteb-generation-gpu-{device}",
            )
            try:
                process.start()
            except BaseException:
                receiver.close()
                raise
            finally:
                sender.close()
            processes[receiver] = process, device
        pending = set(processes)
        while pending:
            for ready in wait(pending, timeout=1):
                assert isinstance(ready, Connection)
                process, device = processes[ready]
                try:
                    kind, payload = ready.recv()
                except EOFError as error:
                    raise RuntimeError(
                        f"Generation process on GPU {device} exited unexpectedly"
                    ) from error
                if kind == "error":
                    raise RuntimeError(f"Generation process on GPU {device} failed ({payload})")
                if kind != "done":
                    raise RuntimeError(f"Unexpected generation process event: {kind}")
                paths.update(payload.paths)
                for version in payload.versions:
                    if version not in versions:
                        versions.append(version)
                pending.remove(ready)
            for connection in pending:
                process, device = processes[connection]
                if process.exitcode is not None and not connection.poll():
                    raise RuntimeError(f"Generation process on GPU {device} exited unexpectedly")
        for process, device in processes.values():
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError(f"Generation process on GPU {device} did not exit cleanly")
        successful = True
    finally:
        for connection, (process, _) in processes.items():
            if not successful:
                _stop_process(process)
            connection.close()
        listener.stop()
        log_queue.close()
        log_queue.join_thread()
    return GenerationResult(versions, paths)
