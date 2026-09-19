"""Shared SentenceTransformers loading and bundled prompt selection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from .constants import EMBEDDING_PROMPT_POLICY, MODELS_DIRECTORY
from .util import (
    as_float,
    as_int,
    capture_model_version,
    check_local_model_conflict,
)


def resolve_embedding_prompt(
    model: Any,
    task_name: str,
    role: str = "text",
    *,
    prompt_names: Mapping[str, Mapping[str, str | None]] | None = None,
) -> dict[str, str | None]:
    """Select only prompts exposed by the loaded SentenceTransformers model."""

    bundled = getattr(model, "prompts", None)
    prompts = bundled if isinstance(bundled, Mapping) else {}
    task = task_name.lower().replace("-", "_")
    task = {"pairclassification": "pair_classification", "summarization": "summarisation"}.get(
        task, task
    )
    configured = (prompt_names or {}).get(task, {})
    if role in configured:
        name = configured[role]
        if name is None:
            return {"name": None, "text": "", "source": "configuration"}
        if name not in prompts:
            raise ValueError(f"prompt_names.{task}.{role}: bundled prompt {name!r} does not exist")
        text = prompts[name]
        if not isinstance(text, str):
            raise ValueError(
                f"prompt_names.{task}.{role}: bundled prompt {name!r} must be a string"
            )
        return {"name": name, "text": text, "source": "configuration"}
    if task in {"retrieval", "reranking"}:
        candidates = {"query": ("query",), "document": ("document", "passage", "corpus")}[role]
        source = "model_role"
    else:
        aliases = {
            "classification": ("Classification",),
            "clustering": ("Clustering",),
            "sts": ("STS",),
            "str": ("STS",),
            "summarisation": ("Summarisation",),
            "pair_classification": ("Classification",),
        }
        candidates = (task, *aliases.get(task, ()))
        source = "model_task"
    name = next((key for key in candidates if key in prompts), None)
    if name is None:
        default = getattr(model, "default_prompt_name", None)
        if isinstance(default, str):
            if default not in prompts:
                raise ValueError("model default_prompt_name does not identify a bundled prompt")
            name = default
            source = "model_default"
    if name is None:
        return {"name": None, "text": "", "source": "none"}
    prompt_text = prompts[name]
    if not isinstance(prompt_text, str):
        raise ValueError("selected model prompt must be a string")
    return {"name": name, "text": prompt_text, "source": source}


def _pooling_module(model: Any) -> Any | None:
    return next((module for module in model if type(module).__name__ == "Pooling"), None)


def _apply_pooling_config(model: Any, config: Mapping[str, object]) -> None:
    pooling_config = config.get("pooling")
    pooling = dict(pooling_config) if isinstance(pooling_config, Mapping) else {}
    include_prompt = pooling.get("include_prompt")
    strategy = str(pooling.get("strategy", "model_default"))
    if include_prompt is not None:
        model.set_pooling_include_prompt(include_prompt)
    if strategy != "model_default":
        module = _pooling_module(model)
        if module is None:
            raise RuntimeError(
                "explicit pooling configuration requires a SentenceTransformers Pooling module"
            )
        modes = {"mean": "mean", "cls": "cls", "last_token": "lasttoken"}
        if not hasattr(module, "pooling_mode") or not hasattr(module, "embedding_dimension"):
            raise RuntimeError("explicit pooling strategy requires the supported ST Pooling API")
        module.pooling_mode = modes[strategy]
        module.pooling_output_dimension = module.embedding_dimension


def _embedding_model_kwargs(torch_module: Any, config: Mapping[str, object]) -> dict[str, object]:
    """Resolve model kwargs without turning the explicit ``auto`` mode into bf16."""

    dtype_name = str(config.get("dtype", "bfloat16")).removeprefix("torch.")
    dtype = {
        "float16": torch_module.float16,
        "float32": torch_module.float32,
        "bfloat16": torch_module.bfloat16,
        "auto": None,
    }[dtype_name]
    model_kwargs: dict[str, object] = {
        "attn_implementation": str(config.get("attn_implementation", "sdpa")),
    }
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    return model_kwargs


def _configure_torch_runtime(torch_module: Any, config: Mapping[str, object]) -> None:
    """Apply the score-affecting PyTorch runtime contract used by evaluation."""

    runtime_seed = as_int(config.get("seed_evaluation", 1337))
    torch_module.manual_seed(runtime_seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(runtime_seed)
    deterministic_algorithms = bool(config.get("deterministic_algorithms", False))
    allow_tf32 = bool(config.get("allow_tf32", True))
    cudnn_benchmark = bool(config.get("cudnn_benchmark", True))
    torch_module.use_deterministic_algorithms(deterministic_algorithms, warn_only=True)
    torch_backends = getattr(torch_module, "backends", None)
    cuda_backend = getattr(torch_backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
        matmul_backend.allow_tf32 = allow_tf32
    cudnn_backend = getattr(torch_backends, "cudnn", None)
    if cudnn_backend is not None:
        if hasattr(cudnn_backend, "deterministic"):
            cudnn_backend.deterministic = deterministic_algorithms
        if hasattr(cudnn_backend, "benchmark"):
            cudnn_backend.benchmark = cudnn_benchmark


def load_sentence_transformer(
    model_name: str, config: Mapping[str, object]
) -> tuple[Any, dict[str, object]]:
    """Load one model using validated settings, with lazy imports of model libraries."""

    check_local_model_conflict(model_name)
    import torch
    from sentence_transformers import SentenceTransformer

    sentence_transformer = cast(Any, SentenceTransformer)
    _configure_torch_runtime(torch, config)
    device_map = config.get("device_map")
    device = config.get("device")
    multi_gpu = isinstance(device, list) and len(device) > 1
    available_gpus = as_int(torch.cuda.device_count())
    requested_devices = device if isinstance(device, list) else [device]
    unavailable = [
        target
        for target in requested_devices
        if isinstance(target, str)
        and (target == "cuda" or target.startswith("cuda:"))
        and (
            not torch.cuda.is_available()
            or (target == "cuda" and available_gpus == 0)
            or (target.startswith("cuda:") and int(target.removeprefix("cuda:")) >= available_gpus)
        )
    ]
    if unavailable:
        raise RuntimeError(
            f"evaluation.devices requests unavailable CUDA devices {unavailable}; found {available_gpus} visible GPUs"
        )
    minimum_gpus = config.get("device_map_min_gpus")
    if minimum_gpus is not None and available_gpus < as_int(minimum_gpus):
        raise RuntimeError(
            f"embedding model requires at least {as_int(minimum_gpus)} GPUs, found {available_gpus}"
        )
    model_kwargs = _embedding_model_kwargs(torch, config)
    if isinstance(device, list):
        device = "cpu" if multi_gpu else device[0]
    if device is None:
        device = "cpu" if device_map else ("cuda" if torch.cuda.is_available() else "cpu")
    constructor: dict[str, object] = {
        "device": str(device),
        "model_kwargs": model_kwargs,
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
        "cache_folder": str(config.get("cache_folder") or Path(MODELS_DIRECTORY).absolute()),
    }
    model = sentence_transformer(model_name, **constructor)
    prompt_names = cast(Mapping[str, Mapping[str, str | None]], config.get("prompt_names", {}))
    for task, roles in prompt_names.items():
        for role in roles:
            resolve_embedding_prompt(model, task, role, prompt_names=prompt_names)
    _apply_pooling_config(model, config)
    maximum_length = config.get("max_sequence_length")
    if maximum_length is not None:
        model.max_seq_length = min(as_int(maximum_length), as_int(model.max_seq_length))
    model.model_name = model_name
    if device_map:
        if not torch.cuda.is_available():
            raise RuntimeError("embedding device_map requires CUDA")
        from accelerate import dispatch_model, infer_auto_device_map

        inner = getattr(model[0], "auto_model", None) or getattr(model[0], "model", None)
        if inner is None:
            raise AttributeError("device_map model has no inner auto_model/model module")
        no_split = getattr(inner, "_no_split_modules", None) or []
        fraction = as_float(config.get("device_map_memory_fraction", 0.3))
        maximum_memory = {
            index: f"{int(torch.cuda.get_device_properties(index).total_memory / 1024**3 * fraction)}GiB"
            for index in range(torch.cuda.device_count())
        }
        inferred = infer_auto_device_map(
            inner,
            max_memory=maximum_memory,
            no_split_module_classes=no_split,
        )
        if any(str(device) in {"cpu", "disk"} for device in inferred.values()):
            raise RuntimeError(
                "embedding device_map would offload to CPU/disk; increase its memory fraction"
            )
        dispatch_model(inner, device_map=inferred)
        original_to = model.to

        def safe_to(device_or_dtype: Any = None, *args: Any, **kwargs: Any) -> Any:
            if isinstance(device_or_dtype, torch.dtype):
                return original_to(device_or_dtype, *args, **kwargs)
            if "dtype" in kwargs and "device" not in kwargs:
                return original_to(dtype=kwargs["dtype"])
            return model

        model.to = safe_to
    pooling = config.get("pooling")
    try:
        first = model[0]
    except (TypeError, IndexError, KeyError):
        first = None
    inner = getattr(first, "auto_model", None) or getattr(first, "model", None) or first
    version = capture_model_version(
        model_name,
        getattr(inner, "config", None),
        getattr(model, "tokenizer", None),
    )
    return model, {
        "version": version,
        "prompt_policy": EMBEDDING_PROMPT_POLICY,
        "include_prompt": pooling.get("include_prompt") if isinstance(pooling, Mapping) else None,
        "device_map": bool(device_map),
        "multi_gpu_requested": multi_gpu,
    }
