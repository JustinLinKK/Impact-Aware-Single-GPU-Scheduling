"""Canonical backend taxonomy shared by configuration, scheduling, and knowledge."""

from __future__ import annotations

from collections.abc import Iterable
import inspect
import warnings


PACKED_BACKEND_MODES = ("cuda_process", "mps_process")
RUNTIME_BACKEND_MODES = ("exclusive", *PACKED_BACKEND_MODES)
BACKEND_NEUTRAL = "backend_neutral"
RUNNER_CONTRACT_SUBPROCESS_V1 = "subprocess_job_v1"
RETIRED_BACKEND_MODES = frozenset(
    {"stream", "cuda_stream", "mps_stream", "stream_mps"}
)

_STREAM_REMOVAL_MESSAGE = (
    "CUDA-stream placement is retired: independently generated training scripts run "
    "as child subprocesses with their own CUDA contexts, so a parent process CUDA "
    "stream cannot control their work. Choose cuda_process or mps_process."
)


def _clean(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def warn_backend_deprecation(message: str, *, stacklevel: int = 1) -> None:
    """Emit a deprecation warning even when MLEvolve's UI mutes warnings.warn."""

    frame = inspect.currentframe()
    try:
        for _ in range(max(1, int(stacklevel))):
            if frame is None or frame.f_back is None:
                break
            frame = frame.f_back
        filename = frame.f_code.co_filename if frame is not None else __file__
        lineno = frame.f_lineno if frame is not None else 1
        module = frame.f_globals.get("__name__") if frame is not None else __name__
        registry = (
            frame.f_globals.setdefault("__warningregistry__", {})
            if frame is not None
            else None
        )
        warnings.warn_explicit(
            message,
            DeprecationWarning,
            filename,
            lineno,
            module=module,
            registry=registry,
        )
    finally:
        del frame


def is_retired_backend(value: object) -> bool:
    return _clean(value) in RETIRED_BACKEND_MODES


def normalize_packing_backend(
    value: object,
    *,
    warn_legacy: bool = True,
) -> str:
    """Return one supported packed mode, warning for the temporary ``mps`` alias."""

    normalized = _clean(value)
    if normalized in RETIRED_BACKEND_MODES:
        raise ValueError(_STREAM_REMOVAL_MESSAGE)
    if normalized == "mps":
        if warn_legacy:
            warn_backend_deprecation(
                "packing backend 'mps' is deprecated; use 'mps_process'. "
                "The legacy value was normalized and will not be persisted.",
                stacklevel=2,
            )
        normalized = "mps_process"
    if normalized not in PACKED_BACKEND_MODES:
        expected = ", ".join(PACKED_BACKEND_MODES)
        raise ValueError(f"Unsupported packing backend {value!r}; expected {expected}")
    return normalized


def normalize_runtime_backend(
    value: object,
    *,
    warn_legacy: bool = True,
) -> str:
    normalized = _clean(value)
    if normalized == "exclusive":
        return normalized
    return normalize_packing_backend(normalized, warn_legacy=warn_legacy)


def normalize_backend_allowlist(
    values: Iterable[object] | None,
    *,
    include_exclusive: bool = True,
    warn_legacy: bool = True,
) -> list[str]:
    normalized: list[str] = []
    for value in values or ():
        backend = normalize_runtime_backend(value, warn_legacy=warn_legacy)
        if backend == "exclusive" and not include_exclusive:
            continue
        if backend not in normalized:
            normalized.append(backend)
    return normalized


def active_backend_matches(
    recorded_backend: object,
    effective_backend: object,
    *,
    allow_exclusive_baseline: bool = False,
    allow_proven_legacy_mps: bool = False,
) -> bool:
    """Hard eligibility check for active empirical evidence."""

    recorded = _clean(recorded_backend)
    if recorded in RETIRED_BACKEND_MODES or not recorded:
        return False
    if recorded == "mps" and not allow_proven_legacy_mps:
        return False
    try:
        recorded = normalize_runtime_backend(recorded, warn_legacy=False)
        effective = normalize_packing_backend(effective_backend, warn_legacy=False)
    except ValueError:
        return False
    return recorded == effective or (
        allow_exclusive_baseline and recorded == "exclusive"
    )


def stream_removal_message() -> str:
    return _STREAM_REMOVAL_MESSAGE
