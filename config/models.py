"""Runtime configuration for the local ML scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import logging
import sys
import tempfile

import yaml

from ..domain.jobs import BATCH_PROBE_SEARCH_MODE_POWER_OF_TWO, normalize_batch_probe_search_mode
from ..redis_cache import RedisCacheSettings


SCHEDULER_MODE_ADAPTIVE = "adaptive"
PREDICTION_MODE_BRANCH_PROFILE = "branch_profile"
PREDICTION_MODE_ML_PREDICTOR = "ml_predictor"
_UNIX_SOCKET_PATH_SAFE_BYTES = 100
logger = logging.getLogger("localml_scheduler")


def normalize_scheduler_mode(value: str | None) -> str:
    normalized = str(value or SCHEDULER_MODE_ADAPTIVE).strip().lower().replace("-", "_")
    if normalized != SCHEDULER_MODE_ADAPTIVE:
        raise ValueError(f"Unsupported scheduler mode: {value}")
    return normalized


def effective_scheduler_mode(value: str | None) -> str:
    return normalize_scheduler_mode(value)


def _cache_socket_path(runtime_root: Path, socket_name: str) -> Path:
    path = runtime_root / socket_name
    if sys.platform == "win32":
        return path
    if len(str(path).encode("utf-8")) < _UNIX_SOCKET_PATH_SAFE_BYTES:
        return path
    digest = hashlib.sha1(str(runtime_root).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"localml-{digest}.sock"


@dataclass(slots=True)
class GpuProfilingSettings:
    warmup_steps: int = 2
    solo_probe_steps: int = 5
    pair_probe_steps: int = 60
    reuse_profile_if_confidence_ge: float = 0.8

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuProfilingSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "solo_probe_steps": self.solo_probe_steps,
            "pair_probe_steps": self.pair_probe_steps,
            "reuse_profile_if_confidence_ge": self.reuse_profile_if_confidence_ge,
        }


@dataclass(slots=True)
class GpuMemorySettings:
    vram_budget_fraction: float = 0.95
    safe_vram_budget_gib: float | None = field(default=None, repr=False)
    hard_stop_memory_fraction: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.vram_budget_fraction is None:
            self.vram_budget_fraction = 0.95
        try:
            fraction = float(self.vram_budget_fraction)
        except (TypeError, ValueError):
            fraction = 0.95
        self.vram_budget_fraction = min(1.0, max(0.0, fraction))
        if self.safe_vram_budget_gib is not None:
            try:
                self.safe_vram_budget_gib = float(self.safe_vram_budget_gib)
            except (TypeError, ValueError):
                self.safe_vram_budget_gib = None
        if self.hard_stop_memory_fraction is not None:
            try:
                self.hard_stop_memory_fraction = float(self.hard_stop_memory_fraction)
            except (TypeError, ValueError):
                self.hard_stop_memory_fraction = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuMemorySettings":
        data = dict(payload or {})
        if "vram_budget_fraction" not in data and "hard_stop_memory_fraction" in data:
            data["vram_budget_fraction"] = data["hard_stop_memory_fraction"]
            logger.warning(
                "gpu_scheduler.memory.hard_stop_memory_fraction is deprecated; "
                "use gpu_scheduler.memory.vram_budget_fraction."
            )
        if "safe_vram_budget_gib" in data:
            logger.warning(
                "gpu_scheduler.memory.safe_vram_budget_gib is deprecated; "
                "safe_vram_budget_mb is now computed from detected VRAM and "
                "gpu_scheduler.memory.vram_budget_fraction."
            )
        return cls(**data)

    def budget_mb(self, total_vram_mb: int | float | None) -> float:
        try:
            total = float(total_vram_mb) if total_vram_mb is not None else 0.0
        except (TypeError, ValueError):
            total = 0.0
        if total > 0:
            return total * float(self.vram_budget_fraction)
        if self.safe_vram_budget_gib is not None:
            return float(self.safe_vram_budget_gib) * 1024.0
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "vram_budget_fraction": self.vram_budget_fraction,
        }


@dataclass(slots=True)
class GpuThresholdSettings:
    pack_prefer_sm_active_lt: float = 0.50
    pack_reject_sm_active_ge: float = 0.80
    pack_reject_max_slowdown: float = 1.30
    latency_sensitive_max_slowdown: float = 1.15
    min_aggregate_gain: float = 1.10

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuThresholdSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_prefer_sm_active_lt": self.pack_prefer_sm_active_lt,
            "pack_reject_sm_active_ge": self.pack_reject_sm_active_ge,
            "pack_reject_max_slowdown": self.pack_reject_max_slowdown,
            "latency_sensitive_max_slowdown": self.latency_sensitive_max_slowdown,
            "min_aggregate_gain": self.min_aggregate_gain,
        }


@dataclass(slots=True)
class GpuTelemetrySettings:
    device_poll_ms: int = 500
    pair_recheck_every_steps: int = 20

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuTelemetrySettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_poll_ms": self.device_poll_ms,
            "pair_recheck_every_steps": self.pair_recheck_every_steps,
        }


@dataclass(slots=True)
class EarlyStopSettings:
    enabled: bool = True
    warmup_samples: int = 10
    patience_samples: int = 10
    min_delta: float = 1e-4
    min_runtime_seconds: float = 120.0
    min_global_step: int = 20
    metric_key: str | None = None
    direction: str = "auto"
    plot_enabled: bool = True

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.plot_enabled = bool(self.plot_enabled)
        self.warmup_samples = max(0, int(self.warmup_samples or 0))
        self.patience_samples = max(1, int(self.patience_samples or 1))
        self.min_global_step = max(0, int(self.min_global_step or 0))
        try:
            self.min_delta = max(0.0, float(self.min_delta))
        except (TypeError, ValueError):
            self.min_delta = 1e-4
        try:
            self.min_runtime_seconds = max(0.0, float(self.min_runtime_seconds))
        except (TypeError, ValueError):
            self.min_runtime_seconds = 120.0
        self.metric_key = str(self.metric_key).strip() if self.metric_key else None
        normalized_direction = str(self.direction or "auto").strip().lower()
        if normalized_direction not in {"auto", "maximize", "minimize"}:
            normalized_direction = "auto"
        self.direction = normalized_direction

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "EarlyStopSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "warmup_samples": self.warmup_samples,
            "patience_samples": self.patience_samples,
            "min_delta": self.min_delta,
            "min_runtime_seconds": self.min_runtime_seconds,
            "min_global_step": self.min_global_step,
            "metric_key": self.metric_key,
            "direction": self.direction,
            "plot_enabled": self.plot_enabled,
        }


@dataclass(slots=True)
class MPSSettings:
    enabled: bool = True
    compute_mode: str = "EXCLUSIVE_PROCESS"
    default_primary_active_thread_pct: int = 60
    default_secondary_active_thread_pct: int = 40
    default_omp_num_threads: int = 6
    default_mkl_num_threads: int = 6
    pipe_directory: str = "/tmp/nvidia-mps"
    log_directory: str = "/tmp/nvidia-mps-log"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MPSSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "compute_mode": self.compute_mode,
            "default_primary_active_thread_pct": self.default_primary_active_thread_pct,
            "default_secondary_active_thread_pct": self.default_secondary_active_thread_pct,
            "default_omp_num_threads": self.default_omp_num_threads,
            "default_mkl_num_threads": self.default_mkl_num_threads,
            "pipe_directory": self.pipe_directory,
            "log_directory": self.log_directory,
        }


@dataclass(slots=True)
class CudaProcessSettings:
    enabled: bool = True
    default_omp_num_threads: int = 6
    default_mkl_num_threads: int = 6

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CudaProcessSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "default_omp_num_threads": self.default_omp_num_threads,
            "default_mkl_num_threads": self.default_mkl_num_threads,
        }


@dataclass(slots=True)
class StreamSettings:
    enabled: bool = False
    host_poll_interval_seconds: float = 0.1
    host_join_timeout_seconds: float = 3.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "StreamSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "host_poll_interval_seconds": self.host_poll_interval_seconds,
            "host_join_timeout_seconds": self.host_join_timeout_seconds,
        }


@dataclass(slots=True)
class AdaptivePlannerSettings:
    exact_search_max_jobs: int = 8
    vram_bucket_mb: int = 128
    frontier_width: int = 32
    finalist_limit: int = 64
    replan_debounce_seconds: float = 1.0
    minimum_throughput_gain_fraction: float = 0.05

    def __post_init__(self) -> None:
        self.exact_search_max_jobs = max(1, int(self.exact_search_max_jobs))
        self.vram_bucket_mb = max(1, int(self.vram_bucket_mb))
        self.frontier_width = max(1, int(self.frontier_width))
        self.finalist_limit = max(1, int(self.finalist_limit))
        self.replan_debounce_seconds = max(0.0, float(self.replan_debounce_seconds))
        self.minimum_throughput_gain_fraction = max(0.0, float(self.minimum_throughput_gain_fraction))

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AdaptivePlannerSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_search_max_jobs": self.exact_search_max_jobs,
            "vram_bucket_mb": self.vram_bucket_mb,
            "frontier_width": self.frontier_width,
            "finalist_limit": self.finalist_limit,
            "replan_debounce_seconds": self.replan_debounce_seconds,
            "minimum_throughput_gain_fraction": self.minimum_throughput_gain_fraction,
        }


@dataclass(slots=True)
class PredictionBranchSettings:
    enabled: bool = True
    fixed_confidence_if_uncalibrated: float = 0.55

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        try:
            self.fixed_confidence_if_uncalibrated = max(0.0, min(1.0, float(self.fixed_confidence_if_uncalibrated)))
        except (TypeError, ValueError):
            self.fixed_confidence_if_uncalibrated = 0.55

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PredictionBranchSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "fixed_confidence_if_uncalibrated": self.fixed_confidence_if_uncalibrated,
        }


@dataclass(slots=True)
class PredictionMLSettings:
    enabled: bool = False
    hardware_key: str | None = None
    checkpoint_path: str | None = None
    calibration_path: str | None = None
    device: str = "cpu"
    cache_size: int = 1024

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.hardware_key = str(self.hardware_key).strip() if self.hardware_key else None
        self.checkpoint_path = str(self.checkpoint_path).strip() if self.checkpoint_path else None
        self.calibration_path = str(self.calibration_path).strip() if self.calibration_path else None
        self.device = str(self.device or "cpu").strip().lower()
        if self.device != "cpu" and self.enabled:
            logger.warning("prediction.ml.device=%s may consume scheduled GPU resources; cpu is recommended.", self.device)
        try:
            self.cache_size = max(0, int(self.cache_size))
        except (TypeError, ValueError):
            self.cache_size = 1024

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PredictionMLSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hardware_key": self.hardware_key,
            "checkpoint_path": self.checkpoint_path,
            "calibration_path": self.calibration_path,
            "device": self.device,
            "cache_size": self.cache_size,
        }


@dataclass(slots=True)
class PredictionSafetyMarginSettings:
    branch: float = 1.20
    ml_uncalibrated: float = 1.25
    ml_calibrated: float = 1.00
    explicit: float = 1.30
    job_local_probe: float = 1.10

    def __post_init__(self) -> None:
        for name in ("branch", "ml_uncalibrated", "ml_calibrated", "explicit", "job_local_probe"):
            try:
                setattr(self, name, max(1.0, float(getattr(self, name))))
            except (TypeError, ValueError):
                setattr(self, name, 1.0)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PredictionSafetyMarginSettings":
        return cls(**(payload or {}))

    def margin_for(self, source: str, *, warnings: tuple[str, ...] = ()) -> float:
        normalized = str(source or "").strip().lower()
        if normalized == "branch":
            return self.branch
        if normalized in {"ml_student", "ml_teacher"}:
            return self.ml_uncalibrated if any("uncalibrated" in warning for warning in warnings) else self.ml_calibrated
        if normalized == "explicit":
            return self.explicit
        if normalized == "job_local_probe":
            return self.job_local_probe
        return 1.30

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "ml_uncalibrated": self.ml_uncalibrated,
            "ml_calibrated": self.ml_calibrated,
            "explicit": self.explicit,
            "job_local_probe": self.job_local_probe,
        }


@dataclass(slots=True)
class PredictionSettings:
    mode: str = PREDICTION_MODE_BRANCH_PROFILE
    timeout_ms: int = 1000
    unknown_value_policy: Any | None = None
    fallback_to_exclusive: bool = True
    branch: PredictionBranchSettings | dict[str, Any] = field(default_factory=PredictionBranchSettings)
    ml: PredictionMLSettings | dict[str, Any] = field(default_factory=PredictionMLSettings)
    safety_margin: PredictionSafetyMarginSettings | dict[str, Any] = field(default_factory=PredictionSafetyMarginSettings)

    def __post_init__(self) -> None:
        self.mode = str(self.mode or PREDICTION_MODE_BRANCH_PROFILE).strip().lower().replace("-", "_")
        if self.mode not in {PREDICTION_MODE_BRANCH_PROFILE, PREDICTION_MODE_ML_PREDICTOR}:
            raise ValueError(f"Unsupported prediction mode: {self.mode}")
        try:
            self.timeout_ms = max(1, int(self.timeout_ms))
        except (TypeError, ValueError):
            self.timeout_ms = 1000
        self.fallback_to_exclusive = bool(self.fallback_to_exclusive)
        if self.branch is None:
            self.branch = PredictionBranchSettings()
        if isinstance(self.branch, dict):
            self.branch = PredictionBranchSettings.from_dict(self.branch)
        if self.ml is None:
            self.ml = PredictionMLSettings()
        if isinstance(self.ml, dict):
            self.ml = PredictionMLSettings.from_dict(self.ml)
        if self.safety_margin is None:
            self.safety_margin = PredictionSafetyMarginSettings()
        if isinstance(self.safety_margin, dict):
            self.safety_margin = PredictionSafetyMarginSettings.from_dict(self.safety_margin)
        if self.mode == PREDICTION_MODE_ML_PREDICTOR and not self.ml.checkpoint_path and not self.fallback_to_exclusive:
            raise ValueError("prediction.mode=ml_predictor requires prediction.ml.checkpoint_path or fallback_to_exclusive=true")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PredictionSettings":
        data = dict(payload or {})
        if "request_timeout_ms" in data and "timeout_ms" not in data:
            data["timeout_ms"] = data.pop("request_timeout_ms")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "timeout_ms": self.timeout_ms,
            "unknown_value_policy": self.unknown_value_policy,
            "fallback_to_exclusive": self.fallback_to_exclusive,
            "branch": self.branch.to_dict(),
            "ml": self.ml.to_dict(),
            "safety_margin": self.safety_margin.to_dict(),
        }


@dataclass(slots=True)
class BaselineCacheSettings:
    warm_queue_policy: str = "top_k"
    warm_queue_top_k: int | None = 2
    entry_capacity: int | None = None
    max_ram_percent: float | None = None
    memory_budget_bytes: int = 2 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        self.warm_queue_policy = str(self.warm_queue_policy or "top_k").strip().lower()
        if self.warm_queue_policy not in {"top_k", "budget_only"}:
            self.warm_queue_policy = "top_k"
        if self.warm_queue_top_k is not None:
            self.warm_queue_top_k = max(0, int(self.warm_queue_top_k))

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "BaselineCacheSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "warm_queue_policy": self.warm_queue_policy,
            "warm_queue_top_k": self.warm_queue_top_k,
            "entry_capacity": self.entry_capacity,
            "max_ram_percent": self.max_ram_percent,
            "memory_budget_bytes": self.memory_budget_bytes,
        }


@dataclass(slots=True)
class LogDBSettings:
    provider: str = "postgres"
    dsn_env: str = "LOCALML_SCHEDULER_LOG_DSN"
    schema: str = "scheduler_logs"
    enabled: bool = False

    def __post_init__(self) -> None:
        self.provider = str(self.provider or "postgres").strip().lower().replace("-", "_")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "LogDBSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "dsn_env": self.dsn_env,
            "schema": self.schema,
            "enabled": self.enabled,
        }


@dataclass(slots=True)
class SchedulerSubmissionDefaults:
    requires_gpu: bool = True
    estimated_vram_mb: int | None = None
    estimated_ram_mb: int | None = None
    packing_eligible: bool = True
    packing_family: str = "mlevolve_script"
    packing_max_slowdown_ratio: float | None = None
    backend_allowlist: list[str] = field(default_factory=list)
    batch_probe_enabled: bool = True
    batch_probe_model_key: str | None = None
    batch_probe_probe_timeout_seconds: int = 45
    batch_probe_startup_timeout_seconds: int = 90
    batch_probe_step_timeout_seconds: int = 30
    batch_probe_optimizer_steps: int = 1
    batch_probe_poll_interval_seconds: float = 0.5
    batch_probe_max_multiplier: int = 32
    batch_probe_search_mode: str = BATCH_PROBE_SEARCH_MODE_POWER_OF_TWO
    runtime_probe_enabled: bool = True
    runtime_probe_target: str | None = None
    runtime_probe_model_key: str | None = None
    runtime_probe_strategy: str = "epoch_1"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SchedulerSubmissionDefaults":
        instance = cls(**(payload or {}))
        if instance.backend_allowlist is None:
            instance.backend_allowlist = []
        else:
            instance.backend_allowlist = [str(item) for item in instance.backend_allowlist]
        instance.batch_probe_search_mode = normalize_batch_probe_search_mode(instance.batch_probe_search_mode)
        instance.runtime_probe_strategy = str(instance.runtime_probe_strategy or "epoch_1").strip().lower().replace("-", "_")
        return instance

    def to_dict(self) -> dict[str, Any]:
        return {
            "requires_gpu": self.requires_gpu,
            "estimated_vram_mb": self.estimated_vram_mb,
            "estimated_ram_mb": self.estimated_ram_mb,
            "packing_eligible": self.packing_eligible,
            "packing_family": self.packing_family,
            "packing_max_slowdown_ratio": self.packing_max_slowdown_ratio,
            "backend_allowlist": list(self.backend_allowlist),
            "batch_probe_enabled": self.batch_probe_enabled,
            "batch_probe_model_key": self.batch_probe_model_key,
            "batch_probe_probe_timeout_seconds": self.batch_probe_probe_timeout_seconds,
            "batch_probe_startup_timeout_seconds": self.batch_probe_startup_timeout_seconds,
            "batch_probe_step_timeout_seconds": self.batch_probe_step_timeout_seconds,
            "batch_probe_optimizer_steps": self.batch_probe_optimizer_steps,
            "batch_probe_poll_interval_seconds": self.batch_probe_poll_interval_seconds,
            "batch_probe_max_multiplier": self.batch_probe_max_multiplier,
            "batch_probe_search_mode": self.batch_probe_search_mode,
            "runtime_probe_enabled": self.runtime_probe_enabled,
            "runtime_probe_target": self.runtime_probe_target,
            "runtime_probe_model_key": self.runtime_probe_model_key,
            "runtime_probe_strategy": self.runtime_probe_strategy,
        }


@dataclass(slots=True)
class GpuSchedulerSettings:
    enabled: bool = True
    mode: str = SCHEDULER_MODE_ADAPTIVE
    backend_priority: list[str] = field(default_factory=lambda: ["stream_mps", "stream", "cuda_process", "mps", "exclusive"])
    max_packed_jobs_per_gpu: int = 8
    candidate_window_size: int = 16
    device_index: int = 0
    fallback_cooldown_seconds: int = 900
    batch_probe_enabled: bool = True
    batch_probe_target_memory_fraction: float | None = field(default=None, repr=False)
    batch_probe_min_batch_size: int = 1
    batch_probe_max_search_rounds: int = 14
    batch_probe_max_batch_size: int | None = 4096
    batch_probe_search_mode: str = BATCH_PROBE_SEARCH_MODE_POWER_OF_TWO
    model_family_probe_enabled: bool = True
    model_family_probe_priority: int = 100
    model_family_probe_timeout_seconds: int | None = 300
    checkpoint_preemption_enabled: bool = True
    checkpoint_preemption_cooldown_seconds: float = 60.0
    checkpoint_preemption_min_runtime_seconds: float = 15.0
    checkpoint_preemption_max_per_job: int = 3
    checkpoint_preemption_min_estimated_gain_seconds: float = 15.0
    checkpoint_preemption_overhead_multiplier: float = 2.0
    checkpoint_preemption_pause_timeout_seconds: float = 60.0
    startpoint_probe_enabled: bool = True
    startpoint_probe_max_models: int | None = None
    derivative_profile_safety_fraction: float = 0.85
    profiling: GpuProfilingSettings = field(default_factory=GpuProfilingSettings)
    memory: GpuMemorySettings = field(default_factory=GpuMemorySettings)
    thresholds: GpuThresholdSettings = field(default_factory=GpuThresholdSettings)
    telemetry: GpuTelemetrySettings = field(default_factory=GpuTelemetrySettings)
    early_stop: EarlyStopSettings = field(default_factory=EarlyStopSettings)
    adaptive: AdaptivePlannerSettings | dict[str, Any] = field(default_factory=AdaptivePlannerSettings)
    submission_defaults: SchedulerSubmissionDefaults = field(default_factory=SchedulerSubmissionDefaults)
    mps: MPSSettings = field(default_factory=MPSSettings)
    cuda_process: CudaProcessSettings = field(default_factory=CudaProcessSettings)
    stream: StreamSettings = field(default_factory=StreamSettings)

    def __post_init__(self) -> None:
        self.mode = normalize_scheduler_mode(self.mode)
        self.batch_probe_search_mode = normalize_batch_probe_search_mode(self.batch_probe_search_mode)
        self.model_family_probe_enabled = bool(self.model_family_probe_enabled)
        try:
            self.model_family_probe_priority = int(self.model_family_probe_priority)
        except (TypeError, ValueError):
            self.model_family_probe_priority = 100
        if self.model_family_probe_timeout_seconds is not None:
            try:
                parsed_probe_timeout = int(self.model_family_probe_timeout_seconds)
            except (TypeError, ValueError):
                parsed_probe_timeout = 300
            self.model_family_probe_timeout_seconds = None if parsed_probe_timeout <= 0 else parsed_probe_timeout
        self.checkpoint_preemption_enabled = bool(self.checkpoint_preemption_enabled)
        try:
            self.checkpoint_preemption_cooldown_seconds = max(0.0, float(self.checkpoint_preemption_cooldown_seconds))
        except (TypeError, ValueError):
            self.checkpoint_preemption_cooldown_seconds = 60.0
        try:
            self.checkpoint_preemption_min_runtime_seconds = max(0.0, float(self.checkpoint_preemption_min_runtime_seconds))
        except (TypeError, ValueError):
            self.checkpoint_preemption_min_runtime_seconds = 15.0
        try:
            self.checkpoint_preemption_max_per_job = max(0, int(self.checkpoint_preemption_max_per_job))
        except (TypeError, ValueError):
            self.checkpoint_preemption_max_per_job = 3
        try:
            self.checkpoint_preemption_min_estimated_gain_seconds = max(0.0, float(self.checkpoint_preemption_min_estimated_gain_seconds))
        except (TypeError, ValueError):
            self.checkpoint_preemption_min_estimated_gain_seconds = 15.0
        try:
            self.checkpoint_preemption_overhead_multiplier = max(0.0, float(self.checkpoint_preemption_overhead_multiplier))
        except (TypeError, ValueError):
            self.checkpoint_preemption_overhead_multiplier = 2.0
        try:
            self.checkpoint_preemption_pause_timeout_seconds = max(1.0, float(self.checkpoint_preemption_pause_timeout_seconds))
        except (TypeError, ValueError):
            self.checkpoint_preemption_pause_timeout_seconds = 60.0
        self.startpoint_probe_enabled = bool(self.startpoint_probe_enabled)
        if self.startpoint_probe_max_models is not None:
            self.startpoint_probe_max_models = max(0, int(self.startpoint_probe_max_models))
        try:
            self.derivative_profile_safety_fraction = float(self.derivative_profile_safety_fraction)
        except (TypeError, ValueError):
            self.derivative_profile_safety_fraction = 0.85
        self.derivative_profile_safety_fraction = min(1.0, max(0.0, self.derivative_profile_safety_fraction))
        if self.backend_priority is None:
            self.backend_priority = ["stream_mps", "stream", "cuda_process", "mps", "exclusive"]
        else:
            self.backend_priority = [str(item) for item in self.backend_priority]
        self.max_packed_jobs_per_gpu = max(1, int(self.max_packed_jobs_per_gpu))
        self.candidate_window_size = max(1, int(self.candidate_window_size))
        if self.profiling is None:
            self.profiling = GpuProfilingSettings()
        if isinstance(self.profiling, dict):
            self.profiling = GpuProfilingSettings.from_dict(self.profiling)
        if self.memory is None:
            self.memory = GpuMemorySettings()
        if isinstance(self.memory, dict):
            self.memory = GpuMemorySettings.from_dict(self.memory)
        if self.thresholds is None:
            self.thresholds = GpuThresholdSettings()
        if isinstance(self.thresholds, dict):
            self.thresholds = GpuThresholdSettings.from_dict(self.thresholds)
        if self.telemetry is None:
            self.telemetry = GpuTelemetrySettings()
        if isinstance(self.telemetry, dict):
            self.telemetry = GpuTelemetrySettings.from_dict(self.telemetry)
        if self.early_stop is None:
            self.early_stop = EarlyStopSettings()
        if isinstance(self.early_stop, dict):
            self.early_stop = EarlyStopSettings.from_dict(self.early_stop)
        if self.adaptive is None:
            self.adaptive = AdaptivePlannerSettings()
        if isinstance(self.adaptive, dict):
            self.adaptive = AdaptivePlannerSettings.from_dict(self.adaptive)
        if self.submission_defaults is None:
            self.submission_defaults = SchedulerSubmissionDefaults()
        if isinstance(self.submission_defaults, dict):
            self.submission_defaults = SchedulerSubmissionDefaults.from_dict(self.submission_defaults)
        if self.mps is None:
            self.mps = MPSSettings()
        if isinstance(self.mps, dict):
            self.mps = MPSSettings.from_dict(self.mps)
        if self.cuda_process is None:
            self.cuda_process = CudaProcessSettings()
        if isinstance(self.cuda_process, dict):
            self.cuda_process = CudaProcessSettings.from_dict(self.cuda_process)
        if self.stream is None:
            self.stream = StreamSettings()
        if isinstance(self.stream, dict):
            self.stream = StreamSettings.from_dict(self.stream)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuSchedulerSettings":
        data = dict(payload or {})
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "backend_priority": list(self.backend_priority),
            "max_packed_jobs_per_gpu": self.max_packed_jobs_per_gpu,
            "candidate_window_size": self.candidate_window_size,
            "device_index": self.device_index,
            "fallback_cooldown_seconds": self.fallback_cooldown_seconds,
            "batch_probe_enabled": self.batch_probe_enabled,
            "batch_probe_min_batch_size": self.batch_probe_min_batch_size,
            "batch_probe_max_search_rounds": self.batch_probe_max_search_rounds,
            "batch_probe_max_batch_size": self.batch_probe_max_batch_size,
            "batch_probe_search_mode": self.batch_probe_search_mode,
            "model_family_probe_enabled": self.model_family_probe_enabled,
            "model_family_probe_priority": self.model_family_probe_priority,
            "model_family_probe_timeout_seconds": self.model_family_probe_timeout_seconds,
            "checkpoint_preemption_enabled": self.checkpoint_preemption_enabled,
            "checkpoint_preemption_cooldown_seconds": self.checkpoint_preemption_cooldown_seconds,
            "checkpoint_preemption_min_runtime_seconds": self.checkpoint_preemption_min_runtime_seconds,
            "checkpoint_preemption_max_per_job": self.checkpoint_preemption_max_per_job,
            "checkpoint_preemption_min_estimated_gain_seconds": self.checkpoint_preemption_min_estimated_gain_seconds,
            "checkpoint_preemption_overhead_multiplier": self.checkpoint_preemption_overhead_multiplier,
            "checkpoint_preemption_pause_timeout_seconds": self.checkpoint_preemption_pause_timeout_seconds,
            "derivative_profile_safety_fraction": self.derivative_profile_safety_fraction,
            "profiling": self.profiling.to_dict(),
            "memory": self.memory.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "telemetry": self.telemetry.to_dict(),
            "early_stop": self.early_stop.to_dict(),
            "adaptive": self.adaptive.to_dict(),
            "submission_defaults": self.submission_defaults.to_dict(),
            "mps": self.mps.to_dict(),
            "cuda_process": self.cuda_process.to_dict(),
            "stream": self.stream.to_dict(),
        }


@dataclass(slots=True)
class SchedulerConfig:
    runtime_root: Path = Path("localml_scheduler/runtime")
    scheduler_poll_interval_seconds: float = 0.5
    command_poll_limit: int = 100
    aging_interval_seconds: float = 180.0
    aging_priority_increment: int = 1
    enable_priority_aging: bool = True
    preempt_check_interval_seconds: float = 0.5
    baseline_cache: BaselineCacheSettings | dict[str, Any] = field(default_factory=BaselineCacheSettings)
    cache_server_host: str = "127.0.0.1"
    cache_server_port: int = 8765
    cache_socket_name: str = "cache_server.sock"
    redis_cache: RedisCacheSettings | dict[str, Any] = field(default_factory=RedisCacheSettings)
    auto_resume_recoverable: bool = False
    prediction: PredictionSettings | dict[str, Any] = field(default_factory=PredictionSettings)
    gpu_scheduler: GpuSchedulerSettings = field(default_factory=GpuSchedulerSettings)
    log_db: LogDBSettings | dict[str, Any] = field(default_factory=LogDBSettings)
    python_executable: str = field(default_factory=lambda: sys.executable)
    sqlite_busy_timeout_ms: int = 10_000
    scheduler_session_id: str | None = None

    db_dir: Path = field(init=False)
    db_path: Path = field(init=False)
    branch_profile_db_path: Path = field(init=False)
    jobs_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    cache_meta_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    events_jsonl_path: Path = field(init=False)
    scheduler_log_path: Path = field(init=False)
    cache_socket_path: Path = field(init=False)
    service_heartbeat_path: Path = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.gpu_scheduler, dict):
            self.gpu_scheduler = GpuSchedulerSettings.from_dict(self.gpu_scheduler)
        if self.baseline_cache is None:
            self.baseline_cache = BaselineCacheSettings()
        if isinstance(self.baseline_cache, dict):
            self.baseline_cache = BaselineCacheSettings.from_dict(self.baseline_cache)
        if self.redis_cache is None:
            self.redis_cache = RedisCacheSettings()
        if isinstance(self.redis_cache, dict):
            self.redis_cache = RedisCacheSettings.from_dict(self.redis_cache)
        if self.prediction is None:
            self.prediction = PredictionSettings()
        if isinstance(self.prediction, dict):
            self.prediction = PredictionSettings.from_dict(self.prediction)
        if self.log_db is None:
            self.log_db = LogDBSettings()
        if isinstance(self.log_db, dict):
            self.log_db = LogDBSettings.from_dict(self.log_db)
        self.runtime_root = Path(self.runtime_root).resolve()
        self.db_dir = self.runtime_root / "db"
        self.db_path = self.db_dir / "scheduler.sqlite3"
        self.branch_profile_db_path = self.db_dir / "branch_profile.sqlite3"
        self.jobs_dir = self.runtime_root / "data" / "jobs"
        self.checkpoints_dir = self.runtime_root / "data" / "checkpoints"
        self.cache_meta_dir = self.runtime_root / "cache_meta"
        self.logs_dir = self.runtime_root / "logs"
        self.events_jsonl_path = self.logs_dir / "events.jsonl"
        self.scheduler_log_path = self.logs_dir / "scheduler.log"
        self.cache_socket_path = _cache_socket_path(self.runtime_root, self.cache_socket_name)
        self.service_heartbeat_path = self.runtime_root / "service_heartbeat.json"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None, **overrides: Any) -> "SchedulerConfig":
        merged = dict(payload or {})
        merged.update(overrides)
        return cls(**merged)

    @classmethod
    def from_file(cls, path: str | Path | None = None, **overrides: Any) -> "SchedulerConfig":
        payload: dict[str, Any] = {}
        if path:
            with Path(path).open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        return cls.from_dict(payload, **overrides)

    def ensure_runtime_layout(self) -> None:
        for directory in (
            self.runtime_root,
            self.db_dir,
            self.jobs_dir,
            self.checkpoints_dir,
            self.cache_meta_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def job_runtime_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def job_command_path(self, job_id: str) -> Path:
        return self.job_runtime_dir(job_id) / "command.json"

    def job_heartbeat_path(self, job_id: str) -> Path:
        return self.job_runtime_dir(job_id) / "heartbeat.json"

    def job_repack_ack_path(self, job_id: str) -> Path:
        return self.job_runtime_dir(job_id) / "repack_ack.json"

    def checkpoints_for_job(self, job_id: str) -> Path:
        return self.checkpoints_dir / job_id

    def cache_address(self) -> str | tuple[str, int]:
        if sys.platform != "win32":
            return str(self.cache_socket_path)
        return (self.cache_server_host, self.cache_server_port)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_root": str(self.runtime_root),
            "scheduler_poll_interval_seconds": self.scheduler_poll_interval_seconds,
            "command_poll_limit": self.command_poll_limit,
            "aging_interval_seconds": self.aging_interval_seconds,
            "aging_priority_increment": self.aging_priority_increment,
            "enable_priority_aging": self.enable_priority_aging,
            "preempt_check_interval_seconds": self.preempt_check_interval_seconds,
            "baseline_cache": self.baseline_cache.to_dict(),
            "cache_server_host": self.cache_server_host,
            "cache_server_port": self.cache_server_port,
            "cache_socket_name": self.cache_socket_name,
            "redis_cache": self.redis_cache.to_dict(),
            "auto_resume_recoverable": self.auto_resume_recoverable,
            "prediction": self.prediction.to_dict(),
            "gpu_scheduler": self.gpu_scheduler.to_dict(),
            "log_db": self.log_db.to_dict(),
            "python_executable": self.python_executable,
            "sqlite_busy_timeout_ms": self.sqlite_busy_timeout_ms,
            "scheduler_session_id": self.scheduler_session_id,
        }


SchedulerSettings = SchedulerConfig
