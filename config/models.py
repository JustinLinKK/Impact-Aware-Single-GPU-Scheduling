"""Runtime configuration for the local ML scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import sys
import tempfile
import warnings

import yaml

from ..backend_mode import (
    PACKED_BACKEND_MODES,
    normalize_backend_allowlist,
    normalize_packing_backend,
    warn_backend_deprecation,
)
from ..redis_cache import RedisCacheSettings

SCHEDULER_MODE_PARALLEL_TIME_AWARE = "parallel_time_aware"
SCHEDULER_DECISION_MODE_BASELINE = "baseline"
SCHEDULER_DECISION_MODE_BACKEND_AWARED = "backend_awared"
PREDICTION_MODE_BRANCH_PROFILE = "branch_profile"
PREDICTION_MODE_ML_PREDICTOR = "ml_predictor"


def normalize_prediction_mode(value: str | None) -> str:
    normalized = (
        str(value or PREDICTION_MODE_BRANCH_PROFILE).strip().lower().replace("-", "_")
    )
    if normalized not in {PREDICTION_MODE_BRANCH_PROFILE, PREDICTION_MODE_ML_PREDICTOR}:
        raise ValueError(f"Unsupported prediction mode: {value}")
    return normalized


def normalize_scheduler_mode(value: str | None) -> str:
    normalized = (
        str(value or SCHEDULER_MODE_PARALLEL_TIME_AWARE)
        .strip()
        .lower()
        .replace("-", "_")
    )
    if normalized != SCHEDULER_MODE_PARALLEL_TIME_AWARE:
        raise ValueError(
            f"Unsupported scheduler mode: {value}. "
            "The production scheduler only supports parallel_time_aware; "
            "VRAM-fill and fixed-width placement modes were removed."
        )
    return normalized


def normalize_scheduler_decision_mode(value: str | None) -> str:
    normalized = (
        str(value or SCHEDULER_DECISION_MODE_BASELINE)
        .strip()
        .lower()
        .replace("-", "_")
    )
    if normalized == "backend_aware":
        normalized = SCHEDULER_DECISION_MODE_BACKEND_AWARED
    if normalized not in {
        SCHEDULER_DECISION_MODE_BASELINE,
        SCHEDULER_DECISION_MODE_BACKEND_AWARED,
    }:
        raise ValueError(
            f"Unsupported scheduler_decision_mode: {value}. "
            "Expected baseline or backend_awared."
        )
    return normalized


@dataclass(slots=True)
class GpuProfilingSettings:
    warmup_steps: int = 30
    solo_probe_steps: int = 80
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
    """Memory safety gates for time-aware admission, never placement scores."""

    gpu_vram_gib: float | None = None
    predicted_budget_fraction: float = 0.85
    live_admission_stop_fraction: float = 0.90
    live_admission_resume_fraction: float = 0.85
    admission_average_window_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.gpu_vram_gib is not None and self.gpu_vram_gib <= 0:
            raise ValueError("gpu_vram_gib must be positive")
        if not 0 < self.predicted_budget_fraction <= 1:
            raise ValueError("predicted_budget_fraction must be in (0, 1]")
        if (
            not 0
            < self.live_admission_resume_fraction
            < self.live_admission_stop_fraction
            <= 1
        ):
            raise ValueError(
                "live admission fractions must satisfy 0 < resume < stop <= 1"
            )
        if self.admission_average_window_seconds <= 0:
            raise ValueError("admission_average_window_seconds must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuMemorySettings":
        raw = dict(payload or {})
        removed = sorted({"safe_vram_budget_gib", "vram_budget_fraction"} & raw.keys())
        if removed:
            raise ValueError(
                "Removed legacy gpu_scheduler.memory settings: "
                + ", ".join(removed)
                + ". Use gpu_vram_gib and predicted_budget_fraction; VRAM is an admission constraint only."
            )
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_vram_gib": self.gpu_vram_gib,
            "predicted_budget_fraction": self.predicted_budget_fraction,
            "live_admission_stop_fraction": self.live_admission_stop_fraction,
            "live_admission_resume_fraction": self.live_admission_resume_fraction,
            "admission_average_window_seconds": self.admission_average_window_seconds,
        }


@dataclass(slots=True)
class TimeObjectiveSettings:
    priority_weight: float = 0.10
    objective_version: str = "time_v6_verified_piecewise_drain"

    def __post_init__(self) -> None:
        if self.priority_weight < 0:
            raise ValueError("priority_weight must be non-negative")
        if not str(self.objective_version).strip():
            raise ValueError("objective_version is required")
        if self.objective_version in {
            "time_v3_flow_only",
            "time_v4_colocation_gain",
            "time_v5_piecewise_drain",
        }:
            previous_version = self.objective_version
            warnings.warn(
                f"{previous_version} is migrated to time_v6_verified_piecewise_drain",
                UserWarning,
                stacklevel=2,
            )
            self.objective_version = "time_v6_verified_piecewise_drain"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TimeObjectiveSettings":
        raw = dict(payload or {})
        removed = sorted(
            {"makespan_weight", "flow_time_weight", "min_aggregate_gain"}.intersection(
                raw
            )
        )
        if removed:
            raise ValueError(
                "gpu_scheduler.objective no longer supports throughput scheduling controls: "
                + ", ".join(removed)
            )
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "priority_weight": self.priority_weight,
            "objective_version": self.objective_version,
        }


@dataclass(slots=True)
class BatchOptionSettings:
    exponent_offsets: list[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    require_power_of_two_original: bool = True

    def __post_init__(self) -> None:
        self.exponent_offsets = [int(value) for value in self.exponent_offsets]
        if len(self.exponent_offsets) != 5 or len(set(self.exponent_offsets)) != 5:
            raise ValueError(
                "batch_options.exponent_offsets must contain five distinct offsets"
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "BatchOptionSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "exponent_offsets": list(self.exponent_offsets),
            "require_power_of_two_original": self.require_power_of_two_original,
        }


@dataclass(slots=True)
class SourceAnalysisSettings:
    cache_enabled: bool = True
    max_source_bytes: int = 2_000_000
    max_unknown_operator_fraction_for_high_confidence: float = 0.05
    max_unknown_operator_fraction_for_medium_confidence: float = 0.25
    peak_tflops_by_dtype: dict[str, float] = field(default_factory=dict)
    memory_bandwidth_gbps: float | None = None

    def __post_init__(self) -> None:
        self.max_source_bytes = max(1, int(self.max_source_bytes))
        self.max_unknown_operator_fraction_for_high_confidence = float(
            self.max_unknown_operator_fraction_for_high_confidence
        )
        self.max_unknown_operator_fraction_for_medium_confidence = float(
            self.max_unknown_operator_fraction_for_medium_confidence
        )
        if not (
            0
            <= self.max_unknown_operator_fraction_for_high_confidence
            <= self.max_unknown_operator_fraction_for_medium_confidence
            <= 1
        ):
            raise ValueError(
                "source_analysis unknown-operator thresholds must satisfy 0 <= high <= medium <= 1"
            )
        self.peak_tflops_by_dtype = {
            str(key).lower(): float(value)
            for key, value in dict(self.peak_tflops_by_dtype or {}).items()
            if float(value) > 0
        }
        if self.memory_bandwidth_gbps is not None:
            self.memory_bandwidth_gbps = float(self.memory_bandwidth_gbps)
            if self.memory_bandwidth_gbps <= 0:
                raise ValueError("source_analysis.memory_bandwidth_gbps must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SourceAnalysisSettings":
        return cls(**dict(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_enabled": self.cache_enabled,
            "max_source_bytes": self.max_source_bytes,
            "max_unknown_operator_fraction_for_high_confidence": self.max_unknown_operator_fraction_for_high_confidence,
            "max_unknown_operator_fraction_for_medium_confidence": self.max_unknown_operator_fraction_for_medium_confidence,
            "peak_tflops_by_dtype": dict(self.peak_tflops_by_dtype),
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
        }


@dataclass(slots=True)
class SourceTrialRankingSettings:
    schema_version: int = 2
    policy: str = "pareto"
    ready_window_size: int = 10
    max_group_size: int = 2
    amortization_factor: float = 3.0
    estimated_setup_seconds: float = 0.0
    require_live_trial_for_unknown: bool = True
    source_analysis: SourceAnalysisSettings = field(default_factory=SourceAnalysisSettings)
    mode_overhead_mb: dict[str, float] = field(
        default_factory=lambda: {
            "cuda_process": 512.0,
            "mps_process": 384.0,
        }
    )
    mps_allocation_templates: list[list[int]] = field(
        default_factory=lambda: [[50, 50], [60, 40], [40, 60]]
    )
    def __post_init__(self) -> None:
        self.schema_version = int(self.schema_version)
        if self.schema_version != 2:
            raise ValueError(
                "source_trial_ranking.schema_version must be 2; version 1 contains retired stream semantics"
            )
        self.policy = str(self.policy or "pareto").strip().lower()
        if self.policy != "pareto":
            raise ValueError("source_trial_ranking.policy must be pareto")
        self.ready_window_size = max(2, int(self.ready_window_size))
        self.max_group_size = int(self.max_group_size)
        if self.max_group_size != 2:
            raise ValueError(
                "source_trial_ranking.max_group_size currently supports only 2"
            )
        self.amortization_factor = float(self.amortization_factor)
        self.estimated_setup_seconds = float(self.estimated_setup_seconds)
        if self.amortization_factor < 0:
            raise ValueError("source_trial_ranking.amortization_factor must be non-negative")
        if self.estimated_setup_seconds < 0:
            raise ValueError("source_trial_ranking.estimated_setup_seconds must be non-negative")
        if self.source_analysis is None:
            self.source_analysis = SourceAnalysisSettings()
        if isinstance(self.source_analysis, dict):
            self.source_analysis = SourceAnalysisSettings.from_dict(self.source_analysis)
        self.mode_overhead_mb = {
            str(key): float(value) for key, value in dict(self.mode_overhead_mb).items()
        }
        unsupported_overheads = sorted(
            set(self.mode_overhead_mb).difference(PACKED_BACKEND_MODES)
        )
        if unsupported_overheads:
            raise ValueError(
                "source_trial_ranking.mode_overhead_mb contains unsupported backends: "
                + ", ".join(unsupported_overheads)
            )
        if any(value < 0 for value in self.mode_overhead_mb.values()):
            raise ValueError("source_trial_ranking.mode_overhead_mb values must be non-negative")
        self.mps_allocation_templates = [
            [int(value) for value in template]
            for template in self.mps_allocation_templates
        ]
        if not self.mps_allocation_templates or any(
            len(template) != 2
            or any(value < 1 or value > 100 for value in template)
            or sum(template) > 100
            for template in self.mps_allocation_templates
        ):
            raise ValueError(
                "source_trial_ranking.mps_allocation_templates must contain valid two-client percentages"
            )
    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SourceTrialRankingSettings":
        raw = dict(payload or {})
        removed = sorted(
            {"stream_offset_templates_in_steps", "stream_offsets"}.intersection(raw)
        )
        if removed:
            raise ValueError(
                "Removed CUDA-stream source-trial settings: " + ", ".join(removed)
            )
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "ready_window_size": self.ready_window_size,
            "max_group_size": self.max_group_size,
            "amortization_factor": self.amortization_factor,
            "estimated_setup_seconds": self.estimated_setup_seconds,
            "require_live_trial_for_unknown": self.require_live_trial_for_unknown,
            "source_analysis": self.source_analysis.to_dict(),
            "mode_overhead_mb": dict(self.mode_overhead_mb),
            "mps_allocation_templates": [list(value) for value in self.mps_allocation_templates],
        }


@dataclass(slots=True)
class DecisionReplaySettings:
    enabled: bool = True
    min_stable_observations: int = 3
    training_time_change_fraction: float = 0.25
    vram_change_fraction: float = 0.25

    def __post_init__(self) -> None:
        self.min_stable_observations = int(self.min_stable_observations)
        self.training_time_change_fraction = float(self.training_time_change_fraction)
        self.vram_change_fraction = float(self.vram_change_fraction)
        if self.min_stable_observations < 1:
            raise ValueError(
                "colocation.decision_replay.min_stable_observations must be at least 1"
            )
        if self.training_time_change_fraction <= 0:
            raise ValueError(
                "colocation.decision_replay.training_time_change_fraction must be positive"
            )
        if self.vram_change_fraction <= 0:
            raise ValueError(
                "colocation.decision_replay.vram_change_fraction must be positive"
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DecisionReplaySettings":
        return cls(**dict(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_stable_observations": self.min_stable_observations,
            "training_time_change_fraction": self.training_time_change_fraction,
            "vram_change_fraction": self.vram_change_fraction,
        }


@dataclass(slots=True)
class ColocationSettings:
    min_gain: float = 1.0
    trial_epochs: int = 2
    trial_decision_timeout_seconds: float = 30.0
    trial_evidence_timeout_min_seconds: float = 300.0
    trial_evidence_timeout_max_seconds: float = 1800.0
    profile_rejection_min_bad_trials: int = 2
    profile_rejection_ttl_seconds: float = 86400.0
    live_trial_enabled: bool = True
    decision_replay: DecisionReplaySettings = field(
        default_factory=DecisionReplaySettings
    )

    def __post_init__(self) -> None:
        self.min_gain = float(self.min_gain)
        self.trial_epochs = int(self.trial_epochs)
        self.trial_decision_timeout_seconds = float(self.trial_decision_timeout_seconds)
        self.trial_evidence_timeout_min_seconds = float(
            self.trial_evidence_timeout_min_seconds
        )
        self.trial_evidence_timeout_max_seconds = float(
            self.trial_evidence_timeout_max_seconds
        )
        self.profile_rejection_min_bad_trials = int(
            self.profile_rejection_min_bad_trials
        )
        self.profile_rejection_ttl_seconds = float(self.profile_rejection_ttl_seconds)
        if self.decision_replay is None:
            self.decision_replay = DecisionReplaySettings()
        if isinstance(self.decision_replay, dict):
            self.decision_replay = DecisionReplaySettings.from_dict(
                self.decision_replay
            )
        if self.min_gain <= 0:
            raise ValueError("colocation.min_gain must be positive")
        if self.trial_epochs < 1:
            raise ValueError("colocation.trial_epochs must be at least 1")
        if self.trial_decision_timeout_seconds <= 0:
            raise ValueError(
                "colocation.trial_decision_timeout_seconds must be positive"
            )
        if self.trial_evidence_timeout_min_seconds <= 0:
            raise ValueError(
                "colocation.trial_evidence_timeout_min_seconds must be positive"
            )
        if (
            self.trial_evidence_timeout_max_seconds
            < self.trial_evidence_timeout_min_seconds
        ):
            raise ValueError(
                "colocation.trial_evidence_timeout_max_seconds must be at least the minimum"
            )
        if self.profile_rejection_min_bad_trials < 1:
            raise ValueError(
                "colocation.profile_rejection_min_bad_trials must be at least 1"
            )
        if self.profile_rejection_ttl_seconds <= 0:
            raise ValueError(
                "colocation.profile_rejection_ttl_seconds must be positive"
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ColocationSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_gain": self.min_gain,
            "trial_epochs": self.trial_epochs,
            "trial_decision_timeout_seconds": self.trial_decision_timeout_seconds,
            "trial_evidence_timeout_min_seconds": self.trial_evidence_timeout_min_seconds,
            "trial_evidence_timeout_max_seconds": self.trial_evidence_timeout_max_seconds,
            "profile_rejection_min_bad_trials": self.profile_rejection_min_bad_trials,
            "profile_rejection_ttl_seconds": self.profile_rejection_ttl_seconds,
            "live_trial_enabled": self.live_trial_enabled,
            "decision_replay": self.decision_replay.to_dict(),
        }


@dataclass(slots=True)
class ExclusiveProbeSettings:
    enabled: bool = True
    drain_without_preemption: bool = True

    def __post_init__(self) -> None:
        if self.enabled and not self.drain_without_preemption:
            raise ValueError(
                "exclusive probes only support non-preemptive drain semantics"
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ExclusiveProbeSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "drain_without_preemption": self.drain_without_preemption,
        }


@dataclass(slots=True)
class EarlyStoppingSettings:
    enabled: bool = False
    metric_name: str = "accuracy"
    mode: str = "max"
    patience_epochs: int = 5
    min_delta: float = 0.0
    min_epochs: int = 1
    save_best_checkpoint: bool = True
    restore_best_checkpoint: bool = False
    missing_metric_policy: str = "ignore"

    def __post_init__(self) -> None:
        self.mode = str(self.mode).strip().lower()
        if self.mode not in {"min", "max"}:
            raise ValueError("early_stopping.mode must be 'min' or 'max'")
        if self.patience_epochs < 1:
            raise ValueError("early_stopping.patience_epochs must be at least 1")
        if self.min_epochs < 0:
            raise ValueError("early_stopping.min_epochs must be non-negative")
        if self.min_delta < 0:
            raise ValueError("early_stopping.min_delta must be non-negative")
        self.missing_metric_policy = str(self.missing_metric_policy).strip().lower()
        if self.missing_metric_policy not in {"ignore", "error"}:
            raise ValueError(
                "early_stopping.missing_metric_policy must be 'ignore' or 'error'"
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "EarlyStoppingSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "metric_name": self.metric_name,
            "mode": self.mode,
            "patience_epochs": self.patience_epochs,
            "min_delta": self.min_delta,
            "min_epochs": self.min_epochs,
            "save_best_checkpoint": self.save_best_checkpoint,
            "restore_best_checkpoint": self.restore_best_checkpoint,
            "missing_metric_policy": self.missing_metric_policy,
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
class PredictionSettings:
    mode: str = PREDICTION_MODE_BRANCH_PROFILE
    registry_path: str | None = None
    conversion_timeout_seconds: float = 15.0
    cache_size: int = 1024
    test_override_enabled: bool = False
    test_model_path: str | None = None

    def __post_init__(self) -> None:
        self.mode = normalize_prediction_mode(self.mode)
        self.conversion_timeout_seconds = max(
            0.1, float(self.conversion_timeout_seconds)
        )
        self.cache_size = max(1, int(self.cache_size))
        if not self.test_override_enabled:
            self.test_model_path = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PredictionSettings":
        raw = dict(payload or {})
        nested = raw.pop("ml", None)
        if isinstance(nested, dict):
            aliases = {
                "source_conversion_timeout_seconds": "conversion_timeout_seconds",
            }
            for key, value in nested.items():
                raw[aliases.get(key, key)] = value
        if "source_conversion_timeout_seconds" in raw:
            raw["conversion_timeout_seconds"] = raw.pop(
                "source_conversion_timeout_seconds"
            )
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ml": {
                "registry_path": self.registry_path,
                "source_conversion_timeout_seconds": self.conversion_timeout_seconds,
                "cache_size": self.cache_size,
                "test_override_enabled": self.test_override_enabled,
                "test_model_path": self.test_model_path,
            },
        }


@dataclass(slots=True)
class GraphDBSettings:
    enabled: bool = True
    mode: str = "mirror"
    provider: str = "neo4j"
    uri: str = "bolt://127.0.0.1:7687"
    username: str = "neo4j"
    password_env: str = "LOCALML_SCHEDULER_NEO4J_PASSWORD"
    database: str = "neo4j"
    bootstrap_constraints: bool = True
    import_sqlite_evidence: bool = True
    sqlite_evidence_path: str | None = None
    allow_sqlite_fallback: bool = True

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.mode = str(self.mode or "mirror").strip().lower().replace("-", "_")
        if self.mode == "primary":
            self.mode = "mirror"
        if self.mode not in {"off", "mirror"}:
            self.mode = "mirror"
        if not self.enabled:
            self.mode = "off"
        self.provider = str(self.provider or "neo4j").strip().lower().replace("-", "_")
        if self.provider == "legacy_sqlite":
            self.provider = "sqlite"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GraphDBSettings":
        raw = dict(payload or {})
        aliases = {
            "auto_import_legacy_sqlite": "import_sqlite_evidence",
            "legacy_sqlite_path": "sqlite_evidence_path",
            "allow_legacy_fallback": "allow_sqlite_fallback",
        }
        for old_name, current_name in aliases.items():
            if old_name in raw and current_name not in raw:
                raw[current_name] = raw.pop(old_name)
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "provider": self.provider,
            "uri": self.uri,
            "username": self.username,
            "password_env": self.password_env,
            "database": self.database,
            "bootstrap_constraints": self.bootstrap_constraints,
            "import_sqlite_evidence": self.import_sqlite_evidence,
            "sqlite_evidence_path": self.sqlite_evidence_path,
            "allow_sqlite_fallback": self.allow_sqlite_fallback,
        }


@dataclass(slots=True)
class HardwareKnowledgeGraphSettings:
    """Connection settings for hardware facts, isolated from profile evidence."""

    enabled: bool = True
    provider: str = "neo4j"
    uri: str = "bolt://127.0.0.1:7688"
    username: str = "neo4j"
    password_env: str = "HARDWARE_KNOWLEDGE_NEO4J_PASSWORD"
    database: str = "neo4j"

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.provider = str(self.provider or "neo4j").strip().lower().replace("-", "_")
        self.uri = str(self.uri or "").strip()
        self.username = str(self.username or "").strip()
        self.password_env = str(self.password_env or "").strip()
        self.database = str(self.database or "").strip()

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "HardwareKnowledgeGraphSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "uri": self.uri,
            "username": self.username,
            "password_env": self.password_env,
            "database": self.database,
        }


@dataclass(slots=True)
class HardwareFeatureDBSettings:
    enabled: bool = True
    provider: str = "qdrant"
    url: str = "http://127.0.0.1:6333"
    api_key_env: str = "LOCALML_SCHEDULER_QDRANT_API_KEY"
    collection_name: str = "hardware_feature_knowledge"
    code_doc_collection_name: str = "code_doc_chunks"
    optimization_recipe_collection_name: str = "optimization_recipe_chunks"
    api_symbol_collection_name: str = "api_symbol_chunks"
    backend_guidance_collection_name: str = "backend_guidance_rules"
    embedding_model_type: str = "local"
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "cpu"
    distance: str = "Cosine"

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.provider = str(self.provider or "qdrant").strip().lower().replace("-", "_")
        self.url = str(self.url or "http://127.0.0.1:6333").strip()
        self.api_key_env = str(
            self.api_key_env or "LOCALML_SCHEDULER_QDRANT_API_KEY"
        ).strip()
        self.collection_name = str(
            self.collection_name or "hardware_feature_knowledge"
        ).strip()
        self.code_doc_collection_name = str(
            self.code_doc_collection_name or "code_doc_chunks"
        ).strip()
        self.optimization_recipe_collection_name = str(
            self.optimization_recipe_collection_name or "optimization_recipe_chunks"
        ).strip()
        self.api_symbol_collection_name = str(
            self.api_symbol_collection_name or "api_symbol_chunks"
        ).strip()
        self.backend_guidance_collection_name = str(
            self.backend_guidance_collection_name or "backend_guidance_rules"
        ).strip()
        self.embedding_model_type = (
            str(self.embedding_model_type or "local").strip().lower()
        )
        self.embedding_model_name = str(
            self.embedding_model_name or "BAAI/bge-base-en-v1.5"
        ).strip()
        self.embedding_device = str(self.embedding_device or "cpu").strip().lower()
        self.distance = str(self.distance or "Cosine").strip()

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "HardwareFeatureDBSettings":
        return cls(**(payload or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "url": self.url,
            "api_key_env": self.api_key_env,
            "collection_name": self.collection_name,
            "code_doc_collection_name": self.code_doc_collection_name,
            "optimization_recipe_collection_name": self.optimization_recipe_collection_name,
            "api_symbol_collection_name": self.api_symbol_collection_name,
            "backend_guidance_collection_name": self.backend_guidance_collection_name,
            "embedding_model_type": self.embedding_model_type,
            "embedding_model_name": self.embedding_model_name,
            "embedding_device": self.embedding_device,
            "distance": self.distance,
        }


@dataclass(slots=True)
class LogDBSettings:
    provider: str = "postgres"
    dsn_env: str = "LOCALML_SCHEDULER_LOG_DSN"
    schema: str = "scheduler_logs"
    enabled: bool = False

    def __post_init__(self) -> None:
        self.provider = (
            str(self.provider or "postgres").strip().lower().replace("-", "_")
        )

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
    estimated_avg_vram_mb: int | None = None
    estimated_ram_mb: int | None = None
    packing_eligible: bool = False
    packing_family: str = "mlevolve_script"
    packing_max_slowdown_ratio: float | None = None
    backend_allowlist: list[str] = field(default_factory=list)
    batch_probe_enabled: bool = True
    batch_probe_model_key: str | None = None
    runtime_probe_enabled: bool = True
    runtime_probe_target: str | None = None
    runtime_probe_model_key: str | None = None
    runtime_probe_strategy: str = "epoch_1"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SchedulerSubmissionDefaults":
        raw = dict(payload or {})
        removed = sorted(
            {
                "batch_probe_probe_timeout_seconds",
                "batch_probe_poll_interval_seconds",
                "batch_probe_max_multiplier",
                "batch_probe_search_mode",
            }
            & raw.keys()
        )
        if removed:
            raise ValueError(
                "Removed legacy submission-default settings: " + ", ".join(removed)
            )
        instance = cls(**raw)
        instance.backend_allowlist = normalize_backend_allowlist(
            instance.backend_allowlist,
            warn_legacy=True,
        )
        instance.runtime_probe_strategy = (
            str(instance.runtime_probe_strategy or "epoch_1")
            .strip()
            .lower()
            .replace("-", "_")
        )
        return instance

    def to_dict(self) -> dict[str, Any]:
        return {
            "requires_gpu": self.requires_gpu,
            "estimated_vram_mb": self.estimated_vram_mb,
            "estimated_avg_vram_mb": self.estimated_avg_vram_mb,
            "estimated_ram_mb": self.estimated_ram_mb,
            "packing_eligible": self.packing_eligible,
            "packing_family": self.packing_family,
            "packing_max_slowdown_ratio": self.packing_max_slowdown_ratio,
            "backend_allowlist": list(self.backend_allowlist),
            "batch_probe_enabled": self.batch_probe_enabled,
            "batch_probe_model_key": self.batch_probe_model_key,
            "runtime_probe_enabled": self.runtime_probe_enabled,
            "runtime_probe_target": self.runtime_probe_target,
            "runtime_probe_model_key": self.runtime_probe_model_key,
            "runtime_probe_strategy": self.runtime_probe_strategy,
        }


@dataclass(slots=True)
class GpuSchedulerSettings:
    """The single production policy: incremental, slowdown-verified placement.

    The scheduler starts one anchor, considers one newcomer at a time, and
    keeps an addition only when measured/predicted drain-time gain passes
    ``colocation.min_gain``. VRAM is solely a hard admission constraint.
    """

    enabled: bool = True
    mode: str = SCHEDULER_MODE_PARALLEL_TIME_AWARE
    scheduler_decision_mode: str = SCHEDULER_DECISION_MODE_BASELINE
    packing_backend: str = "mps_process"
    exclusive_fallback_enabled: bool = True
    mps_unavailable_policy: str = "exclusive"
    # Read compatibility for one deprecation window. Internal code sees at most
    # the authoritative packed backend plus the explicit exclusive fallback.
    backend_priority: list[str] | None = None
    parallel_job_cap: int | None = None
    priority_window_size: int = 8
    oldest_window_size: int = 4
    starvation_timeout_seconds: float = 1800.0
    device_index: int = 0
    fallback_cooldown_seconds: int = 900
    batch_probe_enabled: bool = True
    batch_probe_target_memory_fraction: float = 0.97
    batch_probe_max_batch_size: int | None = None
    startpoint_probe_enabled: bool = False
    startpoint_probe_max_models: int | None = None
    profiling: GpuProfilingSettings = field(default_factory=GpuProfilingSettings)
    memory: GpuMemorySettings = field(default_factory=GpuMemorySettings)
    telemetry: GpuTelemetrySettings = field(default_factory=GpuTelemetrySettings)
    objective: TimeObjectiveSettings = field(default_factory=TimeObjectiveSettings)
    batch_options: BatchOptionSettings = field(default_factory=BatchOptionSettings)
    source_trial_ranking: SourceTrialRankingSettings = field(
        default_factory=SourceTrialRankingSettings
    )
    colocation: ColocationSettings = field(default_factory=ColocationSettings)
    exclusive_probe: ExclusiveProbeSettings = field(
        default_factory=ExclusiveProbeSettings
    )
    submission_defaults: SchedulerSubmissionDefaults = field(
        default_factory=SchedulerSubmissionDefaults
    )
    mps: MPSSettings = field(default_factory=MPSSettings)
    cuda_process: CudaProcessSettings = field(default_factory=CudaProcessSettings)

    def __post_init__(self) -> None:
        self.mode = normalize_scheduler_mode(self.mode)
        self.scheduler_decision_mode = normalize_scheduler_decision_mode(
            self.scheduler_decision_mode
        )
        self.packing_backend = normalize_packing_backend(self.packing_backend)
        self.mps_unavailable_policy = (
            str(self.mps_unavailable_policy or "exclusive")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if self.mps_unavailable_policy not in {"exclusive", "fail"}:
            raise ValueError("mps_unavailable_policy must be 'exclusive' or 'fail'")
        if self.backend_priority is None:
            self.backend_priority = [self.packing_backend]
            if self.exclusive_fallback_enabled:
                self.backend_priority.append("exclusive")
        else:
            legacy_priority = normalize_backend_allowlist(self.backend_priority)
            packed = [item for item in legacy_priority if item != "exclusive"]
            if len(packed) != 1:
                raise ValueError(
                    "gpu_scheduler.backend_priority is ambiguous. Configure exactly one "
                    "packing_backend (cuda_process or mps_process) plus optional exclusive fallback."
                )
            if packed[0] != self.packing_backend:
                raise ValueError(
                    "gpu_scheduler.backend_priority conflicts with authoritative packing_backend: "
                    f"{packed[0]} != {self.packing_backend}"
                )
            self.backend_priority = legacy_priority or [self.packing_backend]
        if self.parallel_job_cap is not None:
            self.parallel_job_cap = max(1, int(self.parallel_job_cap))
        if self.startpoint_probe_max_models is not None:
            self.startpoint_probe_max_models = max(0, int(self.startpoint_probe_max_models))
        self.priority_window_size = max(1, int(self.priority_window_size))
        self.oldest_window_size = max(1, int(self.oldest_window_size))
        self.starvation_timeout_seconds = max(
            0.0, float(self.starvation_timeout_seconds)
        )
        if self.profiling is None:
            self.profiling = GpuProfilingSettings()
        if isinstance(self.profiling, dict):
            self.profiling = GpuProfilingSettings.from_dict(self.profiling)
        if self.memory is None:
            self.memory = GpuMemorySettings()
        if isinstance(self.memory, dict):
            self.memory = GpuMemorySettings.from_dict(self.memory)
        if self.telemetry is None:
            self.telemetry = GpuTelemetrySettings()
        if isinstance(self.telemetry, dict):
            self.telemetry = GpuTelemetrySettings.from_dict(self.telemetry)
        if self.objective is None:
            self.objective = TimeObjectiveSettings()
        if isinstance(self.objective, dict):
            self.objective = TimeObjectiveSettings.from_dict(self.objective)
        if self.batch_options is None:
            self.batch_options = BatchOptionSettings()
        if isinstance(self.batch_options, dict):
            self.batch_options = BatchOptionSettings.from_dict(self.batch_options)
        if self.source_trial_ranking is None:
            self.source_trial_ranking = SourceTrialRankingSettings()
        if isinstance(self.source_trial_ranking, dict):
            self.source_trial_ranking = SourceTrialRankingSettings.from_dict(
                self.source_trial_ranking
            )
        if self.colocation is None:
            self.colocation = ColocationSettings()
        if isinstance(self.colocation, dict):
            self.colocation = ColocationSettings.from_dict(self.colocation)
        if self.exclusive_probe is None:
            self.exclusive_probe = ExclusiveProbeSettings()
        if isinstance(self.exclusive_probe, dict):
            self.exclusive_probe = ExclusiveProbeSettings.from_dict(
                self.exclusive_probe
            )
        if self.memory.predicted_budget_fraction > self.memory.live_admission_stop_fraction:
            warnings.warn(
                "predicted_budget_fraction exceeds live_admission_stop_fraction; a newly admitted pack may close admission immediately",
                UserWarning,
                stacklevel=2,
            )
        if self.submission_defaults is None:
            self.submission_defaults = SchedulerSubmissionDefaults()
        if isinstance(self.submission_defaults, dict):
            self.submission_defaults = SchedulerSubmissionDefaults.from_dict(
                self.submission_defaults
            )
        if not self.submission_defaults.backend_allowlist:
            self.submission_defaults.backend_allowlist = [self.packing_backend]
        if self.mps is None:
            self.mps = MPSSettings()
        if isinstance(self.mps, dict):
            self.mps = MPSSettings.from_dict(self.mps)
        if self.cuda_process is None:
            self.cuda_process = CudaProcessSettings()
        if isinstance(self.cuda_process, dict):
            self.cuda_process = CudaProcessSettings.from_dict(self.cuda_process)
    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuSchedulerSettings":
        raw = dict(payload or {})
        if "stream" in raw:
            raise ValueError(
                "gpu_scheduler.stream was removed. Independently generated subprocess jobs "
                "cannot be controlled by a parent CUDA stream; choose packing_backend."
            )
        legacy_keys = {
            "adaptive",
            "allow_three_way_packing",
            "auto_pack",
            "beam_width",
            "candidate_window_size",
            "concurrent_backend_allowlist",
            "concurrent_groups_enabled",
            "exact_search_max_jobs",
            "max_packed_jobs_per_gpu",
            "parallel_optimizer",
            "thresholds",
            "batch_probe_min_batch_size",
            "batch_probe_max_search_rounds",
            "batch_probe_search_mode",
        }
        removed = sorted(legacy_keys & raw.keys())
        if removed:
            raise ValueError(
                "Removed legacy gpu_scheduler settings: "
                + ", ".join(removed)
                + ". Configure parallel_time_aware with parallel_job_cap, colocation, objective, batch_options, and memory safety gates."
            )
        legacy_priority = raw.get("backend_priority")
        if legacy_priority is not None:
            normalized_priority = normalize_backend_allowlist(legacy_priority)
            packed = [item for item in normalized_priority if item != "exclusive"]
            if len(packed) != 1:
                raise ValueError(
                    "gpu_scheduler.backend_priority is ambiguous. Replace it with one "
                    "packing_backend: cuda_process or mps_process."
                )
            if "packing_backend" in raw:
                configured = normalize_packing_backend(raw["packing_backend"])
                if configured != packed[0]:
                    raise ValueError(
                        "gpu_scheduler.backend_priority conflicts with packing_backend"
                    )
            else:
                warn_backend_deprecation(
                    "gpu_scheduler.backend_priority is deprecated; use packing_backend. "
                    "The single packed backend was migrated.",
                    stacklevel=2,
                )
                raw["packing_backend"] = packed[0]
            raw["backend_priority"] = normalized_priority
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "scheduler_decision_mode": self.scheduler_decision_mode,
            "packing_backend": self.packing_backend,
            "exclusive_fallback_enabled": self.exclusive_fallback_enabled,
            "mps_unavailable_policy": self.mps_unavailable_policy,
            "parallel_job_cap": self.parallel_job_cap,
            "priority_window_size": self.priority_window_size,
            "oldest_window_size": self.oldest_window_size,
            "starvation_timeout_seconds": self.starvation_timeout_seconds,
            "device_index": self.device_index,
            "fallback_cooldown_seconds": self.fallback_cooldown_seconds,
            "batch_probe_enabled": self.batch_probe_enabled,
            "batch_probe_target_memory_fraction": self.batch_probe_target_memory_fraction,
            "batch_probe_max_batch_size": self.batch_probe_max_batch_size,
            "startpoint_probe_enabled": self.startpoint_probe_enabled,
            "startpoint_probe_max_models": self.startpoint_probe_max_models,
            "profiling": self.profiling.to_dict(),
            "memory": self.memory.to_dict(),
            "telemetry": self.telemetry.to_dict(),
            "objective": self.objective.to_dict(),
            "batch_options": self.batch_options.to_dict(),
            "source_trial_ranking": self.source_trial_ranking.to_dict(),
            "colocation": self.colocation.to_dict(),
            "exclusive_probe": self.exclusive_probe.to_dict(),
            "submission_defaults": self.submission_defaults.to_dict(),
            "mps": self.mps.to_dict(),
            "cuda_process": self.cuda_process.to_dict(),
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
    baseline_cache: BaselineCacheSettings | dict[str, Any] = field(
        default_factory=BaselineCacheSettings
    )
    cache_server_host: str = "127.0.0.1"
    cache_server_port: int = 8765
    cache_socket_name: str = "cache_server.sock"
    redis_cache: RedisCacheSettings | dict[str, Any] = field(
        default_factory=RedisCacheSettings
    )
    auto_resume_recoverable: bool = False
    gpu_scheduler: GpuSchedulerSettings = field(default_factory=GpuSchedulerSettings)
    early_stopping: EarlyStoppingSettings | dict[str, Any] = field(
        default_factory=EarlyStoppingSettings
    )
    prediction: PredictionSettings | dict[str, Any] = field(
        default_factory=PredictionSettings
    )
    graph_db: GraphDBSettings | dict[str, Any] = field(default_factory=GraphDBSettings)
    hardware_knowledge_graph: HardwareKnowledgeGraphSettings | dict[str, Any] = field(
        default_factory=HardwareKnowledgeGraphSettings
    )
    hardware_feature_db: HardwareFeatureDBSettings | dict[str, Any] = field(
        default_factory=HardwareFeatureDBSettings
    )
    log_db: LogDBSettings | dict[str, Any] = field(default_factory=LogDBSettings)
    python_executable: str = field(default_factory=lambda: sys.executable)
    sqlite_busy_timeout_ms: int = 10_000

    db_dir: Path = field(init=False)
    db_path: Path = field(init=False)
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
        if self.early_stopping is None:
            self.early_stopping = EarlyStoppingSettings()
        if isinstance(self.early_stopping, dict):
            self.early_stopping = EarlyStoppingSettings.from_dict(self.early_stopping)
        if self.prediction is None:
            self.prediction = PredictionSettings()
        if isinstance(self.prediction, dict):
            self.prediction = PredictionSettings.from_dict(self.prediction)
        if self.baseline_cache is None:
            self.baseline_cache = BaselineCacheSettings()
        if isinstance(self.baseline_cache, dict):
            self.baseline_cache = BaselineCacheSettings.from_dict(self.baseline_cache)
        if self.redis_cache is None:
            self.redis_cache = RedisCacheSettings()
        if isinstance(self.redis_cache, dict):
            self.redis_cache = RedisCacheSettings.from_dict(self.redis_cache)
        if self.graph_db is None:
            self.graph_db = GraphDBSettings()
        if isinstance(self.graph_db, dict):
            self.graph_db = GraphDBSettings.from_dict(self.graph_db)
        if self.hardware_knowledge_graph is None:
            self.hardware_knowledge_graph = HardwareKnowledgeGraphSettings()
        if isinstance(self.hardware_knowledge_graph, dict):
            self.hardware_knowledge_graph = HardwareKnowledgeGraphSettings.from_dict(
                self.hardware_knowledge_graph
            )
        if self.hardware_feature_db is None:
            self.hardware_feature_db = HardwareFeatureDBSettings()
        if isinstance(self.hardware_feature_db, dict):
            self.hardware_feature_db = HardwareFeatureDBSettings.from_dict(
                self.hardware_feature_db
            )
        if self.log_db is None:
            self.log_db = LogDBSettings()
        if isinstance(self.log_db, dict):
            self.log_db = LogDBSettings.from_dict(self.log_db)
        self.runtime_root = Path(self.runtime_root).resolve()
        self.db_dir = self.runtime_root / "db"
        self.db_path = self.db_dir / "scheduler.sqlite3"
        self.jobs_dir = self.runtime_root / "data" / "jobs"
        self.checkpoints_dir = self.runtime_root / "data" / "checkpoints"
        self.cache_meta_dir = self.runtime_root / "cache_meta"
        self.logs_dir = self.runtime_root / "logs"
        self.events_jsonl_path = self.logs_dir / "events.jsonl"
        self.scheduler_log_path = self.logs_dir / "scheduler.log"
        self.cache_socket_path = self.runtime_root / self.cache_socket_name
        self.service_heartbeat_path = self.runtime_root / "service_heartbeat.json"
        if not self.graph_db.sqlite_evidence_path:
            self.graph_db.sqlite_evidence_path = str(self.db_path)

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any] | None = None, **overrides: Any
    ) -> "SchedulerConfig":
        """Build settings from an in-memory mapping.

        Replay tools and embedding applications use this path after applying
        their own overlays, while :meth:`from_file` handles YAML input.
        """
        data = dict(payload or {})
        data.update(overrides)
        return cls(**data)

    @classmethod
    def from_file(
        cls, path: str | Path | None = None, **overrides: Any
    ) -> "SchedulerConfig":
        payload: dict[str, Any] = {}
        if path:
            with Path(path).open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        payload.update(overrides)
        return cls(**payload)

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

    def checkpoints_for_job(self, job_id: str) -> Path:
        return self.checkpoints_dir / job_id

    def cache_address(self) -> str | tuple[str, int]:
        if sys.platform != "win32":
            address = str(self.cache_socket_path)
            # Unix-domain socket paths are commonly limited to roughly 100
            # bytes. Deep temporary/replay roots therefore use a stable short
            # path while all other runtime files remain under runtime_root.
            if len(address.encode()) >= 100:
                digest = hashlib.sha256(address.encode()).hexdigest()[:20]
                return str(
                    Path(tempfile.gettempdir()) / f"localml-scheduler-{digest}.sock"
                )
            return address
        return (self.cache_server_host, self.cache_server_port)

    def to_dict(self) -> dict[str, Any]:
        assert isinstance(self.baseline_cache, BaselineCacheSettings)
        assert isinstance(self.early_stopping, EarlyStoppingSettings)
        assert isinstance(self.prediction, PredictionSettings)
        assert isinstance(self.graph_db, GraphDBSettings)
        assert isinstance(self.hardware_knowledge_graph, HardwareKnowledgeGraphSettings)
        assert isinstance(self.hardware_feature_db, HardwareFeatureDBSettings)
        assert isinstance(self.log_db, LogDBSettings)
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
            "gpu_scheduler": self.gpu_scheduler.to_dict(),
            "early_stopping": self.early_stopping.to_dict(),
            "prediction": self.prediction.to_dict(),
            "graph_db": self.graph_db.to_dict(),
            "hardware_knowledge_graph": self.hardware_knowledge_graph.to_dict(),
            "hardware_feature_db": self.hardware_feature_db.to_dict(),
            "log_db": self.log_db.to_dict(),
            "python_executable": self.python_executable,
            "sqlite_busy_timeout_ms": self.sqlite_busy_timeout_ms,
        }


SchedulerSettings = SchedulerConfig
