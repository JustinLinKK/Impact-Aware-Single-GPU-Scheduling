"""Reusable local ML job scheduler with GPU-aware single-node execution."""

from .config import (
    BaselineCacheSettings,
    DecisionReplaySettings,
    EarlyStoppingSettings,
    GpuSchedulerSettings,
    HardwareFeatureDBSettings,
    SchedulerConfig,
)
from .dto import JobCommandRequest, JobQuery, PreloadRequest, ReportQuery, SubmitJobRequest
from .domain import (
    BatchProbeProfile,
    BatchProbeSpec,
    BatchProbeTrialResult,
    CacheStats,
    CheckpointPolicy,
    JobConfig,
    JobRun,
    JobSpec,
    JobStatus,
    PackingSpec,
    PreloadSource,
    ProgressSnapshot,
    ResourceRequirements,
    RuntimeProbeSpec,
    RuntimeProfile,
    SchedulingClass,
    TrainingJob,
    WorkloadIdentity,
)
from .http_client import SchedulerHTTPClient


def __getattr__(name):
    # HTTP-only agents should not import the scheduler's PyTorch runtime.
    if name == "SchedulerClient":
        from .client import SchedulerClient

        return SchedulerClient
    if name == "SchedulerEngine":
        from .engine import SchedulerEngine

        return SchedulerEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BatchProbeProfile",
    "BatchProbeSpec",
    "BatchProbeTrialResult",
    "CacheStats",
    "CheckpointPolicy",
    "DecisionReplaySettings",
    "EarlyStoppingSettings",
    "JobConfig",
    "JobCommandRequest",
    "JobQuery",
    "JobRun",
    "JobSpec",
    "JobStatus",
    "PackingSpec",
    "PreloadSource",
    "PreloadRequest",
    "ProgressSnapshot",
    "ResourceRequirements",
    "ReportQuery",
    "RuntimeProbeSpec",
    "RuntimeProfile",
    "SchedulingClass",
    "BaselineCacheSettings",
    "GpuSchedulerSettings",
    "HardwareFeatureDBSettings",
    "SchedulerClient",
    "SchedulerConfig",
    "SchedulerEngine",
    "SchedulerHTTPClient",
    "SubmitJobRequest",
    "TrainingJob",
    "WorkloadIdentity",
]
