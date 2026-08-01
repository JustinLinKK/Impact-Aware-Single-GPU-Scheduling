"""Repository protocols for layered scheduler storage."""

from __future__ import annotations

from typing import Any, Iterable, Protocol

from ..domain import (
    BatchProbeProfile,
    BatchSizeObservation,
    CommandType,
    CombinationProfile,
    JobCommand,
    JobStatus,
    PairProfile,
    RuntimeProfile,
    SchedulerReport,
    SoloProfile,
    TrainingJob,
)


class JobRepository(Protocol):
    def save_job(self, job: TrainingJob) -> None:
        ...

    def submit_job(self, job: TrainingJob) -> TrainingJob:
        ...

    def get_job(self, job_id: str) -> TrainingJob | None:
        ...

    def list_jobs(self, statuses: Iterable[JobStatus | str] | None = None) -> list[TrainingJob]:
        ...

    def runnable_jobs(self) -> list[TrainingJob]:
        ...

    def update_job(self, job_id: str, **kwargs: Any) -> TrainingJob:
        ...

    def set_job_status(self, job_id: str, status: JobStatus, *, reason: str | None = None, hold: bool | None = None) -> TrainingJob:
        ...


class CommandRepository(Protocol):
    def enqueue_command(self, command_type: CommandType, *, job_id: str | None = None, payload: dict[str, Any] | None = None) -> int:
        ...

    def fetch_pending_commands(self, limit: int = 100) -> list[JobCommand]:
        ...

    def mark_command_processed(self, command_id: int) -> None:
        ...


class CheckpointRepository(Protocol):
    def record_checkpoint(self, job_id: str, checkpoint_path: str, metadata: dict[str, Any] | None = None) -> None:
        ...

    def latest_checkpoint(self, job_id: str) -> str | None:
        ...


class EventRepository(Protocol):
    def log_event(self, event_type: str, *, job_id: str | None = None, payload: dict[str, Any] | None = None) -> None:
        ...

    def list_events(self, *, job_id: str | None = None, event_type: str | None = None) -> list[dict[str, Any]]:
        ...


class CacheRepository(Protocol):
    def update_cache_metadata(self, model_id: str, baseline_model_path: str, **kwargs: Any) -> None:
        ...

    def cache_metadata_summary(self) -> dict[str, Any]:
        ...


class ProfileRepository(Protocol):
    def upsert_solo_profile(self, profile: SoloProfile) -> SoloProfile:
        ...

    def get_solo_profile(self, signature: str, *, hardware_key: str | None = None) -> SoloProfile | None:
        ...

    def upsert_pair_profile(self, profile: PairProfile) -> PairProfile:
        ...

    def get_pair_profile(self, left_signature: str, right_signature: str, *, hardware_key: str | None = None, backend_name: str | None = None) -> PairProfile | None:
        ...

    def upsert_runtime_profile(self, profile: RuntimeProfile) -> RuntimeProfile:
        ...

    def get_runtime_profile(self, signature: str, *, resolved_batch_size: int, backend_name: str, hardware_key: str | None = None) -> RuntimeProfile | None:
        ...

    def upsert_batch_probe_profile(self, profile: BatchProbeProfile) -> BatchProbeProfile:
        ...

    def get_batch_probe_profile(self, probe_key: str) -> BatchProbeProfile | None:
        ...

    def upsert_batch_size_observation(self, observation: BatchSizeObservation) -> BatchSizeObservation:
        ...

    def get_batch_size_observation(self, **kwargs: Any) -> BatchSizeObservation | None:
        ...

    def list_batch_size_observations(self, **kwargs: Any) -> list[BatchSizeObservation]:
        ...

    def upsert_combination_profile(self, profile: CombinationProfile) -> CombinationProfile:
        ...

    def best_combination_profile(self, **kwargs: Any) -> CombinationProfile | None:
        ...


class ReportingRepository(Protocol):
    def report(self) -> SchedulerReport:
        ...
