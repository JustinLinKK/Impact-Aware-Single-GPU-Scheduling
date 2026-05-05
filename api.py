"""Python API for the scheduler."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import yaml

from .schemas import parse_timestamp, utc_now
from .model_cache.cache_server import CacheClient
from .schemas import (
    BatchProbeProfile,
    BatchSizeObservation,
    CombinationProfile,
    CommandType,
    JobStatus,
    PairProfile,
    SoloProfile,
    TrainingJob,
    stable_job_id,
)
from .scheduler.service import SchedulerService
from .settings import SchedulerSettings
from .storage.sqlite_store import SQLiteStateStore


class LocalMLSchedulerAPI:
    """High-level Python API and CLI backend."""

    def __init__(self, settings: SchedulerSettings | None = None):
        self.settings = settings or SchedulerSettings()
        self.store = SQLiteStateStore(self.settings)

    def create_scheduler_service(self, **kwargs: Any) -> SchedulerService:
        return SchedulerService(self.settings, store=self.store, **kwargs)

    def scheduler_service_heartbeat(self) -> dict[str, Any] | None:
        path = self.settings.service_heartbeat_path
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def scheduler_service_active(self, *, max_staleness_seconds: float | None = None) -> bool:
        heartbeat = self.scheduler_service_heartbeat()
        if not heartbeat:
            return False
        if heartbeat.get("status") != "running":
            return False
        updated_at = parse_timestamp(heartbeat.get("updated_at"))
        if updated_at is None:
            return False
        now = parse_timestamp(utc_now())
        if now is None:
            return False
        stale_after = max_staleness_seconds
        if stale_after is None:
            stale_after = max(5.0, float(self.settings.scheduler_poll_interval_seconds) * 4.0)
        return (now - updated_at).total_seconds() <= float(stale_after)

    def _normalize_job_payload(self, payload: dict[str, Any]) -> TrainingJob:
        payload = dict(payload)
        payload.setdefault("job_id", stable_job_id(payload.get("job_id")))
        payload.setdefault("status", JobStatus.PENDING.value)
        payload.setdefault("submitted_at", utc_now())
        payload.setdefault("metadata", {})
        payload.setdefault("resource_requirements", {})
        payload.setdefault("checkpoint_policy", {})
        payload.setdefault("status_timestamps", {})
        if "config" not in payload:
            payload["config"] = {
                "runner_target": payload.pop("runner_target"),
                "runner_kwargs": payload.pop("runner_kwargs", {}),
                "loader_target": payload.pop("loader_target", None),
                "max_steps": payload.get("max_steps"),
                "max_epochs": payload.get("max_epochs"),
                "seed": payload.pop("seed", None),
                "python_executable": payload.pop("python_executable", None),
                "env": payload.pop("env", {}),
            }
        return TrainingJob.from_dict(payload)

    def submit_job(self, job: TrainingJob) -> TrainingJob:
        return self.store.submit_job(job)

    def submit_job_from_payload(self, payload: dict[str, Any]) -> TrainingJob:
        return self.submit_job(self._normalize_job_payload(payload))

    def submit_job_from_file(self, job_spec_path: str | Path) -> TrainingJob:
        with Path(job_spec_path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return self.submit_job_from_payload(payload)

    def list_jobs(self) -> list[TrainingJob]:
        return self.store.list_jobs()

    def get_job(self, job_id: str) -> TrainingJob | None:
        return self.store.get_job(job_id)

    def pause_job(self, job_id: str) -> int:
        return self.store.enqueue_command(CommandType.PAUSE, job_id=job_id)

    def resume_job(self, job_id: str) -> int:
        return self.store.enqueue_command(CommandType.RESUME, job_id=job_id)

    def cancel_job(self, job_id: str) -> int:
        return self.store.enqueue_command(CommandType.CANCEL, job_id=job_id)

    def preload_model(self, baseline_model_id: str, baseline_model_path: str, *, loader_target: str | None = None, pin: bool = False) -> int:
        return self.store.enqueue_command(
            CommandType.PRELOAD,
            payload={
                "baseline_model_id": baseline_model_id,
                "baseline_model_path": baseline_model_path,
                "loader_target": loader_target,
                "pin": pin,
            },
        )

    def cache_stats(self) -> dict[str, Any]:
        try:
            client = CacheClient(self.settings)
            return client.stats()
        except Exception:
            summary = self.store.cache_metadata_summary()
            return {"stats": summary, "entries": []}

    def report(self) -> dict[str, Any]:
        return self.store.report().to_dict()

    def get_solo_profile(self, signature: str) -> SoloProfile | None:
        return self.store.get_solo_profile(signature)

    def upsert_solo_profile(self, profile: SoloProfile) -> SoloProfile:
        return self.store.upsert_solo_profile(profile)

    def get_pair_profile(self, left_signature: str, right_signature: str) -> PairProfile | None:
        return self.store.get_pair_profile(left_signature, right_signature)

    def upsert_pair_profile(self, profile: PairProfile) -> PairProfile:
        return self.store.upsert_pair_profile(profile)

    def get_batch_probe_profile(self, probe_key: str) -> BatchProbeProfile | None:
        return self.store.get_batch_probe_profile(probe_key)

    def upsert_batch_probe_profile(self, profile: BatchProbeProfile) -> BatchProbeProfile:
        return self.store.upsert_batch_probe_profile(profile)

    def get_batch_size_observation(self, **kwargs: Any) -> BatchSizeObservation | None:
        return self.store.get_batch_size_observation(**kwargs)

    def list_batch_size_observations(self, **kwargs: Any) -> list[BatchSizeObservation]:
        return self.store.list_batch_size_observations(**kwargs)

    def upsert_batch_size_observation(self, observation: BatchSizeObservation) -> BatchSizeObservation:
        return self.store.upsert_batch_size_observation(observation)

    def best_combination_profile(self, **kwargs: Any) -> CombinationProfile | None:
        return self.store.best_combination_profile(**kwargs)

    def list_combination_profiles(self, **kwargs: Any) -> list[CombinationProfile]:
        return self.store.list_combination_profiles(**kwargs)

    def upsert_combination_profile(self, profile: CombinationProfile) -> CombinationProfile:
        return self.store.upsert_combination_profile(profile)

    def dump_jobs_json(self) -> str:
        return json.dumps([job.to_dict() for job in self.list_jobs()], indent=2, sort_keys=True)
