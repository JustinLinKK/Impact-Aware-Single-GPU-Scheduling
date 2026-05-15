"""Public client facade for scheduler commands and queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import yaml

from .config import SchedulerConfig
from .domain import (
    BatchProbeProfile,
    BatchSizeObservation,
    CombinationProfile,
    CommandType,
    JobRun,
    JobSpec,
    JobStatus,
    PairProfile,
    RunProfile,
    RuntimeProfile,
    SchedulerReport,
    SoloProfile,
    TrainingJob,
    parse_timestamp,
    stable_job_id,
    utc_now,
)
from .dto import JobCommandRequest, JobQuery, PreloadRequest, ReportQuery, SubmitJobRequest
from .graph_knowledge import SchedulerKnowledgeBase
from .model_cache.cache_server import CacheClient
from .scheduler.service import SchedulerService
from .storage.state_store import StateStore


class SchedulerClient:
    """Submit work and inspect state through a small command/query surface.

    The new MCP-oriented aggregate queries are forwarded through this client so
    CLI/code callers and MCP callers share the same response shapes.
    """

    def __init__(self, settings: SchedulerConfig | None = None):
        self.settings = settings or SchedulerConfig()
        self.store = StateStore(self.settings)
        self.knowledge = SchedulerKnowledgeBase(self.store)
        self._hardware_feature_store = None

    def create_engine(self):
        from .engine import SchedulerEngine

        return SchedulerEngine(self.settings)

    def create_service(self, **kwargs: Any) -> SchedulerService:
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

    def submit(self, request: SubmitJobRequest | JobSpec | TrainingJob) -> TrainingJob:
        if isinstance(request, TrainingJob):
            return self.store.submit_job(request)
        if isinstance(request, JobSpec):
            request = SubmitJobRequest(spec=request)
        run = request.run or JobRun()
        return self.store.submit_job(TrainingJob.from_parts(request.spec, run))

    def submit_from_payload(self, payload: dict[str, Any]) -> TrainingJob:
        return self.submit(self._normalize_job_payload(payload))

    def submit_from_file(self, job_spec_path: str | Path) -> TrainingJob:
        with Path(job_spec_path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return self.submit_from_payload(payload)

    def command(self, request: JobCommandRequest) -> int:
        return self.store.enqueue_command(request.command_type, job_id=request.job_id)

    def inspect(self, query: JobQuery | str) -> TrainingJob | None:
        job_id = query.job_id if isinstance(query, JobQuery) else str(query)
        return self.store.get_job(job_id)

    def list_jobs(self) -> list[TrainingJob]:
        return self.store.list_jobs()

    def preload(self, request: PreloadRequest) -> int:
        return self.store.enqueue_command(
            CommandType.PRELOAD,
            payload={
                "baseline_model_id": request.model_id,
                "baseline_model_path": str(request.model_path),
                "loader_target": request.loader_target,
                "pin": request.pin,
            },
        )

    def pause(self, job_id: str) -> int:
        return self.command(JobCommandRequest(job_id=job_id, command_type=CommandType.PAUSE))

    def resume(self, job_id: str) -> int:
        return self.command(JobCommandRequest(job_id=job_id, command_type=CommandType.RESUME))

    def cancel(self, job_id: str) -> int:
        return self.command(JobCommandRequest(job_id=job_id, command_type=CommandType.CANCEL))

    def cache_stats(self) -> dict[str, Any]:
        try:
            client = CacheClient(self.settings)
            return client.stats()
        except Exception:
            summary = self.store.cache_metadata_summary()
            summary.update(
                {
                    "memory_budget_bytes": self.settings.baseline_cache.memory_budget_bytes,
                    "entry_capacity": self.settings.baseline_cache.entry_capacity,
                    "max_ram_percent": self.settings.baseline_cache.max_ram_percent,
                    "system_total_memory_bytes": None,
                    "effective_memory_budget_bytes": self.settings.baseline_cache.memory_budget_bytes,
                }
            )
            return {"stats": summary, "entries": []}

    def report(self, query: ReportQuery | None = None) -> dict[str, Any]:
        del query
        report: SchedulerReport = self.store.report()
        return report.to_dict()

    def list_events(self, *, job_id: str | None = None, event_type: str | None = None) -> list[dict[str, Any]]:
        return self.store.list_events(job_id=job_id, event_type=event_type)

    def get_solo_profile(self, signature: str) -> SoloProfile | None:
        return self.store.get_solo_profile(signature)

    def upsert_solo_profile(self, profile: SoloProfile) -> SoloProfile:
        return self.store.upsert_solo_profile(profile)

    def get_pair_profile(self, left_signature: str, right_signature: str, *, backend_name: str | None = None) -> PairProfile | None:
        return self.store.get_pair_profile(left_signature, right_signature, backend_name=backend_name)

    def upsert_pair_profile(self, profile: PairProfile) -> PairProfile:
        return self.store.upsert_pair_profile(profile)

    def get_runtime_profile(self, signature: str, *, resolved_batch_size: int, backend_name: str) -> RuntimeProfile | None:
        return self.store.get_runtime_profile(signature, resolved_batch_size=resolved_batch_size, backend_name=backend_name)

    def list_runtime_profiles(self, **kwargs: Any) -> list[RuntimeProfile]:
        return self.store.list_runtime_profiles(**kwargs)

    def upsert_runtime_profile(self, profile: RuntimeProfile) -> RuntimeProfile:
        return self.store.upsert_runtime_profile(profile)

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

    def list_run_profiles(self, **kwargs: Any) -> list[RunProfile]:
        return self.knowledge.list_run_profiles(**kwargs)

    def get_job_graph_context(self, job_id: str) -> dict[str, Any]:
        return self.knowledge.get_job_graph_context(job_id)

    def search_hardware(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.knowledge.search_hardware(**kwargs)

    def get_hardware_context(self, hardware_key: str = "current", include_scheduler_limits: bool = True) -> dict[str, Any]:
        return self.knowledge.get_hardware_context(
            hardware_key=hardware_key,
            include_scheduler_limits=include_scheduler_limits,
        )

    def get_job_design_context(self, candidate: dict[str, Any], limit: int = 5) -> dict[str, Any]:
        return self.knowledge.get_job_design_context(candidate=candidate, limit=limit)

    def search_profiles(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.knowledge.search_profiles(**kwargs)

    def get_runtime_estimate(self, **kwargs: Any) -> dict[str, Any]:
        return self.knowledge.get_runtime_estimate(**kwargs)

    def recommend_batch_size(self, **kwargs: Any) -> dict[str, Any]:
        return self.knowledge.recommend_batch_size(**kwargs)

    def recommend_epochs(self, **kwargs: Any) -> dict[str, Any]:
        return self.knowledge.recommend_epochs(**kwargs)

    def get_packet_compatibility(self, **kwargs: Any) -> dict[str, Any]:
        return self.knowledge.get_packet_compatibility(**kwargs)

    def search_profile_summaries(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.knowledge.search_profile_summaries(**kwargs)

    def record_tuning_outcome(self, **kwargs: Any) -> dict[str, Any]:
        return self.knowledge.record_tuning_outcome(**kwargs)

    def _feature_store(self):
        if self._hardware_feature_store is None:
            from .hardware_features import HardwareFeatureStore

            self._hardware_feature_store = HardwareFeatureStore(self.settings)
        return self._hardware_feature_store

    def ingest_hardware_features(
        self,
        *,
        source: str | Path | None = None,
        recreate: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return self._feature_store().ingest_source(source, recreate=recreate, dry_run=dry_run)

    def search_hardware_features(
        self,
        *,
        query: str,
        hardware_key: str = "current",
        architecture: str | None = None,
        vendor: str | None = None,
        workload_type: str | None = None,
        framework: str | None = "pytorch",
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        hardware_context = self.get_hardware_context(hardware_key, include_scheduler_limits=True)
        return self._feature_store().search(
            query=query,
            hardware_context=hardware_context,
            architecture=architecture,
            vendor=vendor,
            workload_type=workload_type,
            framework=framework,
            limit=limit,
        )

    def get_hardware_feature_context(
        self,
        *,
        hardware_key: str = "current",
        workload_type: str | None = None,
        model_family: str | None = None,
        framework: str | None = "pytorch",
        limit: int = 8,
    ) -> dict[str, Any]:
        hardware_context = self.get_hardware_context(hardware_key, include_scheduler_limits=True)
        hardware = hardware_context.get("hardware") or {}
        query_parts = [
            str(hardware.get("gpu_name") or "current hardware"),
            str(hardware.get("compute_capability") or ""),
            str(workload_type or ""),
            str(model_family or ""),
            str(framework or ""),
            "training optimization precision memory dataloader checkpointing",
        ]
        query = " ".join(part for part in query_parts if part.strip())
        matches = self._feature_store().search(
            query=query,
            hardware_context=hardware_context,
            workload_type=workload_type,
            framework=framework,
            limit=limit,
        )
        return {
            "found": bool(matches),
            "hardware_context": hardware_context,
            "query": query,
            "matches": matches,
            "evidence_refs": [f"hardware_feature:{match['record_id']}" for match in matches if match.get("record_id")],
        }

    def get_hardware_optimization_context(self, *, candidate: dict[str, Any], limit: int = 8) -> dict[str, Any]:
        job_design_context = self.get_job_design_context(candidate, limit=limit)
        feature_context = self.get_hardware_feature_context(
            hardware_key="current",
            workload_type=candidate.get("workload_type") or candidate.get("task_type"),
            model_family=candidate.get("model_family") or candidate.get("packing_family"),
            framework=str(candidate.get("framework") or "pytorch"),
            limit=limit,
        )
        recommendations: list[str] = []
        batch_recommendation = job_design_context.get("batch_size_recommendation") or {}
        if batch_recommendation.get("found") and batch_recommendation.get("recommended_batch_size") is not None:
            recommendations.append(f"Use graph-recommended batch size {batch_recommendation['recommended_batch_size']} as the starting physical batch size.")
        epoch_recommendation = job_design_context.get("epoch_recommendation") or {}
        if epoch_recommendation.get("found") and epoch_recommendation.get("recommended_epochs") is not None:
            recommendations.append(f"Use historical epoch recommendation {epoch_recommendation['recommended_epochs']} as the initial epoch budget.")
        risk_notes = list(job_design_context.get("risk_flags") or [])
        for match in feature_context.get("matches") or []:
            for pattern in match.get("recommended_patterns") or []:
                if pattern not in recommendations:
                    recommendations.append(pattern)
            for pattern in match.get("avoid_patterns") or []:
                if pattern not in risk_notes:
                    risk_notes.append(pattern)
        evidence_refs = list(job_design_context.get("evidence_refs") or [])
        evidence_refs.extend(feature_context.get("evidence_refs") or [])
        feature_confidences = [
            float(match.get("confidence"))
            for match in feature_context.get("matches") or []
            if match.get("confidence") is not None
        ]
        graph_confidence = float(job_design_context.get("confidence") or 0.0)
        feature_confidence = max(feature_confidences) if feature_confidences else 0.0
        confidence = round(max(graph_confidence, feature_confidence), 3)
        return {
            "hardware_context": job_design_context.get("hardware_context"),
            "job_design_context": job_design_context,
            "feature_context": feature_context,
            "combined_recommendations": recommendations[: max(1, int(limit))],
            "risk_notes": risk_notes,
            "evidence_refs": evidence_refs,
            "confidence": confidence,
        }

    def dump_jobs_json(self) -> str:
        return json.dumps([job.to_dict() for job in self.list_jobs()], indent=2, sort_keys=True)
