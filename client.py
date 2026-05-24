"""Public client facade for scheduler commands and queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import uuid

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
from .redis_cache import invalidate_graph_cache
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
        self._code_knowledge_store = None

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

    def submit_many(self, requests: list[SubmitJobRequest | JobSpec | TrainingJob]) -> list[TrainingJob]:
        """Submit a scheduler-visible round before callers start polling results."""
        submitted: list[TrainingJob] = []
        for request in requests:
            submitted.append(self.submit(request))
        return submitted

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

    def rebuild_evidence_graph(self, *, dry_run: bool = True, wipe: bool = False) -> dict[str, Any]:
        jobs = [job for job in self.store.list_jobs() if job.status.is_terminal]
        solo_profiles = list(self.store.list_solo_profiles()) if hasattr(self.store, "list_solo_profiles") else []
        pair_profiles = list(self.store.list_pair_profiles()) if hasattr(self.store, "list_pair_profiles") else []
        runtime_profiles = list(self.store.list_runtime_profiles()) if hasattr(self.store, "list_runtime_profiles") else []
        batch_probe_profiles = list(self.store.list_batch_probe_profiles()) if hasattr(self.store, "list_batch_probe_profiles") else []
        batch_size_observations = list(self.store.list_batch_size_observations()) if hasattr(self.store, "list_batch_size_observations") else []
        combination_profiles = list(self.store.list_combination_profiles()) if hasattr(self.store, "list_combination_profiles") else []
        counts = {
            "terminal_jobs": len(jobs),
            "solo_profiles": len(solo_profiles),
            "pair_profiles": len(pair_profiles),
            "runtime_profiles": len(runtime_profiles),
            "batch_probe_profiles": len(batch_probe_profiles),
            "batch_size_observations": len(batch_size_observations),
            "combination_profiles": len(combination_profiles),
        }
        if dry_run:
            return {"ok": True, "dry_run": True, "would_write": counts, "wipe": False}
        from .storage.neo4j_store import Neo4jStateStore

        target = getattr(self.store, "mirror_backend", None)
        if target is None or not isinstance(target, Neo4jStateStore):
            target = Neo4jStateStore(self.settings)
        if wipe:
            target._run_write("MATCH (n) DETACH DELETE n")
            target._apply_constraints()
        for job in jobs:
            target.record_scheduler_job_evidence(job)
        for profile in solo_profiles:
            target.record_solo_profile_evidence(profile)
        for profile in pair_profiles:
            target.record_pair_profile_evidence(profile)
        for profile in runtime_profiles:
            target.record_runtime_profile_evidence(profile)
        for profile in batch_probe_profiles:
            target.record_batch_probe_evidence(profile)
        for observation in batch_size_observations:
            target.record_batch_size_observation_evidence(observation)
        for profile in combination_profiles:
            target.record_combination_profile_evidence(profile)
        invalidate_graph_cache(self.settings)
        return {"ok": True, "dry_run": False, "written": counts, "wipe": bool(wipe)}

    def _feature_store(self):
        if self._hardware_feature_store is None:
            from .hardware_features import HardwareFeatureStore

            self._hardware_feature_store = HardwareFeatureStore(self.settings)
        return self._hardware_feature_store

    def _code_store(self):
        if self._code_knowledge_store is None:
            from .code_knowledge import CodeKnowledgeStore

            self._code_knowledge_store = CodeKnowledgeStore(self.settings)
        return self._code_knowledge_store

    def ingest_hardware_features(
        self,
        *,
        source: str | Path | None = None,
        recreate: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return self._feature_store().ingest_source(source, recreate=recreate, dry_run=dry_run)

    def ingest_code_knowledge(
        self,
        *,
        source: str | Path | None = None,
        recreate: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return self._code_store().ingest_source(source, recreate=recreate, dry_run=dry_run)

    def ingest_schema_knowledge(
        self,
        *,
        schema_root: str | Path = "schema",
        recreate: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        from .code_knowledge.records import convert_hardware_feature_records, load_code_knowledge_records
        from .hardware_features.records import load_feature_records

        root = Path(schema_root)
        hardware_path = root / "hardware_feature_records"
        code_doc_path = root / "code_doc_chunks"
        api_symbol_path = root / "api_symbol_chunks"

        hardware_records = load_feature_records(hardware_path)
        code_doc_records = load_code_knowledge_records(code_doc_path)
        api_symbol_records = load_code_knowledge_records(api_symbol_path)
        converted_records = convert_hardware_feature_records(hardware_records)
        code_records = code_doc_records + api_symbol_records + converted_records

        hardware_result = self._feature_store().ingest_records(
            hardware_records,
            recreate=recreate,
            dry_run=dry_run,
        )
        code_result = self._code_store().ingest_records(
            code_records,
            recreate=recreate,
            dry_run=dry_run,
        )
        return {
            "ok": bool(hardware_result.get("ok")) and bool(code_result.get("ok")),
            "dry_run": bool(dry_run),
            "schema_root": str(root),
            "collections": {
                "hardware_feature_knowledge": {
                    "target": self.settings.hardware_feature_db.collection_name,
                    "record_count": len(hardware_records),
                    "result": hardware_result,
                },
                "code_doc_chunks": {
                    "target": self.settings.hardware_feature_db.code_doc_collection_name,
                    "record_count": sum(1 for record in code_records if record.get("record_type") == "code_doc_chunks"),
                },
                "optimization_recipe_chunks": {
                    "target": self.settings.hardware_feature_db.optimization_recipe_collection_name,
                    "record_count": sum(1 for record in code_records if record.get("record_type") == "optimization_recipe_chunks"),
                },
                "api_symbol_chunks": {
                    "target": self.settings.hardware_feature_db.api_symbol_collection_name,
                    "record_count": sum(1 for record in code_records if record.get("record_type") == "api_symbol_chunks"),
                },
            },
            "code_knowledge_result": code_result,
            "source_counts": {
                "hardware_feature_records": len(hardware_records),
                "code_doc_chunks": len(code_doc_records),
                "api_symbol_chunks": len(api_symbol_records),
                "converted_from_hardware": len(converted_records),
            },
        }

    def get_profile_evidence(self, *, candidate: dict[str, Any], limit: int = 8) -> dict[str, Any]:
        return self.knowledge.get_profile_evidence(candidate=candidate, limit=limit)

    def search_code_knowledge(
        self,
        *,
        query: str,
        filters: dict[str, Any] | None = None,
        record_types: list[str] | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        return self._code_store().search(
            query=query,
            filters=filters or {},
            record_types=record_types,
            limit=limit,
        )

    def _vector_filters_from_context(
        self,
        candidate: dict[str, Any],
        graph_context: dict[str, Any],
    ) -> dict[str, Any]:
        hardware_context = graph_context.get("hardware_context") or {}
        hardware = hardware_context.get("hardware") or {}
        diagnosis = graph_context.get("derived_diagnosis") or {}
        filters: dict[str, Any] = {
            "framework": str(candidate.get("framework") or "pytorch"),
        }
        if candidate.get("model_family") or candidate.get("packing_family"):
            filters["model_families"] = candidate.get("model_family") or candidate.get("packing_family")
        if candidate.get("workload_type") or candidate.get("task_type"):
            filters["workload_types"] = candidate.get("workload_type") or candidate.get("task_type")
        if diagnosis.get("profile_symptoms"):
            filters["profile_symptoms"] = list(diagnosis["profile_symptoms"])
        if diagnosis.get("optimization_targets"):
            filters["optimization_targets"] = list(diagnosis["optimization_targets"])
        hardware_features = list((hardware.get("technology_keys") or []))
        if not hardware_features and hardware.get("compute_capability"):
            hardware_features.append(f"cuda_capability_{str(hardware['compute_capability']).replace('.', '')}")
        if hardware_features:
            filters["hardware_feature_keys"] = hardware_features
        return {key: value for key, value in filters.items() if value}

    def _code_query_from_context(self, candidate: dict[str, Any], graph_context: dict[str, Any]) -> str:
        diagnosis = graph_context.get("derived_diagnosis") or {}
        parts = [
            str(candidate.get("framework") or "pytorch"),
            str(candidate.get("model_family") or candidate.get("packing_family") or ""),
            str(candidate.get("workload_type") or candidate.get("task_type") or ""),
            " ".join(diagnosis.get("profile_symptoms") or []),
            " ".join(diagnosis.get("optimization_targets") or []),
            "training optimization code batch size precision throughput vram",
        ]
        return " ".join(part for part in parts if part.strip())

    def get_code_optimization_context(
        self,
        *,
        candidate: dict[str, Any],
        graph_context: dict[str, Any] | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        graph_context = graph_context or self.get_profile_evidence(candidate=candidate, limit=limit)
        query = self._code_query_from_context(candidate, graph_context)
        filters = self._vector_filters_from_context(candidate, graph_context)
        results = self.search_code_knowledge(
            query=query,
            filters=filters,
            record_types=["optimization_recipe_chunks", "code_doc_chunks", "api_symbol_chunks"],
            limit=limit,
        )
        if not results and filters:
            relaxed_filters = {"framework": filters.get("framework", "pytorch")}
            results = self.search_code_knowledge(
                query=query,
                filters=relaxed_filters,
                record_types=["optimization_recipe_chunks", "code_doc_chunks", "api_symbol_chunks"],
                limit=limit,
            )
        recipes = [item for item in results if item.get("record_type") == "optimization_recipe_chunks"]
        docs = [item for item in results if item.get("record_type") == "code_doc_chunks"]
        api_symbols = [item for item in results if item.get("record_type") == "api_symbol_chunks"]
        return {
            "found": bool(results),
            "query": query,
            "filters": filters,
            "vector_evidence": {
                "recipes": recipes,
                "docs": docs,
                "api_symbols": api_symbols,
            },
            "evidence_refs": [
                f"code_knowledge:{item.get('record_type')}:{item.get('record_id')}"
                for item in results
                if item.get("record_id")
            ],
        }

    def get_optimization_context(self, *, candidate: dict[str, Any], limit: int = 8) -> dict[str, Any]:
        graph_context = self.get_profile_evidence(candidate=candidate, limit=limit)
        code_context = self.get_code_optimization_context(candidate=candidate, graph_context=graph_context, limit=limit)
        recommendations: list[str] = []
        risks: list[str] = list(graph_context.get("risk_flags") or [])
        batch_recommendation = graph_context.get("batch_size_recommendation") or {}
        if batch_recommendation.get("found") and batch_recommendation.get("recommended_batch_size") is not None:
            recommendations.append(f"Use graph-recommended physical batch size {batch_recommendation['recommended_batch_size']} as the starting point.")
        epoch_recommendation = graph_context.get("epoch_recommendation") or {}
        if epoch_recommendation.get("found") and epoch_recommendation.get("recommended_epochs") is not None:
            recommendations.append(f"Use historical epoch budget {epoch_recommendation['recommended_epochs']} unless the agent has a scoring reason to differ.")
        vector_evidence = code_context.get("vector_evidence") or {}
        for item in (vector_evidence.get("recipes") or []) + (vector_evidence.get("docs") or []):
            for pattern in item.get("recommended_patterns") or []:
                if pattern not in recommendations:
                    recommendations.append(pattern)
            for pattern in item.get("avoid_patterns") or []:
                if pattern not in risks:
                    risks.append(pattern)
        graph_confidence = float(graph_context.get("confidence") or 0.0)
        vector_confidences = [
            float(item.get("confidence"))
            for group in vector_evidence.values()
            for item in (group or [])
            if item.get("confidence") is not None
        ]
        vector_confidence = max(vector_confidences) if vector_confidences else 0.0
        confidence = round(max(graph_confidence, vector_confidence), 3)
        return {
            "hardware_context": graph_context.get("hardware_context"),
            "graph_evidence": graph_context.get("graph_evidence") or {"exact_profiles": [], "similar_profiles": [], "packed_profiles": []},
            "derived_diagnosis": graph_context.get("derived_diagnosis") or {"profile_symptoms": [], "optimization_targets": []},
            "vector_evidence": vector_evidence,
            "recommendations": recommendations[: max(1, int(limit))],
            "risk_flags": risks,
            "evidence_refs": list(graph_context.get("evidence_refs") or []) + list(code_context.get("evidence_refs") or []),
            "confidence": confidence,
        }

    def plan_job_packet(self, *, candidates: list[dict[str, Any]], limit: int = 8) -> dict[str, Any]:
        """Plan a round of jobs together so probes/packing evidence can be considered before dispatch."""
        normalized_candidates = [dict(candidate or {}) for candidate in candidates]
        packet_id = f"packet-{uuid.uuid4().hex[:12]}"
        jobs: list[dict[str, Any]] = []
        evidence_refs: list[str] = []
        confidences: list[float] = []
        for index, candidate in enumerate(normalized_candidates):
            try:
                context = self.get_optimization_context(candidate=candidate, limit=limit)
            except Exception as exc:
                context = {
                    "hardware_context": None,
                    "graph_evidence": {"exact_profiles": [], "similar_profiles": [], "packed_profiles": []},
                    "derived_diagnosis": {"profile_symptoms": ["context_lookup_failed"], "optimization_targets": []},
                    "vector_evidence": {},
                    "recommendations": [],
                    "risk_flags": [f"context lookup failed: {exc}"],
                    "evidence_refs": [],
                    "confidence": 0.0,
                }
            evidence_refs.extend(str(ref) for ref in context.get("evidence_refs") or [])
            confidences.append(float(context.get("confidence") or 0.0))
            jobs.append(
                {
                    "index": index,
                    "node_id": candidate.get("node_id"),
                    "candidate": candidate,
                    "optimization_context": context,
                }
            )

        compatibilities: list[dict[str, Any]] = []
        for left_idx, left in enumerate(normalized_candidates):
            left_model = str(left.get("model_key") or left.get("script_signature") or left.get("packing_family") or "")
            if not left_model:
                continue
            for right_idx in range(left_idx + 1, len(normalized_candidates)):
                right = normalized_candidates[right_idx]
                right_model = str(right.get("model_key") or right.get("script_signature") or right.get("packing_family") or "")
                if not right_model:
                    continue
                try:
                    compatibility = self.get_packet_compatibility(
                        model_a=left_model,
                        model_b=right_model,
                        hardware="current",
                    )
                except Exception as exc:
                    compatibility = {"found": False, "reason": str(exc)}
                compatibilities.append(
                    {
                        "left_index": left_idx,
                        "right_index": right_idx,
                        "left_model": left_model,
                        "right_model": right_model,
                        "compatibility": compatibility,
                    }
                )

        recommendations = []
        if normalized_candidates:
            recommendations.append(
                "Submit this round as one scheduler packet so batch probing, placement, and packing decisions can see all runnable jobs before dispatch."
            )
        compatible_pairs = [
            item for item in compatibilities if (item.get("compatibility") or {}).get("compatible") is True
        ]
        if compatible_pairs:
            recommendations.append(f"Historical pair evidence marks {len(compatible_pairs)} job pair(s) as pack-compatible.")

        return {
            "found": bool(jobs),
            "packet_id": packet_id,
            "jobs": jobs,
            "packet_compatibility": compatibilities,
            "recommendations": recommendations,
            "evidence_refs": sorted(set(evidence_refs)),
            "confidence": round(max(confidences) if confidences else 0.0, 3),
        }

    def optimize_job_packet(self, *, candidates: list[dict[str, Any]], limit: int = 8) -> dict[str, Any]:
        return self.plan_job_packet(candidates=candidates, limit=limit)

    def get_model_design_hardware_context(
        self,
        *,
        workload_type: str | None = None,
        task_type: str | None = None,
        candidate_families: list[str] | None = None,
        hardware_key: str = "current",
        limit: int = 8,
    ) -> dict[str, Any]:
        """Rank model-family choices using hardware feature/code knowledge before draft generation."""
        workload = workload_type or task_type or "mlevolve_training"
        hardware_context = self.get_hardware_context(hardware_key, include_scheduler_limits=True)
        families = candidate_families or self._default_model_families_for_workload(workload)
        hardware = hardware_context.get("hardware") or {}
        gpu_name = hardware.get("gpu_name") or hardware.get("hardware_key") or "current GPU"
        options: list[dict[str, Any]] = []
        evidence_refs: list[str] = []
        risk_flags: list[str] = []

        for family in families[: max(1, int(limit))]:
            query = (
                f"{gpu_name} {workload} {family} pytorch training throughput utilization "
                "tensor cores bf16 fp16 fp8 transformer engine convolution dataloader"
            )
            try:
                matches = self.search_code_knowledge(
                    query=query,
                    filters={"framework": "pytorch", "workload_types": workload, "model_families": family},
                    record_types=["optimization_recipe_chunks", "code_doc_chunks", "api_symbol_chunks"],
                    limit=max(1, min(4, int(limit))),
                )
            except Exception:
                matches = []
            if not matches:
                try:
                    matches = self.search_hardware_features(
                        query=query,
                        hardware_key=hardware_key,
                        workload_type=workload,
                        framework="pytorch",
                        limit=max(1, min(4, int(limit))),
                    )
                except Exception:
                    matches = []
            refs = [
                f"code_knowledge:{item.get('record_type', 'hardware_feature')}:{item.get('record_id')}"
                for item in matches
                if item.get("record_id")
            ]
            evidence_refs.extend(refs)
            recommended_patterns: list[str] = []
            avoid_patterns: list[str] = []
            for item in matches:
                recommended_patterns.extend(str(pattern) for pattern in item.get("recommended_patterns") or [])
                avoid_patterns.extend(str(pattern) for pattern in item.get("avoid_patterns") or [])
            feature_words = self._hardware_feature_words(matches)
            confidence = max([float(item.get("confidence") or 0.0) for item in matches] or [0.0])
            score = round((confidence or 0.1) + min(0.4, 0.05 * len(matches)), 3) if matches else 0.05
            options.append(
                {
                    "model_family": family,
                    "score": score,
                    "confidence": round(confidence, 3),
                    "rationale": self._model_family_rationale(family, workload, feature_words, bool(matches)),
                    "hardware_features": feature_words[:8],
                    "expected_benefits": recommended_patterns[:4],
                    "risks": avoid_patterns[:4],
                    "evidence_refs": refs,
                }
            )
            risk_flags.extend(avoid_patterns[:4])

        options.sort(key=lambda item: (float(item.get("score") or 0.0), float(item.get("confidence") or 0.0)), reverse=True)
        recommendations = [
            "Select the highest task-suitable model family with hardware evidence; do not sacrifice the competition metric only for speed.",
            "Prefer tensor-core-friendly precision and shapes when supported by both the model family and the installed PyTorch/CUDA stack.",
            "Use conservative baseline-compatible models when evidence is low-confidence or required weights/packages are unavailable.",
        ]
        return {
            "found": bool(options),
            "hardware_context": hardware_context,
            "workload_type": workload,
            "model_options": options[: max(1, int(limit))],
            "recommendations": recommendations,
            "risk_flags": sorted(set(risk_flags))[: max(1, int(limit))],
            "evidence_refs": sorted(set(evidence_refs)),
            "confidence": round(max([float(item.get("confidence") or 0.0) for item in options] or [0.0]), 3),
        }

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
        # Deprecated compatibility wrapper. Prefer search_code_knowledge(...).
        filters: dict[str, Any] = {"framework": framework}
        if workload_type:
            filters["workload_types"] = workload_type
        matches = self.search_code_knowledge(
            query=query,
            filters=filters,
            record_types=["code_doc_chunks", "optimization_recipe_chunks"],
            limit=limit,
        )
        if matches:
            return matches
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
        # Deprecated compatibility wrapper. Prefer get_code_optimization_context(...).
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
        matches = self.search_code_knowledge(
            query=query,
            filters={
                "framework": framework,
                "workload_types": workload_type,
                "model_families": model_family,
            },
            record_types=["code_doc_chunks", "optimization_recipe_chunks"],
            limit=limit,
        )
        if not matches:
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
            "evidence_refs": [f"code_knowledge:{match.get('record_type', 'hardware_feature')}:{match['record_id']}" for match in matches if match.get("record_id")],
        }

    def get_hardware_optimization_context(self, *, candidate: dict[str, Any], limit: int = 8) -> dict[str, Any]:
        # Deprecated compatibility wrapper. Prefer get_optimization_context(...).
        context = self.get_optimization_context(candidate=candidate, limit=limit)
        job_design_context = self.get_job_design_context(candidate, limit=limit)
        feature_context = {
            "found": any(context.get("vector_evidence", {}).values()),
            "matches": (
                list(context.get("vector_evidence", {}).get("recipes") or [])
                + list(context.get("vector_evidence", {}).get("docs") or [])
                + list(context.get("vector_evidence", {}).get("api_symbols") or [])
            ),
        }
        return {
            "hardware_context": context.get("hardware_context"),
            "job_design_context": job_design_context,
            "feature_context": feature_context,
            "combined_recommendations": context.get("recommendations", []),
            "risk_notes": context.get("risk_flags", []),
            "evidence_refs": context.get("evidence_refs", []),
            "confidence": context.get("confidence", 0.0),
        }

    def _default_model_families_for_workload(self, workload_type: str | None) -> list[str]:
        workload = str(workload_type or "").lower()
        if "vision" in workload:
            return ["convnet", "efficientnet", "convnext", "vision_transformer", "hybrid_cnn_transformer"]
        if "transformer" in workload or "text" in workload or "nlp" in workload:
            return ["transformer", "small_transformer", "lora_transformer", "sequence_cnn"]
        if "audio" in workload:
            return ["cnn", "conformer", "spectrogram_transformer"]
        if "tabular" in workload:
            return ["lightgbm", "xgboost", "tabular_mlp", "tab_transformer"]
        return ["baseline_compatible", "cnn", "transformer", "tree_ensemble"]

    def _hardware_feature_words(self, matches: list[dict[str, Any]]) -> list[str]:
        features: list[str] = []
        for item in matches:
            for key in ("hardware_feature_keys", "technology_keys", "api_symbols"):
                raw_values = item.get(key) or []
                if isinstance(raw_values, str):
                    raw_values = [raw_values]
                for value in raw_values:
                    text = str(value)
                    if text and text not in features:
                        features.append(text)
            for text in (
                str(item.get("title") or ""),
                str(item.get("summary_text") or ""),
                " ".join(str(pattern) for pattern in item.get("recommended_patterns") or []),
            ):
                lowered = text.lower()
                for token in ("tensor core", "bf16", "fp16", "fp8", "fp4", "transformer engine", "torch.compile", "flash attention", "dataloader"):
                    if token in lowered and token not in features:
                        features.append(token)
        return features

    def _model_family_rationale(
        self,
        family: str,
        workload_type: str | None,
        feature_words: list[str],
        has_evidence: bool,
    ) -> str:
        if has_evidence:
            feature_text = ", ".join(feature_words[:4]) if feature_words else "matched scheduler/code knowledge"
            return f"{family} has hardware/code evidence for {workload_type or 'this workload'} using {feature_text}."
        return (
            f"{family} is a baseline-compatible option for {workload_type or 'this workload'}, "
            "but no strong hardware-specific evidence was found."
        )

    def dump_jobs_json(self) -> str:
        return json.dumps([job.to_dict() for job in self.list_jobs()], indent=2, sort_keys=True)
