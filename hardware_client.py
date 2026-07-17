"""Independent hardware-knowledge facade for MLEvolve agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import subprocess
import sys

from .graph_knowledge import SchedulerKnowledgeBase
from .hardware import HardwareProfile
from .hardware_knowledge import HardwareKnowledgeGraphStore, HardwareKnowledgeSettings
from .redis_cache import RedisLRUCache, graph_cache_enabled
from .runtime_environment import (
    detect_runtime_environment,
    validate_generated_training_code as _validate_generated_training_code,
)
from .storage import BranchProfileReader


_PIPELINE_HARDWARE_STAGES = ("model_structure", "datatype", "training_parameters")

_HARDWARE_STAGE_ALIASES = {
    "data": "datatype",
    "data_type": "datatype",
    "data_processing": "datatype",
    "data_processing_and_feature_engineering": "datatype",
    "feature_engineering": "datatype",
    "model_design": "model_structure",
    "architecture": "model_structure",
    "optimizer_selection": "training_parameters",
    "loss": "training_parameters",
    "training": "training_parameters",
    "training_evaluation": "training_parameters",
    "training_parameters": "training_parameters",
    "training_params": "training_parameters",
    "pre_submit_training_review": "training_parameters",
}

_COMPOSITE_HARDWARE_STAGE_ALIASES = {
    "stage1": ("datatype", "model_structure"),
    "stage_1": ("datatype", "model_structure"),
    "hardware_context_lookup": ("datatype", "model_structure"),
    "stage1_candidate_construction": ("datatype", "model_structure"),
    "candidate_construction": ("datatype", "model_structure"),
    "datatype_precision": ("datatype", "training_parameters"),
    "precision": ("datatype", "training_parameters"),
    "training_evaluation": ("training_parameters",),
}

_AGENT_STAGE_HARDWARE_STAGES = {
    "draft": ("model_structure", "datatype", "training_parameters"),
    "improve": ("model_structure", "training_parameters"),
    "evolution": ("model_structure", "training_parameters"),
    "fusion": ("model_structure", "training_parameters"),
    "debug": ("training_parameters",),
    "code_review": ("training_parameters",),
    "aggregation": ("training_parameters",),
    "model_design": ("model_structure",),
    "datatype_precision": ("datatype", "training_parameters"),
    "training_evaluation": ("training_parameters",),
}


def _sanitize_agent_response(value: Any) -> Any:
    if isinstance(value, list):
        return [_sanitize_agent_response(item) for item in value if item not in (None, "", [], {})]
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            for cleaned in [_sanitize_agent_response(item)]
            if cleaned not in (None, "", [], {})
        }
    return value


class _EmptyProfileStore:
    def __init__(self, settings: HardwareKnowledgeSettings, client: "HardwareKnowledgeClient") -> None:
        self.settings = settings
        self._client = client

    def hardware_profile(self) -> HardwareProfile:
        return self._client.hardware_profile()

    def hardware_key(self) -> str:
        return self.hardware_profile().hardware_key

    def get_job(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def list_jobs(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def list_runtime_profiles(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def list_solo_profiles(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def list_pair_profiles(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def list_batch_probe_profiles(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def list_batch_size_observations(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def list_combination_profiles(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []


class _ProfileReaderProxy(_EmptyProfileStore):
    def __init__(self, reader: BranchProfileReader, client: "HardwareKnowledgeClient", *, include_profile_evidence: bool) -> None:
        super().__init__(reader.settings, client)
        self._reader = reader
        self._include_profile_evidence = include_profile_evidence

    def get_job(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("list_") and not self._include_profile_evidence:
            return lambda *args, **kwargs: []
        return getattr(self._reader, name)


class HardwareKnowledgeClient:
    """Prompt/evidence client that is independent of scheduler storage."""

    def __init__(
        self,
        settings: HardwareKnowledgeSettings | dict[str, Any] | None = None,
        *,
        include_profile_evidence: bool = True,
        probe_timeout_seconds: float = 10.0,
    ) -> None:
        if settings is None:
            settings = HardwareKnowledgeSettings()
        if isinstance(settings, dict):
            settings = HardwareKnowledgeSettings.from_dict(settings)
        self.settings = settings
        self.settings.ensure_runtime_layout()
        self.include_profile_evidence = bool(include_profile_evidence)
        self.probe_timeout_seconds = max(0.1, float(probe_timeout_seconds or 10.0))
        self._probe_status: dict[str, Any] | None = None
        self._scheduler_client: Any | None = None
        self._hardware_knowledge_store: HardwareKnowledgeGraphStore | None = None
        self._hardware_neighborhood_cache = RedisLRUCache.from_settings(self.settings) if graph_cache_enabled(self.settings) else None
        self.store = self._build_profile_store()
        self.knowledge = SchedulerKnowledgeBase(self.store, redis_cache=self._hardware_neighborhood_cache)
        self.profile_evidence_used = False

    def _build_profile_store(self) -> _EmptyProfileStore:
        path = Path(self.settings.branch_profile_db_path)
        if not self.include_profile_evidence or not path.exists():
            return _EmptyProfileStore(self.settings, self)
        try:
            return _ProfileReaderProxy(BranchProfileReader(self.settings), self, include_profile_evidence=True)
        except Exception:
            return _EmptyProfileStore(self.settings, self)

    def attach_scheduler_client(self, scheduler_client: Any | None) -> None:
        self._scheduler_client = scheduler_client

    @property
    def probe_status(self) -> dict[str, Any]:
        return dict(self.probe_current_hardware())

    @property
    def scheduler_context_attached(self) -> bool:
        return self._scheduler_client is not None

    def _hardware_graph_store(self) -> HardwareKnowledgeGraphStore:
        if self._hardware_knowledge_store is None:
            self._hardware_knowledge_store = HardwareKnowledgeGraphStore(self.settings)
        return self._hardware_knowledge_store

    def ingest_hardware_knowledge_graph(
        self,
        *,
        schema_root: str | Path = "schema",
        recreate: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return self._hardware_graph_store().ingest_schema_root(schema_root=schema_root, recreate=recreate, dry_run=dry_run)

    def probe_current_hardware(self) -> dict[str, Any]:
        if self._probe_status is not None:
            return self._probe_status

        device_index = int(getattr(self.settings, "device_index", 0) or 0)
        script = (
            "import json, sys\n"
            "from localml_scheduler.hardware import detect_hardware_profile\n"
            "profile = detect_hardware_profile(device_index=int(sys.argv[1]))\n"
            "print(json.dumps(profile.to_dict(), sort_keys=True))\n"
        )
        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[1])
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = repo_root if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script, str(device_index)],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.probe_timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            self._probe_status = {
                "ok": False,
                "source": "hardware_probe_subprocess",
                "device_index": device_index,
                "reason": f"hardware probe timed out after {self.probe_timeout_seconds:g}s",
                "stderr": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "",
            }
            return self._probe_status
        except Exception as exc:
            self._probe_status = {
                "ok": False,
                "source": "hardware_probe_subprocess",
                "device_index": device_index,
                "reason": str(exc),
            }
            return self._probe_status

        if completed.returncode != 0:
            self._probe_status = {
                "ok": False,
                "source": "hardware_probe_subprocess",
                "device_index": device_index,
                "reason": f"hardware probe exited with code {completed.returncode}",
                "stderr": (completed.stderr or "")[-1000:],
            }
            return self._probe_status
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except Exception as exc:
            self._probe_status = {
                "ok": False,
                "source": "hardware_probe_subprocess",
                "device_index": device_index,
                "reason": f"hardware probe produced invalid JSON: {exc}",
                "stdout": (completed.stdout or "")[-1000:],
                "stderr": (completed.stderr or "")[-1000:],
            }
            return self._probe_status

        self._probe_status = {
            "ok": True,
            "source": "hardware_probe_subprocess",
            "device_index": device_index,
            "hardware_profile": payload,
        }
        return self._probe_status

    def hardware_profile(self) -> HardwareProfile:
        status = self.probe_current_hardware()
        if not status.get("ok"):
            raise RuntimeError(str(status.get("reason") or "hardware probe failed"))
        payload = dict(status.get("hardware_profile") or {})
        return HardwareProfile(
            hardware_key=str(payload.get("hardware_key") or ""),
            os_name=str(payload.get("os_name") or ""),
            gpu_name=str(payload.get("gpu_name") or ""),
            total_vram_mb=payload.get("total_vram_mb"),
            compute_capability=payload.get("compute_capability"),
            cuda_runtime=payload.get("cuda_runtime"),
            torch_version=str(payload.get("torch_version") or ""),
        )

    def search_hardware(self, **kwargs: Any) -> list[dict[str, Any]]:
        return _sanitize_agent_response(self.knowledge.search_hardware(**kwargs))

    def get_hardware_context(self, hardware_key: str = "current", include_scheduler_limits: bool = True) -> dict[str, Any]:
        probe = self.probe_current_hardware() if str(hardware_key or "current") == "current" else {}
        if str(hardware_key or "current") == "current" and not probe.get("ok"):
            return {
                "found": False,
                "hardware": None,
                "accelerator": None,
                "toolkit": None,
                "backend_capabilities": {},
                "scheduler_limits": {},
                "source": "hardware_probe_subprocess",
                "hardware_probe_source": probe.get("source"),
                "hardware_probe_success": False,
                "reason": probe.get("reason"),
            }
        expose_scheduler = bool(include_scheduler_limits and self._scheduler_client is not None)
        result = _sanitize_agent_response(
            self.knowledge.get_hardware_context(hardware_key=hardware_key, include_scheduler_limits=expose_scheduler)
        )
        if not expose_scheduler:
            result["backend_capabilities"] = {}
            result["scheduler_limits"] = {}
        result["hardware_probe_source"] = probe.get("source")
        result["hardware_probe_success"] = probe.get("ok")
        result["profile_evidence_enabled"] = self.include_profile_evidence
        return result

    def get_job_design_context(self, *, candidate: dict[str, Any], limit: int = 5) -> dict[str, Any]:
        return self.knowledge.get_job_design_context(candidate=candidate, limit=limit)

    def get_profile_evidence(self, *, candidate: dict[str, Any], limit: int = 8) -> dict[str, Any]:
        if not self.include_profile_evidence:
            return {
                "hardware_context": self.get_hardware_context("current", include_scheduler_limits=False),
                "graph_evidence": {"exact_profiles": [], "similar_profiles": [], "packed_profiles": []},
                "derived_diagnosis": {"profile_symptoms": [], "optimization_targets": []},
                "evidence_refs": [],
                "confidence": 0.0,
            }
        result = self.knowledge.get_profile_evidence(candidate=candidate, limit=limit)
        graph = result.get("graph_evidence") or {}
        self.profile_evidence_used = any(graph.get(key) for key in ("exact_profiles", "similar_profiles", "packed_profiles"))
        return result

    @staticmethod
    def _normalize_hardware_stage_name(stage: Any) -> str | None:
        normalized = str(stage or "").strip().lower().replace("-", "_")
        if not normalized or normalized == "all":
            return None
        normalized = _HARDWARE_STAGE_ALIASES.get(normalized, normalized)
        return normalized if normalized in _PIPELINE_HARDWARE_STAGES else None

    @classmethod
    def _normalize_hardware_stage_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        raw_items = list(value) if isinstance(value, (list, tuple, set)) else [item for item in str(value).replace(";", ",").split(",") if str(item).strip()]
        stages: list[str] = []
        for item in raw_items:
            normalized = str(item or "").strip().lower().replace("-", "_")
            expanded = _COMPOSITE_HARDWARE_STAGE_ALIASES.get(normalized)
            if expanded:
                for stage in expanded:
                    if stage not in stages:
                        stages.append(stage)
                continue
            stage = cls._normalize_hardware_stage_name(normalized)
            if stage and stage not in stages:
                stages.append(stage)
        return stages

    @classmethod
    def _hardware_stages_for_candidate(cls, candidate: dict[str, Any]) -> list[str]:
        for key in ("hardware_pipeline_stages", "hardware_pipeline_stage", "pipeline_stages", "pipeline_stage"):
            stages = cls._normalize_hardware_stage_list(candidate.get(key))
            if stages:
                return stages
        agent_stage = str(candidate.get("stage") or "").strip().lower().replace("-", "_")
        if agent_stage in _AGENT_STAGE_HARDWARE_STAGES:
            return list(_AGENT_STAGE_HARDWARE_STAGES[agent_stage])
        direct_stage = cls._normalize_hardware_stage_name(agent_stage)
        return [direct_stage] if direct_stage else []

    @staticmethod
    def _stage_feature_context_from_static_graph(*, hardware_name: str, stages: list[str], limit: int) -> dict[str, Any]:
        try:
            from .hardware_knowledge.feature_filter import query_hardware_features, query_hardware_node
        except Exception as exc:
            return {
                "found": False,
                "hardware": None,
                "stage_filter": stages,
                "stages": [],
                "features": [],
                "feature_count": 0,
                "source": "hardware_knowledge_graph.json",
                "reason": str(exc),
            }
        stage_payloads: list[dict[str, Any]] = []
        merged_features: list[dict[str, Any]] = []
        seen_features: set[str] = set()
        hardware_payload: dict[str, Any] | None = None
        reason = "hardware not found"
        per_stage_limit = max(1, int(limit))
        for stage in stages:
            node_payload = query_hardware_node(hardware_name, stage)
            feature_payload = query_hardware_features(hardware_name, stage)
            if not node_payload.get("found") and not feature_payload.get("found"):
                reason = str(node_payload.get("reason") or feature_payload.get("reason") or reason)
                continue
            if hardware_payload is None:
                hardware_payload = {
                    "gpu_name": node_payload.get("gpu_name") or feature_payload.get("gpu_name"),
                    "architecture": node_payload.get("architecture"),
                    "vram_MB": node_payload.get("vram_MB"),
                    "compute_capability": node_payload.get("compute_capability"),
                }
                hardware_payload = {key: value for key, value in hardware_payload.items() if value not in (None, "", [], {})}
            stage_features = list(feature_payload.get("features") or [])[:per_stage_limit]
            for feature in stage_features:
                feature_id = str(feature.get("feature_id") or "")
                key = feature_id if feature_id else repr(feature)
                if key in seen_features:
                    continue
                seen_features.add(key)
                merged_features.append(dict(feature, pipeline_stage=stage))
            stage_payloads.append(
                {
                    "stage": stage,
                    "node": node_payload,
                    "features": stage_features,
                    "feature_count": int(feature_payload.get("feature_count") or len(stage_features)),
                }
            )
        return _sanitize_agent_response(
            {
                "found": bool(stage_payloads),
                "hardware": hardware_payload,
                "stage_filter": stages[0] if len(stages) == 1 else list(stages),
                "stages": stage_payloads,
                "features": merged_features,
                "feature_count": sum(int(item.get("feature_count") or 0) for item in stage_payloads),
                "source": "hardware_knowledge_graph.json",
                "reason": None if stage_payloads else reason,
            }
        )

    def get_stage_hardware_features(
        self,
        hardware_id: str = "current",
        *,
        pipeline_stage: str | list[str] | tuple[str, ...] | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        stages = self._normalize_hardware_stage_list(pipeline_stage)
        if not stages:
            stages = list(_PIPELINE_HARDWARE_STAGES)
        hardware_context = self.get_hardware_context(hardware_id, include_scheduler_limits=False)
        hardware = hardware_context.get("hardware") or {}
        hardware_name = str(hardware.get("gpu_name") or hardware.get("hardware_key") or hardware_id)
        result = self._stage_feature_context_from_static_graph(hardware_name=hardware_name, stages=stages, limit=limit)
        result["hardware_context"] = hardware_context
        return result

    @staticmethod
    def _feature_index_from_stage_context(stage_context: dict[str, Any]) -> list[dict[str, Any]]:
        index: list[dict[str, Any]] = []
        for item in stage_context.get("features") or []:
            index.append(
                {
                    "feature_id": item.get("feature_id"),
                    "feature_name": item.get("feature_name") or item.get("name") or item.get("title"),
                    "category": item.get("category"),
                    "support_level": item.get("support_level"),
                    "recommended": bool(item.get("recommended")),
                    "performance_impact": item.get("performance_impact"),
                    "frameworks": list(item.get("frameworks") or []),
                    "tags": list(item.get("tags") or []),
                    "confidence": item.get("confidence"),
                }
            )
        return index

    def get_hardware_feature_index(self, hardware_id: str = "current", limit: int = 256) -> dict[str, Any]:
        context = self.get_stage_hardware_features(hardware_id, pipeline_stage=list(_PIPELINE_HARDWARE_STAGES), limit=max(1, int(limit)))
        return _sanitize_agent_response(
            {
                "found": bool(context.get("found")),
                "hardware": context.get("hardware"),
                "features": self._feature_index_from_stage_context(context)[: max(1, int(limit))],
                "feature_count": context.get("feature_count"),
                "source": context.get("source"),
            }
        )

    def get_hardware_feature_details(self, *, hardware_id: str = "current", feature_ids: list[str], limit: int = 64) -> dict[str, Any]:
        requested = [str(item) for item in feature_ids or [] if str(item).strip()]
        if not requested:
            return {"found": False, "hardware": None, "features": [], "requested_feature_ids": [], "missing_feature_ids": [], "source": "empty_request"}
        context = self.get_stage_hardware_features(hardware_id, pipeline_stage=list(_PIPELINE_HARDWARE_STAGES), limit=max(int(limit), len(requested), 64))
        by_id = {str(item.get("feature_id")): item for item in context.get("features") or [] if item.get("feature_id")}
        selected = [by_id[feature_id] for feature_id in requested if feature_id in by_id]
        missing = [feature_id for feature_id in requested if feature_id not in by_id]
        return _sanitize_agent_response(
            {
                "found": bool(selected),
                "hardware": context.get("hardware"),
                "features": selected[: max(1, int(limit))],
                "requested_feature_ids": requested,
                "missing_feature_ids": missing,
                "source": context.get("source"),
            }
        )

    def prewarm_current_hardware_neighborhood(self, hardware_id: str = "current", *, limit: int = 256) -> dict[str, Any]:
        result = self.get_hardware_feature_index(hardware_id=hardware_id, limit=limit)
        return {
            "ok": bool(result.get("found")),
            "hardware_id": ((result.get("hardware") or {}).get("hardware_key")),
            "hardware_name": ((result.get("hardware") or {}).get("gpu_name")),
            "feature_count": len(result.get("features") or []),
            "source": result.get("source"),
            "cache_namespace": "hardware:neighborhood" if self._hardware_neighborhood_cache is not None else None,
            "reason": result.get("reason"),
        }

    def get_runtime_environment(self, *, include_package_versions: bool = True, include_precision_checks: bool = True) -> dict[str, Any]:
        return _sanitize_agent_response(
            detect_runtime_environment(
                include_package_versions=include_package_versions,
                include_precision_checks=include_precision_checks,
                device_index=int(getattr(self.settings, "device_index", 0) or 0),
            )
        )

    def validate_generated_training_code(self, code: str, stage: str = "code_review") -> dict[str, Any]:
        return _sanitize_agent_response(_validate_generated_training_code(code, stage=stage))

    @staticmethod
    def _default_model_families_for_workload(workload_type: str | None) -> list[str]:
        workload = str(workload_type or "").lower()
        if "vision" in workload:
            return ["resnet50", "efficientnet-b0", "convnext-tiny", "vit-base", "swin-tiny"]
        if "transformer" in workload or "text" in workload or "nlp" in workload:
            return ["transformer", "small-transformer", "lora-transformer", "sequence-cnn"]
        if "audio" in workload:
            return ["cnn", "conformer", "spectrogram-transformer"]
        if "tabular" in workload:
            return ["lightgbm", "xgboost", "tabular-mlp", "tab-transformer"]
        return ["baseline-compatible", "resnet50", "transformer", "tree-ensemble"]

    @staticmethod
    def _hardware_feature_words(features: list[dict[str, Any]]) -> list[str]:
        words: list[str] = []
        for item in features:
            for value in (item.get("feature_id"), item.get("category"), item.get("name"), item.get("feature_name")):
                text = str(value or "").strip()
                if text and text not in words:
                    words.append(text)
        return words

    @staticmethod
    def _model_family_rationale(family: str, workload_type: str | None, feature_words: list[str]) -> str:
        feature_text = ", ".join(feature_words[:4]) if feature_words else "the current hardware profile"
        return f"{family} is a candidate branch for {workload_type or 'this workload'} with context from {feature_text}."

    def get_model_design_hardware_context(
        self,
        *,
        workload_type: str | None = None,
        task_type: str | None = None,
        candidate_families: list[str] | None = None,
        hardware_key: str = "current",
        limit: int = 8,
    ) -> dict[str, Any]:
        workload = workload_type or task_type or "mlevolve_training"
        hardware_context = self.get_hardware_context(hardware_key, include_scheduler_limits=False)
        feature_context = self.get_stage_hardware_features(hardware_key, pipeline_stage="model_structure", limit=max(4, int(limit)))
        feature_words = self._hardware_feature_words(list(feature_context.get("features") or []))
        families = candidate_families or self._default_model_families_for_workload(workload)
        options = [
            {
                "model_family": family,
                "branch_name": family,
                "score": round(0.2 + min(0.5, 0.05 * len(feature_words)), 3),
                "confidence": round(0.25 + min(0.5, 0.03 * len(feature_words)), 3),
                "rationale": self._model_family_rationale(family, workload, feature_words),
                "hardware_features": feature_words[:8],
                "expected_benefits": [],
                "risks": [],
                "evidence_refs": [f"hardware_feature:{item.get('feature_id')}" for item in feature_context.get("features") or [] if item.get("feature_id")][:4],
            }
            for family in families[: max(1, int(limit))]
        ]
        return _sanitize_agent_response(
            {
                "found": bool(options),
                "hardware_context": hardware_context,
                "hardware_feature_index": {
                    "found": bool(feature_context.get("found")),
                    "features": self._feature_index_from_stage_context(feature_context),
                    "source": feature_context.get("source"),
                },
                "workload_type": workload,
                "model_options": options,
                "recommendations": [
                    "Prefer a branch that matches the task metric first, then use hardware facts to choose precision and batch-size strategy.",
                    "Reuse existing branch profiles when branch_name matches the mother model.",
                ],
                "risk_flags": [],
                "evidence_refs": sorted({ref for option in options for ref in option.get("evidence_refs", [])}),
                "confidence": round(max([float(item.get("confidence") or 0.0) for item in options] or [0.0]), 3),
            }
        )

    def get_optimization_context(self, *, candidate: dict[str, Any], limit: int = 8) -> dict[str, Any]:
        graph_context = self.get_profile_evidence(candidate=candidate, limit=limit)
        pipeline_stages = self._hardware_stages_for_candidate(candidate)
        stage_hardware_features = (
            self.get_stage_hardware_features("current", pipeline_stage=pipeline_stages, limit=max(2, int(limit)))
            if pipeline_stages
            else {}
        )
        recommendations: list[str] = []
        risks: list[str] = list(graph_context.get("risk_flags") or [])
        batch_recommendation = graph_context.get("batch_size_recommendation") or {}
        if batch_recommendation.get("found") and batch_recommendation.get("recommended_batch_size") is not None:
            recommendations.append(f"Use profile-recommended physical batch size {batch_recommendation['recommended_batch_size']} as the starting point.")
        epoch_recommendation = graph_context.get("epoch_recommendation") or {}
        if epoch_recommendation.get("found") and epoch_recommendation.get("recommended_epochs") is not None:
            recommendations.append(f"Use historical epoch budget {epoch_recommendation['recommended_epochs']} unless the scoring metric suggests otherwise.")
        for feature in stage_hardware_features.get("features") or []:
            if feature.get("recommended") and feature.get("name"):
                recommendations.append(str(feature["name"]))
        result = {
            "hardware_context": graph_context.get("hardware_context"),
            "graph_evidence": graph_context.get("graph_evidence") or {"exact_profiles": [], "similar_profiles": [], "packed_profiles": []},
            "derived_diagnosis": graph_context.get("derived_diagnosis") or {"profile_symptoms": [], "optimization_targets": []},
            "stage_hardware_features": stage_hardware_features,
            "recommendations": recommendations[: max(1, int(limit))],
            "risk_flags": risks,
            "evidence_refs": list(graph_context.get("evidence_refs") or []),
            "confidence": round(float(graph_context.get("confidence") or 0.0), 3),
        }
        return _sanitize_agent_response(result)

    def plan_job_packet(self, *, candidates: list[dict[str, Any]], limit: int = 8) -> dict[str, Any]:
        jobs: list[dict[str, Any]] = []
        evidence_refs: list[str] = []
        confidences: list[float] = []
        for index, candidate in enumerate(list(candidates or [])):
            context = self.get_optimization_context(candidate=dict(candidate or {}), limit=limit)
            evidence_refs.extend(str(ref) for ref in context.get("evidence_refs") or [])
            confidences.append(float(context.get("confidence") or 0.0))
            jobs.append({"index": index, "node_id": candidate.get("node_id"), "candidate": candidate, "optimization_context": context})
        return {
            "found": bool(jobs),
            "jobs": jobs,
            "packet_compatibility": [],
            "recommendations": [],
            "evidence_refs": sorted(set(evidence_refs)),
            "confidence": round(max(confidences) if confidences else 0.0, 3),
        }

    def optimize_job_packet(self, *, candidates: list[dict[str, Any]], limit: int = 8) -> dict[str, Any]:
        return self.plan_job_packet(candidates=candidates, limit=limit)

    def search_hardware_features(self, *, query: str, hardware_key: str = "current", architecture: str | None = None, vendor: str | None = None, workload_type: str | None = None, framework: str | None = "pytorch", limit: int = 8) -> list[dict[str, Any]]:
        del workload_type
        hardware_context = self.get_hardware_context(hardware_key, include_scheduler_limits=False)
        hardware = hardware_context.get("hardware") or {}
        hardware_lookup = str(hardware.get("gpu_name") or hardware.get("hardware_key") or hardware_key)
        try:
            return _sanitize_agent_response(
                self._hardware_graph_store().search(
                    query=query,
                    hardware=hardware_lookup,
                    architecture=architecture,
                    vendor=vendor,
                    framework=framework,
                    limit=limit,
                )
            )
        except Exception:
            return []

    def get_hardware_feature_context(self, *, hardware_key: str = "current", workload_type: str | None = None, model_family: str | None = None, framework: str | None = "pytorch", limit: int = 8) -> dict[str, Any]:
        query = " ".join(part for part in (workload_type or "", model_family or "", framework or "", "training optimization precision memory") if part)
        matches = self.search_hardware_features(query=query, hardware_key=hardware_key, workload_type=workload_type, framework=framework, limit=limit)
        return {"found": bool(matches), "hardware_context": self.get_hardware_context(hardware_key), "query": query, "matches": matches}

    def get_hardware_optimization_context(self, *, candidate: dict[str, Any], limit: int = 8) -> dict[str, Any]:
        return self.get_optimization_context(candidate=candidate, limit=limit)
