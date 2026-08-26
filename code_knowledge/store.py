"""Qdrant-backed code-knowledge vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import os
from datetime import datetime, timezone

import numpy as np

from localml_scheduler.hardware_features.records import load_seed_records
from localml_scheduler.backend_mode import (
    BACKEND_NEUTRAL,
    RETIRED_BACKEND_MODES,
    RUNNER_CONTRACT_SUBPROCESS_V1,
    normalize_runtime_backend,
)

from .records import (
    API_SYMBOL_SCHEMA_VERSION,
    BACKEND_GUIDANCE_SCHEMA_VERSION,
    CODE_DOC_SCHEMA_VERSION,
    OPTIMIZATION_RECIPE_SCHEMA_VERSION,
    convert_hardware_feature_records,
    load_backend_guidance_seed_records,
    load_code_knowledge_records,
    record_to_search_text,
    validate_code_knowledge_record,
)


_SCHEMA_BY_RECORD_TYPE = {
    "code_doc_chunks": CODE_DOC_SCHEMA_VERSION,
    "optimization_recipe_chunks": OPTIMIZATION_RECIPE_SCHEMA_VERSION,
    "api_symbol_chunks": API_SYMBOL_SCHEMA_VERSION,
    "docs": CODE_DOC_SCHEMA_VERSION,
    "recipes": OPTIMIZATION_RECIPE_SCHEMA_VERSION,
    "api_symbols": API_SYMBOL_SCHEMA_VERSION,
    "backend_guidance_rules": BACKEND_GUIDANCE_SCHEMA_VERSION,
    "backend_guidance": BACKEND_GUIDANCE_SCHEMA_VERSION,
}


@dataclass(slots=True)
class CodeKnowledgeSearchResult:
    record: dict[str, Any]
    score: float
    collection_name: str

    def to_public_dict(self) -> dict[str, Any]:
        record = self.record
        return {
            "record_id": record.get("record_id"),
            "record_type": record.get("record_type"),
            "collection_name": self.collection_name,
            "title": record.get("title"),
            "text": record.get("text"),
            "summary_text": record.get("solution_summary") or record.get("usage_summary") or record.get("text"),
            "score": self.score,
            "framework": record.get("framework"),
            "frameworks": list(record.get("frameworks") or []),
            "framework_version": record.get("framework_version"),
            "framework_versions": list(record.get("framework_versions") or []),
            "toolkits": list(record.get("toolkits") or []),
            "toolkit_versions": list(record.get("toolkit_versions") or []),
            "driver_versions": list(record.get("driver_versions") or []),
            "compute_capabilities": list(record.get("compute_capabilities") or []),
            "accelerator_names": list(record.get("accelerator_names") or []),
            "gpu_architectures": list(record.get("gpu_architectures") or []),
            "backend_keys": list(record.get("backend_keys") or []),
            "technology_keys": list(record.get("technology_keys") or []),
            "hardware_feature_keys": list(record.get("hardware_feature_keys") or []),
            "model_families": list(record.get("model_families") or []),
            "workload_types": list(record.get("workload_types") or []),
            "optimization_targets": list(record.get("optimization_targets") or []),
            "profile_symptoms": list(record.get("profile_symptoms") or []),
            "api_symbols": list(record.get("api_symbols") or []),
            "recommended_patterns": list(record.get("recommended_patterns") or []),
            "avoid_patterns": list(record.get("avoid_patterns") or []),
            "risk_level": record.get("risk_level"),
            "confidence": record.get("confidence"),
            "source_id": record.get("source_id"),
            "source_type": record.get("source_type"),
            "source_title": record.get("source_title"),
            "source_url": record.get("source_url"),
            "source_version": record.get("source_version"),
            "retrieved_or_verified_date": record.get("retrieved_or_verified_date"),
            "backend_modes": list(record.get("backend_modes") or []),
            "runner_contracts": list(record.get("runner_contracts") or []),
            "pipeline_stages": list(record.get("pipeline_stages") or []),
            "rule_type": record.get("rule_type"),
            "owner": record.get("owner"),
            "strength": record.get("strength"),
            "transferability": record.get("transferability"),
            "review_status": record.get("review_status"),
            "hardware_constraints": dict(record.get("hardware_constraints") or {}),
            "source_refs": list(record.get("source_refs") or []),
            "last_verified": record.get("last_verified"),
            "applicability": dict(record.get("applicability") or {}),
            "support_status": record.get("support_status"),
            "applicability_support": dict(record.get("applicability_support") or {}),
            "cuda_docs_cache_key": record.get("cuda_docs_cache_key"),
            "query_template_version": record.get("query_template_version"),
            "remote_tool_schema_hash": record.get("remote_tool_schema_hash"),
            "backend_config_hash": record.get("backend_config_hash"),
            "verified_source": bool(record.get("verified_source", False)),
        }


class CodeKnowledgeStore:
    """Thin Qdrant adapter for docs, optimization recipes, and API symbols."""

    def __init__(
        self,
        settings: Any,
        *,
        qdrant_client: Any | None = None,
        qdrant_models: Any | None = None,
        embedding_model: Any | None = None,
    ):
        self.settings = settings
        self.config = settings.hardware_feature_db
        self._client = qdrant_client
        self._models = qdrant_models
        self._embedding_model = embedding_model

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "enabled", False)) and getattr(self.config, "provider", "qdrant") == "qdrant"

    @property
    def collection_names(self) -> dict[str, str]:
        return {
            CODE_DOC_SCHEMA_VERSION: self.config.code_doc_collection_name,
            OPTIMIZATION_RECIPE_SCHEMA_VERSION: self.config.optimization_recipe_collection_name,
            API_SYMBOL_SCHEMA_VERSION: self.config.api_symbol_collection_name,
            BACKEND_GUIDANCE_SCHEMA_VERSION: self.config.backend_guidance_collection_name,
        }

    def _qdrant_models(self) -> Any:
        if self._models is None:
            from qdrant_client import models

            self._models = models
        return self._models

    def _qdrant_client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            api_key = os.getenv(self.config.api_key_env) if self.config.api_key_env else None
            self._client = QdrantClient(url=self.config.url, api_key=api_key)
        return self._client

    def _embedder(self) -> Any:
        if self._embedding_model is None:
            from agents.memory.embedding_models import EmbeddingModel

            self._embedding_model = EmbeddingModel(
                model_type=self.config.embedding_model_type,
                model_name=self.config.embedding_model_name,
                device=self.config.embedding_device,
            )
        return self._embedding_model

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self._embedder().encode(texts, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)

    def _dimension(self) -> int:
        return int(getattr(self._embedder(), "dimension"))

    def _distance(self) -> Any:
        models = self._qdrant_models()
        normalized = str(self.config.distance or "Cosine").strip().upper()
        if hasattr(models, "Distance"):
            return getattr(models.Distance, normalized, getattr(models.Distance, "COSINE"))
        return normalized

    def _collection_exists(self, client: Any, collection_name: str) -> bool:
        if hasattr(client, "collection_exists"):
            return bool(client.collection_exists(collection_name))
        try:
            client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def _collection_for_record(self, record: dict[str, Any]) -> str:
        schema_version = str(record.get("schema_version") or "")
        return self.collection_names[schema_version]

    def _point_id(self, collection_name: str, record_id: str) -> str:
        return hashlib.sha256(f"{collection_name}:{record_id}".encode("utf-8")).hexdigest()

    def _record_types(self, record_types: list[str] | None = None) -> list[str]:
        if not record_types:
            return [CODE_DOC_SCHEMA_VERSION, OPTIMIZATION_RECIPE_SCHEMA_VERSION, API_SYMBOL_SCHEMA_VERSION, BACKEND_GUIDANCE_SCHEMA_VERSION]
        schemas: list[str] = []
        for item in record_types:
            schema = _SCHEMA_BY_RECORD_TYPE.get(str(item))
            if schema and schema not in schemas:
                schemas.append(schema)
        return schemas or [CODE_DOC_SCHEMA_VERSION, OPTIMIZATION_RECIPE_SCHEMA_VERSION, API_SYMBOL_SCHEMA_VERSION, BACKEND_GUIDANCE_SCHEMA_VERSION]

    def ensure_collections(self, *, recreate: bool = False) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "reason": "code knowledge database disabled"}
        client = self._qdrant_client()
        models = self._qdrant_models()
        dimension = self._dimension()
        created: list[str] = []
        for collection_name in self.collection_names.values():
            if recreate and hasattr(client, "delete_collection"):
                try:
                    client.delete_collection(collection_name)
                except Exception:
                    pass
            if recreate or not self._collection_exists(client, collection_name):
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=dimension, distance=self._distance()),
                )
                created.append(collection_name)
        return {"ok": True, "collections": list(self.collection_names.values()), "dimension": dimension, "created": created}

    def ingest_records(self, records: list[dict[str, Any]], *, recreate: bool = False, dry_run: bool = False) -> dict[str, Any]:
        normalized = [validate_code_knowledge_record(record) for record in records]
        if dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "collections": list(self.collection_names.values()),
                "record_count": len(normalized),
                "record_ids": [record["record_id"] for record in normalized],
            }
        if not self.enabled:
            return {"ok": False, "reason": "code knowledge database disabled", "record_count": len(normalized)}
        collection_result = self.ensure_collections(recreate=recreate)
        client = self._qdrant_client()
        models = self._qdrant_models()
        by_collection: dict[str, list[dict[str, Any]]] = {}
        for record in normalized:
            by_collection.setdefault(self._collection_for_record(record), []).append(record)
        for collection_name, collection_records in by_collection.items():
            texts = [record_to_search_text(record) for record in collection_records]
            vectors = self._encode(texts)
            points = [
                models.PointStruct(
                    id=self._point_id(collection_name, record["record_id"]),
                    vector=vectors[index].tolist(),
                    payload={**record, "search_text": texts[index]},
                )
                for index, record in enumerate(collection_records)
            ]
            if points:
                client.upsert(collection_name=collection_name, points=points)
        return {
            "ok": True,
            "dry_run": False,
            "collections": list(by_collection.keys()),
            "record_count": len(normalized),
            "record_ids": [record["record_id"] for record in normalized],
            "collection_result": collection_result,
        }

    def ingest_source(self, source: str | Path | None = None, *, recreate: bool = False, dry_run: bool = False) -> dict[str, Any]:
        records = (
            [
                *convert_hardware_feature_records(load_seed_records()),
                *load_backend_guidance_seed_records(),
            ]
            if source is None
            else load_code_knowledge_records(source)
        )
        return self.ingest_records(records, recreate=recreate, dry_run=dry_run)

    def _match_condition(self, key: str, value: Any) -> Any:
        models = self._qdrant_models()
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    def _build_filter(self, filters: dict[str, Any] | None) -> Any | None:
        filters = filters or {}
        must = []
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, list):
                if value:
                    normalized = [str(item).strip() for item in value if str(item).strip()]
                    if not normalized:
                        continue
                    models = self._qdrant_models()
                    if hasattr(models, "MatchAny"):
                        must.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=normalized),
                            )
                        )
                    else:
                        must.append(self._match_condition(key, normalized[0]))
            elif isinstance(value, bool):
                must.append(self._match_condition(key, value))
            elif str(value).strip():
                must.append(self._match_condition(key, str(value).strip()))
        if not must:
            return None
        return self._qdrant_models().Filter(must=must)

    def _payload_from_point(self, point: Any) -> dict[str, Any]:
        payload = getattr(point, "payload", None)
        if payload is None and isinstance(point, dict):
            payload = point.get("payload")
        return dict(payload or {})

    def _score_from_point(self, point: Any) -> float:
        score = getattr(point, "score", None)
        if score is None and isinstance(point, dict):
            score = point.get("score")
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    def _query_points(self, collection_name: str, query_vector: list[float], query_filter: Any | None, limit: int) -> list[Any]:
        client = self._qdrant_client()
        if hasattr(client, "query_points"):
            result = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            return list(getattr(result, "points", result) or [])
        return list(
            client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            or []
        )

    def search(
        self,
        *,
        query: str,
        filters: dict[str, Any] | None = None,
        record_types: list[str] | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            client = self._qdrant_client()
            schemas = self._record_types(record_types)
            query_vector = self._encode([query])[0].tolist()
            query_filter = self._build_filter(filters)
            results: list[dict[str, Any]] = []
            for schema in schemas:
                collection_name = self.collection_names[schema]
                if not self._collection_exists(client, collection_name):
                    continue
                points = self._query_points(collection_name, query_vector, query_filter, max(1, int(limit)))
                for point in points:
                    payload = self._payload_from_point(point)
                    payload.pop("search_text", None)
                    if (
                        payload.get("source_type") == "nvidia_cuda_docs"
                        and not self._nvidia_cuda_payload_visible(payload, filters)
                    ):
                        continue
                    if (
                        schema == OPTIMIZATION_RECIPE_SCHEMA_VERSION
                        and payload.get("source_type") == "nvidia_cuda_docs"
                        and payload.get("review_status") != "reviewed"
                    ):
                        # Draft/retired hosted interpretations remain stored
                        # for audit but are not published into general agent
                        # retrieval.
                        continue
                    results.append(
                        CodeKnowledgeSearchResult(
                            record=payload,
                            score=self._score_from_point(point),
                            collection_name=collection_name,
                        ).to_public_dict()
                    )
        except Exception:
            return []
        results.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return results[: max(1, int(limit))]

    @staticmethod
    def _nvidia_cuda_payload_visible(
        payload: dict[str, Any], filters: dict[str, Any]
    ) -> bool:
        """Require an exact cache key or complete applicability at read time."""

        cache_key = str(filters.get("cuda_docs_cache_key") or "").strip()
        if cache_key and cache_key == str(payload.get("cuda_docs_cache_key") or ""):
            return bool(filters.get("verified_source"))
        if filters.get("source_type") != "nvidia_cuda_docs" or filters.get(
            "verified_source"
        ) is not True:
            return False
        applicability = dict(payload.get("applicability") or {})
        required = (
            "gpu_architecture",
            "compute_capability",
            "driver_major_minor",
            "cuda_major_minor",
            "framework",
            "framework_major_minor",
            "backend_mode",
            "backend_config_hash",
            "runner_contract",
        )
        for key in required:
            expected = str(filters.get(f"applicability.{key}") or "").strip()
            if not expected or expected != str(applicability.get(key) or "").strip():
                return False
        max_age_days = (
            90
            if payload.get("record_type") == "optimization_recipe_chunks"
            else 30
        )
        value = str(payload.get("retrieved_or_verified_date") or "")
        try:
            retrieved = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if retrieved.tzinfo is None:
                retrieved = retrieved.replace(tzinfo=timezone.utc)
            if (
                datetime.now(timezone.utc) - retrieved
            ).total_seconds() > max_age_days * 86400:
                return False
        except ValueError:
            return False
        return True

    def get_cuda_doc_chunks(
        self,
        *,
        cache_key: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Return only raw NVIDIA chunks for one exact applicability key."""

        if not str(cache_key).strip():
            return []
        return self.search(
            query=str(cache_key),
            filters={
                "cuda_docs_cache_key": str(cache_key),
                "source_type": "nvidia_cuda_docs",
                "verified_source": True,
            },
            record_types=["code_doc_chunks"],
            limit=max(1, int(limit)),
        )

    @staticmethod
    def _hardware_rule_exclusion(
        record: dict[str, Any], hardware_context: dict[str, Any]
    ) -> str | None:
        constraints = dict(record.get("hardware_constraints") or {})
        hardware = dict(hardware_context.get("hardware") or hardware_context or {})
        required_vendor = str(constraints.get("vendor") or "").strip().lower()
        actual_vendor = str(hardware.get("vendor") or "").strip().lower()
        if required_vendor and actual_vendor and required_vendor != actual_vendor:
            return f"hardware_vendor_mismatch:{actual_vendor}"
        minimum = str(constraints.get("min_compute_capability") or "").strip()
        actual = str(hardware.get("compute_capability") or "").strip()
        if minimum and actual:
            try:
                if tuple(map(int, actual.split("."))) < tuple(map(int, minimum.split("."))):
                    return f"compute_capability_below:{minimum}"
            except ValueError:
                return "compute_capability_unparseable"
        return None

    def get_backend_design_guidance(
        self,
        *,
        effective_backend: str,
        runner_contract: str = RUNNER_CONTRACT_SUBPROCESS_V1,
        pipeline_stage: str = "model_design",
        hardware_context: dict[str, Any] | None = None,
        query: str | None = None,
        limit: int = 32,
    ) -> dict[str, Any]:
        """Return deterministic exact-backend rules before optional vector ranking."""

        backend = normalize_runtime_backend(effective_backend)
        if runner_contract != RUNNER_CONTRACT_SUBPROCESS_V1:
            raise ValueError(
                f"Unsupported runner contract {runner_contract!r}; expected {RUNNER_CONTRACT_SUBPROCESS_V1}"
            )
        if pipeline_stage not in {
            "model_design",
            "datatype_precision",
            "training_evaluation",
        }:
            raise ValueError(f"Unsupported backend-guidance pipeline stage: {pipeline_stage}")
        records = load_backend_guidance_seed_records()
        eligible: list[dict[str, Any]] = []
        excluded: list[dict[str, str]] = []
        for record in records:
            rule_id = str(record.get("record_id") or record.get("rule_id") or "")
            modes = set(record.get("backend_modes") or [])
            reason: str | None = None
            if modes.intersection(RETIRED_BACKEND_MODES):
                reason = "retired_backend"
            elif not record.get("active", True) or record.get("deprecated"):
                reason = "inactive"
            elif backend not in modes and BACKEND_NEUTRAL not in modes:
                reason = f"backend_mismatch:{backend}"
            elif runner_contract not in set(record.get("runner_contracts") or []):
                reason = f"runner_contract_mismatch:{runner_contract}"
            elif pipeline_stage not in set(record.get("pipeline_stages") or []):
                reason = f"pipeline_stage_mismatch:{pipeline_stage}"
            else:
                reason = self._hardware_rule_exclusion(
                    record, hardware_context or {}
                )
            if reason:
                excluded.append({"rule_id": rule_id, "reason": reason})
            else:
                eligible.append(record)

        vector_scores: dict[str, float] = {}
        if self.enabled and query:
            for match in self.search(
                query=query,
                filters={
                    "backend_modes": [BACKEND_NEUTRAL, backend],
                    "runner_contracts": runner_contract,
                    "pipeline_stages": pipeline_stage,
                },
                record_types=["backend_guidance_rules"],
                limit=max(1, int(limit)),
            ):
                vector_scores[str(match.get("record_id") or "")] = float(
                    match.get("score") or 0.0
                )

        strength_order = {"hard": 0, "preferred": 1, "informational": 2}
        eligible.sort(
            key=lambda item: (
                strength_order.get(str(item.get("strength")), 3),
                -vector_scores.get(str(item.get("record_id") or ""), 0.0),
                str(item.get("record_id") or ""),
            )
        )
        selected = eligible[: max(1, int(limit))]
        public_rules = [
            CodeKnowledgeSearchResult(
                record=record,
                score=vector_scores.get(str(record.get("record_id") or ""), 1.0),
                collection_name=self.config.backend_guidance_collection_name,
            ).to_public_dict()
            for record in selected
        ]
        hard_rules = [item for item in public_rules if item.get("strength") == "hard"]
        preferred_rules = [
            item for item in public_rules if item.get("strength") == "preferred"
        ]
        neutral_rules = [
            item
            for item in public_rules
            if BACKEND_NEUTRAL in (item.get("backend_modes") or [])
        ]
        preferred_patterns = list(
            dict.fromkeys(
                pattern
                for item in public_rules
                for pattern in (item.get("recommended_patterns") or [])
            )
        )
        avoid_patterns = list(
            dict.fromkeys(
                pattern
                for item in public_rules
                for pattern in (item.get("avoid_patterns") or [])
            )
        )
        return {
            "effective_backend": backend,
            "runner_contract": runner_contract,
            "pipeline_stage": pipeline_stage,
            "hard_rules": hard_rules,
            "preferred_rules": preferred_rules,
            "backend_neutral_rules": neutral_rules,
            "preferred_patterns": preferred_patterns,
            "avoid_patterns": avoid_patterns,
            "excluded_rules": excluded,
            "selected_rule_ids": [item.get("record_id") for item in public_rules],
            "evidence_refs": [
                f"backend_guidance:{item.get('record_id')}" for item in public_rules
            ],
            "confidence": min(
                (float(item.get("confidence") or 0.0) for item in public_rules),
                default=0.0,
            ),
            "semantic_ranking_used": bool(vector_scores),
            "knowledge_corpus_version": BACKEND_GUIDANCE_SCHEMA_VERSION,
        }
