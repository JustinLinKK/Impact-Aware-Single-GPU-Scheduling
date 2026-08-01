"""Qdrant-backed code-knowledge vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import os

import numpy as np

from localml_scheduler.hardware_features.records import load_seed_records

from .records import (
    API_SYMBOL_SCHEMA_VERSION,
    CODE_DOC_SCHEMA_VERSION,
    OPTIMIZATION_RECIPE_SCHEMA_VERSION,
    convert_hardware_feature_records,
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
            return [CODE_DOC_SCHEMA_VERSION, OPTIMIZATION_RECIPE_SCHEMA_VERSION, API_SYMBOL_SCHEMA_VERSION]
        schemas: list[str] = []
        for item in record_types:
            schema = _SCHEMA_BY_RECORD_TYPE.get(str(item))
            if schema and schema not in schemas:
                schemas.append(schema)
        return schemas or [CODE_DOC_SCHEMA_VERSION, OPTIMIZATION_RECIPE_SCHEMA_VERSION, API_SYMBOL_SCHEMA_VERSION]

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
        records = convert_hardware_feature_records(load_seed_records()) if source is None else load_code_knowledge_records(source)
        return self.ingest_records(records, recreate=recreate, dry_run=dry_run)

    def _match_condition(self, key: str, value: str) -> Any:
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
                    must.append(self._match_condition(key, str(value[0])))
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
