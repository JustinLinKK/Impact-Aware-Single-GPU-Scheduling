"""Qdrant-backed hardware feature vector store."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable
import uuid

import numpy as np

from localml_scheduler.redis_cache import RedisLRUCache, vector_cache_enabled

from .records import load_feature_records, load_seed_records, record_to_search_text


HardwareContextProvider = Callable[[str], dict[str, Any]]


@dataclass(slots=True)
class HardwareFeatureSearchResult:
    record: dict[str, Any]
    score: float
    hardware_match: dict[str, Any]

    def to_public_dict(self) -> dict[str, Any]:
        record = self.record
        return {
            "record_id": record.get("record_id"),
            "title": record.get("title"),
            "summary_text": record.get("summary_text"),
            "detail_text": record.get("detail_text"),
            "score": self.score,
            "hardware_match": self.hardware_match,
            "tags": list(record.get("tags") or []),
            "recommended_patterns": list(record.get("recommended_patterns") or []),
            "avoid_patterns": list(record.get("avoid_patterns") or []),
            "source_refs": list(record.get("source_refs") or []),
            "last_verified": record.get("last_verified"),
            "confidence": record.get("confidence"),
        }


class HardwareFeatureStore:
    """Thin Qdrant adapter for curated hardware feature records."""

    def __init__(
        self,
        settings: Any,
        *,
        qdrant_client: Any | None = None,
        qdrant_models: Any | None = None,
        embedding_model: Any | None = None,
        redis_cache: Any | None = None,
    ):
        self.settings = settings
        self.config = settings.hardware_feature_db
        self._client = qdrant_client
        self._models = qdrant_models
        self._embedding_model = embedding_model
        self._redis_cache = redis_cache
        self._redis_cache_checked = redis_cache is not None

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "enabled", False)) and getattr(self.config, "provider", "qdrant") == "qdrant"

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

    def _cache(self) -> Any | None:
        if self._redis_cache is not None:
            return self._redis_cache
        if self._redis_cache_checked or not vector_cache_enabled(self.settings):
            return None
        self._redis_cache_checked = True
        self._redis_cache = RedisLRUCache.from_settings(self.settings)
        return self._redis_cache

    def _cache_payload(
        self,
        *,
        query: str,
        hardware_context: dict[str, Any] | None,
        architecture: str | None,
        vendor: str | None,
        workload_type: str | None,
        framework: str | None,
        limit: int,
    ) -> dict[str, Any]:
        return {
            "store": "hardware_features",
            "query": query,
            "hardware_context": hardware_context or {},
            "architecture": architecture,
            "vendor": vendor,
            "workload_type": workload_type,
            "framework": framework,
            "limit": max(1, int(limit)),
            "collection_name": self.config.collection_name,
            "embedding_model_type": getattr(self.config, "embedding_model_type", None),
            "embedding_model_name": getattr(self.config, "embedding_model_name", None),
            "embedding_dimension": getattr(self.config, "embedding_dimension", None),
        }

    def _embedder(self) -> Any:
        if self._embedding_model is None:
            from agents.memory.embedding_models import EmbeddingModel

            self._embedding_model = EmbeddingModel(
                model_type=self.config.embedding_model_type,
                model_name=self.config.embedding_model_name,
                api_key=(
                    getattr(self.config, "embedding_api_key", "")
                    or os.getenv(getattr(self.config, "embedding_api_key_env", "OPENROUTER_API_KEY") or "")
                )
                if getattr(self.config, "embedding_model_type", "") == "openrouter"
                else None,
                api_key_env=getattr(self.config, "embedding_api_key_env", "OPENROUTER_API_KEY"),
                base_url=getattr(self.config, "embedding_base_url", None),
                dimension=getattr(self.config, "embedding_dimension", None),
                batch_size=getattr(self.config, "embedding_batch_size", 32),
                max_retries=getattr(self.config, "embedding_max_retries", 4),
                retry_delay_seconds=getattr(self.config, "embedding_retry_delay_seconds", 1.0),
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

    def _point_id(self, record_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mlevolve:hardware_feature:{record_id}"))

    def _collection_exists(self, client: Any) -> bool:
        if hasattr(client, "collection_exists"):
            return bool(client.collection_exists(self.config.collection_name))
        try:
            client.get_collection(self.config.collection_name)
            return True
        except Exception:
            return False

    def _collection_vector_size(self, client: Any, collection_name: str) -> int | None:
        if not hasattr(client, "get_collection"):
            return None
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            return None
        vectors = getattr(getattr(getattr(collection, "config", None), "params", None), "vectors", None)
        if isinstance(vectors, dict):
            first = next(iter(vectors.values()), None)
            return int(getattr(first, "size", 0) or 0) or None
        size = getattr(vectors, "size", None)
        try:
            return int(size) if size is not None else None
        except (TypeError, ValueError):
            return None

    def ensure_collection(self, *, recreate: bool = False) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "reason": "hardware feature database disabled"}
        client = self._qdrant_client()
        models = self._qdrant_models()
        collection_name = self.config.collection_name
        dimension = self._dimension()
        if recreate and hasattr(client, "delete_collection"):
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass
        if recreate or not self._collection_exists(client):
            vector_params = models.VectorParams(size=dimension, distance=self._distance())
            client.create_collection(collection_name=collection_name, vectors_config=vector_params)
            created = True
        else:
            created = False
            existing_dimension = self._collection_vector_size(client, collection_name)
            if existing_dimension is not None and existing_dimension != dimension:
                raise ValueError(
                    f"Qdrant collection {collection_name!r} has vector size {existing_dimension}, "
                    f"but configured embedding dimension is {dimension}. Recreate the collection or update embedding_dimension."
                )
        if recreate or created:
            cache = self._cache()
            if cache is not None:
                cache.invalidate_namespace("vector:hardware_features")
        return {"ok": True, "collection_name": collection_name, "dimension": dimension, "created": created}

    def ingest_records(self, records: list[dict[str, Any]], *, recreate: bool = False, dry_run: bool = False) -> dict[str, Any]:
        if dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "collection_name": self.config.collection_name,
                "record_count": len(records),
                "record_ids": [record["record_id"] for record in records],
            }
        if not self.enabled:
            return {"ok": False, "reason": "hardware feature database disabled", "record_count": len(records)}
        collection = self.ensure_collection(recreate=recreate)
        client = self._qdrant_client()
        models = self._qdrant_models()
        texts = [record_to_search_text(record) for record in records]
        vectors = self._encode(texts)
        points = [
            models.PointStruct(
                id=self._point_id(record["record_id"]),
                vector=vectors[index].tolist(),
                payload={**record, "search_text": texts[index]},
            )
            for index, record in enumerate(records)
        ]
        if points:
            client.upsert(collection_name=self.config.collection_name, points=points)
        cache = self._cache()
        if cache is not None:
            cache.invalidate_namespace("vector:hardware_features")
        return {
            "ok": True,
            "dry_run": False,
            "collection_name": self.config.collection_name,
            "record_count": len(records),
            "record_ids": [record["record_id"] for record in records],
            "collection": collection,
        }

    def ingest_source(self, source: str | Path | None = None, *, recreate: bool = False, dry_run: bool = False) -> dict[str, Any]:
        records = load_seed_records() if source is None else load_feature_records(source)
        return self.ingest_records(records, recreate=recreate, dry_run=dry_run)

    def _match_any_condition(self, key: str, value: str) -> Any:
        models = self._qdrant_models()
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    def _build_filter(self, filters: dict[str, str | None]) -> Any | None:
        models = self._qdrant_models()
        must = [
            self._match_any_condition(key, value)
            for key, value in filters.items()
            if value is not None and str(value).strip()
        ]
        if not must:
            return None
        return models.Filter(must=must)

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

    def _query_points(self, query_vector: list[float], query_filter: Any | None, limit: int) -> list[Any]:
        client = self._qdrant_client()
        if hasattr(client, "query_points"):
            result = client.query_points(
                collection_name=self.config.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            return list(getattr(result, "points", result) or [])
        return list(
            client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            or []
        )

    def _hardware_match(self, record: dict[str, Any], hardware_context: dict[str, Any] | None) -> dict[str, Any]:
        hardware = dict((hardware_context or {}).get("hardware") or {})
        gpu_name = str(hardware.get("gpu_name") or "").lower()
        compute_capability = str(hardware.get("compute_capability") or "")
        accelerator_names = [str(value).lower().replace(" ", "_") for value in record.get("accelerator_names") or []]
        exact_accelerator = any(name and name in gpu_name.replace(" ", "_") for name in accelerator_names)
        exact_compute = bool(compute_capability and compute_capability in {str(value) for value in record.get("compute_capabilities") or []})
        return {
            "hardware_key": hardware.get("hardware_key"),
            "gpu_name": hardware.get("gpu_name"),
            "exact_accelerator": exact_accelerator,
            "exact_compute_capability": exact_compute,
            "matched": bool(exact_accelerator or exact_compute),
        }

    def search(
        self,
        *,
        query: str,
        hardware_context: dict[str, Any] | None = None,
        architecture: str | None = None,
        vendor: str | None = None,
        workload_type: str | None = None,
        framework: str | None = "pytorch",
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        cache = self._cache()
        cache_payload = self._cache_payload(
            query=query,
            hardware_context=hardware_context,
            architecture=architecture,
            vendor=vendor,
            workload_type=workload_type,
            framework=framework,
            limit=limit,
        )
        if cache is not None:
            cached = cache.get("vector:hardware_features", cache_payload)
            if isinstance(cached, list):
                return cached
        try:
            client = self._qdrant_client()
            if not self._collection_exists(client):
                return []
            query_vector = self._encode([query])[0].tolist()
            query_filter = self._build_filter(
                {
                    "architectures": architecture,
                    "vendor": vendor,
                    "workload_types": workload_type,
                    "frameworks": framework,
                }
            )
            points = self._query_points(query_vector, query_filter, max(1, int(limit)))
        except Exception:
            return []
        results: list[dict[str, Any]] = []
        for point in points:
            payload = self._payload_from_point(point)
            payload.pop("search_text", None)
            results.append(
                HardwareFeatureSearchResult(
                    record=payload,
                    score=self._score_from_point(point),
                    hardware_match=self._hardware_match(payload, hardware_context),
                ).to_public_dict()
            )
        if cache is not None:
            cache.set("vector:hardware_features", cache_payload, results)
        return results
