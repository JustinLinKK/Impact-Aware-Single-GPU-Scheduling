from __future__ import annotations

from types import SimpleNamespace
import hashlib
import tempfile
import unittest

import numpy as np

from localml_scheduler.config import SchedulerConfig
from localml_scheduler.code_knowledge import CodeKnowledgeStore, convert_hardware_feature_records, validate_code_knowledge_record
from localml_scheduler.hardware_features import load_seed_records, validate_feature_record
from localml_scheduler.hardware_features.records import HardwareFeatureRecordError
from localml_scheduler.hardware_features.store import HardwareFeatureStore


class _FakeEmbeddingModel:
    dimension = 3

    def encode(self, texts, show_progress_bar=False):
        del show_progress_bar
        rows = []
        for text in texts:
            rows.append([float(len(text)), float(text.count("cuda")), 1.0])
        return np.asarray(rows, dtype=np.float32)


class _FakeModels:
    class Distance:
        COSINE = "Cosine"

    class VectorParams:
        def __init__(self, size, distance):
            self.size = size
            self.distance = distance

    class PointStruct:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload

    class MatchValue:
        def __init__(self, value):
            self.value = value

    class FieldCondition:
        def __init__(self, key, match):
            self.key = key
            self.match = match

    class Filter:
        def __init__(self, must):
            self.must = must


class _FakeQdrantClient:
    def __init__(self):
        self.collections = {}
        self.points = {}
        self.created_vector_params = None
        self.last_filter = None

    def collection_exists(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.collections[collection_name] = vectors_config
        self.created_vector_params = vectors_config

    def delete_collection(self, collection_name):
        self.collections.pop(collection_name, None)
        self.points.clear()

    def upsert(self, collection_name, points):
        self.collections.setdefault(collection_name, None)
        for point in points:
            self.points[point.id] = point

    def query_points(self, collection_name, query, query_filter, limit, with_payload):
        del collection_name, query, with_payload
        self.last_filter = query_filter
        rows = []
        for point in self.points.values():
            if self._matches(point.payload, query_filter):
                rows.append(SimpleNamespace(payload=point.payload, score=0.99))
        return SimpleNamespace(points=rows[:limit])

    def _matches(self, payload, query_filter):
        if query_filter is None:
            return True
        for condition in query_filter.must:
            value = condition.match.value
            payload_value = payload.get(condition.key)
            if isinstance(payload_value, list):
                if value not in payload_value:
                    return False
            elif payload_value != value:
                return False
        return True


class HardwareFeatureRecordTest(unittest.TestCase):
    def test_seed_records_are_valid(self) -> None:
        records = load_seed_records()

        self.assertGreaterEqual(len(records), 5)
        self.assertTrue(all(record["schema_version"] == "hardware_feature_record_v1" for record in records))

    def test_validation_rejects_missing_required_field(self) -> None:
        record = dict(load_seed_records()[0])
        record.pop("title")

        with self.assertRaises(HardwareFeatureRecordError):
            validate_feature_record(record)

    def test_validation_rejects_bad_confidence(self) -> None:
        record = dict(load_seed_records()[0])
        record["confidence"] = 1.5

        with self.assertRaises(HardwareFeatureRecordError):
            validate_feature_record(record)

    def test_validation_rejects_unsupported_schema(self) -> None:
        record = dict(load_seed_records()[0])
        record["schema_version"] = "future_schema"

        with self.assertRaises(HardwareFeatureRecordError):
            validate_feature_record(record)


class HardwareFeatureStoreTest(unittest.TestCase):
    def _store(self):
        settings = SchedulerConfig(runtime_root=tempfile.mkdtemp())
        fake_client = _FakeQdrantClient()
        store = HardwareFeatureStore(
            settings,
            qdrant_client=fake_client,
            qdrant_models=_FakeModels,
            embedding_model=_FakeEmbeddingModel(),
        )
        return store, fake_client

    def test_collection_creation_uses_embedding_dimension_and_cosine_distance(self) -> None:
        store, fake_client = self._store()

        result = store.ensure_collection(recreate=True)

        self.assertTrue(result["ok"])
        self.assertEqual(result["dimension"], 3)
        self.assertEqual(fake_client.created_vector_params.size, 3)
        self.assertEqual(fake_client.created_vector_params.distance, "Cosine")

    def test_ingestion_uses_deterministic_ids_and_idempotent_upsert(self) -> None:
        store, fake_client = self._store()
        record = load_seed_records()[0]

        first = store.ingest_records([record])
        second = store.ingest_records([record])

        expected_id = hashlib.sha256(record["record_id"].encode("utf-8")).hexdigest()
        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertEqual(list(fake_client.points.keys()), [expected_id])
        self.assertEqual(fake_client.points[expected_id].payload["record_id"], record["record_id"])

    def test_retrieval_filters_and_hardware_match_are_applied(self) -> None:
        store, fake_client = self._store()
        store.ingest_records(load_seed_records())

        results = store.search(
            query="blackwell pytorch vision training tensor cores",
            hardware_context={
                "hardware": {
                    "hardware_key": "hw-5090",
                    "gpu_name": "NVIDIA GeForce RTX 5090",
                    "compute_capability": "12.0",
                }
            },
            architecture="blackwell",
            vendor="nvidia",
            workload_type="vision_training",
            framework="pytorch",
            limit=8,
        )

        self.assertGreaterEqual(len(results), 1)
        self.assertIsNotNone(fake_client.last_filter)
        self.assertTrue(all(str(result["record_id"]).startswith("nvidia.blackwell") for result in results))
        self.assertTrue(any(result["hardware_match"]["matched"] for result in results))
        self.assertIn("recommended_patterns", results[0])


class CodeKnowledgeStoreTest(unittest.TestCase):
    def _store(self):
        settings = SchedulerConfig(runtime_root=tempfile.mkdtemp())
        fake_client = _FakeQdrantClient()
        store = CodeKnowledgeStore(
            settings,
            qdrant_client=fake_client,
            qdrant_models=_FakeModels,
            embedding_model=_FakeEmbeddingModel(),
        )
        return store, fake_client

    def test_hardware_feature_records_convert_to_code_knowledge(self) -> None:
        records = convert_hardware_feature_records([load_seed_records()[0]])

        self.assertGreaterEqual(len(records), 1)
        self.assertTrue(all(record["record_type"] in {"code_doc_chunks", "optimization_recipe_chunks"} for record in records))
        self.assertTrue(any(record["record_type"] == "code_doc_chunks" for record in records))

    def test_code_knowledge_validation_accepts_recipe(self) -> None:
        record = validate_code_knowledge_record(
            {
                "schema_version": "optimization_recipe_chunk_v1",
                "recipe_id": "recipe.test.amp",
                "title": "Use AMP for low SM utilization",
                "problem_statement": "FP32 training has low SM utilization.",
                "solution_summary": "Use torch.amp.autocast.",
                "text": "Wrap forward and loss with autocast.",
                "technology_keys": ["pytorch_amp"],
                "hardware_feature_keys": ["tensor_core"],
                "optimization_targets": ["improve_throughput"],
                "profile_symptoms": ["low_sm_utilization"],
                "recommended_patterns": ["Use torch.amp.autocast."],
                "avoid_patterns": [],
                "confidence": 0.8,
            }
        )

        self.assertEqual(record["record_type"], "optimization_recipe_chunks")
        self.assertEqual(record["record_id"], "recipe.test.amp")

    def test_code_knowledge_store_uses_three_physical_collections(self) -> None:
        store, fake_client = self._store()
        records = convert_hardware_feature_records(load_seed_records()[:1])

        ingest = store.ingest_records(records, recreate=True)
        results = store.search(
            query="pytorch training tensor core optimization",
            filters={"framework": "pytorch"},
            record_types=["code_doc_chunks", "optimization_recipe_chunks", "api_symbol_chunks"],
            limit=8,
        )

        self.assertTrue(ingest["ok"])
        self.assertIn("code_doc_chunks", fake_client.collections)
        self.assertIn("optimization_recipe_chunks", fake_client.collections)
        self.assertIn("api_symbol_chunks", fake_client.collections)
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all(result["record_type"] in {"code_doc_chunks", "optimization_recipe_chunks"} for result in results))


if __name__ == "__main__":
    unittest.main()
