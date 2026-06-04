from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import yaml

from agents.memory.embedding_models import EmbeddingModel
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerConfig
from localml_scheduler.code_knowledge import CodeKnowledgeStore, convert_hardware_feature_records, validate_code_knowledge_record
from localml_scheduler.code_knowledge.records import load_code_knowledge_records
from localml_scheduler.hardware_features import load_seed_records, validate_feature_record
from localml_scheduler.hardware_features.records import HardwareFeatureRecordError, load_feature_records
from localml_scheduler.hardware_features.store import HardwareFeatureStore
from localml_scheduler.hardware_knowledge.records import load_hardware_knowledge_from_schema


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
        self.query_count = 0

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
        self.query_count += 1
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


class _MemoryCache:
    def __init__(self):
        self.values = {}
        self.invalidated = []

    def _key(self, namespace, payload):
        return (namespace, repr(sorted(payload.items())))

    def get(self, namespace, payload):
        return self.values.get(self._key(namespace, payload))

    def set(self, namespace, payload, value):
        self.values[self._key(namespace, payload)] = value

    def invalidate_namespace(self, namespace):
        self.invalidated.append(namespace)
        for key in [key for key in self.values if key[0] == namespace]:
            self.values.pop(key, None)


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

    def test_load_feature_records_accepts_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record = load_seed_records()[0]
            path = Path(tmp) / "record.yaml"
            path.write_text(yaml.safe_dump(record), encoding="utf-8")

            loaded = load_feature_records(tmp)

        self.assertEqual([item["record_id"] for item in loaded], [record["record_id"]])

    def test_hardware_knowledge_loader_accepts_generated_graph_json(self) -> None:
        bundle = load_hardware_knowledge_from_schema("schema")

        self.assertGreater(len(bundle["hardware"]), 0)
        self.assertGreater(len(bundle["features"]), 0)
        self.assertGreater(len(bundle["relationships"]), 0)
        self.assertTrue(any(item["hardware_id"] == "nvidia.blackwell.geforce_rtx_5090.spec" for item in bundle["hardware"]))
        self.assertTrue(any(item["feature_id"] == "bf16" for item in bundle["features"]))


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

        expected_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"mlevolve:hardware_feature:{record['record_id']}"))
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

    def test_code_knowledge_search_uses_redis_cache_when_available(self) -> None:
        settings = SchedulerConfig(runtime_root=tempfile.mkdtemp())
        fake_client = _FakeQdrantClient()
        cache = _MemoryCache()
        store = CodeKnowledgeStore(
            settings,
            qdrant_client=fake_client,
            qdrant_models=_FakeModels,
            embedding_model=_FakeEmbeddingModel(),
            redis_cache=cache,
        )
        store.ingest_records(convert_hardware_feature_records(load_seed_records()[:1]))
        fake_client.query_count = 0

        first = store.search(query="pytorch tensor core", filters={"framework": "pytorch"}, limit=4)
        first_query_count = fake_client.query_count
        second = store.search(query="pytorch tensor core", filters={"framework": "pytorch"}, limit=4)

        self.assertEqual(first, second)
        self.assertGreater(first_query_count, 0)
        self.assertEqual(fake_client.query_count, first_query_count)
        self.assertIn("vector:code_knowledge", cache.invalidated)

    def test_load_code_knowledge_records_accepts_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "code_doc_chunk_v1",
                        "chunk_id": "doc.test.amp",
                        "title": "AMP note",
                        "text": "Use autocast for tensor-core-friendly training.",
                        "technology_keys": ["pytorch_amp"],
                        "hardware_feature_keys": ["tensor_core"],
                        "workload_types": ["vision_training"],
                        "confidence": 0.8,
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_code_knowledge_records(tmp)

        self.assertEqual([item["record_id"] for item in loaded], ["doc.test.amp"])


class OpenRouterEmbeddingTest(unittest.TestCase):
    def test_openrouter_config_requires_explicit_model_and_dimension(self) -> None:
        with self.assertRaises(ValueError):
            SchedulerConfig(
                runtime_root=tempfile.mkdtemp(),
                hardware_feature_db={
                    "embedding_model_type": "openrouter",
                    "embedding_dimension": 2,
                },
            )
        with self.assertRaises(ValueError):
            SchedulerConfig(
                runtime_root=tempfile.mkdtemp(),
                hardware_feature_db={
                    "embedding_model_type": "openrouter",
                    "embedding_model_name": "openai/text-embedding-3-small",
                },
            )

    def test_openrouter_direct_api_key_is_not_serialized(self) -> None:
        config = SchedulerConfig(
            runtime_root=tempfile.mkdtemp(),
            hardware_feature_db={
                "embedding_model_type": "openrouter",
                "embedding_model_name": "test/embed",
                "embedding_dimension": 2,
                "embedding_api_key": "local-secret",
            },
        )

        self.assertEqual(config.hardware_feature_db.embedding_api_key, "local-secret")
        self.assertEqual(config.to_dict()["hardware_feature_db"]["embedding_api_key"], "")

    def test_openrouter_request_construction_and_batching(self) -> None:
        calls = []
        model = EmbeddingModel(
            model_type="openrouter",
            model_name="test/embed",
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            dimension=2,
            batch_size=2,
            retry_delay_seconds=0,
        )

        def fake_post(url, payload, headers):
            calls.append((url, payload, headers))
            return {"data": [{"embedding": [float(index), 1.0]} for index, _ in enumerate(payload["input"])]}

        model._openrouter_post_json = fake_post  # type: ignore[method-assign]

        result = model.encode(["a", "b", "c"])

        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "https://openrouter.ai/api/v1/embeddings")
        self.assertEqual(calls[0][1]["model"], "test/embed")
        self.assertEqual(calls[0][1]["encoding_format"], "float")
        self.assertEqual(calls[0][1]["input"], ["a", "b"])
        self.assertEqual(calls[0][2]["Authorization"], "Bearer test-key")


class SchemaIngestionTest(unittest.TestCase):
    def test_schema_ingestion_dry_run_counts_directory_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "schema"
            hardware_dir = root / "hardware_feature_records"
            code_dir = root / "code_doc_chunks"
            api_dir = root / "api_symbol_chunks"
            hardware_dir.mkdir(parents=True)
            code_dir.mkdir()
            api_dir.mkdir()

            hardware_record = load_seed_records()[0]
            (hardware_dir / "hardware.yaml").write_text(yaml.safe_dump(hardware_record), encoding="utf-8")
            (code_dir / "doc.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "code_doc_chunk_v1",
                        "chunk_id": "doc.test",
                        "title": "Doc",
                        "text": "A code doc chunk.",
                        "technology_keys": ["pytorch"],
                        "hardware_feature_keys": ["tensor_core"],
                        "workload_types": ["vision_training"],
                        "confidence": 0.7,
                    }
                ),
                encoding="utf-8",
            )
            (api_dir / "api.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "api_symbol_chunk_v1",
                        "api_symbol_id": "api.test",
                        "api_symbol": "torch.autocast",
                        "title": "torch.autocast",
                        "usage_summary": "Automatic mixed precision context manager.",
                        "text": "Use autocast around forward pass.",
                        "api_symbols": ["torch.autocast"],
                        "technology_keys": ["pytorch_amp"],
                        "hardware_feature_keys": ["tensor_core"],
                        "confidence": 0.9,
                    }
                ),
                encoding="utf-8",
            )

            client = SchedulerClient(SchedulerConfig(runtime_root=Path(tmp) / "runtime"))
            result = client.ingest_schema_knowledge(schema_root=root, dry_run=True)

        self.assertTrue(result["ok"])
        self.assertEqual(result["source_counts"]["hardware_feature_records"], 1)
        self.assertEqual(result["source_counts"]["code_doc_chunks"], 1)
        self.assertEqual(result["source_counts"]["api_symbol_chunks"], 1)
        self.assertGreaterEqual(result["source_counts"]["converted_from_hardware"], 1)


if __name__ == "__main__":
    unittest.main()
