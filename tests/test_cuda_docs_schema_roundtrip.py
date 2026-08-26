from __future__ import annotations

from types import SimpleNamespace
import hashlib

import numpy as np

from localml_scheduler.config import SchedulerConfig
from localml_scheduler.code_knowledge import CodeKnowledgeStore
from localml_scheduler.code_knowledge.records import validate_code_knowledge_record
from localml_scheduler.cuda_mcp_bridge import HardwareFacts, to_records


class _Embedding:
    dimension = 3

    def encode(self, texts, show_progress_bar=False):
        del show_progress_bar
        return np.asarray(
            [
                [float(len(text)), float(text.lower().count("cuda")), 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


class _Models:
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


class _Qdrant:
    def __init__(self):
        self.collections: dict[str, object] = {}
        self.points: dict[str, dict[str, object]] = {}

    def collection_exists(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.collections[collection_name] = vectors_config
        self.points.setdefault(collection_name, {})

    def delete_collection(self, collection_name):
        self.collections.pop(collection_name, None)
        self.points.pop(collection_name, None)

    def upsert(self, collection_name, points):
        collection = self.points.setdefault(collection_name, {})
        for point in points:
            collection[point.id] = point

    def query_points(self, collection_name, query, query_filter, limit, with_payload):
        del query, with_payload
        rows = []
        for point in self.points.get(collection_name, {}).values():
            if self._matches(point.payload, query_filter):
                rows.append(SimpleNamespace(payload=point.payload, score=0.99))
        return SimpleNamespace(points=rows[:limit])

    @staticmethod
    def _matches(payload, query_filter):
        if query_filter is None:
            return True
        for condition in query_filter.must:
            expected = condition.match.value
            actual = payload
            for part in str(condition.key).split("."):
                actual = actual.get(part) if isinstance(actual, dict) else None
            if isinstance(actual, list):
                if expected not in actual:
                    return False
            elif actual != expected:
                return False
        return True


def _facts(
    *, gpu: str = "NVIDIA A10", capability=(8, 6), backend_hash="cfg-a"
) -> HardwareFacts:
    return HardwareFacts(
        gpu_name=gpu,
        gpu_architecture="ampere" if capability >= (8, 0) else "volta",
        compute_capability=capability,
        cuda_version="12.4.1",
        driver_version="550.54.15",
        torch_version="2.4.1+cu124",
        backend_config_hash=backend_hash,
    )


def _source(version: str = "12.4") -> list[dict[str, str]]:
    return [
        {
            "title": "CUDA C Programming Guide",
            "url": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
            "source_version": version,
            "retrieved_or_verified_date": "2026-08-26",
        }
    ]


def _record(
    *,
    facts: HardwareFacts | None = None,
    backend: str = "cuda_process",
    cache_key: str = "localml:cuda_docs:v2:key-a",
    source_version: str = "12.4",
    text: str = "CUDA graphs retain source-labelled context without Markdown bullets.",
):
    records = to_records(
        topic="verify CUDA execution behavior",
        answer=text,
        facts=facts or _facts(),
        source_refs=_source(source_version),
        verified_date="2026-08-26",
        effective_backend=backend,
        cache_key=cache_key,
        remote_tool_schema_hash="tool-schema-sha256",
    )
    assert len(records) == 1
    return records[0]


def test_bridge_validator_qdrant_exact_key_round_trip_preserves_metadata(
    tmp_path,
) -> None:
    raw = _record()
    normalized = validate_code_knowledge_record(raw)
    qdrant = _Qdrant()
    settings = SchedulerConfig(runtime_root=tmp_path)
    store = CodeKnowledgeStore(
        settings,
        qdrant_client=qdrant,
        qdrant_models=_Models,
        embedding_model=_Embedding(),
    )

    result = store.ingest_records([normalized])
    rows = store.get_cuda_doc_chunks(cache_key=raw["cuda_docs_cache_key"])

    assert result["ok"] is True
    assert len(rows) == 1
    row = rows[0]
    assert row["source_url"] == _source()[0]["url"]
    assert row["source_refs"][0]["source_version"] == "12.4"
    assert row["retrieved_or_verified_date"] == "2026-08-26"
    assert row["compute_capabilities"] == ["8.6"]
    assert row["accelerator_names"] == ["nvidia_a10"]
    assert row["backend_keys"] == ["cuda_process"]
    assert row["toolkit_versions"] == ["12.4.1"]
    assert row["framework_versions"] == ["2.4.1+cu124"]
    assert row["driver_versions"] == ["550.54.15"]
    assert row["applicability"]["gpu_architecture"] == "ampere"
    assert row["applicability"]["backend_config_hash"] == "cfg-a"
    assert row["remote_tool_schema_hash"] == "tool-schema-sha256"
    assert "Markdown bullets" in row["text"]


def test_gpu_backend_and_source_versions_have_distinct_ids_and_keys() -> None:
    v100 = _record(
        facts=_facts(gpu="Tesla V100", capability=(7, 0), backend_hash="v100"),
        cache_key="localml:cuda_docs:v2:v100-process",
    )
    a10 = _record(cache_key="localml:cuda_docs:v2:a10-process")
    mps = _record(backend="mps_process", cache_key="localml:cuda_docs:v2:a10-mps")
    new_source = _record(
        cache_key="localml:cuda_docs:v2:a10-process",
        source_version="12.5",
    )

    ids = {item["chunk_id"] for item in (v100, a10, mps, new_source)}
    assert len(ids) == 4
    assert v100["compute_capabilities"] == ["7.0"]
    assert a10["compute_capabilities"] == ["8.6"]
    assert mps["backend_modes"] == ["mps_process"]
    assert a10["backend_modes"] == ["cuda_process"]


def test_record_id_is_source_and_content_specific() -> None:
    first = _record(text="First retrieved source chunk.")
    second = _record(text="Second retrieved source chunk.")
    assert first["chunk_id"] != second["chunk_id"]
    assert first["chunk_id"].startswith("nvidia.cuda_mcp.")
    assert len(hashlib.sha256(first["text"].encode()).hexdigest()) == 64


def test_capability_status_distinguishes_functional_native_and_unsupported() -> None:
    v100 = _facts(gpu="Tesla V100", capability=(7, 0), backend_hash="v100")
    a10 = _facts(capability=(8, 6))
    assert v100.support_status("bf16") == "functionally_supported"
    assert v100.support_status("tf32") == "unsupported"
    assert a10.support_status("bf16") == "natively_accelerated"
    assert a10.support_status("not-in-table") == "unknown_pending_local_verification"


def test_general_search_requires_complete_exact_applicability_for_nvidia_records(
    tmp_path,
) -> None:
    qdrant = _Qdrant()
    store = CodeKnowledgeStore(
        SchedulerConfig(runtime_root=tmp_path),
        qdrant_client=qdrant,
        qdrant_models=_Models,
        embedding_model=_Embedding(),
    )
    a10 = _record(cache_key="localml:cuda_docs:v2:a10")
    v100 = _record(
        facts=_facts(gpu="Tesla V100", capability=(7, 0), backend_hash="v100"),
        cache_key="localml:cuda_docs:v2:v100",
    )
    mps = _record(
        backend="mps_process",
        cache_key="localml:cuda_docs:v2:mps",
    )
    store.ingest_records([a10, v100, mps])

    assert (
        store.search(
            query="CUDA execution",
            filters={"framework": "pytorch"},
            record_types=["code_doc_chunks"],
        )
        == []
    )
    exact = {
        "source_type": "nvidia_cuda_docs",
        "verified_source": True,
        **{
            f"applicability.{key}": value
            for key, value in a10["applicability"].items()
            if key != "remote_tool_schema_hash"
        },
    }
    rows = store.search(
        query="CUDA execution",
        filters=exact,
        record_types=["code_doc_chunks"],
    )
    assert [row["record_id"] for row in rows] == [a10["chunk_id"]]
    assert rows[0]["backend_modes"] == ["cuda_process"]
    assert rows[0]["compute_capabilities"] == ["8.6"]
