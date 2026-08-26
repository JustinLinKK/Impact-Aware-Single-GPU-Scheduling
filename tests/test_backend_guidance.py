from __future__ import annotations

from types import SimpleNamespace

import pytest

from localml_scheduler.code_knowledge import (
    BACKEND_GUIDANCE_SCHEMA_VERSION,
    CodeKnowledgeRecordError,
    CodeKnowledgeStore,
    load_backend_guidance_seed_records,
    validate_code_knowledge_record,
    validate_backend_guidance_corpus,
    validate_backend_guidance_record,
)
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import PairProfile, RuntimeProfile


def _store(tmp_path) -> CodeKnowledgeStore:
    settings = SchedulerSettings(
        runtime_root=tmp_path,
        hardware_feature_db={"enabled": False},
    )
    return CodeKnowledgeStore(settings)


def test_backend_guidance_is_exact_even_without_vector_database(tmp_path) -> None:
    store = _store(tmp_path)
    hardware = {
        "hardware": {"vendor": "nvidia", "compute_capability": "8.9"}
    }
    cuda = store.get_backend_design_guidance(
        effective_backend="cuda_process",
        pipeline_stage="model_design",
        hardware_context=hardware,
    )
    mps = store.get_backend_design_guidance(
        effective_backend="mps_process",
        pipeline_stage="model_design",
        hardware_context=hardware,
    )

    assert cuda["semantic_ranking_used"] is False
    assert mps["semantic_ranking_used"] is False
    assert any(rule.startswith("cuda_process.") for rule in cuda["selected_rule_ids"])
    assert not any(rule.startswith("mps_process.") for rule in cuda["selected_rule_ids"])
    assert any(rule.startswith("mps_process.") for rule in mps["selected_rule_ids"])
    assert not any(rule.startswith("cuda_process.") for rule in mps["selected_rule_ids"])
    assert set(cuda["selected_rule_ids"]).intersection(mps["selected_rule_ids"]) == {
        "shared.subprocess.task_quality_and_elasticity",
        "shared.subprocess.isolated_artifacts_and_telemetry",
    }


def test_backend_guidance_corpus_is_reachable_and_has_no_retired_modes() -> None:
    records = load_backend_guidance_seed_records()
    result = validate_backend_guidance_corpus(records)
    assert result["ok"]
    assert result["retired_backend_rule_ids"] == []
    assert result["unreachable_backends"] == []
    assert all(record["frameworks"] for record in records)
    assert all(record["review_status"] == "reviewed" for record in records)
    assert all(record["last_verified"] for record in records)
    assert all(record["source_refs"] for record in records)


def test_backend_guidance_validator_rejects_retired_mode() -> None:
    record = {
        "schema_version": BACKEND_GUIDANCE_SCHEMA_VERSION,
        "rule_id": "invalid.retired",
        "title": "invalid",
        "text": "invalid",
        "backend_modes": ["stream"],
        "runner_contracts": ["subprocess_job_v1"],
        "pipeline_stages": ["model_design"],
        "rule_type": "safety",
        "owner": "scheduler",
        "strength": "hard",
        "transferability": "exact_backend",
    }
    with pytest.raises(CodeKnowledgeRecordError, match="backend_modes"):
        validate_backend_guidance_record(record)


def test_every_knowledge_schema_rejects_retired_backend_scope() -> None:
    with pytest.raises(CodeKnowledgeRecordError, match="backend_modes"):
        validate_code_knowledge_record(
            {
                "schema_version": "code_doc_chunk_v1",
                "chunk_id": "invalid-retired-scope",
                "title": "invalid",
                "text": "invalid",
                "backend_modes": ["cuda_stream"],
            }
        )


def test_qdrant_list_filters_use_match_any(tmp_path) -> None:
    class MatchValue:
        def __init__(self, *, value):
            self.value = value

    class MatchAny:
        def __init__(self, *, any):
            self.any = any

    class FieldCondition:
        def __init__(self, *, key, match):
            self.key = key
            self.match = match

    class Filter:
        def __init__(self, *, must):
            self.must = must

    models = SimpleNamespace(
        MatchValue=MatchValue,
        MatchAny=MatchAny,
        FieldCondition=FieldCondition,
        Filter=Filter,
    )
    store = CodeKnowledgeStore(
        SchedulerSettings(runtime_root=tmp_path), qdrant_models=models
    )
    query_filter = store._build_filter(
        {"backend_modes": ["backend_neutral", "mps_process"]}
    )
    assert query_filter.must[0].match.any == [
        "backend_neutral",
        "mps_process",
    ]


def test_framework_only_vector_fallback_keeps_backend_and_runner_filters(
    tmp_path, monkeypatch
) -> None:
    client = SchedulerClient(
        SchedulerSettings(
            runtime_root=tmp_path,
            gpu_scheduler={"packing_backend": "cuda_process"},
        )
    )
    calls: list[dict] = []

    def search_code_knowledge(**kwargs):
        calls.append(dict(kwargs["filters"]))
        return []

    monkeypatch.setattr(client, "search_code_knowledge", search_code_knowledge)
    result = client.get_code_optimization_context(
        candidate={"stage": "draft", "framework": "pytorch"},
        graph_context={"hardware_context": {}, "derived_diagnosis": {}},
    )
    assert result["found"] is False
    assert len(calls) == 2
    assert all(
        call["backend_modes"] == ["backend_neutral", "cuda_process"]
        and call["runner_contracts"] == "subprocess_job_v1"
        and call["pipeline_stages"]
        == ["model_design", "datatype_precision", "training_evaluation"]
        for call in calls
    )


def test_graph_evidence_excludes_cross_backend_and_labels_exclusive_baseline(
    tmp_path,
) -> None:
    client = SchedulerClient(
        SchedulerSettings(
            runtime_root=tmp_path,
            gpu_scheduler={"packing_backend": "cuda_process"},
        )
    )
    hardware_key = client.store.hardware_key()
    for backend in ("cuda_process", "mps_process", "exclusive"):
        client.upsert_runtime_profile(
            RuntimeProfile.create(
                signature="candidate-signature",
                hardware_key=hardware_key,
                backend_name=backend,
                resolved_batch_size=8,
                strategy="epoch_1",
                epoch_1_seconds=10.0,
                estimated_total_runtime_seconds=100.0,
                confidence=0.9,
            )
        )
    for backend in ("cuda_process", "mps_process"):
        client.upsert_pair_profile(
            PairProfile.create(
                "candidate-signature",
                "neighbor-signature",
                backend_name=backend,
                hardware_key=hardware_key,
                compatible=True,
            )
        )

    evidence = client.get_profile_evidence(
        candidate={
            "script_signature": "candidate-signature",
            "model_key": "candidate-signature",
        },
        limit=20,
    )
    exact = evidence["graph_evidence"]["exact_profiles"]
    assert {
        (item["data"].get("backend_name"), item["transferability"])
        for item in exact
        if item["kind"] == "runtime_profile"
    } == {
        ("cuda_process", "exact_backend"),
        ("exclusive", "exclusive_baseline"),
    }
    assert {
        item["data"]["backend_name"]
        for item in evidence["graph_evidence"]["packed_profiles"]
    } == {"cuda_process"}
    assert any(
        item["backend_name"] == "mps_process"
        and item["reason"] == "backend_mismatch:cuda_process"
        for item in evidence["excluded_evidence"]
    )
