from __future__ import annotations

import json

from localml_scheduler.hardware_knowledge import feature_filter as canonical_filter


def _ids(payload):
    return {item["feature_id"] for item in payload.get("features") or []}


def test_composite_pipeline_stage_contract_and_parallelism_visibility() -> None:
    assert canonical_filter.PIPELINE_STAGES == (
        "model_design",
        "datatype_precision",
        "training_evaluation",
    )
    design = canonical_filter.query_hardware_features(
        "GeForce RTX 5090", "model_design"
    )
    assert design["stage_filter"] == "model_design"
    ids = _ids(design)
    assert "dataset_decomposition" in ids
    assert "tensor_cores" in ids
    assert "async_tensor_parallel" in ids


def test_runtime_backend_guidance_is_not_an_unconditioned_hardware_feature() -> None:
    for stage in canonical_filter.PIPELINE_STAGES:
        payload = canonical_filter.query_hardware_features(
            "GeForce RTX 5090", stage
        )
        ids = _ids(payload)
        assert "cuda_stream_scheduler_compatibility" not in ids
        assert "cuda_process_scheduler_compatibility" not in ids
        assert "mps_scheduler_compatibility" not in ids


def test_noncanonical_stage_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unsupported pipeline stage"):
        canonical_filter.query_hardware_features("GeForce RTX 5090", "model")


def test_filtered_payload_has_no_source_urls() -> None:
    payload = canonical_filter.query_hardware_features(
        "NVIDIA GeForce RTX 5090", "datatype_precision"
    )
    serialized = json.dumps(payload)
    assert payload["found"] is True
    assert "source_urls" not in serialized
    assert "http://" not in serialized
    assert "https://" not in serialized
