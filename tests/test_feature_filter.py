"""Test Filter1 & Filter2 with real graph DB data."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_filter_module():
    spec = importlib.util.spec_from_file_location(
        "feature_filter",
        str(Path(__file__).resolve().parents[1] / "hardware_knowledge" / "feature_filter.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _feature_keys(rows):
    return {row[0] if isinstance(row, list) and row else row for row in rows}


def _assert_feature_key_pairs(rows):
    assert rows
    for row in rows:
        assert isinstance(row, list)
        assert len(row) == 2
        assert all(isinstance(value, str) for value in row)
        assert row[0]
        assert row[1]
        assert "..." not in row[1]
        assert row[1].endswith(".")
        assert "mainly used for" in row[1]
        assert "http://" not in row[1]
        assert "https://" not in row[1]


def test_pipeline_stage_categories_align_with_contract():
    mod = load_filter_module()

    assert mod.PIPELINE_STAGES == ("model_structure", "datatype", "training_parameters")

    datatype = mod.query_hardware_features("GeForce RTX 5090", "datatype")
    datatype_ids = {feature["feature_id"] for feature in datatype["features"]}
    assert datatype["stage_filter"] == "datatype"
    assert "dataset_decomposition" in datatype_ids
    assert "nvimagecodec_gpu_decode" in datatype_ids
    assert "bf16" not in datatype_ids
    assert "muon_optimizer" not in datatype_ids

    training = mod.query_hardware_features("GeForce RTX 5090", "training_parameters")
    training_ids = {feature["feature_id"] for feature in training["features"]}
    assert training["stage_filter"] == "training_parameters"
    assert {"muon_optimizer", "soap_optimizer", "ademamix_optimizer"} <= training_ids
    assert "gram_newton_schulz_symmetric_gemm" in training_ids
    assert {"bf16", "fp8_rowwise_scaling", "async_tensor_parallel"} <= training_ids


def test_removed_stage_names_are_rejected():
    mod = load_filter_module()

    for stage in ("model", "optimizer", "tuning"):
        with pytest.raises(ValueError, match="unsupported hardware pipeline stage"):
            mod.query_hardware_features("GeForce RTX 5090", stage)
        with pytest.raises(ValueError, match="unsupported hardware pipeline stage"):
            mod.query_hardware_node("GeForce RTX 5090", stage)


def test_5090_training_filter_marks_unconfirmed_candidates():
    mod = load_filter_module()

    result = mod.query_hardware_features("GeForce RTX 5090", "training_parameters")
    by_id = {feature["feature_id"]: feature for feature in result["features"]}

    assert result["stage_filter"] == "training_parameters"
    assert "muon_optimizer" in by_id
    assert "gram_newton_schulz_symmetric_gemm" in by_id
    assert "ns_steps=5" in by_id["muon_optimizer"]["notes"]
    assert "match_rms_adamw" in by_id["muon_optimizer"]["notes"]
    assert "embedding/bias/norm/head/non-2D params" in by_id["muon_optimizer"]["notes"]
    assert by_id["soap_optimizer"]["support_level"] == "experimental"
    assert by_id["ademamix_optimizer"]["support_level"] == "experimental"
    assert by_id["soap_optimizer"]["recommended"] is False
    assert by_id["ademamix_optimizer"]["recommended"] is False
    assert "not widely confirmed" in by_id["soap_optimizer"]["limitations"]
    assert "not widely confirmed" in by_id["ademamix_optimizer"]["limitations"]


def test_5090_model_structure_node_exposes_source_free_feature_lists():
    mod = load_filter_module()

    result = mod.query_hardware_node("GeForce RTX 5090", "model_structure")

    assert result["stage_filter"] == "model_structure"
    assert "node_id" not in result
    assert "source_urls" not in result
    assert "recommended_patterns" in result
    assert "avoid_patterns" in result
    assert any("sm_120" in item for item in result["recommended_patterns"])
    assert any("sm_100 launch assumptions" in item for item in result["avoid_patterns"])
    _assert_feature_key_pairs(result["stage_feature_keys"])
    stage_keys = _feature_keys(result["stage_feature_keys"])
    assert "tensor_cores" in stage_keys
    assert "sm_120" in stage_keys
    assert "tensor_cores_5gen" in stage_keys
    assert "soap_optimizer" not in stage_keys
    assert "http://" not in json.dumps(result)
    assert "https://" not in json.dumps(result)


def test_training_parameters_feature_key_mapping_is_merged_and_source_free():
    mod = load_filter_module()

    datatype = mod.query_hardware_node("GeForce RTX 5090", "datatype")
    training = mod.query_hardware_node("GeForce RTX 5090", "training_parameters")

    for payload in (datatype, training):
        for field in (
            "stage_feature_keys",
            "recommended_feature_keys",
            "not_recommended_feature_keys",
            "conditional_feature_keys",
        ):
            if payload.get(field):
                _assert_feature_key_pairs(payload[field])
    datatype_keys = _feature_keys(datatype["stage_feature_keys"])
    training_keys = _feature_keys(training["stage_feature_keys"])
    training_not_recommended = _feature_keys(training["not_recommended_feature_keys"])
    assert {"dataset_decomposition", "nvimagecodec_gpu_decode"} <= datatype_keys
    assert "tensor_cores" not in datatype_keys
    assert "muon_optimizer" not in datatype_keys
    assert {"bf16", "fp8_rowwise_scaling"} <= training_keys
    assert {"muon_optimizer", "gram_newton_schulz_symmetric_gemm"} <= training_keys
    assert {"soap_optimizer", "ademamix_optimizer"} <= training_not_recommended
    assert any("gradient accumulation" in item.lower() for item in training["recommended_patterns"])
    assert any("nvfp4" in item.lower() or "mxfp8" in item.lower() for item in training["recommended_patterns"])
    assert any("fp4" in item.lower() or "mxfp8" in item.lower() for item in training["avoid_patterns"])
    combined = json.dumps({"datatype": datatype, "training_parameters": training})
    assert "http://" not in combined
    assert "https://" not in combined


def test_vendor_prefixed_hardware_names_resolve():
    mod = load_filter_module()

    result = mod.query_hardware_features("NVIDIA GeForce RTX 5090", "training_parameters")
    by_key = mod.query_hardware_features("nvidia.blackwell.geforce_rtx_5090.spec", "training_parameters")

    assert result["found"] is True
    assert result["gpu_name"] == "GeForce RTX 5090"
    assert "node_id" not in result
    assert "source_urls" not in json.dumps(result)
    assert "http://" not in json.dumps(result)
    assert "https://" not in json.dumps(result)
    assert "soap_optimizer" in {feature["feature_id"] for feature in result["features"]}
    assert by_key["found"] is True
    assert "soap_optimizer" in {feature["feature_id"] for feature in by_key["features"]}


def test_unconfirmed_optimizers_are_experimental_for_all_hardware():
    graph_path = Path(__file__).resolve().parents[2] / "schema" / "hardware_knowledge_graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    optimizer_ids = {"soap_optimizer", "ademamix_optimizer"}

    hardware_by_node_id = {
        node["id"]: node
        for node in graph["nodes"]
        if node.get("label") == "Hardware"
    }
    optimizer_edges_by_hardware: dict[str, set[str]] = {}
    for edge in graph["edges"]:
        feature_id = str(edge.get("to") or "").split(":", 1)[-1]
        if edge.get("type") != "HAS_FEATURE" or feature_id not in optimizer_ids:
            continue
        props = edge.get("properties") or {}
        assert props.get("support_level") == "experimental"
        assert props.get("recommended") is False
        assert props.get("verified") is False
        assert props.get("recommendation_scope") == "unconfirmed_software_optimizer_candidate"
        assert "not widely confirmed" in props.get("limitations", "")
        optimizer_edges_by_hardware.setdefault(edge["from"], set()).add(feature_id)

    for node_id, node in hardware_by_node_id.items():
        props = node.get("properties") or {}
        recipes = set(props.get("recipes") or [])
        experimental = set(props.get("experimental_recipes") or [])
        assert not (recipes & optimizer_ids)
        if node_id in optimizer_edges_by_hardware:
            assert optimizer_edges_by_hardware[node_id] <= experimental


def main():
    mod = load_filter_module()

    which = input("Which filter? (1 / 2): ").strip()
    gpu_name = input("Enter GPU name (e.g. rtx 4090): ").strip() or "rtx 4090"
    stage = input("Pipeline stage (model_structure / datatype / training_parameters / all): ").strip()
    if not stage or stage == "all":
        stage = None

    if which == "2":
        result = mod.query_hardware_features(gpu_name, stage)
        label = f'query_hardware_features("{gpu_name}", "{stage or "None"}")'
    else:
        result = mod.query_hardware_node(gpu_name, stage)
        label = f'query_hardware_node("{gpu_name}", "{stage or "None"}")'

    print(f"\n{'=' * 50}")
    print(label)
    print(f"{'=' * 50}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
