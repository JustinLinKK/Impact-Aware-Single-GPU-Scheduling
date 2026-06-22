"""Test Filter1 & Filter2 with real graph DB data."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path


def load_filter_module():
    spec = importlib.util.spec_from_file_location(
        "feature_filter",
        str(Path(__file__).resolve().parents[1] / "hardware_knowledge" / "feature_filter.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pipeline_stage_categories_align_with_contract():
    mod = load_filter_module()

    assert mod.PIPELINE_STAGES == ("datatype", "model", "optimizer", "tuning")

    datatype = mod.query_hardware_features("GeForce RTX 5090", "datatype")
    datatype_ids = {feature["feature_id"] for feature in datatype["features"]}
    assert datatype["stage_filter"] == "datatype"
    assert "dataset_decomposition" in datatype_ids
    assert "nvimagecodec_gpu_decode" in datatype_ids
    assert "bf16" not in datatype_ids

    tuning = mod.query_hardware_features("GeForce RTX 5090", "tuning")
    tuning_ids = {feature["feature_id"] for feature in tuning["features"]}
    assert "bf16" in tuning_ids
    assert "fp8_rowwise_scaling" in tuning_ids
    assert "async_tensor_parallel" in tuning_ids


def test_legacy_stage_aliases_still_work():
    mod = load_filter_module()

    model = mod.query_hardware_features("GeForce RTX 5090", "model")
    legacy_model = mod.query_hardware_features("GeForce RTX 5090", "model_structure")
    assert legacy_model["stage_filter"] == "model"
    assert {f["feature_id"] for f in legacy_model["features"]} == {
        f["feature_id"] for f in model["features"]
    }

    tuning = mod.query_hardware_features("GeForce RTX 5090", "training_parameters")
    assert tuning["stage_filter"] == "tuning"
    assert "bf16" in {feature["feature_id"] for feature in tuning["features"]}


def test_5090_optimizer_filter_marks_unconfirmed_candidates():
    mod = load_filter_module()

    result = mod.query_hardware_features("GeForce RTX 5090", "optimizer")
    by_id = {feature["feature_id"]: feature for feature in result["features"]}

    assert result["stage_filter"] == "optimizer"
    assert "muon_optimizer" in by_id
    assert "gram_newton_schulz_symmetric_gemm" in by_id
    assert by_id["soap_optimizer"]["support_level"] == "experimental"
    assert by_id["ademamix_optimizer"]["support_level"] == "experimental"
    assert by_id["soap_optimizer"]["recommended"] is False
    assert by_id["ademamix_optimizer"]["recommended"] is False
    assert "not widely confirmed" in by_id["soap_optimizer"]["limitations"]
    assert "not widely confirmed" in by_id["ademamix_optimizer"]["limitations"]


def test_5090_node_exposes_direct_feature_lists():
    mod = load_filter_module()

    result = mod.query_hardware_node("GeForce RTX 5090", "optimizer")

    assert result["stage_filter"] == "optimizer"
    assert "fp8" in result["datatypes"]
    assert "muon_optimizer" in result["recipes"]
    assert "soap_optimizer" in result["experimental_recipes"]
    assert "ademamix_optimizer" in result["experimental_recipes"]


def test_vendor_prefixed_hardware_names_resolve():
    mod = load_filter_module()

    result = mod.query_hardware_features("NVIDIA GeForce RTX 5090", "optimizer")
    by_key = mod.query_hardware_features("nvidia.blackwell.geforce_rtx_5090.spec", "optimizer")

    assert result["found"] is True
    assert result["gpu_name"] == "GeForce RTX 5090"
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
    stage = input("Pipeline stage (datatype / model / optimizer / tuning / all): ").strip()
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
