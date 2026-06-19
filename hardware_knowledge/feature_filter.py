"""Filter1 & Filter2: query Hardware and Feature nodes from the knowledge graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pipeline stage keyword definitions
# ---------------------------------------------------------------------------

PIPELINE_STAGES = ("model_structure", "datatype", "training_parameters")

_STAGE_KEYWORDS: dict[str, list[str]] = {
    "model_structure": [
        "attention", "flash-attn", "sdpa", "channels_last", "cnn", "unet",
        "transformer", "activation checkpointing", "sequence packing",
        "kv-cache", "tiled", "chunked", "image", "video", "wsi",
        "model", "architecture", "layer", "kernel", "extension", "sm_",
        "mig", "topology", "nvlink",
    ],
    "datatype": [
        "bf16", "fp16", "fp8", "fp4", "fp64", "tf32", "int8",
        "precision", "autocast", "mixed precision", "quantiz",
        "mxfp4", "mxfp8", "nvfp4", "gradscaler",
        "transformer engine", "tensor core",
    ],
    "training_parameters": [
        "batch_size", "batch size", "batch", "grad_accum",
        "gradient accumulation", "learning rate", "lr", "epoch",
        "optimizer", "adamw", "muon", "weight_decay", "momentum",
        "ns_steps", "nesterov", "adjust_lr_fn", "scheduler",
        "vram", "memory budget", "memory plan",
        "throughput", "configurable",
    ],
}

_PRECISION_FEATURE_IDS = {
    "fp32", "tf32", "fp16", "bf16", "fp64", "fp8", "fp8_e4m3", "fp8_e5m2",
    "fp4", "int8",
}

_PRECISION_DISPLAY = {
    "fp32": "FP32", "tf32": "TF32", "fp16": "FP16", "bf16": "BF16",
    "fp64": "FP64", "fp8": "FP8", "fp8_e4m3": "FP8_E4M3",
    "fp8_e5m2": "FP8_E5M2", "fp4": "FP4", "int8": "INT8",
}

_DEFAULT_GRAPH_PATH = Path(__file__).resolve().parents[2] / "schema" / "hardware_knowledge_graph.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_hardware_node(
    hardware_name: str,
    agent_stage: str | None = None,
    *,
    graph_path: str | Path | None = None,
) -> dict[str, Any]:
    """Query a hardware node by GPU name, filtered by pipeline stage.

    Parameters
    ----------
    hardware_name:
        GPU name to search, e.g. ``"rtx 4090"``.
    agent_stage:
        ``"model_structure"`` / ``"datatype"`` / ``"training_parameters"``
        or ``None`` for all patterns.
    """
    graph = _load_graph(graph_path)
    node, edges = _lookup_node(graph, hardware_name)
    if node is None:
        return {"found": False, "hardware_name": hardware_name}

    props = node.get("properties", node)
    raw_cap = _first_list_entry(props.get("compute_capabilities"))

    precisions = _precisions_from_edges(edges) if edges else []
    recommended = _as_str_list(props.get("recommended_patterns"))
    avoid = _as_str_list(props.get("avoid_patterns"))

    if agent_stage:
        keywords = _STAGE_KEYWORDS.get(agent_stage, [])
        recommended = [p for p in recommended if any(k in p.lower() for k in keywords)]
        avoid = [p for p in avoid if any(k in p.lower() for k in keywords)]

    return {
        "found": True,
        "node_id": node["id"],
        "gpu_name": _as_str(props.get("name")),
        "architecture": _as_str(
            _first_list_entry(props.get("architectures"))
            or props.get("architecture")
        ),
        "vram_MB": _as_int(props.get("vram_MB")),
        "supported_precisions": precisions,
        "sm_count": _as_int(props.get("sm_count")),
        "compute_capability": raw_cap,
        "recommended_patterns": recommended,
        "avoid_patterns": avoid,
    }


# ---------------------------------------------------------------------------
# Filter2: list feature details for a hardware node
# ---------------------------------------------------------------------------

_STAGE_CATEGORIES: dict[str, list[str]] = {
    "datatype": ["precision"],
    "model_structure": ["kernel", "tensor_core", "interconnect", "compute_capability"],
    "training_parameters": ["precision", "kernel"],
}


def query_hardware_features(
    hardware_name: str,
    agent_stage: str | None = None,
    *,
    graph_path: str | Path | None = None,
) -> dict[str, Any]:
    """List all Feature node details for a hardware, optionally filtered by stage.

    Parameters
    ----------
    hardware_name:
        GPU name to search, e.g. ``"rtx 4090"``.
    agent_stage:
        ``"datatype"`` returns only precision features;
        ``"model_structure"`` returns kernel/tensor_core/interconnect features;
        ``None`` returns all features.
    """
    graph = _load_graph(graph_path)
    node, edges = _lookup_node(graph, hardware_name)
    if node is None:
        return {"found": False, "hardware_name": hardware_name, "features": []}

    feat_nodes = {
        n["id"]: n["properties"]
        for n in graph["nodes"] if n["label"] == "Feature"
    }

    categories = _STAGE_CATEGORIES.get(agent_stage) if agent_stage else None

    features: list[dict[str, Any]] = []
    for edge in edges:
        feat_id = edge.get("to", "")
        feat_props = feat_nodes.get(feat_id)
        if feat_props is None:
            continue

        category = feat_props.get("category", "")
        if categories and category not in categories:
            continue

        edge_props = edge.get("properties", {})
        features.append({
            "feature_id": feat_props.get("feature_id"),
            "name": feat_props.get("name"),
            "category": category,
            "description": feat_props.get("description"),
            "example_code": feat_props.get("example_code"),
            "api_symbols": feat_props.get("api_symbols", []),
            "source_url": feat_props.get("source_url"),
            "support_level": edge_props.get("support_level"),
            "min_compute_capability": edge_props.get("min_compute_capability"),
        })

    return {
        "found": True,
        "node_id": node["id"],
        "gpu_name": node.get("properties", {}).get("name"),
        "stage_filter": agent_stage,
        "feature_count": len(features),
        "features": features,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_graph(graph_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(graph_path) if graph_path else _DEFAULT_GRAPH_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _lookup_node(
    graph: dict[str, Any], hardware_name: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    query = hardware_name.lower()
    for node in graph["nodes"]:
        if node["label"] != "Hardware":
            continue
        props = node.get("properties", {})
        searchable = [props.get("name", "")] + list(props.get("aliases") or [])
        if any(query in s.lower() for s in searchable):
            hw_id = node["id"]
            edges = [
                e for e in graph["edges"]
                if e.get("from") == hw_id and e.get("type") == "HAS_FEATURE"
            ]
            return node, edges
    return None, []


def _precisions_from_edges(edges: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for edge in edges:
        fid = edge.get("feature_id") or ""
        if not fid:
            to_field = str(edge.get("to") or "")
            fid = to_field.split(":")[-1] if ":" in to_field else to_field
        fid = fid.strip().lower()
        if fid in _PRECISION_FEATURE_IDS:
            display = _PRECISION_DISPLAY.get(fid, fid.upper())
            if display not in result:
                result.append(display)
    return result


def _first_list_entry(value: Any) -> str | None:
    if isinstance(value, list):
        return str(value[0]).strip() if value else None
    return str(value).strip() if value is not None else None


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
