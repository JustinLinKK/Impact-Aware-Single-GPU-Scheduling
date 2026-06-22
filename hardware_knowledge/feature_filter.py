"""Filter1 & Filter2: query Hardware and Feature nodes from the knowledge graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pipeline stage keyword definitions
# ---------------------------------------------------------------------------

PIPELINE_STAGES = ("datatype", "model", "optimizer", "tuning")

_STAGE_ALIASES = {
    "data_type": "datatype",
    "model_structure": "model",
    "training": "tuning",
    "training_parameters": "tuning",
    "training_params": "tuning",
    "precision": "tuning",
}

_STAGE_KEYWORDS: dict[str, list[str]] = {
    "datatype": [
        "data", "dataset", "dataloader", "modality", "shape",
        "sequence", "packing", "bucket", "decomposition", "decode",
        "image", "video", "wsi", "token", "tiled", "chunked",
        "channels_last",
    ],
    "model": [
        "attention", "flash-attn", "sdpa", "channels_last", "cnn", "unet",
        "transformer", "activation checkpointing", "sequence packing",
        "kv-cache", "tiled", "chunked", "image", "video", "wsi",
        "model", "architecture", "layer", "kernel", "extension", "sm_",
        "mig", "topology", "nvlink",
    ],
    "optimizer": [
        "optimizer", "adamw", "muon", "soap", "ademamix", "loss",
        "cross-entropy", "cross entropy", "learning rate", "lr",
        "weight_decay", "momentum", "ns_steps", "nesterov",
        "adjust_lr_fn", "scheduler",
    ],
    "tuning": [
        "bf16", "fp16", "fp8", "fp4", "fp64", "tf32", "int8",
        "precision", "autocast", "mixed precision", "quantiz",
        "mxfp4", "mxfp8", "nvfp4", "gradscaler",
        "transformer engine", "tensor core",
        "batch_size", "batch size", "batch", "grad_accum",
        "gradient accumulation", "epoch",
        "vram", "memory budget", "memory plan",
        "throughput", "configurable",
    ],
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
        ``"datatype"`` / ``"model"`` / ``"optimizer"`` / ``"tuning"``
        or ``None`` for all patterns. Legacy stage names are accepted.
    """
    graph = _load_graph(graph_path)
    node, _edges = _lookup_node(graph, hardware_name)
    if node is None:
        return {"found": False, "hardware_name": hardware_name}

    props = node.get("properties", node)
    raw_cap = _first_list_entry(props.get("compute_capabilities"))

    recommended = _as_str_list(props.get("recommended_patterns"))
    avoid = _as_str_list(props.get("avoid_patterns"))
    stage = _normalize_stage(agent_stage)

    if stage:
        keywords = _STAGE_KEYWORDS.get(stage, [])
        recommended = [p for p in recommended if any(k in p.lower() for k in keywords)]
        avoid = [p for p in avoid if any(k in p.lower() for k in keywords)]

    return _compact_dict({
        "found": True,
        "node_id": node["id"],
        "gpu_name": _as_str(props.get("name")),
        "architecture": _as_str(
            _first_list_entry(props.get("architectures"))
            or props.get("architecture")
        ),
        "vram_MB": _as_int(props.get("vram_MB")),
        "datatypes": _as_str_list(
            props.get("datatypes")
            or props.get("supported_precisions")
            or props.get("precisions")
        ),
        "software_features": _as_str_list(props.get("software_features")),
        "recipes": _as_str_list(props.get("recipes")),
        "experimental_recipes": _as_str_list(props.get("experimental_recipes")),
        "sm_count": _as_int(props.get("sm_count")),
        "compute_capability": raw_cap,
        "stage_filter": stage,
        "recommended_patterns": recommended,
        "avoid_patterns": avoid,
    })


# ---------------------------------------------------------------------------
# Filter2: list feature details for a hardware node
# ---------------------------------------------------------------------------

_STAGE_CATEGORIES: dict[str, set[str]] = {
    "datatype": {"data_pipeline"},
    "model": {"compute_capability", "interconnect", "kernel_optimization", "tensor_core"},
    "optimizer": {"optimizer"},
    "tuning": {"data_pipeline", "interconnect", "kernel_optimization", "parallelism", "precision"},
}

_STAGE_FEATURE_IDS: dict[str, set[str]] = {
    "optimizer": {"gram_newton_schulz_symmetric_gemm"},
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
        ``"datatype"`` returns data-pipeline/data-shape features;
        ``"model"`` returns architecture/kernel/tensor-core features;
        ``"optimizer"`` returns optimizer features and optimizer-adjacent kernels;
        ``"tuning"`` returns precision, dataloader, parallelism, and runtime features;
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

    stage = _normalize_stage(agent_stage)
    categories = _STAGE_CATEGORIES.get(stage) if stage else None
    feature_ids = _STAGE_FEATURE_IDS.get(stage, set()) if stage else set()

    features: list[dict[str, Any]] = []
    for edge in edges:
        feat_id = edge.get("to", "")
        feat_props = feat_nodes.get(feat_id)
        if feat_props is None:
            continue

        category = feat_props.get("category", "")
        feature_id = feat_props.get("feature_id")
        category_match = bool(categories and category in categories)
        feature_match = feature_id in feature_ids
        if (categories or feature_ids) and not (category_match or feature_match):
            continue

        edge_props = edge.get("properties", {})
        features.append(_compact_dict({
            "feature_id": feature_id,
            "name": _feature_name(feat_props),
            "category": category,
            "description": feat_props.get("description"),
            "example_code": feat_props.get("example_code"),
            "api_symbols": feat_props.get("api_symbols", []),
            "source_urls": _source_urls(feat_props),
            "usage": feat_props.get("usage"),
            "recommended_patterns": feat_props.get("recommended_patterns", []),
            "avoid_patterns": feat_props.get("avoid_patterns", []),
            "support_level": edge_props.get("support_level"),
            "recommended": edge_props.get("recommended"),
            "verified": edge_props.get("verified"),
            "performance_impact": edge_props.get("performance_impact"),
            "min_compute_capability": edge_props.get("min_compute_capability"),
            "recommendation_scope": (
                edge_props.get("recommendation_scope")
                or feat_props.get("default_recommendation_scope")
            ),
            "limitations": edge_props.get("limitations") or feat_props.get("limitations"),
            "notes": edge_props.get("notes") or feat_props.get("notes"),
        }))

    return {
        "found": True,
        "node_id": node["id"],
        "gpu_name": node.get("properties", {}).get("name"),
        "stage_filter": stage,
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


def _normalize_stage(agent_stage: str | None) -> str | None:
    if agent_stage is None:
        return None
    stage = str(agent_stage).strip().lower()
    if not stage or stage == "all":
        return None
    return _STAGE_ALIASES.get(stage, stage)


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


def _source_urls(feat_props: dict[str, Any]) -> list[str]:
    urls = _as_str_list(feat_props.get("source_urls"))
    for url in _as_str_list(feat_props.get("source_url")):
        if url not in urls:
            urls.append(url)
    return urls


def _feature_name(feat_props: dict[str, Any]) -> str | None:
    feature_id = _as_str(feat_props.get("feature_id"))
    name = _as_str(feat_props.get("name"))
    if name == feature_id:
        return None
    return name


def _compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if value is not None and value != "" and value != [] and value != {}
    }


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
