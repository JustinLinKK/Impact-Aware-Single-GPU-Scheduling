"""Filter1 & Filter2: query Hardware and Feature nodes from the knowledge graph."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pipeline stage keyword definitions
# ---------------------------------------------------------------------------

PIPELINE_STAGES = ("model_design", "datatype_precision", "training_evaluation")
LEGACY_HARDWARE_STAGES = ("datatype", "model", "optimizer", "tuning")

_STAGE_ALIASES = {
    "data_type": "datatype",
    "data_processing": "datatype",
    "data_processing_and_feature_engineering": "datatype",
    "feature_engineering": "datatype",
    "stage1": "model_design",
    "stage_1": "model_design",
    "stage1_candidate_construction": "model_design",
    "candidate_construction": "model_design",
    "model-design": "model_design",
    "model_design": "model_design",
    "model_structure": "model",
    "datatype-precision": "datatype_precision",
    "datatype_precision": "datatype_precision",
    "datatype_quantization": "datatype_precision",
    "datatype-quantization": "datatype_precision",
    "quantization": "datatype_precision",
    "training": "training_evaluation",
    "training-evaluation": "training_evaluation",
    "training_evaluation": "training_evaluation",
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
    node, edges = _lookup_node(graph, hardware_name)
    if node is None:
        return {"found": False, "hardware_name": hardware_name}

    props = node.get("properties", node)
    raw_cap = _first_list_entry(props.get("compute_capabilities"))
    stage = _normalize_stage(agent_stage)

    feature_index = _stage_feature_key_index(graph, edges, stage)

    return _compact_dict({
        "found": True,
        "gpu_name": _as_str(props.get("name")),
        "architecture": _as_str(
            _first_list_entry(props.get("architectures"))
            or props.get("architecture")
        ),
        "vram_MB": _as_int(props.get("vram_MB")),
        "sm_count": _as_int(props.get("sm_count")),
        "compute_capability": raw_cap,
        "stage_filter": stage,
        **feature_index,
        "recommended_patterns": props.get("recommended_patterns", []),
        "avoid_patterns": props.get("avoid_patterns", []),
        "feature_query": {
            "tool": "query_hardware_features",
            "hardware_name": _as_str(props.get("name")) or hardware_name,
            "stage": stage or "all",
        },
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

_COMPOSITE_STAGE_SPECS: dict[str, tuple[tuple[str, set[str], set[str]], ...]] = {
    "model_design": (
        ("datatype", _STAGE_CATEGORIES["datatype"], _STAGE_FEATURE_IDS.get("datatype", set())),
        ("model", _STAGE_CATEGORIES["model"], _STAGE_FEATURE_IDS.get("model", set())),
    ),
    "datatype_precision": (
        ("tuning", {"precision"}, set()),
    ),
    "training_evaluation": (
        ("optimizer", _STAGE_CATEGORIES["optimizer"], _STAGE_FEATURE_IDS.get("optimizer", set())),
        ("tuning", {"interconnect", "kernel_optimization", "parallelism"}, set()),
    ),
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
    stage_specs = _stage_specs(stage)

    features: list[dict[str, Any]] = []
    for edge in edges:
        feat_id = edge.get("to", "")
        feat_props = feat_nodes.get(feat_id)
        if feat_props is None:
            continue

        category = feat_props.get("category", "")
        feature_id = feat_props.get("feature_id")
        if stage_specs and not _feature_matches_stage(str(category), str(feature_id or ""), stage_specs):
            continue

        edge_props = edge.get("properties", {})
        features.append(_compact_dict({
            "feature_id": feature_id,
            "name": _feature_name(feat_props),
            "category": category,
            "description": feat_props.get("description"),
            "example_code": feat_props.get("example_code"),
            "api_symbols": feat_props.get("api_symbols", []),
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
            "notes": _merge_text_values(feat_props.get("notes"), edge_props.get("notes")),
        }))

    return {
        "found": True,
        "gpu_name": _sanitize_public_value(node.get("properties", {}).get("name")),
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


def _stage_specs(stage: str | None) -> tuple[tuple[str, set[str], set[str]], ...]:
    if not stage:
        return ()
    if stage in _COMPOSITE_STAGE_SPECS:
        return _COMPOSITE_STAGE_SPECS[stage]
    if stage in LEGACY_HARDWARE_STAGES:
        return ((stage, _STAGE_CATEGORIES.get(stage, set()), _STAGE_FEATURE_IDS.get(stage, set())),)
    return ()


def _feature_matches_stage(
    category: str,
    feature_id: str,
    specs: tuple[tuple[str, set[str], set[str]], ...],
) -> bool:
    for _component, categories, explicit_ids in specs:
        if category in categories or feature_id in explicit_ids:
            return True
    return False


def _stage_feature_key_index(
    graph: dict[str, Any],
    edges: list[dict[str, Any]],
    stage: str | None,
) -> dict[str, Any]:
    feat_nodes = {
        n["id"]: n.get("properties", {})
        for n in graph["nodes"] if n.get("label") == "Feature"
    }
    stage_specs = _stage_specs(stage)
    all_keys: list[list[str]] = []
    recommended_keys: list[list[str]] = []
    not_recommended_keys: list[list[str]] = []
    conditional_keys: list[list[str]] = []

    for edge in edges:
        feat_props = feat_nodes.get(edge.get("to", ""))
        if not feat_props:
            continue
        feature_id = _as_str(feat_props.get("feature_id"))
        if not feature_id:
            continue
        category = str(feat_props.get("category") or "")
        if stage_specs and not _feature_matches_stage(category, feature_id, stage_specs):
            continue
        edge_props = edge.get("properties") or {}
        description = _feature_short_description(feature_id, feat_props)
        _append_unique_feature_pair(all_keys, feature_id, description)
        if edge_props.get("recommended") is False:
            _append_unique_feature_pair(not_recommended_keys, feature_id, description)
        elif edge_props.get("recommended") is True:
            _append_unique_feature_pair(recommended_keys, feature_id, description)
        support_level = str(edge_props.get("support_level") or "").lower()
        if (
            edge_props.get("recommended") is False
            or support_level in {"experimental", "limited"}
            or bool(edge_props.get("limitations"))
            or bool(edge_props.get("notes"))
            or bool(feat_props.get("limitations"))
            or bool(feat_props.get("notes"))
        ):
            _append_unique_feature_pair(conditional_keys, feature_id, description)

    return {
        "stage_feature_keys": all_keys,
        "recommended_feature_keys": recommended_keys,
        "not_recommended_feature_keys": not_recommended_keys,
        "conditional_feature_keys": conditional_keys,
    }


def _merge_text_values(*values: Any) -> str:
    merged: list[str] = []
    for value in values:
        text = _strip_urls(str(value or "").strip())
        if text and text not in merged:
            merged.append(text)
    return "; ".join(merged)


def _feature_short_description(feature_id: str, feat_props: dict[str, Any]) -> str:
    capability = _feature_capability_phrase(feature_id, feat_props)
    usage = _feature_main_use_phrase(feature_id, feat_props)
    sentence = f"{capability}, mainly used for {usage}."
    sentence = re.sub(r"\s+", " ", _strip_urls(sentence)).strip()
    return sentence.replace("..", ".")


def _feature_capability_phrase(feature_id: str, feat_props: dict[str, Any]) -> str:
    feature_name = _display_feature_name(feature_id, feat_props)
    category = str(feat_props.get("category") or "").strip().lower()
    description = _clean_feature_text(feat_props.get("description"))

    special = {
        "cut_cross_entropy": "Cut Cross-Entropy fuses the classifier projection with cross-entropy loss to avoid materializing large logits",
        "gram_newton_schulz_symmetric_gemm": "Gram Newton-Schulz symmetric GEMM optimizes the symmetric-product kernel used by Newton-Schulz updates",
        "tensor_cores": "Tensor Cores accelerate mixed-precision matrix and convolution operations",
        "tensor_cores_3gen": "Third-generation Tensor Cores accelerate Ampere mixed-precision matrix operations",
        "tensor_cores_4gen": "Fourth-generation Tensor Cores accelerate Ada/Hopper mixed-precision matrix operations",
        "tensor_cores_5gen": "Fifth-generation Tensor Cores accelerate Blackwell mixed-precision and low-precision matrix operations",
        "nvimagecodec_gpu_decode": "nvImageCodec GPU decode moves image decoding directly onto the GPU",
        "dataset_decomposition": "Dataset decomposition buckets training data into curriculum-friendly chunks",
        "async_tensor_parallel": "Asynchronous tensor parallelism overlaps distributed collectives with compute",
        "fp8_all_gather_fsdp2": "FP8 all-gather for FSDP2 reduces distributed parameter communication volume",
    }
    if feature_id in special:
        return special[feature_id]

    if category == "compute_capability":
        return f"{feature_id} identifies the GPU compute capability and kernel target"
    if category == "precision":
        return f"{feature_name} defines a numeric precision or scaling policy"
    if category == "interconnect":
        return f"{feature_name} describes the host-to-GPU interconnect capability"
    if category == "optimizer":
        clause = _description_action_clause(description)
        return clause or f"{feature_name} describes a training optimizer candidate"
    if category == "parallelism":
        clause = _description_action_clause(description)
        return clause or f"{feature_name} describes a distributed training parallelism feature"
    if category == "data_pipeline":
        clause = _description_action_clause(description)
        return clause or f"{feature_name} describes a data loading or preprocessing feature"
    if category == "kernel_optimization":
        clause = _description_action_clause(description)
        return clause or f"{feature_name} describes a kernel-level optimization"
    if category == "tensor_core":
        return f"{feature_name} describes Tensor Core acceleration support"
    return f"{feature_name} describes a hardware feature"


def _feature_main_use_phrase(feature_id: str, feat_props: dict[str, Any]) -> str:
    category = str(feat_props.get("category") or "").strip().lower()
    description = _clean_feature_text(feat_props.get("description")).lower()
    usage = _clean_feature_text(feat_props.get("usage")).lower()
    scope = _clean_feature_text(feat_props.get("default_recommendation_scope")).lower()

    if feature_id == "cut_cross_entropy" or "large-vocab" in description or "large vocabulary" in description:
        return "large-vocabulary language-model heads"
    if feature_id == "gram_newton_schulz_symmetric_gemm" or "muon_newton_schulz" in scope:
        return "Muon or Newton-Schulz optimizer-step hotspots"
    if feature_id == "muon_optimizer" or "matmul_heavy_training" in scope:
        return "matmul-heavy neural-network training"
    if feature_id in {"soap_optimizer", "ademamix_optimizer"} or "candidate_optimizer" in scope:
        return "optimizer experiments that are benchmarked against a stable AdamW baseline"
    if feature_id == "nvimagecodec_gpu_decode" or "cv/imaging" in description:
        return "computer-vision and imaging data loaders"
    if feature_id == "dataset_decomposition":
        return "long-context or variable-length dataset pipelines"
    if feature_id in {"async_tensor_parallel", "fp8_all_gather_fsdp2"} or "multi-gpu" in description or "distributed" in usage:
        return "multi-GPU distributed training"

    category_usage = {
        "compute_capability": "hardware capability checks and CUDA kernel targeting",
        "data_pipeline": "data loading and preprocessing pipelines",
        "interconnect": "host-to-GPU transfer and placement planning",
        "kernel_optimization": "kernel-level training or inference bottlenecks",
        "optimizer": "training optimizer selection",
        "parallelism": "distributed or parallel training",
        "precision": "datatype and mixed-precision policy",
        "tensor_core": "mixed-precision tensor-core workloads",
    }
    return category_usage.get(category, "hardware-aware optimization decisions")


def _description_action_clause(description: str) -> str | None:
    if not description:
        return None
    text = re.sub(r"\([^)]*\)", "", description)
    if ":" in text:
        subject, rest = text.split(":", 1)
        subject = subject.strip()
        rest = rest.strip()
        clause = re.split(r"[.;]", rest, maxsplit=1)[0].strip()
        if clause and len(clause) <= 120:
            if re.match(r"^(fuses|decodes|uses|replaces|reduces|overlaps|decomposes|accelerates)\b", clause, re.I):
                return f"{subject} {clause}"
    sentence = re.split(r"[.;]", text, maxsplit=1)[0].strip()
    if sentence and len(sentence) <= 120 and not sentence.lower().startswith(("cuda c++ programming guide", "nvidia blackwell architecture")):
        return sentence
    return None


def _display_feature_name(feature_id: str, feat_props: dict[str, Any]) -> str:
    name = _as_str(feat_props.get("name"))
    if name and name != feature_id:
        return name
    return _humanize_feature_key(feature_id).rstrip(".")


def _clean_feature_text(value: Any) -> str:
    return re.sub(r"\s+", " ", _strip_urls(str(value or "").replace("\n", " "))).strip()


def _humanize_feature_key(feature_id: str) -> str:
    words = [word for word in str(feature_id or "").replace("-", "_").split("_") if word]
    if not words:
        return "Hardware feature."
    text = " ".join(word.upper() if word.lower() in {"fp8", "fp4", "fp16", "bf16", "tf32", "int8"} else word for word in words)
    return text[:1].upper() + text[1:] + "."


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _append_unique_feature_pair(values: list[list[str]], feature_id: str, description: str) -> None:
    if not feature_id:
        return
    if any(item and item[0] == feature_id for item in values):
        return
    values.append([feature_id, description or _humanize_feature_key(feature_id)])


def _lookup_node(
    graph: dict[str, Any], hardware_name: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    query = hardware_name.lower()
    for node in graph["nodes"]:
        if node["label"] != "Hardware":
            continue
        props = node.get("properties", {})
        searchable = [
            node.get("id", ""),
            props.get("hardware_id", ""),
            props.get("name", ""),
            props.get("name_key", ""),
        ] + list(props.get("aliases") or [])
        if any(query in s.lower() or s.lower() in query for s in searchable if s):
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


def _feature_name(feat_props: dict[str, Any]) -> str | None:
    feature_id = _as_str(feat_props.get("feature_id"))
    name = _as_str(feat_props.get("name"))
    if name == feature_id:
        return None
    return name


def _compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        cleaned = _sanitize_public_value(value)
        if cleaned not in (None, "", [], {}):
            result[key] = cleaned
    return result


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sanitize_public_value(value: Any) -> Any:
    if isinstance(value, str):
        return _strip_urls(value)
    if isinstance(value, list):
        cleaned = [_sanitize_public_value(item) for item in value]
        return [item for item in cleaned if item not in (None, "", [], {})]
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if key not in {"source_url", "source_urls"}
            for cleaned in [_sanitize_public_value(item)]
            if cleaned not in (None, "", [], {})
        }
    return value


def _strip_urls(value: str) -> str:
    text = re.sub(r"\s*\[[^\]]*https?://[^\]]+\]", "", str(value or ""))
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()
