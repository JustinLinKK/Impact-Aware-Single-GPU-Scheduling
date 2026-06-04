"""Hardware capability graph record loading and migration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

import yaml

from localml_scheduler.hardware_features.records import load_feature_records, validate_feature_record


HARDWARE_KNOWLEDGE_SCHEMA_VERSION = "hardware_knowledge_graph_v1"

HARDWARE_TYPES = {"GPU", "CPU", "FPGA", "TPU", "NPU", "ASIC", "ACCELERATOR"}
FEATURE_MATURITY = {"experimental", "beta", "production", "deprecated"}
SUPPORT_LEVELS = {"native", "optimized", "supported", "experimental", "limited"}

_PRECISION_FEATURES = {"fp32", "tf32", "fp16", "bf16", "fp64", "int8", "fp8", "fp8_e4m3", "fp8_e5m2", "fp4"}
_HIGH_IMPACT_FEATURES = _PRECISION_FEATURES | {"tensor_cores", "tensor_cores_3gen", "tensor_cores_4gen", "tensor_cores_5gen"}
_MEDIUM_IMPACT_FEATURES = {"cuda_graphs", "mps", "mig", "nvlink", "pcie5_x16", "amp"}
_NATIVE_FEATURES = _PRECISION_FEATURES | {
    "tensor_cores",
    "tensor_cores_3gen",
    "tensor_cores_4gen",
    "tensor_cores_5gen",
    "mig",
    "nvlink",
    "pcie5_x16",
}
_DISPLAY_NAMES = {
    "amp": "Automatic Mixed Precision",
    "bf16": "BF16 Mixed Precision Training",
    "fp16": "FP16 Mixed Precision",
    "fp32": "FP32 Training",
    "fp64": "FP64 Training",
    "tf32": "TF32 Matmul",
    "int8": "INT8 Quantization",
    "fp8": "FP8 Precision",
    "fp8_e4m3": "FP8 E4M3 Precision",
    "fp8_e5m2": "FP8 E5M2 Precision",
    "fp4": "FP4 Quantization",
    "tensor_cores": "Tensor Cores",
    "tensor_cores_3gen": "3rd-generation Tensor Cores",
    "tensor_cores_4gen": "4th-generation Tensor Cores",
    "tensor_cores_5gen": "5th-generation Tensor Cores",
    "cuda_graphs": "CUDA Graphs",
    "mps": "NVIDIA MPS",
    "mig": "MIG Partitioning",
    "nvlink": "NVLink",
    "pcie5_x16": "PCIe 5.0 x16",
}
_PRECISION_NAMES = {
    "fp32": "FP32",
    "tf32": "TF32",
    "fp16": "FP16",
    "bf16": "BF16",
    "fp64": "FP64",
    "int8": "INT8",
    "fp8": "FP8",
    "fp4": "FP4",
}
_STACK_NAMES = {
    "cuda": "CUDA",
    "cudnn": "cuDNN",
    "nccl": "NCCL",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "jax": "JAX",
    "rocm": "ROCm",
}


class HardwareKnowledgeRecordError(ValueError):
    """Raised when hardware graph records are invalid."""


def _as_string(value: Any) -> str:
    return str(value or "").strip()


def _as_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise HardwareKnowledgeRecordError(f"{field_name} must be a list")
    return [str(item).strip() for item in value if str(item).strip()]


def _canonical_name_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _display_stack(value: str) -> str:
    key = value.strip().lower()
    return _STACK_NAMES.get(key, value.strip())


def _feature_name(feature_id: str) -> str:
    return _DISPLAY_NAMES.get(feature_id, feature_id.replace("_", " ").title())


def _feature_category(feature_id: str) -> str:
    if feature_id in _PRECISION_FEATURES or feature_id == "amp":
        return "precision_optimization"
    if feature_id.startswith("tensor_cores") or feature_id.startswith("sm_"):
        return "compute_acceleration"
    if feature_id in {"nvlink", "pcie5_x16"}:
        return "interconnect"
    if feature_id in {"cuda_graphs", "mps", "mig"}:
        return "runtime_optimization"
    return "hardware_capability"


def _feature_how_to_use(feature_id: str) -> str:
    if feature_id == "bf16":
        return "Enable automatic mixed precision in the ML framework using BF16."
    if feature_id == "tf32":
        return "Enable TF32 matmul policy for FP32-heavy CUDA training workloads."
    if feature_id == "fp16":
        return "Use automatic mixed precision with FP16 and gradient scaling when numerically stable."
    if feature_id == "cuda_graphs":
        return "Capture stable training-step regions and replay them with CUDA Graphs when shapes are static."
    if feature_id == "mps":
        return "Use NVIDIA MPS when multiple CUDA processes should share one GPU."
    return f"Use {feature_id.replace('_', ' ')} when the framework and workload support it."


def _feature_when_to_use(feature_id: str) -> str:
    if feature_id in _PRECISION_FEATURES:
        return "Use when the hardware has native support and the model is stable under the selected precision."
    if feature_id.startswith("tensor_cores"):
        return "Use for tensor-core-friendly matrix multiply, convolution, and attention workloads."
    if feature_id in {"nvlink", "pcie5_x16"}:
        return "Use when data movement or multi-GPU communication affects throughput."
    return "Use when it matches the workload and framework runtime."


def _feature_when_not_to_use(feature_id: str) -> str:
    if feature_id in _PRECISION_FEATURES:
        return "Avoid when the hardware lacks efficient support or the model is numerically unstable."
    if feature_id == "cuda_graphs":
        return "Avoid when shapes or control flow change frequently across training steps."
    if feature_id == "mps":
        return "Avoid when exclusive GPU access is required for reliable measurement or isolation."
    return "Avoid when framework support is missing or the optimization does not fit the workload."


def _feature_sample_code(feature_id: str) -> str:
    if feature_id == "bf16":
        return "with torch.autocast(device_type='cuda', dtype=torch.bfloat16):\n    output = model(input)\n    loss = loss_fn(output, target)"
    if feature_id == "fp16":
        return "with torch.autocast(device_type='cuda', dtype=torch.float16):\n    output = model(input)"
    if feature_id == "tf32":
        return "torch.set_float32_matmul_precision('high')"
    if feature_id == "cuda_graphs":
        return "graph = torch.cuda.CUDAGraph()\nwith torch.cuda.graph(graph):\n    output = model(static_input)"
    return ""


def _max_verified_date(record: dict[str, Any]) -> str:
    dates = [
        str(ref.get("retrieved_or_verified_date") or "").strip()
        for ref in record.get("source_refs") or []
        if isinstance(ref, dict)
    ]
    dates = [value for value in dates if value]
    return max(dates) if dates else str(record.get("last_verified") or "").strip()


def _aliases_for_record(record: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    name = str(record["title"]).strip()
    if name.lower().startswith("nvidia "):
        aliases.append(name[7:])
    for accelerator in record.get("accelerator_names") or []:
        text = str(accelerator).strip()
        if not text:
            continue
        aliases.append(text)
        aliases.append(text.replace("_", " "))
    compact = re.sub(r"[^A-Za-z0-9]+", " ", name).strip()
    if compact:
        aliases.append(compact)
    seen: set[str] = set()
    deduped: list[str] = []
    for alias in aliases:
        key = alias.lower()
        if key and key not in seen and alias != name:
            seen.add(key)
            deduped.append(alias)
    return deduped


def validate_hardware_spec(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HardwareKnowledgeRecordError("HardwareSpec record must be an object")
    hardware_id = _as_string(payload.get("hardware_id"))
    name = _as_string(payload.get("name"))
    name_key = _canonical_name_key(payload.get("name_key") or name)
    hardware_type = _as_string(payload.get("hardware_type") or "GPU").upper()
    if not hardware_id:
        raise HardwareKnowledgeRecordError("hardware_id is required")
    if not name:
        raise HardwareKnowledgeRecordError("name is required")
    if not name_key:
        raise HardwareKnowledgeRecordError("name_key is required")
    if hardware_type not in HARDWARE_TYPES:
        raise HardwareKnowledgeRecordError(f"unsupported hardware_type: {hardware_type}")
    memory_gb = payload.get("memory_gb")
    if memory_gb is not None:
        try:
            memory_gb = float(memory_gb)
        except (TypeError, ValueError) as exc:
            raise HardwareKnowledgeRecordError("memory_gb must be a number") from exc
    return {
        "schema_version": HARDWARE_KNOWLEDGE_SCHEMA_VERSION,
        "hardware_id": hardware_id,
        "name": name,
        "name_key": name_key,
        "aliases": _as_string_list(payload.get("aliases"), field_name="aliases"),
        "vendor": _as_string(payload.get("vendor")),
        "hardware_type": hardware_type,
        "architecture": _as_string(payload.get("architecture")),
        "description": _as_string(payload.get("description")),
        "memory_gb": memory_gb,
        "memory_type": _as_string(payload.get("memory_type")),
        "supported_precisions": _as_string_list(payload.get("supported_precisions"), field_name="supported_precisions"),
        "software_stack": _as_string_list(payload.get("software_stack"), field_name="software_stack"),
        "created_at": _as_string(payload.get("created_at")),
        "updated_at": _as_string(payload.get("updated_at")),
    }


def validate_feature(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HardwareKnowledgeRecordError("Feature record must be an object")
    feature_id = _as_string(payload.get("feature_id"))
    name = _as_string(payload.get("name"))
    maturity = _as_string(payload.get("maturity") or "production").lower()
    if not feature_id:
        raise HardwareKnowledgeRecordError("feature_id is required")
    if not name:
        raise HardwareKnowledgeRecordError("name is required")
    if maturity not in FEATURE_MATURITY:
        raise HardwareKnowledgeRecordError(f"unsupported maturity: {maturity}")
    return {
        "schema_version": HARDWARE_KNOWLEDGE_SCHEMA_VERSION,
        "feature_id": feature_id,
        "name": name,
        "category": _as_string(payload.get("category") or "hardware_capability"),
        "description": _as_string(payload.get("description")),
        "how_to_use": _as_string(payload.get("how_to_use")),
        "when_to_use": _as_string(payload.get("when_to_use")),
        "when_not_to_use": _as_string(payload.get("when_not_to_use")),
        "sample_code": _as_string(payload.get("sample_code")),
        "frameworks": _as_string_list(payload.get("frameworks"), field_name="frameworks"),
        "maturity": maturity,
    }


def validate_has_feature(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HardwareKnowledgeRecordError("HAS_FEATURE record must be an object")
    hardware_id = _as_string(payload.get("hardware_id"))
    feature_id = _as_string(payload.get("feature_id"))
    support_level = _as_string(payload.get("support_level") or "supported").lower()
    if not hardware_id:
        raise HardwareKnowledgeRecordError("hardware_id is required")
    if not feature_id:
        raise HardwareKnowledgeRecordError("feature_id is required")
    if support_level not in SUPPORT_LEVELS:
        raise HardwareKnowledgeRecordError(f"unsupported support_level: {support_level}")
    return {
        "hardware_id": hardware_id,
        "feature_id": feature_id,
        "support_level": support_level,
        "recommended": bool(payload.get("recommended", False)),
        "performance_impact": _as_string(payload.get("performance_impact") or "low"),
        "min_driver_version": _as_string(payload.get("min_driver_version")),
        "min_framework_version": _as_string(payload.get("min_framework_version")),
        "software_requirements": _as_string_list(payload.get("software_requirements"), field_name="software_requirements"),
        "limitations": _as_string(payload.get("limitations")),
        "hardware_specific_how_to_use": _as_string(payload.get("hardware_specific_how_to_use")),
        "hardware_specific_sample_code": _as_string(payload.get("hardware_specific_sample_code")),
        "verified": bool(payload.get("verified", False)),
        "last_verified_at": _as_string(payload.get("last_verified_at")),
    }


def load_feature_ontology(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return {}
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    keys = payload.get("keys") if isinstance(payload, dict) else {}
    return dict(keys or {}) if isinstance(keys, dict) else {}


def feature_from_key(feature_id: str, *, ontology: dict[str, dict[str, Any]] | None = None, frameworks: list[str] | None = None) -> dict[str, Any]:
    ontology_payload = dict((ontology or {}).get(feature_id) or {})
    description = _as_string(ontology_payload.get("description"))
    return validate_feature(
        {
            "feature_id": feature_id,
            "name": _as_string(ontology_payload.get("name")) or _feature_name(feature_id),
            "category": _as_string(ontology_payload.get("category")) or _feature_category(feature_id),
            "description": description or f"{_feature_name(feature_id)} capability.",
            "how_to_use": _as_string(ontology_payload.get("how_to_use")) or _feature_how_to_use(feature_id),
            "when_to_use": _as_string(ontology_payload.get("when_to_use")) or _feature_when_to_use(feature_id),
            "when_not_to_use": _as_string(ontology_payload.get("when_not_to_use")) or _feature_when_not_to_use(feature_id),
            "sample_code": _as_string(ontology_payload.get("sample_code")) or _feature_sample_code(feature_id),
            "frameworks": list(frameworks or ["PyTorch", "TensorFlow", "JAX"]),
            "maturity": _as_string(ontology_payload.get("maturity")) or "production",
        }
    )


def _recommended_for_feature(feature_id: str, patterns: list[str]) -> bool:
    aliases = {feature_id.lower(), feature_id.lower().replace("_", " ")}
    if feature_id.startswith("tensor_cores"):
        aliases.add("tensor core")
    if feature_id == "amp":
        aliases.add("autocast")
        aliases.add("mixed precision")
    for pattern in patterns:
        lowered = pattern.lower()
        if any(alias and alias in lowered for alias in aliases):
            return True
    return False


def _relationship_defaults(record: dict[str, Any], feature_id: str, feature: dict[str, Any]) -> dict[str, Any]:
    recommended_patterns = list(record.get("recommended_patterns") or [])
    avoid_patterns = list(record.get("avoid_patterns") or [])
    matching_recommendation = next((pattern for pattern in recommended_patterns if _recommended_for_feature(feature_id, [pattern])), "")
    matching_limitation = next((pattern for pattern in avoid_patterns if _recommended_for_feature(feature_id, [pattern])), "")
    impact = "high" if feature_id in _HIGH_IMPACT_FEATURES or feature_id.startswith("sm_") else "medium" if feature_id in _MEDIUM_IMPACT_FEATURES else "low"
    verified_at = _max_verified_date(record)
    return validate_has_feature(
        {
            "hardware_id": record["record_id"],
            "feature_id": feature_id,
            "support_level": "native" if feature_id in _NATIVE_FEATURES or feature_id.startswith("sm_") else "supported",
            "recommended": bool(matching_recommendation),
            "performance_impact": impact,
            "software_requirements": [_display_stack(item) for item in list(record.get("toolkits") or []) + list(record.get("frameworks") or [])],
            "limitations": matching_limitation,
            "hardware_specific_how_to_use": matching_recommendation or feature.get("how_to_use", ""),
            "hardware_specific_sample_code": feature.get("sample_code", ""),
            "verified": bool(record.get("source_refs")),
            "last_verified_at": verified_at,
        }
    )


def convert_hardware_feature_records_to_graph(records: list[dict[str, Any]], *, ontology: dict[str, dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    hardware_specs: list[dict[str, Any]] = []
    features_by_id: dict[str, dict[str, Any]] = {}
    relationships: list[dict[str, Any]] = []
    for raw_record in records:
        record = validate_feature_record(raw_record)
        verified_at = _max_verified_date(record)
        precision_values = ["FP32"]
        for feature_id in record.get("features") or []:
            if feature_id in _PRECISION_NAMES and _PRECISION_NAMES[feature_id] not in precision_values:
                precision_values.append(_PRECISION_NAMES[feature_id])
        frameworks = [_display_stack(item) for item in record.get("frameworks") or []]
        hardware_specs.append(
            validate_hardware_spec(
                {
                    "hardware_id": record["record_id"],
                    "name": record["title"],
                    "name_key": record["title"],
                    "aliases": _aliases_for_record(record),
                    "vendor": "NVIDIA" if record.get("vendor") == "nvidia" else str(record.get("vendor") or "").upper(),
                    "hardware_type": "GPU",
                    "architecture": (record.get("architectures") or [""])[0],
                    "description": record.get("summary_text") or record.get("detail_text") or "",
                    "memory_gb": float(record.get("vram_MB") or 0) / 1024.0 if record.get("vram_MB") else None,
                    "memory_type": record.get("vram_type") or "",
                    "supported_precisions": precision_values,
                    "software_stack": [_display_stack(item) for item in record.get("toolkits") or []] + frameworks,
                    "created_at": verified_at,
                    "updated_at": verified_at,
                }
            )
        )
        for feature_id in record.get("features") or []:
            feature = features_by_id.setdefault(
                feature_id,
                feature_from_key(feature_id, ontology=ontology, frameworks=frameworks or ["PyTorch"]),
            )
            relationships.append(_relationship_defaults(record, feature_id, feature))
    return {
        "hardware": hardware_specs,
        "features": list(features_by_id.values()),
        "relationships": relationships,
    }


def load_hardware_knowledge_from_schema(schema_root: str | Path = "schema") -> dict[str, list[dict[str, Any]]]:
    root = Path(schema_root)
    graph_json = root / "hardware_knowledge_graph.json"
    if graph_json.exists():
        return load_hardware_knowledge_from_graph_json(graph_json)
    records = load_feature_records(root / "hardware_feature_records")
    ontology = load_feature_ontology(root / "ontology" / "hardware_feature_keys.yaml")
    return convert_hardware_feature_records_to_graph(records, ontology=ontology)


def load_hardware_knowledge_from_graph_json(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    nodes = list(payload.get("nodes") or [])
    edges = list(payload.get("edges") or [])
    hardware_by_node_id: dict[str, dict[str, Any]] = {}
    feature_by_node_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        label = str(node.get("label") or "")
        props = dict(node.get("properties") or {})
        if label in {"Hardware", "HardwareSpec"}:
            hardware = validate_hardware_spec(
                {
                    "hardware_id": props.get("hardware_id"),
                    "name": props.get("name"),
                    "name_key": props.get("name_key") or props.get("name"),
                    "aliases": props.get("aliases") or [],
                    "vendor": props.get("vendor"),
                    "hardware_type": props.get("hardware_type") or "GPU",
                    "architecture": props.get("architecture"),
                    "description": props.get("description"),
                    "memory_gb": props.get("memory_gb") or props.get("vram_gb"),
                    "memory_type": props.get("memory_type"),
                    "supported_precisions": props.get("supported_precisions") or props.get("precisions") or [],
                    "software_stack": props.get("software_stack") or [],
                    "created_at": props.get("created_at"),
                    "updated_at": props.get("updated_at"),
                }
            )
            hardware_by_node_id[str(node.get("id"))] = hardware
        elif label == "Feature":
            feature = validate_feature(
                {
                    "feature_id": props.get("feature_id"),
                    "name": props.get("name") or props.get("feature_id"),
                    "category": props.get("category") or _feature_category(str(props.get("feature_id") or "")),
                    "description": props.get("description"),
                    "how_to_use": props.get("how_to_use") or props.get("description"),
                    "when_to_use": props.get("when_to_use"),
                    "when_not_to_use": props.get("when_not_to_use"),
                    "sample_code": props.get("sample_code") or props.get("example_code"),
                    "frameworks": props.get("frameworks") or [],
                    "maturity": props.get("maturity") or "production",
                }
            )
            feature_by_node_id[str(node.get("id"))] = feature

    relationships: list[dict[str, Any]] = []
    for edge in edges:
        if edge.get("type") != "HAS_FEATURE":
            continue
        hardware = hardware_by_node_id.get(str(edge.get("from")))
        feature = feature_by_node_id.get(str(edge.get("to")))
        if not hardware or not feature:
            continue
        props = dict(edge.get("properties") or {})
        relationships.append(
            validate_has_feature(
                {
                    "hardware_id": hardware["hardware_id"],
                    "feature_id": feature["feature_id"],
                    "support_level": props.get("support_level") or "supported",
                    "recommended": props.get("recommended", False),
                    "performance_impact": props.get("performance_impact") or "low",
                    "min_driver_version": props.get("min_driver_version"),
                    "min_framework_version": props.get("min_framework_version"),
                    "software_requirements": props.get("software_requirements") or [],
                    "limitations": props.get("limitations"),
                    "hardware_specific_how_to_use": props.get("hardware_specific_how_to_use") or feature.get("how_to_use"),
                    "hardware_specific_sample_code": props.get("hardware_specific_sample_code") or feature.get("sample_code"),
                    "verified": props.get("verified", False),
                    "last_verified_at": props.get("last_verified_at"),
                }
            )
        )

    return {
        "hardware": list(hardware_by_node_id.values()),
        "features": list(feature_by_node_id.values()),
        "relationships": relationships,
    }
