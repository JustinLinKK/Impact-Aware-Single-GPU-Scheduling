"""Versioned, explicit contracts for safe PerfSeer source capture.

The source is still executed in the isolated CPU worker. This module validates
the workload description, builds only declared inputs, and fingerprints local
Python imports. It deliberately does not infer a model, loss, or training recipe
from an arbitrary training script.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


CONTRACT_VERSION = "mlevolve.perfseer_workload.v1"
DEFAULT_MAX_INPUT_NUMEL = 16_000_000
HARD_MAX_INPUT_NUMEL = 64_000_000
SUPPORTED_LOSSES = {"mean", "mse", "cross_entropy"}
SUPPORTED_OPTIMIZERS = {"adam", "adamw", "sgd"}
SUPPORTED_PRECISION = {"fp32_ieee"}
_BLOCKED_PRETRAIN_CALLS = {
    "from_pretrained", "hf_hub_download", "snapshot_download",
    "load_state_dict_from_url", "download_url",
}
_SAFE_DECORATORS = {"dataclass", "staticmethod", "classmethod", "property"}
_SAFE_MODULE_INSTANCE_CONSTRUCTORS = {
    "Linear", "Bilinear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d",
    "ConvTranspose2d", "ConvTranspose3d", "Embedding", "EmbeddingBag",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm",
    "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "ReLU", "GELU",
    "Sigmoid", "Tanh", "SiLU", "LeakyReLU", "Dropout", "Dropout2d",
    "MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "Flatten", "Identity",
    "Sequential", "ModuleList", "ModuleDict",
}


class ContractError(ValueError):
    """A structured-workload contract is invalid or unsupported."""

    def __init__(self, code: str, message: str, *, context: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "context": self.context}


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("invalid_contract", f"{label} must be an object")
    return dict(value)


def _positive_int(value: Any, label: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError("invalid_contract", f"{label} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ContractError("input_bound_exceeded", f"{label} exceeds its allowed maximum", context={"value": value, "maximum": maximum})
    return value


@dataclass(frozen=True)
class WorkloadContract:
    raw: dict[str, Any]
    mode: str
    entry: str
    constructor_args: tuple[Any, ...]
    constructor_kwargs: dict[str, Any]
    precision: str
    optimizer: dict[str, Any]
    loss: dict[str, Any]
    max_input_numel: int

    @classmethod
    def parse(cls, raw: Any, *, request: Mapping[str, Any], metadata: Mapping[str, Any]) -> "WorkloadContract":
        """Parse explicit v1 metadata or adapt the previous explicit fields.

        Legacy requests remain supported only when entry, shapes, loss, and
        optimizer were explicitly supplied by the caller. A missing item never
        receives a guessed training value.
        """
        if raw is None:
            return cls._legacy(request=request, metadata=metadata)
        data = _as_mapping(raw, "workload_contract")
        if data.get("version") != CONTRACT_VERSION:
            raise ContractError("unsupported_contract_version", f"workload_contract.version must equal {CONTRACT_VERSION!r}")
        mode = data.get("mode")
        if mode not in {"module", "candidate_adapter"}:
            raise ContractError("unsupported_contract_mode", "mode must be 'module' or 'candidate_adapter'")
        allowed_contract_fields = {"version", "mode", "entry", "import_roots", "constructor_args", "constructor_kwargs", "max_input_numel", "inputs", "loss", "training", "adapter"}
        unknown_contract_fields = set(data) - allowed_contract_fields
        if unknown_contract_fields:
            raise ContractError("unsupported_contract_fields", "unknown workload contract fields cannot be ignored", context={"fields": sorted(unknown_contract_fields)})
        roots = data.get("import_roots", [])
        if not isinstance(roots, list) or any(not isinstance(root, str) for root in roots):
            raise ContractError("invalid_import_roots", "import_roots must be a list of paths")
        entry = data.get("entry")
        if not isinstance(entry, str) or not entry.strip():
            raise ContractError("missing_entry", "workload_contract.entry is required")
        args = data.get("constructor_args", [])
        kwargs = data.get("constructor_kwargs", {})
        if not isinstance(args, list) or not isinstance(kwargs, Mapping):
            raise ContractError("invalid_contract", "constructor_args must be a list and constructor_kwargs an object")
        training = _as_mapping(data.get("training"), "training")
        unknown_training_fields = set(training) - {"precision", "gradient_accumulation_steps", "optimizer", "scheduler", "backend"}
        if unknown_training_fields:
            raise ContractError("unsupported_training_fields", "unknown training fields cannot be ignored", context={"fields": sorted(unknown_training_fields)})
        precision = _precision(training.get("precision"), request, metadata)
        if training.get("gradient_accumulation_steps") != 1:
            raise ContractError("unsupported_accumulation", "source capture supports only explicitly declared gradient_accumulation_steps=1")
        optimizer = _optimizer(training.get("optimizer"))
        scheduler = _as_mapping(training.get("scheduler"), "training.scheduler")
        if not isinstance(scheduler.get("name"), str) or not scheduler["name"].strip():
            raise ContractError("missing_scheduler", "training.scheduler.name must be explicit (use 'none' for no scheduler)")
        if not isinstance(training.get("backend"), str) or not training["backend"].strip():
            raise ContractError("missing_backend", "training.backend must identify the declared training backend")
        loss = _as_mapping(data.get("loss"), "loss")
        if mode == "module":
            if "adapter" in data:
                raise ContractError("invalid_contract", "module mode cannot contain adapter configuration")
            inputs = _as_mapping(data["inputs"], "inputs")
            if set(inputs) != {"positional", "keyword"} or not isinstance(inputs["positional"], list) or not isinstance(inputs["keyword"], Mapping):
                raise ContractError("invalid_input_spec", "inputs must contain positional list and keyword object")
            if loss.get("kind") not in SUPPORTED_LOSSES:
                raise ContractError("unsupported_loss", "module mode requires an explicit supported loss kind")
            unsupported_loss_fields = set(loss) - {"kind", "target", "output_path"}
            if unsupported_loss_fields:
                raise ContractError("unsupported_loss_config", "module loss options are not implemented and cannot be ignored", context={"fields": sorted(unsupported_loss_fields)})
            if "target" not in loss:
                raise ContractError("missing_target_spec", "module mode requires an explicit target TensorSpec")
            if "inputs" not in data:
                raise ContractError("missing_input_spec", "module mode requires explicit positional/keyword input specs")
            _validate_tree_spec(data["inputs"], "inputs")
            _validate_tree_spec(loss["target"], "loss.target")
            if loss["kind"] == "cross_entropy":
                target_spec = loss["target"]
                if not isinstance(target_spec, Mapping) or target_spec.get("dtype") != "int64":
                    raise ContractError("invalid_target_spec", "cross_entropy requires an explicit int64 class-index target TensorSpec")
            if "output_path" in loss and not isinstance(loss["output_path"], list):
                raise ContractError("invalid_contract", "loss.output_path must be a list of string/integer keys")
            if any(isinstance(key, bool) or not isinstance(key, (str, int)) for key in loss.get("output_path", [])):
                raise ContractError("invalid_contract", "loss.output_path components must be strings or integers")
        else:
            if "inputs" in data:
                raise ContractError("invalid_contract", "candidate_adapter mode cannot contain module input specs")
            if loss.get("kind") != "candidate_adapter_training_step":
                raise ContractError("unsupported_loss", "candidate_adapter mode must explicitly select candidate_adapter_training_step")
            if set(loss) != {"kind"}:
                raise ContractError("unsupported_loss_config", "candidate_adapter loss options are defined by training_step and cannot be ignored", context={"fields": sorted(set(loss) - {"kind"})})
            adapter = _as_mapping(data.get("adapter"), "adapter")
            for name in ("context", "scenario", "batch_spec"):
                if name not in adapter:
                    raise ContractError("missing_adapter_contract", f"adapter.{name} is required")
            if not isinstance(adapter["context"], Mapping) or not isinstance(adapter["scenario"], Mapping):
                raise ContractError("invalid_contract", "adapter context and scenario must be objects")
            unknown_adapter_fields = set(adapter) - {"context", "scenario", "batch_spec", "fixture", "criterion"}
            if unknown_adapter_fields:
                raise ContractError("unsupported_adapter_fields", "unknown adapter fields cannot be ignored", context={"fields": sorted(unknown_adapter_fields)})
            _validate_batch_spec(adapter["batch_spec"], "adapter.batch_spec")
        max_numel = _positive_int(data.get("max_input_numel", DEFAULT_MAX_INPUT_NUMEL), "max_input_numel", maximum=HARD_MAX_INPUT_NUMEL)
        batch_size = _positive_int(request.get("batch_size"), "batch_size")
        if mode == "module":
            declared_numel = (_check_declared_numel(data["inputs"], max_numel, batch_size=batch_size)
                              + _check_declared_numel(loss["target"], max_numel, batch_size=batch_size))
            if declared_numel > max_numel:
                raise ContractError("input_bound_exceeded", "declared inputs and target exceed max_input_numel", context={"declared_numel": declared_numel, "maximum": max_numel})
        else:
            _check_declared_numel(data["adapter"]["batch_spec"], max_numel, batch_size=int(request["batch_size"]))
        return cls(data, mode, entry.strip(), tuple(args), dict(kwargs), precision, optimizer, loss, max_numel)

    @classmethod
    def _legacy(cls, *, request: Mapping[str, Any], metadata: Mapping[str, Any]) -> "WorkloadContract":
        required = ("entry", "input_shapes", "loss", "optimizer")
        absent = [key for key in required if request.get(key) is None and metadata.get(key) is None]
        if absent:
            raise ContractError("missing_contract_fields", "legacy source request lacks explicit required fields", context={"missing": absent})
        entry = request.get("entry") or metadata.get("entry")
        shapes = request.get("input_shapes") or metadata.get("input_shapes")
        dtypes = request.get("input_dtypes") or ["float32"] * len(shapes or [])
        if not shapes or len(shapes) != len(dtypes):
            raise ContractError("invalid_input_spec", "legacy input_shapes and input_dtypes must be explicit and have matching lengths")
        batch = _positive_int(request.get("batch_size"), "batch_size")
        input_spec = {
            "positional": [
                {"shape": [batch if str(dim) == "$batch" else dim for dim in shape],
                 "dtype": dtype, "distribution": "normal" if str(dtype).startswith("float") else None,
                 **({"min": 0, "max": 128} if not str(dtype).startswith("float") else {})}
                for shape, dtype in zip(shapes, dtypes, strict=True)
            ],
            "keyword": {},
        }
        # Previous released-worker behavior used a zero target shaped from the
        # model output. Retain it only in the explicitly marked compatibility
        # contract; all new v1 contracts require a declared target.
        loss_name = request.get("loss", metadata.get("loss"))
        optimizer_name = request.get("optimizer", metadata.get("optimizer"))
        raw_opt = metadata.get("optimizer_config", request.get("optimizer_config", {})) or {}
        if not isinstance(raw_opt, Mapping):
            raise ContractError("invalid_optimizer_config", "legacy optimizer_config must be an object")
        lr = raw_opt.get("lr", metadata.get("learning_rate", request.get("learning_rate", 1e-3)))
        weight_decay = raw_opt.get("weight_decay", metadata.get("weight_decay", request.get("weight_decay", 0.0)))
        raw = {
            "version": "legacy-explicit-fields", "mode": "module", "entry": str(entry),
            "constructor_args": request.get("constructor_args", []),
            "constructor_kwargs": request.get("constructor_kwargs", {}),
            "inputs": input_spec,
            "loss": {"kind": loss_name, "target": {"legacy_output_shape": True, "dtype": "float32", "fill": 0.0}},
            "training": {"precision": request.get("precision") or metadata.get("precision") or metadata.get("precision_mode"), "optimizer": {"name": optimizer_name,
                "lr": lr, "weight_decay": weight_decay},
                "scheduler": dict(metadata.get("scheduler") or {"name": "none"}),
                "backend": metadata.get("execution_backend", "unknown"),
                "gradient_accumulation_steps": metadata.get("gradient_accumulation_steps", 1)},
            "max_input_numel": DEFAULT_MAX_INPUT_NUMEL,
        }
        # Older explicit source metadata specified optimizer name but not its
        # hyperparameters. Keep that named legacy behavior; the v1 schema below
        # requires all three optimizer fields.
        if loss_name not in SUPPORTED_LOSSES or optimizer_name not in SUPPORTED_OPTIMIZERS:
            raise ContractError("unsupported_legacy_recipe", "legacy loss/optimizer names must be explicit supported values")
        if raw["training"]["gradient_accumulation_steps"] != 1:
            raise ContractError("unsupported_accumulation", "source capture supports only gradient_accumulation_steps=1")
        precision = _precision(raw["training"]["precision"], request, metadata)
        optimizer = _optimizer({**raw["training"]["optimizer"], "foreach": None, "fused": None}, legacy_defaults=True)
        return cls(raw, "module", str(entry), tuple(raw["constructor_args"]), dict(raw["constructor_kwargs"]),
                   precision, optimizer, raw["loss"], DEFAULT_MAX_INPUT_NUMEL)


def _precision(value: Any, request: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    declared = value or request.get("precision") or metadata.get("precision") or metadata.get("precision_mode")
    aliases = {"fp32": "fp32_ieee", "float32": "fp32_ieee", "fp32_ieee": "fp32_ieee"}
    normalized = aliases.get(str(declared).strip().lower()) if declared is not None else None
    if normalized not in SUPPORTED_PRECISION:
        raise ContractError("unsupported_precision", "source capture supports only an explicit FP32 recipe; non-FP32 workloads require an explicit training_graph_path with strict backward replay", context={"declared": declared})
    request_precision = request.get("precision")
    if request_precision is not None and aliases.get(str(request_precision).strip().lower(), request_precision) != normalized:
        raise ContractError("precision_mismatch", "request precision differs from workload_contract.training.precision")
    if metadata.get("uses_amp") is True:
        raise ContractError("unsupported_precision", "AMP source recipes require an explicit training_graph_path with strict backward replay")
    return normalized


def _optimizer(value: Any, *, legacy_defaults: bool = False) -> dict[str, Any]:
    data = _as_mapping(value, "training.optimizer")
    name = data.get("name")
    if name not in SUPPORTED_OPTIMIZERS:
        raise ContractError("unsupported_optimizer", "optimizer.name must be adam, adamw, or sgd")
    allowed = ({"name", "lr", "weight_decay", "betas", "eps", "amsgrad", "foreach", "fused"}
               if name in {"adam", "adamw"} else
               {"name", "lr", "weight_decay", "momentum", "dampening", "nesterov", "foreach", "fused"})
    unknown = set(data) - allowed
    if unknown:
        raise ContractError("unsupported_optimizer_config", "unknown optimizer fields cannot be ignored", context={"fields": sorted(unknown)})
    missing_dispatch = {"foreach", "fused"} - set(data)
    if missing_dispatch and not legacy_defaults:
        raise ContractError("missing_optimizer_config", "optimizer.foreach and optimizer.fused must be explicit booleans or null for the pinned PyTorch default", context={"missing": sorted(missing_dispatch)})
    required = {"lr", "weight_decay"}
    missing = sorted(required - set(data))
    if missing:
        raise ContractError("missing_optimizer_config", "optimizer algorithm settings must be explicit", context={"missing": missing})
    lr, decay = float(data["lr"]), float(data["weight_decay"])
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(decay) or decay < 0:
        raise ContractError("invalid_optimizer_config", "optimizer lr must be positive and weight_decay nonnegative")
    if name in {"adam", "adamw"}:
        betas = data.get("betas", [0.9, 0.999])
        if not isinstance(betas, (list, tuple)) or len(betas) != 2 or any(not 0 <= float(beta) < 1 for beta in betas):
            raise ContractError("invalid_optimizer_config", "Adam betas must be two explicit values in [0, 1)")
        eps = data.get("eps", 1e-8)
        if not math.isfinite(float(eps)) or float(eps) <= 0:
            raise ContractError("invalid_optimizer_config", "Adam eps must be positive")
        amsgrad = data.get("amsgrad", False)
        foreach, fused = data.get("foreach"), data.get("fused")
        if not isinstance(amsgrad, bool) or (foreach is not None and not isinstance(foreach, bool)) or (fused is not None and not isinstance(fused, bool)):
            raise ContractError("invalid_optimizer_config", "Adam foreach/fused must be booleans or null for the pinned PyTorch default; amsgrad defaults to false")
        if foreach and fused:
            raise ContractError("invalid_optimizer_config", "Adam foreach and fused cannot both be true")
        return {**data, "name": name, "lr": lr, "weight_decay": decay, "betas": [float(b) for b in betas], "eps": float(eps), "amsgrad": amsgrad, "foreach": foreach, "fused": fused}
    momentum = data.get("momentum", 0.0)
    dampening = data.get("dampening", 0.0)
    nesterov = data.get("nesterov", False)
    foreach, fused = data.get("foreach"), data.get("fused")
    for key in ("momentum", "dampening"):
        if not math.isfinite(float(momentum if key == "momentum" else dampening)) or float(momentum if key == "momentum" else dampening) < 0:
            raise ContractError("invalid_optimizer_config", f"SGD {key} must be nonnegative")
    if not isinstance(nesterov, bool) or (foreach is not None and not isinstance(foreach, bool)) or (fused is not None and not isinstance(fused, bool)):
        raise ContractError("invalid_optimizer_config", "SGD foreach/fused must be booleans or null for the pinned PyTorch default; nesterov defaults to false")
    if foreach and fused:
        raise ContractError("invalid_optimizer_config", "SGD foreach and fused cannot both be true")
    return {**data, "name": name, "lr": lr, "weight_decay": decay, "momentum": float(momentum), "dampening": float(dampening), "nesterov": nesterov, "foreach": foreach, "fused": fused}


def _validate_tree_spec(value: Any, path: str) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_tree_spec(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        if "literal" in value:
            if set(value) != {"literal"} or not isinstance(value["literal"], (str, int, float, bool, type(None))):
                raise ContractError("invalid_input_spec", f"{path} literal must be a JSON scalar")
            return
        required = {"shape", "dtype"}
        if not required.issubset(value):
            for key, child in value.items():
                _validate_tree_spec(child, f"{path}.{key}")
            return
        allowed = {"shape", "dtype", "fill", "distribution", "min", "max"}
        unknown = set(value) - allowed
        if unknown:
            raise ContractError("invalid_input_spec", f"{path} has unsupported TensorSpec keys", context={"keys": sorted(unknown)})
        shape = value["shape"]
        if not isinstance(shape, list):
            raise ContractError("invalid_input_spec", f"{path}.shape must be a list")
        for dim in shape:
            if isinstance(dim, bool) or not isinstance(dim, (int, str)) or (isinstance(dim, int) and dim <= 0) or (isinstance(dim, str) and dim != "$batch"):
                raise ContractError("invalid_input_spec", f"{path}.shape dimensions must be positive integers or '$batch'")
        dtype = str(value["dtype"])
        if dtype not in {"float32", "int64", "int32", "int16", "int8", "uint8", "bool"}:
            raise ContractError("unsupported_input_dtype", f"{path} dtype {dtype!r} is unsupported")
        if dtype.startswith("int") or dtype == "uint8":
            low, high = value.get("min"), value.get("max")
            if isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, int) or not isinstance(high, int) or high <= low:
                raise ContractError("missing_integer_bounds", f"{path} integer inputs require explicit min and max (half-open)")
            if dtype == "uint8" and (low < 0 or high > 256):
                raise ContractError("invalid_integer_bounds", f"{path} uint8 bounds must be inside [0, 256)")
            limits = {"int8": (-128, 128), "int16": (-32768, 32768), "int32": (-(2**31), 2**31), "int64": (-(2**63), 2**63)}
            if dtype in limits and (low < limits[dtype][0] or high > limits[dtype][1]):
                raise ContractError("invalid_integer_bounds", f"{path} bounds exceed {dtype} range")
            if "fill" in value or "distribution" in value:
                raise ContractError("invalid_input_spec", f"{path} integer TensorSpec cannot use float initialization fields")
        elif dtype == "bool":
            if "min" in value or "max" in value or "fill" in value or "distribution" in value:
                raise ContractError("invalid_input_spec", f"{path} bool TensorSpec accepts no bounds or float initialization fields")
        else:
            if "min" in value or "max" in value:
                raise ContractError("invalid_input_spec", f"{path} float TensorSpec cannot use integer bounds")
            if "fill" in value and (isinstance(value["fill"], bool) or not isinstance(value["fill"], (int, float)) or not math.isfinite(float(value["fill"]))):
                raise ContractError("invalid_input_spec", f"{path}.fill must be finite numeric")
            if value.get("distribution") not in (None, "normal"):
                raise ContractError("invalid_input_spec", f"{path}.distribution must be 'normal'")
            if "fill" not in value and value.get("distribution") != "normal":
                raise ContractError("missing_float_initializer", f"{path} float input requires explicit fill or distribution='normal'")
        return
    if isinstance(value, Mapping):
        return
    raise ContractError("invalid_input_spec", f"{path} must be an object, list, or explicit literal")


def _validate_batch_spec(value: Any, path: str) -> None:
    """Validate adapter output structure; tensor values come from the adapter itself."""
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_batch_spec(child, f"{path}[{index}]")
        return
    if isinstance(value, Mapping) and "literal" in value:
        if set(value) != {"literal"} or not isinstance(value["literal"], (str, int, float, bool, type(None))):
            raise ContractError("invalid_batch_spec", f"{path} literal must be a JSON scalar")
        return
    if isinstance(value, Mapping) and "shape" in value and "dtype" in value:
        if set(value) != {"shape", "dtype"}:
            raise ContractError("invalid_batch_spec", f"{path} TensorSpec may declare only shape and dtype; adapter-generated values are not initialized by the converter")
        shape = value["shape"]
        if not isinstance(shape, list):
            raise ContractError("invalid_batch_spec", f"{path}.shape must be a list")
        for dim in shape:
            if isinstance(dim, bool) or not isinstance(dim, (int, str)) or (isinstance(dim, int) and dim <= 0) or (isinstance(dim, str) and dim != "$batch"):
                raise ContractError("invalid_batch_spec", f"{path}.shape dimensions must be positive integers or '$batch'")
        if value["dtype"] not in {"float32", "int64", "int32", "int16", "int8", "uint8", "bool"}:
            raise ContractError("unsupported_input_dtype", f"{path} dtype {value['dtype']!r} is unsupported")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_batch_spec(child, f"{path}.{key}")
        return
    raise ContractError("invalid_batch_spec", f"{path} must be a TensorSpec, pytree, or explicit literal")


def _check_declared_numel(spec: Any, maximum: int, *, batch_size: int | None = None) -> int:
    total = 0
    if isinstance(spec, Mapping):
        if "shape" in spec and "dtype" in spec:
            dims = [batch_size if dimension == "$batch" else int(dimension) for dimension in spec["shape"]]
            total += math.prod(dims)
        elif "literal" not in spec:
            for child in spec.values():
                total += _check_declared_numel(child, maximum, batch_size=batch_size)
    elif isinstance(spec, list):
        for child in spec:
            total += _check_declared_numel(child, maximum, batch_size=batch_size)
    if total > maximum:
        raise ContractError("input_bound_exceeded", "declared tensor inputs exceed max_input_numel", context={"declared_numel": total, "maximum": maximum})
    return total


def build_tree(spec: Any, *, batch_size: int, torch: Any, generator: Any) -> Any:
    """Materialize a JSON tensor/pytree spec only after its size was checked."""
    if isinstance(spec, list):
        return [build_tree(value, batch_size=batch_size, torch=torch, generator=generator) for value in spec]
    if isinstance(spec, Mapping):
        if "literal" in spec:
            return spec["literal"]
        if "shape" in spec and "dtype" in spec:
            shapes = [batch_size if value == "$batch" else int(value) for value in spec["shape"]]
            dtype = getattr(torch, spec["dtype"])
            if spec["dtype"] == "bool":
                return torch.randint(0, 2, shapes, dtype=torch.int64, generator=generator).bool()
            if spec["dtype"].startswith("int") or spec["dtype"] == "uint8":
                return torch.randint(int(spec["min"]), int(spec["max"]), shapes, dtype=dtype, generator=generator)
            if "fill" in spec:
                return torch.full(shapes, float(spec["fill"]), dtype=dtype)
            return torch.randn(shapes, dtype=dtype, generator=generator)
        return {str(key): build_tree(value, batch_size=batch_size, torch=torch, generator=generator) for key, value in spec.items()}
    raise ContractError("invalid_input_spec", "all source inputs must be described by a TensorSpec, pytree, or literal")


def tree_tensor_numel(value: Any) -> int:
    """Count tensor elements in an actual nested input/batch."""
    try:
        import torch
    except ImportError:
        return 0
    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, Mapping):
        return sum(tree_tensor_numel(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tree_tensor_numel(child) for child in value)
    return 0


def at_path(value: Any, path: Sequence[Any]) -> Any:
    current = value
    for component in path:
        if isinstance(current, Mapping) and isinstance(component, str):
            if component not in current:
                raise ContractError("output_path_missing", f"output key {component!r} was not present")
            current = current[component]
        elif isinstance(current, (tuple, list)) and isinstance(component, int):
            try:
                current = current[component]
            except IndexError as exc:
                raise ContractError("output_path_missing", f"output index {component} was not present") from exc
        else:
            raise ContractError("output_path_type_mismatch", "output_path does not match the model output structure")
    return current


def _source_ast(path: Path, entry: str | None = None) -> ast.Module:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ContractError("source_parse_failed", f"could not parse source file: {exc}", context={"source_path": str(path)}) from exc
    if entry:
        _reject_reachable_pretrained_calls(tree, entry)
    return tree


def _reject_reachable_pretrained_calls(tree: ast.Module, entry: str) -> None:
    definitions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    entry_node = definitions.get(entry)
    if entry_node is None:
        # For an explicitly named instance, inspect its assigned expression later;
        # unrelated helper functions should not invalidate this selected entry.
        return
    pending = [entry_node]
    seen: set[str] = set()
    while pending:
        node = pending.pop()
        key = getattr(node, "name", "")
        if key in seen:
            continue
        seen.add(key)
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                fn = child.func.attr if isinstance(child.func, ast.Attribute) else child.func.id if isinstance(child.func, ast.Name) else None
                if fn in _BLOCKED_PRETRAIN_CALLS:
                    raise ContractError("pretrained_loading_unsupported", "selected source entry reaches a pretrained/download loader; use a local capture fixture", context={"call": fn, "line": getattr(child, "lineno", None)})
                if isinstance(child.func, ast.Attribute) and child.func.attr in {"import_module", "load_module"}:
                    raise ContractError("dynamic_import_unsupported", "selected source entry uses dynamic imports that cannot be fingerprinted", context={"line": getattr(child, "lineno", None)})
                if isinstance(child.func, ast.Name) and child.func.id in definitions:
                    pending.append(definitions[child.func.id])
            elif isinstance(child, ast.ClassDef):
                pending.extend(item for item in child.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))


def _safe_module_tree(tree: ast.Module, source_path: Path, entry: str | None = None) -> ast.Module:
    body: list[ast.stmt] = []
    nn_aliases = {"torch.nn"}
    imported_nn_constructors: set[str] = set()
    path_aliases: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name == "torch.nn":
                    nn_aliases.add(alias.asname or "torch.nn")
                elif alias.name == "torch":
                    nn_aliases.add((alias.asname or "torch") + ".nn")
        elif isinstance(statement, ast.ImportFrom) and statement.module == "torch.nn":
            imported_nn_constructors.update(alias.asname or alias.name for alias in statement.names if alias.name in _SAFE_MODULE_INSTANCE_CONSTRUCTORS)
        elif isinstance(statement, ast.ImportFrom) and statement.module == "torch":
            if any(alias.name == "nn" for alias in statement.names):
                imported_alias = next(alias.asname or alias.name for alias in statement.names if alias.name == "nn")
                nn_aliases.add(imported_alias)
        elif isinstance(statement, ast.ImportFrom) and statement.module == "pathlib":
            if any(alias.name == "Path" for alias in statement.names):
                path_aliases.add(next(alias.asname or alias.name for alias in statement.names if alias.name == "Path"))
        elif isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name == "pathlib":
                    path_aliases.add((alias.asname or "pathlib") + ".Path")
    safe_path_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decorators = getattr(node, "decorator_list", ())
            if any(not _safe_decorator(item) for item in decorators):
                raise ContractError("unsafe_source_decorator", "only dataclass/staticmethod/classmethod/property decorators are accepted", context={"line": node.lineno})
            body.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None:
                body.append(node)
                continue
            try:
                ast.literal_eval(value)
            except (ValueError, TypeError):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                is_selected_instance = entry is not None and any(isinstance(target, ast.Name) and target.id == entry for target in targets)
                is_path_constant = _safe_path_expression(value, path_aliases, safe_path_names)
                if not is_path_constant and (not is_selected_instance or not _safe_nn_constructor(value, nn_aliases, imported_nn_constructors)):
                    raise ContractError("unsafe_module_initializer", "top-level assignments must be literal constants, safe pathlib constants, or a selected module instance built with safe torch.nn constructors", context={"line": node.lineno})
                if is_path_constant:
                    safe_path_names.update(target.id for target in targets if isinstance(target, ast.Name))
            body.append(node)
        elif isinstance(node, ast.If) and _is_main_guard(node.test):
            continue
        else:
            raise ContractError("unsafe_module_initializer", "top-level executable statements are unsupported", context={"line": getattr(node, "lineno", None), "source_path": str(source_path)})
    tree.body = body
    return ast.fix_missing_locations(tree)


def _safe_nn_constructor(node: ast.AST, aliases: set[str], imported: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        valid = node.func.id in imported
    elif isinstance(node.func, ast.Attribute):
        dotted = _ast_dotted_name(node.func.value)
        valid = dotted in aliases and node.func.attr in _SAFE_MODULE_INSTANCE_CONSTRUCTORS
    else:
        valid = False
    if not valid:
        return False
    for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
        try:
            ast.literal_eval(argument)
        except (ValueError, TypeError):
            if not _safe_nn_constructor(argument, aliases, imported):
                return False
    return True


def _ast_dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _ast_dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _safe_path_expression(node: ast.AST, aliases: set[str], names: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, int))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _safe_path_expression(node.left, aliases, names) and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
    if isinstance(node, ast.Attribute) and node.attr in {"parent", "name", "stem", "suffix"}:
        return _safe_path_expression(node.value, aliases, names)
    if isinstance(node, ast.Call) and _ast_dotted_name(node.func) in aliases and len(node.args) == 1 and not node.keywords:
        arg = node.args[0]
        return isinstance(arg, ast.Name) and arg.id == "__file__" or isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    return False


def _is_main_guard(test: ast.expr) -> bool:
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"
            and len(test.ops) == 1 and isinstance(test.ops[0], (ast.Eq, ast.Is))
            and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__")


def _safe_decorator(node: ast.expr) -> bool:
    if isinstance(node, ast.Name) and node.id in _SAFE_DECORATORS:
        return True
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id == "torch" and node.attr in {"no_grad", "inference_mode"}
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return node.func.value.id == "torch" and node.func.attr in {"no_grad", "inference_mode"} and not node.args and not node.keywords
    return False


def import_source_module(source_path: str | Path, *, import_roots: Sequence[str] = (), entry: str | None = None) -> ModuleType:
    """Import declarations/constants only; skip `__main__` pipelines and reject executable top-level code."""
    source = Path(source_path).expanduser().resolve()
    if not source.is_file() or source.suffix != ".py":
        raise ContractError("source_not_python", f"source_path must be an existing .py file: {source}")
    tree = _safe_module_tree(_source_ast(source, entry), source, entry)
    roots = [str(Path(root).expanduser().resolve()) for root in import_roots]
    roots.insert(0, str(source.parent))
    old_prefix = list(sys.path)
    for root in reversed(list(dict.fromkeys(roots))):
        if root not in sys.path:
            sys.path.insert(0, root)
    module_name = f"_mlevolve_capture_{hashlib.sha256(str(source).encode()).hexdigest()[:16]}"
    package_context_roots = [*(Path(root) for root in roots[1:]), source.parent]
    package_root, package_parts = _package_context(source, package_context_roots)
    package_namespace_name = module_name + "_pkg"
    package_name = package_namespace_name + ("." + ".".join(package_parts) if package_parts else "")
    if package_parts:
        root_namespace = ModuleType(package_namespace_name)
        root_namespace.__package__ = package_namespace_name
        root_namespace.__path__ = [str(package_root)]  # type: ignore[attr-defined]
        sys.modules[package_namespace_name] = root_namespace
        for index in range(len(package_parts)):
            current_name = package_namespace_name + "." + ".".join(package_parts[:index + 1])
            current_path = package_root.joinpath(*package_parts[:index + 1])
            namespace = ModuleType(current_name)
            namespace.__package__ = current_name
            namespace.__path__ = [str(current_path)]  # type: ignore[attr-defined]
            sys.modules[current_name] = namespace
            parent = sys.modules[package_namespace_name if index == 0 else package_namespace_name + "." + ".".join(package_parts[:index])]
            setattr(parent, package_parts[index], namespace)
    module = ModuleType(module_name)
    module.__file__ = str(source)
    module.__package__ = package_name
    sys.modules[module_name] = module
    try:
        code = compile(tree, str(source), "exec")
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path[:] = old_prefix
    return module


def load_entry(module: ModuleType, contract: WorkloadContract, *, torch: Any, request: Mapping[str, Any]) -> tuple[Any, Any | None, Any]:
    """Resolve an explicitly named module/factory/class or CandidateAdapter."""
    if not hasattr(module, contract.entry):
        raise ContractError("entry_not_found", f"source module does not define {contract.entry!r}", context={"entry": contract.entry})
    target = getattr(module, contract.entry)
    if contract.mode == "module":
        if isinstance(target, torch.nn.Module):
            model = target
        elif isinstance(target, type) and issubclass(target, torch.nn.Module):
            model = target(*contract.constructor_args, **contract.constructor_kwargs)
        elif callable(target):
            model = target(*contract.constructor_args, **contract.constructor_kwargs)
        else:
            raise ContractError("invalid_model_entry", "module entry must be an nn.Module instance, class, or factory")
        if not isinstance(model, torch.nn.Module):
            raise ContractError("entry_not_module", f"module entry returned {type(model).__name__}, expected nn.Module")
        return model.cpu().train(), None, None
    if not isinstance(target, type):
        raise ContractError("invalid_adapter_entry", "candidate_adapter entry must name an adapter class")
    adapter = target(*contract.constructor_args, **contract.constructor_kwargs)
    required = ("build_model", "build_train_batch", "training_step")
    absent = [name for name in required if not callable(getattr(adapter, name, None))]
    if absent:
        raise ContractError("adapter_protocol_incomplete", "CandidateAdapter is missing required methods", context={"missing": absent})
    adapter_spec = contract.raw["adapter"]
    context = dict(adapter_spec["context"])
    context["device"] = "cpu"
    model = adapter.build_model(context)
    if not isinstance(model, torch.nn.Module):
        raise ContractError("entry_not_module", "CandidateAdapter.build_model must return nn.Module")
    scenario = dict(adapter_spec["scenario"])
    scenario["batch_size"] = int(request["batch_size"])
    scenario["fixture"] = adapter_spec.get("fixture", scenario.get("fixture", _scenario_fixture(adapter_spec["batch_spec"])))
    model = model.cpu().train()
    return model, adapter, {"context": context, "scenario": scenario, "spec": adapter_spec, "module": module}


def _scenario_fixture(spec: Any) -> Any:
    """Convert batch TensorSpecs into the per-example shape hints expected by MLEvolve adapters."""
    if isinstance(spec, list):
        return [_scenario_fixture(value) for value in spec]
    if isinstance(spec, Mapping):
        if "shape" in spec and "dtype" in spec:
            shape = list(spec["shape"])
            if shape and shape[0] == "$batch":
                shape = shape[1:]
            return {"shape": shape}
        if "literal" in spec:
            return spec["literal"]
        return {str(key): _scenario_fixture(value) for key, value in spec.items()}
    return spec


def build_adapter_batch(adapter: Any, model: Any, details: Mapping[str, Any], *, torch: Any, max_input_numel: int, optimizer_spec: Mapping[str, Any]) -> tuple[Any, Any]:
    """Build and validate the CandidateAdapter's declared synthetic batch."""
    scenario = dict(details["scenario"])
    batch_size = int(scenario["batch_size"])
    spec = details["spec"]
    declared = spec["batch_spec"]
    _check_declared_numel(declared, max_input_numel, batch_size=batch_size)
    batch = adapter.build_train_batch(scenario, "cpu")
    actual_numel = tree_tensor_numel(batch)
    if actual_numel > max_input_numel:
        raise ContractError("input_bound_exceeded", "CandidateAdapter batch exceeded max_input_numel", context={"actual_numel": actual_numel, "maximum": max_input_numel})
    _assert_batch_matches(batch, declared, batch_size)
    context = dict(details["context"])
    if isinstance(context.get("criterion"), Mapping):
        context["criterion"] = _declared_criterion(context["criterion"], torch, details.get("module"))
    if ("criterion" not in context or context.get("criterion") is None) and "criterion" in spec:
        context["criterion"] = _declared_criterion(spec["criterion"], torch, details.get("module"))
    optimizer_builder = getattr(adapter, "build_optimizer", None)
    if callable(optimizer_builder):
        optimizer = optimizer_builder(model, context)
        _check_adapter_optimizer(optimizer, model, optimizer_spec, torch)
    return batch, context


def _check_adapter_optimizer(optimizer: Any, model: Any, optimizer_spec: Mapping[str, Any], torch: Any) -> None:
    expected_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[optimizer_spec["name"]]
    if type(optimizer) is not expected_class:
        raise ContractError("optimizer_mismatch", "CandidateAdapter.build_optimizer differs from declared optimizer", context={"declared": expected_class.__name__, "actual": type(optimizer).__name__})
    expected_values = {key: optimizer_spec[key] for key in ("lr", "weight_decay", "betas", "eps", "amsgrad", "momentum", "dampening", "nesterov", "foreach", "fused") if key in optimizer_spec}
    defaults = getattr(optimizer, "defaults", {})
    mismatches = {key: {"declared": value, "actual": _canonical_optimizer_value(defaults.get(key))} for key, value in expected_values.items() if _canonical_optimizer_value(defaults.get(key)) != _canonical_optimizer_value(value)}
    if mismatches:
        raise ContractError("optimizer_config_mismatch", "CandidateAdapter.build_optimizer differs from resolved optimizer recipe", context={"mismatches": mismatches})
    group_mismatches = []
    for index, group in enumerate(optimizer.param_groups):
        changed = {key: {"declared": value, "actual": _canonical_optimizer_value(group.get(key))}
                   for key, value in expected_values.items()
                   if _canonical_optimizer_value(group.get(key)) != _canonical_optimizer_value(value)}
        if changed:
            group_mismatches.append({"group": index, "fields": changed})
    if group_mismatches:
        raise ContractError("optimizer_group_mismatch", "CandidateAdapter optimizer parameter groups override the declared recipe", context={"groups": group_mismatches})
    actual_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    trainable_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(trainable_ids):
        raise ContractError("optimizer_parameter_coverage", "CandidateAdapter optimizer must cover every trainable model parameter exactly once", context={"optimizer_parameter_count": len(actual_ids), "trainable_parameter_count": len(trainable_ids)})


def _canonical_optimizer_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_canonical_optimizer_value(item) for item in value]
    if isinstance(value, list):
        return [_canonical_optimizer_value(item) for item in value]
    return value


def _declared_criterion(value: Any, torch: Any, module: ModuleType | None = None) -> Any:
    if not isinstance(value, Mapping):
        raise ContractError("unsupported_adapter_criterion", "adapter criterion declaration must be an object")
    unknown = set(value) - {"kind", "entry", "constructor_args", "constructor_kwargs"}
    if unknown:
        raise ContractError("unsupported_adapter_criterion", "unsupported criterion options cannot be ignored", context={"fields": sorted(unknown)})
    if value.get("kind") == "source_entry":
        entry = value.get("entry")
        if not isinstance(entry, str) or module is None or not hasattr(module, entry):
            raise ContractError("unsupported_adapter_criterion", "source_entry criterion must name a class in the selected source module")
        target = getattr(module, entry)
        args, kwargs = value.get("constructor_args", []), value.get("constructor_kwargs", {})
        if not isinstance(args, list) or not isinstance(kwargs, Mapping):
            raise ContractError("unsupported_adapter_criterion", "criterion constructor arguments must be explicit JSON list/object")
        criterion = target(*args, **dict(kwargs)) if isinstance(target, type) else target
        if not isinstance(criterion, torch.nn.Module):
            raise ContractError("unsupported_adapter_criterion", "source_entry criterion must resolve to nn.Module")
        return criterion.cpu()
    if value.get("kind") not in {"mse", "cross_entropy"}:
        raise ContractError("unsupported_adapter_criterion", "criterion kind must be mse, cross_entropy, or source_entry")
    if value["kind"] == "mse":
        return torch.nn.MSELoss()
    return torch.nn.CrossEntropyLoss()


def _assert_batch_matches(batch: Any, spec: Any, batch_size: int, path: str = "batch") -> None:
    try:
        import torch
    except ImportError:
        return
    if isinstance(spec, Mapping) and "literal" in spec:
        if batch != spec["literal"]:
            raise ContractError("adapter_batch_literal_mismatch", f"{path} literal differs from declared batch_spec")
    elif isinstance(spec, Mapping) and "shape" in spec and "dtype" in spec:
        if not isinstance(batch, torch.Tensor):
            raise ContractError("adapter_batch_structure_mismatch", f"{path} was declared tensor but CandidateAdapter returned {type(batch).__name__}")
        expected_shape = tuple(batch_size if dim == "$batch" else int(dim) for dim in spec["shape"])
        if tuple(batch.shape) != expected_shape:
            raise ContractError("adapter_batch_shape_mismatch", f"{path} shape differs from the declared contract", context={"expected": expected_shape, "actual": tuple(batch.shape)})
        expected_dtype = getattr(torch, spec["dtype"])
        if batch.dtype != expected_dtype:
            raise ContractError("adapter_batch_dtype_mismatch", f"{path} dtype differs from the declared contract", context={"expected": str(expected_dtype), "actual": str(batch.dtype)})
    elif isinstance(spec, Mapping):
        if not isinstance(batch, Mapping) or set(batch) != set(spec):
            raise ContractError("adapter_batch_structure_mismatch", f"{path} keys differ from declared batch_spec")
        for key, value in spec.items():
            _assert_batch_matches(batch[key], value, batch_size, f"{path}.{key}")
    elif isinstance(spec, list):
        if not isinstance(batch, (tuple, list)) or len(batch) != len(spec):
            raise ContractError("adapter_batch_structure_mismatch", f"{path} sequence differs from declared batch_spec")
        for index, value in enumerate(spec):
            _assert_batch_matches(batch[index], value, batch_size, f"{path}[{index}]")


def dependency_fingerprint(source_path: str | Path, import_roots: Sequence[str] = ()) -> dict[str, Any]:
    """Hash statically discoverable local Python dependencies without importing them.

    `complete` is false when an import is dynamic or cannot be resolved inside
    the declared local roots. Callers should include the evidence in cache keys
    and disable persistent caching when dependency closure is incomplete.
    """
    source = Path(source_path).expanduser().resolve()
    roots: list[Path] = list(dict.fromkeys([source.parent, *(Path(root).expanduser().resolve() for root in import_roots)]))
    pending = [source]
    files: dict[str, str] = {}
    unresolved: list[str] = []
    dynamic = False
    while pending:
        path = pending.pop().resolve()
        if str(path) in files:
            continue
        try:
            payload = path.read_bytes()
            tree = ast.parse(payload, filename=str(path))
        except (OSError, SyntaxError) as exc:
            unresolved.append(f"{path}: {type(exc).__name__}")
            continue
        files[str(path)] = hashlib.sha256(payload).hexdigest()
        imports: set[str] = set()
        from_imports: list[tuple[str, list[str]]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    package_parts = _package_parts(path, roots)
                    up = node.level - 1
                    base = package_parts[:max(0, len(package_parts) - up)]
                    module = ".".join([*base, *([module] if module else [])])
                if module:
                    imports.add(module)
                from_imports.append((module, [alias.name for alias in node.names if alias.name != "*"]))
            elif isinstance(node, ast.Call) and ((isinstance(node.func, ast.Name) and node.func.id in {"__import__", "import_module"}) or (isinstance(node.func, ast.Attribute) and node.func.attr in {"import_module", "__import__"})):
                if isinstance(node.func, ast.Attribute):
                    dynamic = True
                elif node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    imports.add(node.args[0].value)
                else:
                    dynamic = True
        for name in sorted(imports):
            parts = name.split(".")
            candidates = []
            for root in roots:
                base = root.joinpath(*parts)
                candidates.extend((base.with_suffix(".py"), base / "__init__.py"))
            found = next((candidate for candidate in candidates if candidate.is_file()), None)
            if found is not None:
                pending.append(found)
            elif any((root / name.split(".")[0]).exists() or (root / (name.split(".")[0] + ".py")).exists() for root in roots):
                unresolved.append(name)
        for module_name, imported_names in from_imports:
            for child in imported_names:
                full = ".".join(part for part in (module_name, child) if part)
                if not full:
                    continue
                parts = full.split(".")
                for root in roots:
                    base = root.joinpath(*parts)
                    candidate = base.with_suffix(".py")
                    package_file = base / "__init__.py"
                    if candidate.is_file():
                        pending.append(candidate)
                        break
                    if package_file.is_file():
                        pending.append(package_file)
                        break
    payload = json.dumps(sorted(files.items()), separators=(",", ":")).encode()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "files": [{"path": path, "sha256": digest} for path, digest in sorted(files.items())],
        "unresolved_local_imports": sorted(set(unresolved)),
        "dynamic_imports": dynamic,
        "complete": not unresolved and not dynamic,
    }


def _package_parts(path: Path, roots: Sequence[Path]) -> list[str]:
    """Infer package components only when an explicit import root proves them."""
    return _package_context(path, roots)[1]


def _package_context(path: Path, roots: Sequence[Path]) -> tuple[Path, list[str]]:
    candidates: list[tuple[int, Path, list[str]]] = []
    for root in sorted(roots, key=lambda item: len(item.parts), reverse=True):
        try:
            relative = path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        parts = list(relative.parent.parts)
        if all((root.joinpath(*parts[:index]) / "__init__.py").is_file() for index in range(1, len(parts) + 1)):
            candidates.append((len(parts), root, parts))
    if candidates:
        _, root, parts = max(candidates, key=lambda candidate: candidate[0])
        return root, parts
    return path.parent, []


def fingerprint_source(source_path: str | Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Parent-facing dependency receipt API; accepts full perfseer_model metadata."""
    contract = metadata.get("workload_contract", {}) if isinstance(metadata, Mapping) else {}
    roots = contract.get("import_roots", metadata.get("import_roots", [])) if isinstance(contract, Mapping) else metadata.get("import_roots", [])
    if not isinstance(roots, list) or any(not isinstance(root, str) for root in roots):
        raise ContractError("invalid_import_roots", "import_roots must be a list of paths")
    receipt = dependency_fingerprint(source_path, roots)
    receipt["import_roots"] = [str(Path(root).expanduser().resolve()) for root in roots]
    return receipt


def disable_network() -> None:
    """Best-effort network denial for user-source execution inside the child."""
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})

    def denied(*args: Any, **kwargs: Any) -> Any:
        raise ContractError("network_access_denied", "source capture worker has no network access")

    socket.create_connection = denied  # type: ignore[assignment]
    socket.socket.connect = denied  # type: ignore[method-assign]


__all__ = [
    "CONTRACT_VERSION", "DEFAULT_MAX_INPUT_NUMEL", "ContractError", "WorkloadContract",
    "at_path", "build_adapter_batch", "build_tree", "dependency_fingerprint", "fingerprint_source",
    "disable_network", "import_source_module", "load_entry", "tree_tensor_numel", "_check_adapter_optimizer",
]
