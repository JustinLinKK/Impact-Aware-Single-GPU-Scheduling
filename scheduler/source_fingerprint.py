"""Deterministic, CPU-only source fingerprints used to order live trials.

The analyzer in this module never imports or executes submitted code.  Its
output is deliberately descriptive: measured colocation evidence remains the
only authority for accepting a placement.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..domain import TrainingJob

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PhaseFingerprint:
    resource_sequence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StaticJobFingerprint:
    schema_version: int
    source_hash: str
    graph_hash: str
    batch_size: int
    dtype: str
    steps_per_epoch: int | None
    predicted_epoch_seconds: float
    predicted_vram_bytes: int
    step_seconds: float | None
    forward_flops: float | None
    training_step_flops: float | None
    estimated_bytes_per_step: float | None
    parameter_bytes: int | None
    activation_bytes: int | None
    gradient_bytes: int | None
    optimizer_bytes: int | None
    compute_pressure: float | None
    memory_pressure: float | None
    operator_count: int
    operator_histogram: Mapping[str, int]
    operator_flop_histogram: Mapping[str, float]
    largest_op_fraction: float | None
    small_op_fraction: float | None
    tensor_core_eligible_fraction: float | None
    reduction_fraction: float | None
    irregular_memory_fraction: float | None
    explicit_sync_count: int
    blocking_transfer_count: int
    async_transfer_count: int
    dataloader_worker_count: int | None
    cpu_augmentation_flag: bool | None
    checkpoint_frequency: float | None
    evaluation_frequency: float | None
    forward_phase: PhaseFingerprint | None
    backward_phase: PhaseFingerprint | None
    optimizer_phase: PhaseFingerprint | None
    unknown_operator_fraction: float
    dynamic_control_flow: bool
    custom_operation_flag: bool
    resource_class: str
    confidence: str
    analysis_warnings: tuple[str, ...]

    @property
    def analysis_uncertainty(self) -> float:
        missing = sum(
            value is None
            for value in (
                self.steps_per_epoch,
                self.training_step_flops,
                self.estimated_bytes_per_step,
            )
        )
        return min(
            1.0,
            self.unknown_operator_fraction
            + (0.25 if self.dynamic_control_flow else 0.0)
            + (0.25 if self.custom_operation_flag else 0.0)
            + 0.15 * missing,
        )

    @property
    def execution_signature(self) -> str:
        """Epoch-independent identity for exact interference profiles."""
        return (
            f"static-v{self.schema_version}:{self.source_hash}:"
            f"{self.graph_hash}:{self.dtype}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_hash": self.source_hash,
            "graph_hash": self.graph_hash,
            "batch_size": self.batch_size,
            "dtype": self.dtype,
            "steps_per_epoch": self.steps_per_epoch,
            "predicted_epoch_seconds": self.predicted_epoch_seconds,
            "predicted_vram_bytes": self.predicted_vram_bytes,
            "step_seconds": self.step_seconds,
            "forward_flops": self.forward_flops,
            "training_step_flops": self.training_step_flops,
            "estimated_bytes_per_step": self.estimated_bytes_per_step,
            "parameter_bytes": self.parameter_bytes,
            "activation_bytes": self.activation_bytes,
            "gradient_bytes": self.gradient_bytes,
            "optimizer_bytes": self.optimizer_bytes,
            "compute_pressure": self.compute_pressure,
            "memory_pressure": self.memory_pressure,
            "operator_count": self.operator_count,
            "operator_histogram": dict(self.operator_histogram),
            "operator_flop_histogram": dict(self.operator_flop_histogram),
            "largest_op_fraction": self.largest_op_fraction,
            "small_op_fraction": self.small_op_fraction,
            "tensor_core_eligible_fraction": self.tensor_core_eligible_fraction,
            "reduction_fraction": self.reduction_fraction,
            "irregular_memory_fraction": self.irregular_memory_fraction,
            "explicit_sync_count": self.explicit_sync_count,
            "blocking_transfer_count": self.blocking_transfer_count,
            "async_transfer_count": self.async_transfer_count,
            "dataloader_worker_count": self.dataloader_worker_count,
            "cpu_augmentation_flag": self.cpu_augmentation_flag,
            "checkpoint_frequency": self.checkpoint_frequency,
            "evaluation_frequency": self.evaluation_frequency,
            "unknown_operator_fraction": self.unknown_operator_fraction,
            "dynamic_control_flow": self.dynamic_control_flow,
            "custom_operation_flag": self.custom_operation_flag,
            "resource_class": self.resource_class,
            "confidence": self.confidence,
            "analysis_warnings": list(self.analysis_warnings),
        }


def linear_flops(batch: int, input_features: int, output_features: int) -> float:
    return float(2 * batch * input_features * output_features)


def convolution_flops(
    batch: int,
    output_height: int,
    output_width: int,
    output_channels: int,
    input_channels: int,
    kernel_height: int,
    kernel_width: int,
    *,
    groups: int = 1,
) -> float:
    return float(
        2
        * batch
        * output_height
        * output_width
        * output_channels
        * (input_channels / max(1, groups))
        * kernel_height
        * kernel_width
    )


def attention_flops(batch: int, heads: int, sequence: int, head_dim: int) -> float:
    # QK^T and attention-value products.
    return float(4 * batch * heads * sequence * sequence * head_dim)


def tensor_elementwise_flops(
    elements: int, operations_per_element: float = 1.0
) -> float:
    return float(max(0, elements) * max(0.0, operations_per_element))


_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "attention",
        (
            "attention",
            "multiheadattention",
            "scaled_dot_product_attention",
            "flash_attn",
        ),
    ),
    ("convolution", ("conv1d", "conv2d", "conv3d", "convolution")),
    ("gemm", ("linear", "matmul", "mm", "bmm", "einsum", "addmm")),
    ("normalization", ("batchnorm", "layernorm", "groupnorm", "rmsnorm", "normalize")),
    (
        "reduction",
        ("softmax", "log_softmax", "sum", "mean", "amax", "amin", "cross_entropy"),
    ),
    ("embedding", ("embedding", "gather", "scatter", "index_select")),
    ("pooling", ("pool", "adaptiveavgpool", "adaptivemaxpool")),
    ("recurrent", ("lstm", "gru", "rnn")),
    ("optimizer", ("adam", "adamw", "sgd", "rmsprop", "optimizer.step", ".step")),
    (
        "data_movement",
        (
            "permute",
            "transpose",
            "contiguous",
            "reshape",
            "view",
            ".to",
            ".cuda",
            ".cpu",
        ),
    ),
    (
        "activation",
        ("relu", "gelu", "silu", "sigmoid", "tanh", "dropout", "activation"),
    ),
)


def normalize_operator_name(call_name: str) -> str:
    lowered = call_name.lower()
    for category, patterns in _CATEGORY_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return category
    return "unknown"


def _call_name(node: ast.Call) -> str:
    parts: list[str] = []
    current: ast.AST | None = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _constant_int(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    return None


def _number(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        try:
            if value is not None and float(value) >= 0:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


class StaticJobAnalyzer:
    """Analyze source syntax and submitted configuration without importing it."""

    def __init__(
        self,
        *,
        cache_enabled: bool = True,
        max_source_bytes: int = 2_000_000,
        high_unknown_fraction: float = 0.05,
        medium_unknown_fraction: float = 0.25,
        peak_tflops_by_dtype: Mapping[str, float] | None = None,
        memory_bandwidth_gbps: float | None = None,
    ) -> None:
        self.cache_enabled = bool(cache_enabled)
        self.max_source_bytes = max(1, int(max_source_bytes))
        self.high_unknown_fraction = float(high_unknown_fraction)
        self.medium_unknown_fraction = float(medium_unknown_fraction)
        self.peak_tflops_by_dtype = {
            str(key).lower(): float(value)
            for key, value in dict(peak_tflops_by_dtype or {}).items()
            if float(value) > 0
        }
        self.memory_bandwidth_gbps = (
            float(memory_bandwidth_gbps)
            if memory_bandwidth_gbps is not None and float(memory_bandwidth_gbps) > 0
            else None
        )
        self._cache: dict[str, StaticJobFingerprint] = {}

    @staticmethod
    def _source(job: TrainingJob, max_source_bytes: int) -> tuple[str, str | None]:
        kwargs = dict(job.config.runner_kwargs or {})
        for key in ("architecture_source", "source_code", "model_source"):
            value = kwargs.get(key) or job.metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value[:max_source_bytes], None
        paths = [
            kwargs.get("script_path"),
            kwargs.get("source_path"),
            kwargs.get("model_source_path"),
            job.baseline_model_path,
        ]
        for raw_path in paths:
            if not raw_path:
                continue
            path = Path(str(raw_path))
            if path.suffix.lower() not in {".py", ".pyw"} or not path.is_file():
                continue
            try:
                raw = path.read_bytes()
            except OSError as exc:
                return "", f"source_read_failed:{type(exc).__name__}"
            if len(raw) > max_source_bytes:
                return (
                    raw[:max_source_bytes].decode("utf-8", errors="replace"),
                    "source_truncated",
                )
            return raw.decode("utf-8", errors="replace"), None
        return "", "source_unavailable"

    @staticmethod
    def _dtype(job: TrainingJob) -> str:
        values = (
            job.config.runner_kwargs.get("precision"),
            job.config.runner_kwargs.get("dtype"),
            job.config.runner_kwargs.get("amp_dtype"),
            job.metadata.get("precision"),
            job.metadata.get("dtype"),
        )
        raw = next((str(value).lower() for value in values if value), "unknown")
        aliases = {
            "16-mixed": "float16",
            "fp16": "float16",
            "half": "float16",
            "bf16-mixed": "bfloat16",
            "bf16": "bfloat16",
            "fp32": "float32",
            "32": "float32",
        }
        return aliases.get(raw, raw)

    @staticmethod
    def _steps_per_epoch(job: TrainingJob) -> int | None:
        candidates = (
            job.metadata.get("runtime_steps_per_epoch"),
            job.config.runner_kwargs.get("steps_per_epoch"),
            job.batch_probe.shape_hints.get("steps_per_epoch"),
        )
        for value in candidates:
            try:
                if value is not None and int(value) > 0:
                    return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def analyze(
        self,
        job: TrainingJob,
        batch_size: int,
        *,
        predicted_epoch_seconds: float = 0.0,
        predicted_vram_bytes: int = 0,
    ) -> StaticJobFingerprint:
        source, source_warning = self._source(job, self.max_source_bytes)
        warnings: list[str] = [source_warning] if source_warning else []
        try:
            tree = ast.parse(source) if source else ast.parse("")
            normalized_source = ast.dump(
                tree, annotate_fields=True, include_attributes=False
            )
        except SyntaxError:
            tree = ast.parse("")
            normalized_source = " ".join(source.split())
            warnings.append("source_syntax_unsupported")
        source_hash = sha256(normalized_source.encode("utf-8")).hexdigest()
        dtype = self._dtype(job)
        graph_payload = {
            "batch_size": int(batch_size),
            "dtype": dtype,
            "normalized_source": normalized_source,
            "optimizer": job.config.runner_kwargs.get("optimizer")
            or job.metadata.get("optimizer"),
            "shape_hints": job.batch_probe.shape_hints,
            "gradient_accumulation": job.config.runner_kwargs.get(
                "gradient_accumulation_steps", 1
            ),
            "fallback_model_identity": (
                {
                    "architecture_key": job.workload_identity.architecture_key,
                    "architecture_family": job.workload_identity.architecture_family,
                    "baseline_model_id": job.baseline_model_id,
                    "runner_target": job.config.runner_target,
                }
                if not source
                else None
            ),
        }
        graph_hash = sha256(
            json.dumps(
                graph_payload, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        ).hexdigest()
        explicit = {
            **job.batch_probe.shape_hints,
            **job.metadata,
            **job.config.runner_kwargs,
        }
        cache_key = sha256(
            json.dumps(
                {
                    "graph_hash": graph_hash,
                    "epoch": float(predicted_epoch_seconds or 0.0),
                    "vram": int(predicted_vram_bytes or 0),
                    "steps": self._steps_per_epoch(job),
                    "explicit_flops": _number(
                        explicit, "training_step_flops", "step_flops"
                    ),
                    "explicit_bytes": _number(
                        explicit, "estimated_bytes_per_step", "bytes_per_step"
                    ),
                    "schema": SCHEMA_VERSION,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        call_names = [_call_name(call) for call in calls]
        categories = [normalize_operator_name(name) for name in call_names]
        histogram = Counter(categories)
        operator_count = sum(histogram.values())
        unknown_fraction = (
            histogram.get("unknown", 0) / operator_count if operator_count else 1.0
        )
        if not source:
            warnings.append("operator_graph_unavailable")
        elif unknown_fraction > 0:
            warnings.append("unknown_operators_present")

        explicit_sync_count = sum(
            "cuda.synchronize" in name.lower()
            or name.lower().endswith("device_synchronize")
            for name in call_names
        )
        blocking_transfer_count = 0
        async_transfer_count = 0
        dataloader_worker_count: int | None = None
        for call, name in zip(calls, call_names, strict=True):
            lowered = name.lower()
            leaf_name = lowered.rsplit(".", 1)[-1]
            non_blocking = any(
                keyword.arg == "non_blocking"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in call.keywords
            )
            if leaf_name in {"to", "cuda"} and non_blocking:
                async_transfer_count += 1
            elif leaf_name in {"cpu", "numpy", "item", "to", "cuda"}:
                blocking_transfer_count += 1
            if lowered.endswith("dataloader"):
                for keyword in call.keywords:
                    if keyword.arg == "num_workers":
                        dataloader_worker_count = _constant_int(keyword.value)

        dynamic_control_flow = any(
            isinstance(node, (ast.While, ast.AsyncFor)) for node in ast.walk(tree)
        ) or any(name.lower() in {"eval", "exec", "compile"} for name in call_names)
        custom_operation_flag = any(
            marker in name.lower()
            for name in call_names
            for marker in (
                "torch.ops",
                "load_inline",
                "cpp_extension",
                "triton",
                "custom_op",
                "flash_attn",
            )
        )
        if dynamic_control_flow:
            warnings.append("dynamic_control_flow")
        if custom_operation_flag:
            warnings.append("custom_or_fused_operation")

        checkpoint_count = sum(
            any(
                marker in name.lower()
                for marker in ("torch.save", "save_checkpoint", "checkpoint")
            )
            for name in call_names
        )
        evaluation_count = sum(
            any(
                marker in name.lower() for marker in ("validate", "evaluation", ".eval")
            )
            for name in call_names
        )
        cpu_augmentation = (
            any(
                any(
                    marker in name.lower()
                    for marker in ("transform", "augment", "albument", "opencv", "cv2")
                )
                for name in call_names
            )
            if source
            else None
        )

        count = max(1, operator_count)
        compute_categories = {"gemm", "convolution", "attention", "recurrent"}
        memory_categories = {
            "embedding",
            "data_movement",
            "pooling",
            "normalization",
            "reduction",
        }
        compute_fraction = (
            sum(histogram.get(key, 0) for key in compute_categories) / count
        )
        memory_fraction = (
            sum(histogram.get(key, 0) for key in memory_categories) / count
        )
        reduction_fraction = histogram.get("reduction", 0) / count
        irregular_fraction = histogram.get("embedding", 0) / count
        tensor_core_fraction = (
            sum(histogram.get(key, 0) for key in ("gemm", "convolution", "attention"))
            / count
            if dtype in {"float16", "bfloat16", "tf32", "float8"}
            else 0.0
        )
        small_fraction = (
            sum(
                histogram.get(key, 0)
                for key in ("activation", "normalization", "data_movement")
            )
            / count
        )
        largest_fraction = (
            max(histogram.values(), default=0) / count if operator_count else None
        )

        steps = self._steps_per_epoch(job)
        epoch_seconds = max(0.0, float(predicted_epoch_seconds or 0.0))
        step_seconds = epoch_seconds / steps if steps and epoch_seconds > 0 else None
        training_flops = _number(explicit, "training_step_flops", "step_flops")
        forward_flops = _number(explicit, "forward_flops")
        estimated_bytes = _number(
            explicit, "estimated_bytes_per_step", "bytes_per_step"
        )
        peak_tflops = self.peak_tflops_by_dtype.get(dtype)
        compute_pressure = (
            training_flops / (peak_tflops * 1e12 * step_seconds)
            if training_flops is not None and peak_tflops and step_seconds
            else None
        )
        memory_pressure = (
            estimated_bytes / (self.memory_bandwidth_gbps * 1e9 * step_seconds)
            if estimated_bytes is not None
            and self.memory_bandwidth_gbps
            and step_seconds
            else None
        )
        if compute_pressure is None:
            warnings.append("compute_pressure_unavailable")
        if memory_pressure is None:
            warnings.append("memory_pressure_unavailable")

        if compute_pressure is not None and memory_pressure is not None:
            if compute_pressure > memory_pressure * 1.25:
                resource_class = "compute_leaning"
            elif memory_pressure > compute_pressure * 1.25:
                resource_class = "memory_leaning"
            else:
                resource_class = "balanced"
        elif compute_fraction > memory_fraction * 1.25:
            resource_class = "compute_leaning"
        elif memory_fraction > compute_fraction * 1.25:
            resource_class = "memory_leaning"
        elif operator_count:
            resource_class = "balanced"
        else:
            resource_class = "unknown"

        missing_core = (
            steps is None or not job.batch_probe.shape_hints or dtype == "unknown"
        )
        if (
            unknown_fraction <= self.high_unknown_fraction
            and not missing_core
            and not dynamic_control_flow
            and not custom_operation_flag
        ):
            confidence = "HIGH"
        elif (
            unknown_fraction <= self.medium_unknown_fraction
            and source
            and not custom_operation_flag
        ):
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        sequence: list[str] = []
        if compute_fraction:
            sequence.append("compute")
        if reduction_fraction:
            sequence.append("reduction")
        if memory_fraction:
            sequence.append("memory")
        phase = PhaseFingerprint(tuple(sequence or ["unknown"]))
        fingerprint = StaticJobFingerprint(
            schema_version=SCHEMA_VERSION,
            source_hash=source_hash,
            graph_hash=graph_hash,
            batch_size=int(batch_size),
            dtype=dtype,
            steps_per_epoch=steps,
            predicted_epoch_seconds=epoch_seconds,
            predicted_vram_bytes=max(0, int(predicted_vram_bytes)),
            step_seconds=step_seconds,
            forward_flops=forward_flops,
            training_step_flops=training_flops,
            estimated_bytes_per_step=estimated_bytes,
            parameter_bytes=int(_number(explicit, "parameter_bytes") or 0) or None,
            activation_bytes=int(_number(explicit, "activation_bytes") or 0) or None,
            gradient_bytes=int(_number(explicit, "gradient_bytes") or 0) or None,
            optimizer_bytes=int(_number(explicit, "optimizer_bytes") or 0) or None,
            compute_pressure=compute_pressure,
            memory_pressure=memory_pressure,
            operator_count=operator_count,
            operator_histogram=MappingProxyType(dict(sorted(histogram.items()))),
            operator_flop_histogram=MappingProxyType({}),
            largest_op_fraction=largest_fraction,
            small_op_fraction=small_fraction if operator_count else None,
            tensor_core_eligible_fraction=(
                tensor_core_fraction if operator_count else None
            ),
            reduction_fraction=reduction_fraction if operator_count else None,
            irregular_memory_fraction=irregular_fraction if operator_count else None,
            explicit_sync_count=explicit_sync_count,
            blocking_transfer_count=blocking_transfer_count,
            async_transfer_count=async_transfer_count,
            dataloader_worker_count=dataloader_worker_count,
            cpu_augmentation_flag=cpu_augmentation,
            checkpoint_frequency=float(checkpoint_count) if checkpoint_count else None,
            evaluation_frequency=float(evaluation_count) if evaluation_count else None,
            forward_phase=phase,
            backward_phase=phase,
            optimizer_phase=(
                PhaseFingerprint(("optimizer_memory",))
                if histogram.get("optimizer", 0)
                else None
            ),
            unknown_operator_fraction=unknown_fraction,
            dynamic_control_flow=dynamic_control_flow,
            custom_operation_flag=custom_operation_flag,
            resource_class=resource_class,
            confidence=confidence,
            analysis_warnings=tuple(dict.fromkeys(warnings)),
        )
        if self.cache_enabled:
            self._cache[cache_key] = fingerprint
        return fingerprint
