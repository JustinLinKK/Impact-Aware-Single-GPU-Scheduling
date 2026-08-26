"""Deterministic backend-specific risk vectors for colocation trials."""

from __future__ import annotations

from itertools import combinations
from types import MappingProxyType

from ..backend_mode import normalize_packing_backend
from .source_fingerprint import StaticJobFingerprint
from .trial_candidate import BackendTrialConfig, CompatibilityAssessment

_COMPUTE_CATEGORIES = {"gemm", "convolution", "attention", "recurrent"}
_MEMORY_CATEGORIES = {
    "embedding",
    "data_movement",
    "normalization",
    "pooling",
    "reduction",
}


def _operator_fraction(
    fingerprint: StaticJobFingerprint, categories: set[str]
) -> float:
    count = max(1, fingerprint.operator_count)
    return sum(fingerprint.operator_histogram.get(key, 0) for key in categories) / count


def _compute(fingerprint: StaticJobFingerprint) -> float:
    if fingerprint.compute_pressure is not None:
        return max(0.0, fingerprint.compute_pressure)
    return _operator_fraction(fingerprint, _COMPUTE_CATEGORIES)


def _memory(fingerprint: StaticJobFingerprint) -> float:
    if fingerprint.memory_pressure is not None:
        return max(0.0, fingerprint.memory_pressure)
    return _operator_fraction(fingerprint, _MEMORY_CATEGORIES)


def _large(fingerprint: StaticJobFingerprint) -> float:
    return max(0.0, float(fingerprint.largest_op_fraction or 0.0))


def _uncertainty(fingerprints: tuple[StaticJobFingerprint, ...]) -> float:
    return max((item.analysis_uncertainty for item in fingerprints), default=1.0)


def _confidence(fingerprints: tuple[StaticJobFingerprint, ...]) -> str:
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    return max(
        (item.confidence for item in fingerprints),
        key=lambda item: order.get(item, 2),
        default="LOW",
    )


def _pairwise_product(values: list[float]) -> float:
    return sum(left * right for left, right in combinations(values, 2))


class BackendCompatibilityPolicy:
    def evaluate(
        self,
        fingerprints: tuple[StaticJobFingerprint, ...],
        *,
        backend_name: str,
        backend_config: BackendTrialConfig,
    ) -> CompatibilityAssessment:
        normalized = normalize_packing_backend(backend_name, warn_legacy=False)
        if normalized == "mps_process":
            return self._mps(fingerprints, normalized, backend_config)
        return self._process(fingerprints, normalized, backend_config)

    def _mps(
        self,
        fingerprints: tuple[StaticJobFingerprint, ...],
        backend_name: str,
        backend_config: BackendTrialConfig,
    ) -> CompatibilityAssessment:
        computes = [_compute(item) for item in fingerprints]
        memories = [_memory(item) for item in fingerprints]
        larges = [_large(item) for item in fingerprints]
        irregular = [
            float(item.irregular_memory_fraction or 0.0) for item in fingerprints
        ]
        risks = {
            "compute_excess": max(0.0, sum(computes) - 1.0),
            "bandwidth_excess": max(0.0, sum(memories) - 1.0),
            "same_resource_conflict": _pairwise_product(computes)
            + _pairwise_product(memories),
            "large_operation_conflict": _pairwise_product(larges),
            "irregular_memory_conflict": _pairwise_product(irregular),
            "analysis_uncertainty": _uncertainty(fingerprints),
        }
        reasons: list[str] = []
        if risks["compute_excess"] > 0:
            reasons.append("MPS_COMPUTE_EXCESS")
        if risks["bandwidth_excess"] > 0:
            reasons.append("MPS_BANDWIDTH_EXCESS")
        if len(fingerprints) == 2 and {
            fingerprints[0].resource_class,
            fingerprints[1].resource_class,
        } == {"compute_leaning", "memory_leaning"}:
            reasons.append("MPS_COMPUTE_MEMORY_COMPLEMENT")
        if risks["large_operation_conflict"] >= 0.25:
            reasons.append("MPS_BOTH_LARGE_OP_DOMINATED")
        if sum(computes) < 1 and sum(memories) < 1:
            reasons.append("MPS_BOTH_UNDERFILLED_PROXY")
        if risks["irregular_memory_conflict"] > 0:
            reasons.append("MPS_IRREGULAR_MEMORY_CONFLICT")
        allocations = backend_config.allocation_percentages
        if allocations:
            reasons.append("MPS_ALLOCATION_" + "_".join(map(str, allocations)))
        if _confidence(fingerprints) == "LOW":
            reasons.append("MPS_ANALYSIS_LOW_CONFIDENCE")
        rejection = ()
        if allocations and len(allocations) != len(fingerprints):
            rejection = ("MPS_ALLOCATION_MEMBER_MISMATCH",)
        return CompatibilityAssessment(
            backend_name=backend_name,
            hard_rejection_reasons=rejection,
            risk_components=MappingProxyType(risks),
            reason_codes=tuple(reasons),
            analysis_confidence=_confidence(fingerprints),
            analysis_uncertainty=_uncertainty(fingerprints),
        )

    def _process(
        self,
        fingerprints: tuple[StaticJobFingerprint, ...],
        backend_name: str,
        backend_config: BackendTrialConfig,
    ) -> CompatibilityAssessment:
        pressures = [max(_compute(item), _memory(item)) for item in fingerprints]
        larges = [_large(item) for item in fingerprints]
        host_phase = [
            float(bool(item.cpu_augmentation_flag))
            + float(item.checkpoint_frequency or 0.0)
            + float(item.evaluation_frequency or 0.0)
            for item in fingerprints
        ]
        sync = [
            float(item.explicit_sync_count + item.blocking_transfer_count)
            / max(1, item.operator_count)
            for item in fingerprints
        ]
        risks = {
            "continuous_gpu_conflict": _pairwise_product(pressures),
            "large_operation_conflict": _pairwise_product(larges),
            "host_gap_alignment": _pairwise_product(host_phase),
            "context_memory_pressure": sum(
                item.predicted_vram_bytes for item in fingerprints
            )
            / max(1.0, 1024**3),
            "synchronization_pressure": sum(sync),
            "analysis_uncertainty": _uncertainty(fingerprints),
        }
        reasons: list[str] = []
        if any(host_phase) and any(value == 0 for value in host_phase):
            reasons.append("PROCESS_CPU_GPU_COMPLEMENT")
        if any(item.dataloader_worker_count for item in fingerprints):
            reasons.append("PROCESS_DATALOADER_GAP_OPPORTUNITY")
        if any(item.checkpoint_frequency for item in fingerprints):
            reasons.append("PROCESS_CHECKPOINT_PHASE_OPPORTUNITY")
        if sum(pressures) < 1:
            reasons.append("PROCESS_SHORT_BURST_PAIR")
        if risks["continuous_gpu_conflict"] >= 0.25:
            reasons.append("PROCESS_BOTH_CONTINUOUS_GPU")
        if risks["large_operation_conflict"] >= 0.25:
            reasons.append("PROCESS_LONG_KERNEL_CONFLICT")
        if risks["host_gap_alignment"] > 0:
            reasons.append("PROCESS_HOST_PHASE_ALIGNMENT")
        if _confidence(fingerprints) == "LOW":
            reasons.append("PROCESS_ANALYSIS_LOW_CONFIDENCE")
        return CompatibilityAssessment(
            backend_name=backend_name,
            risk_components=MappingProxyType(risks),
            reason_codes=tuple(reasons),
            analysis_confidence=_confidence(fingerprints),
            analysis_uncertainty=_uncertainty(fingerprints),
        )
