"""Backend-aware trial candidate and compatibility value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping

from ..domain import TrainingJob
from .source_fingerprint import StaticJobFingerprint


@dataclass(frozen=True, slots=True)
class BackendTrialConfig:
    allocation_percentages: tuple[int, ...] = ()
    stream_offset_steps: float | None = None
    mps_clients: int | None = None
    streams_per_client: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "allocation_percentages": list(self.allocation_percentages),
            "stream_offset_steps": self.stream_offset_steps,
            "mps_clients": self.mps_clients,
            "streams_per_client": self.streams_per_client,
        }

    @property
    def stable_key(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class CompatibilityAssessment:
    backend_name: str
    hard_rejection_reasons: tuple[str, ...] = ()
    risk_components: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    reason_codes: tuple[str, ...] = ()
    analysis_confidence: str = "LOW"
    analysis_uncertainty: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "backend_name": self.backend_name,
            "hard_rejection_reasons": list(self.hard_rejection_reasons),
            "risk_components": dict(self.risk_components),
            "reason_codes": list(self.reason_codes),
            "analysis_confidence": self.analysis_confidence,
            "analysis_uncertainty": self.analysis_uncertainty,
        }


@dataclass(slots=True)
class TrialCandidate:
    jobs: tuple[TrainingJob, ...]
    fingerprints: tuple[StaticJobFingerprint, ...]
    batch_sizes: tuple[int, ...]
    backend_name: str
    backend_config: BackendTrialConfig
    hardware_key: str
    predicted_vram_bytes: int
    vram_headroom_bytes: int
    optimistic_makespan_gain_seconds: float
    estimated_trial_cost_seconds: float
    exact_profile_status: str = "unknown"
    compatibility: CompatibilityAssessment | None = None
    priority_key: tuple[object, ...] = ()
    profile_key: str | None = None
    profile: object | None = None
    pareto_front: int = -1
    final_rank: int = -1
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def stable_candidate_id(self) -> str:
        members = sorted(
            (
                {
                    "batch_size": fingerprint.batch_size,
                    "dtype": fingerprint.dtype,
                    "graph_hash": fingerprint.graph_hash,
                    "job_id": job.job_id,
                    "source_hash": fingerprint.source_hash,
                }
                for job, fingerprint in zip(self.jobs, self.fingerprints, strict=True)
            ),
            key=lambda item: str(item["job_id"]),
        )
        payload = {
            "backend_config": self.backend_config.to_dict(),
            "backend_name": self.backend_name,
            "hardware_key": self.hardware_key,
            "members": members,
            "schema_version": 1,
        }
        digest = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        return f"trial:{self.backend_name}:{digest}"

    @property
    def uncertainty(self) -> float:
        if self.compatibility is not None:
            return self.compatibility.analysis_uncertainty
        return max(
            (fingerprint.analysis_uncertainty for fingerprint in self.fingerprints),
            default=1.0,
        )

    def to_decision_record(self, *, selected: bool = False) -> dict[str, object]:
        return {
            "candidate_id": self.stable_candidate_id,
            "job_ids": [job.job_id for job in self.jobs],
            "batch_sizes": list(self.batch_sizes),
            "backend": self.backend_name,
            "backend_config": self.backend_config.to_dict(),
            "predicted_vram_bytes": self.predicted_vram_bytes,
            "vram_headroom_bytes": self.vram_headroom_bytes,
            "optimistic_makespan_gain_s": self.optimistic_makespan_gain_seconds,
            "estimated_trial_cost_s": self.estimated_trial_cost_seconds,
            "exact_profile_status": self.exact_profile_status,
            "pareto_front": self.pareto_front,
            "risks": (
                dict(self.compatibility.risk_components)
                if self.compatibility is not None
                else {}
            ),
            "confidence": (
                self.compatibility.analysis_confidence
                if self.compatibility is not None
                else "LOW"
            ),
            "reason_codes": (
                list(self.compatibility.reason_codes)
                if self.compatibility is not None
                else []
            ),
            "analysis_warnings": sorted(
                {
                    warning
                    for fingerprint in self.fingerprints
                    for warning in fingerprint.analysis_warnings
                }
            ),
            "final_rank": self.final_rank,
            "selected": selected,
        }
