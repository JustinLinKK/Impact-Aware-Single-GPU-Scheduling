"""Runtime state owned by SchedulerService.

These small data objects are kept separate from the scheduler loop so the
state that survives between scheduling decisions is easy to inspect. The
colocation and placement-replay records provide explicit serialization
boundaries because they are persisted across service restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain import WorkloadIdentity, utc_now
from .telemetry import GpuTelemetrySample


@dataclass(slots=True)
class ActiveRun:
    """A worker group currently contributing telemetry and profile evidence."""

    group_id: str
    mode: str
    backend_name: str
    job_ids: tuple[str, ...]
    opened_at: str = field(default_factory=utc_now)
    batch_overrides: dict[str, int] = field(default_factory=dict)
    fallback_order: list[str] = field(default_factory=list)
    hardware_key: str = ""
    group_signature: str = ""
    samples: list[GpuTelemetrySample] = field(default_factory=list)
    fallback_triggered: bool = False
    fallback_reason: str | None = None
    overlapped: bool = False
    objective_breakdown: dict[str, object] = field(default_factory=dict)
    objective_version: str | None = None
    mandatory_anchor_job_id: str | None = None


@dataclass(slots=True)
class ColocationTrialState:
    """Durable decision barrier for evaluating one newcomer in a live stack."""

    trial_id: str
    candidate_job_id: str
    preexisting_job_ids: tuple[str, ...]
    started_at: str
    start_epoch: int
    target_epoch: int
    backend_name: str
    profile_key: str
    candidate_solo_epoch_seconds: float
    pretrial_epoch_seconds: dict[str, float] = field(default_factory=dict)
    member_start_epochs: dict[str, int] = field(default_factory=dict)
    evidence_deadline_at: str = ""
    scheduler_decision_mode: str = "baseline"
    estimated_trial_cost_seconds: float = 0.0
    setup_cost_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Serialize ColocationTrialState for durable scheduler state."""
        return {
            "trial_id": self.trial_id,
            "candidate_job_id": self.candidate_job_id,
            "preexisting_job_ids": list(self.preexisting_job_ids),
            "started_at": self.started_at,
            "start_epoch": self.start_epoch,
            "target_epoch": self.target_epoch,
            "backend_name": self.backend_name,
            "profile_key": self.profile_key,
            "candidate_solo_epoch_seconds": self.candidate_solo_epoch_seconds,
            "pretrial_epoch_seconds": dict(self.pretrial_epoch_seconds),
            "member_start_epochs": dict(self.member_start_epochs),
            "evidence_deadline_at": self.evidence_deadline_at,
            "scheduler_decision_mode": self.scheduler_decision_mode,
            "estimated_trial_cost_seconds": self.estimated_trial_cost_seconds,
            "setup_cost_seconds": self.setup_cost_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ColocationTrialState":
        """Restore ColocationTrialState from durable scheduler state."""
        return cls(
            trial_id=str(payload["trial_id"]),
            candidate_job_id=str(payload["candidate_job_id"]),
            preexisting_job_ids=tuple(
                str(value) for value in payload.get("preexisting_job_ids", [])
            ),
            started_at=str(payload["started_at"]),
            start_epoch=int(payload["start_epoch"]),
            target_epoch=int(payload["target_epoch"]),
            backend_name=str(payload["backend_name"]),
            profile_key=str(payload["profile_key"]),
            candidate_solo_epoch_seconds=float(payload["candidate_solo_epoch_seconds"]),
            pretrial_epoch_seconds={
                str(key): float(value)
                for key, value in dict(
                    payload.get("pretrial_epoch_seconds", {})
                ).items()
            },
            member_start_epochs={
                str(key): int(value)
                for key, value in dict(payload.get("member_start_epochs", {})).items()
            },
            evidence_deadline_at=str(payload.get("evidence_deadline_at") or ""),
            scheduler_decision_mode=str(
                payload.get("scheduler_decision_mode") or "baseline"
            ),
            estimated_trial_cost_seconds=float(
                payload.get("estimated_trial_cost_seconds") or 0.0
            ),
            setup_cost_seconds=float(payload.get("setup_cost_seconds") or 0.0),
        )


@dataclass(frozen=True, slots=True)
class TrialEpochEvidence:
    """Fresh epoch samples collected for one member during a live trial."""

    seconds_per_epoch: float | None
    sample_count: int
    samples: tuple[float, ...] = ()


@dataclass(slots=True)
class ColocationStallState:
    """Block repeated rejected additions until the active membership changes."""

    preexisting_job_ids: tuple[str, ...]
    candidate_job_id: str
    profile_key: str
    reason: str
    started_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, object]:
        """Serialize ColocationStallState for durable scheduler state."""
        return {
            "preexisting_job_ids": list(self.preexisting_job_ids),
            "candidate_job_id": self.candidate_job_id,
            "profile_key": self.profile_key,
            "reason": self.reason,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ColocationStallState":
        """Restore ColocationStallState from durable scheduler state."""
        return cls(
            preexisting_job_ids=tuple(
                str(value) for value in payload.get("preexisting_job_ids", [])
            ),
            candidate_job_id=str(payload["candidate_job_id"]),
            profile_key=str(payload["profile_key"]),
            reason=str(payload["reason"]),
            started_at=str(payload.get("started_at") or utc_now()),
        )


@dataclass(frozen=True, slots=True)
class PlacementProfileSnapshot:
    """Stable predictor inputs used to compare placement observations."""

    batch_size: int
    total_training_seconds: float
    avg_vram_mb: float
    source: str
    confidence: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize PlacementProfileSnapshot for durable scheduler state."""
        return {
            "batch_size": self.batch_size,
            "total_training_seconds": self.total_training_seconds,
            "avg_vram_mb": self.avg_vram_mb,
            "source": self.source,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PlacementProfileSnapshot":
        """Restore PlacementProfileSnapshot from durable scheduler state."""
        confidence = payload.get("confidence")
        return cls(
            batch_size=int(payload["batch_size"]),
            total_training_seconds=float(payload["total_training_seconds"]),
            avg_vram_mb=float(payload["avg_vram_mb"]),
            source=str(payload.get("source") or "unknown"),
            confidence=float(confidence) if confidence is not None else None,
        )


@dataclass(slots=True)
class PlacementPatternObservation:
    """One verified placement for a repeatable workload identity."""

    identity: WorkloadIdentity
    hardware_key: str
    scheduler_mode: str
    target_width: int
    backend_name: str
    slot_profiles: list[PlacementProfileSnapshot]
    member_job_ids: tuple[str, ...]
    reason: str
    observed_at: str = field(default_factory=utc_now)

    @property
    def member_fingerprint(self) -> str:
        """Return an order-independent identity for the observed members."""
        return "|".join(sorted(self.member_job_ids))

    def to_dict(self) -> dict[str, object]:
        """Serialize PlacementPatternObservation for durable scheduler state."""
        return {
            "identity": self.identity.to_dict(),
            "hardware_key": self.hardware_key,
            "scheduler_mode": self.scheduler_mode,
            "target_width": self.target_width,
            "backend_name": self.backend_name,
            "slot_profiles": [profile.to_dict() for profile in self.slot_profiles],
            "member_job_ids": list(self.member_job_ids),
            "reason": self.reason,
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PlacementPatternObservation":
        """Restore PlacementPatternObservation from durable scheduler state."""
        return cls(
            identity=WorkloadIdentity.from_dict(dict(payload.get("identity") or {})),
            hardware_key=str(payload.get("hardware_key") or ""),
            scheduler_mode=str(payload.get("scheduler_mode") or ""),
            target_width=int(payload["target_width"]),
            backend_name=str(payload["backend_name"]),
            slot_profiles=[
                PlacementProfileSnapshot.from_dict(dict(item))
                for item in list(payload.get("slot_profiles") or [])
                if isinstance(item, dict)
            ],
            member_job_ids=tuple(
                str(item) for item in list(payload.get("member_job_ids") or [])
            ),
            reason=str(payload.get("reason") or "unknown"),
            observed_at=str(payload.get("observed_at") or utc_now()),
        )


@dataclass(slots=True)
class PlacementReplayTemplate:
    """Learned placement shape that may be replayed after validation."""

    identity: WorkloadIdentity
    hardware_key: str
    scheduler_mode: str
    target_width: int
    backend_name: str
    slot_profiles: list[PlacementProfileSnapshot]
    observation_count: int
    activated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, object]:
        """Serialize PlacementReplayTemplate for durable scheduler state."""
        return {
            "identity": self.identity.to_dict(),
            "hardware_key": self.hardware_key,
            "scheduler_mode": self.scheduler_mode,
            "target_width": self.target_width,
            "backend_name": self.backend_name,
            "slot_profiles": [profile.to_dict() for profile in self.slot_profiles],
            "observation_count": self.observation_count,
            "activated_at": self.activated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PlacementReplayTemplate":
        """Restore PlacementReplayTemplate from durable scheduler state."""
        return cls(
            identity=WorkloadIdentity.from_dict(dict(payload.get("identity") or {})),
            hardware_key=str(payload.get("hardware_key") or ""),
            scheduler_mode=str(payload.get("scheduler_mode") or ""),
            target_width=int(payload["target_width"]),
            backend_name=str(payload["backend_name"]),
            slot_profiles=[
                PlacementProfileSnapshot.from_dict(dict(item))
                for item in list(payload.get("slot_profiles") or [])
                if isinstance(item, dict)
            ],
            observation_count=int(payload.get("observation_count") or 0),
            activated_at=str(payload.get("activated_at") or utc_now()),
        )


@dataclass(slots=True)
class PlacementReplayState:
    """Persisted learning state and probe-suppression counters."""

    observations: list[PlacementPatternObservation] = field(default_factory=list)
    pending_observation: PlacementPatternObservation | None = None
    template: PlacementReplayTemplate | None = None
    suppressed_probes: int = 0
    suppressed_trials: int = 0
    suppressed_decisions: int = 0

    def to_dict(self) -> dict[str, object]:
        """Serialize PlacementReplayState for durable scheduler state."""
        return {
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
            "pending_observation": (
                self.pending_observation.to_dict() if self.pending_observation else None
            ),
            "template": self.template.to_dict() if self.template else None,
            "suppressed_probes": self.suppressed_probes,
            "suppressed_trials": self.suppressed_trials,
            "suppressed_decisions": self.suppressed_decisions,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PlacementReplayState":
        """Restore PlacementReplayState from durable scheduler state."""
        pending = payload.get("pending_observation")
        template = payload.get("template")
        return cls(
            observations=[
                PlacementPatternObservation.from_dict(dict(item))
                for item in list(payload.get("observations") or [])
                if isinstance(item, dict)
            ],
            pending_observation=(
                PlacementPatternObservation.from_dict(dict(pending))
                if isinstance(pending, dict)
                else None
            ),
            template=(
                PlacementReplayTemplate.from_dict(dict(template))
                if isinstance(template, dict)
                else None
            ),
            suppressed_probes=int(payload.get("suppressed_probes") or 0),
            suppressed_trials=int(payload.get("suppressed_trials") or 0),
            suppressed_decisions=int(payload.get("suppressed_decisions") or 0),
        )
