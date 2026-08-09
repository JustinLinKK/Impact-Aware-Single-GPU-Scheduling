"""Learn and replay stable placement decisions.

Repeated workloads often produce the same safe placement. This module
observes verified placements, waits for a configurable number of stable
observations, and then replays the learned width/backend/batch vector. Every
replay is revalidated against identity, predictor confidence, compatibility,
and the current VRAM budget before it reaches dispatch.
"""

from __future__ import annotations

from math import isfinite
from statistics import fmean, median

from ..config import SCHEDULER_MODE_PARALLEL_TIME_AWARE
from ..domain import BatchResolution, TrainingJob, WorkloadIdentity
from .planner_types import DispatchPlan
from .queue import RunnableJobQueue
from .service_state import (
    PlacementPatternObservation,
    PlacementProfileSnapshot,
    PlacementReplayTemplate,
)


class PlacementReplayMixin:
    """Own the placement-learning and replay state machine."""

    @staticmethod
    def _member_descriptor(
        job: TrainingJob, fallback_backend: str
    ) -> dict[str, object]:
        """Build the stable member description used in placement profiles."""
        return {
            "signature": job.packing.signature or job.job_id,
            "batch_size": BatchResolution.resolved_batch_size(job),
            "backend_name": str(
                job.metadata.get("placement_backend") or fallback_backend
            ),
        }

    def _workload_identity(self, job: TrainingJob) -> WorkloadIdentity:
        """Derive a replay-safe identity from explicit and inferred metadata."""
        metadata = dict(job.metadata or {})
        supplied = job.workload_identity
        identity = WorkloadIdentity(
            task_key=supplied.task_key or metadata.get("task_key") or job.workflow_id,
            dataset_key=supplied.dataset_key or metadata.get("dataset_key"),
            architecture_key=(
                supplied.architecture_key
                or metadata.get("architecture_key")
                or metadata.get("branch_name")
                or metadata.get("model_name")
                or job.packing.family
            ),
            architecture_family=(
                supplied.architecture_family
                or metadata.get("architecture_family")
                or metadata.get("architecture_type")
                or metadata.get("model_family")
                or job.packing.family
            ),
        )
        generic_architectures = {
            "unknown",
            "generic",
            "mlevolve-script",
            "mlevolve-candidate",
        }
        if identity.architecture_key in generic_architectures:
            identity.architecture_key = None
        if identity.architecture_family in generic_architectures:
            identity.architecture_family = None
        return identity

    @staticmethod
    def _identities_match(left: WorkloadIdentity, right: WorkloadIdentity) -> bool:
        """Compare canonical workload identities for exact replay eligibility."""
        return left.to_dict() == right.to_dict()

    @staticmethod
    def _replay_slot(job: TrainingJob, target_width: int | None = None) -> int | None:
        """Read and validate a job's assigned slot in a replay template."""
        raw_slot = job.metadata.get("placement_replay_slot")
        if raw_slot is None:
            return None
        try:
            slot = int(raw_slot)
        except (TypeError, ValueError):
            return None
        if slot < 0 or (target_width is not None and slot >= target_width):
            return None
        return slot

    def _placement_profile_snapshot(
        self,
        job: TrainingJob,
        *,
        backend_name: str,
        batch_size: int,
    ) -> PlacementProfileSnapshot | None:
        """Capture validated runtime and VRAM evidence for one slot."""
        total_epochs = job.max_epochs or job.config.max_epochs
        try:
            planned_epochs = int(total_epochs) if total_epochs is not None else 0
        except (TypeError, ValueError):
            return None
        if planned_epochs <= 0:
            return None
        options = self.planner.estimator.estimate_batch_options(
            job,
            backend_name,
            [max(1, int(batch_size))],
        )
        if not options:
            return None
        option = options[0]
        snapshot = PlacementProfileSnapshot(
            batch_size=option.batch_size,
            total_training_seconds=float(option.seconds_per_epoch) * planned_epochs,
            avg_vram_mb=float(option.avg_vram_mb),
            source=str(option.source),
            confidence=option.confidence,
        )
        if not (
            isfinite(snapshot.total_training_seconds)
            and snapshot.total_training_seconds > 0
            and isfinite(snapshot.avg_vram_mb)
            and snapshot.avg_vram_mb > 0
        ):
            return None
        source = snapshot.source.lower().replace("-", "_")
        if (
            "out_of_distribution" in source
            or source == "ood"
            or source.startswith("ood_")
            or source.endswith("_ood")
            or "invalid" in source
            or "missing" in source
        ):
            return None
        if (
            snapshot.confidence is None
            or snapshot.confidence
            < self.settings.gpu_scheduler.profiling.reuse_profile_if_confidence_ge
        ):
            return None
        return snapshot

    def _profile_change(
        self,
        reference: PlacementProfileSnapshot,
        candidate: PlacementProfileSnapshot,
    ) -> tuple[bool, dict[str, float]]:
        """Detect material runtime or memory drift from learned evidence."""
        runtime_ratio = max(
            reference.total_training_seconds, candidate.total_training_seconds
        ) / max(
            1e-9,
            min(reference.total_training_seconds, candidate.total_training_seconds),
        )
        vram_ratio = max(reference.avg_vram_mb, candidate.avg_vram_mb) / max(
            1e-9,
            min(reference.avg_vram_mb, candidate.avg_vram_mb),
        )
        settings = self.settings.gpu_scheduler.colocation.decision_replay
        significant = (
            runtime_ratio >= 1.0 + settings.training_time_change_fraction
            or vram_ratio >= 1.0 + settings.vram_change_fraction
        )
        return significant, {
            "training_time_ratio": runtime_ratio,
            "vram_ratio": vram_ratio,
        }

    def _build_pattern_observation(
        self,
        jobs: list[TrainingJob],
        *,
        target_width: int,
        backend_name: str,
        reason: str,
    ) -> PlacementPatternObservation | None:
        """Build a replay observation only from complete trusted profiles."""
        replay_settings = self.settings.gpu_scheduler.colocation.decision_replay
        if not replay_settings.enabled or not jobs or target_width < 1:
            return None
        ordered = sorted(
            jobs,
            key=lambda job: (
                (
                    self._replay_slot(job)
                    if self._replay_slot(job) is not None
                    else 1_000_000
                ),
                job.queue_sequence,
                job.job_id,
            ),
        )[:target_width]
        if len(ordered) != target_width:
            return None
        identity = self._workload_identity(ordered[0])
        if not identity.replay_eligible:
            return None
        if any(
            not self._identities_match(identity, self._workload_identity(job))
            for job in ordered[1:]
        ):
            return None
        stable_backend = "exclusive" if target_width == 1 else str(backend_name)
        profiles: list[PlacementProfileSnapshot] = []
        for job in ordered:
            profile = self._placement_profile_snapshot(
                job,
                backend_name=stable_backend,
                batch_size=BatchResolution.resolved_batch_size(job),
            )
            if profile is None:
                return None
            profiles.append(profile)
        return PlacementPatternObservation(
            identity=identity,
            hardware_key=self.store.hardware_key(),
            scheduler_mode=self.settings.gpu_scheduler.mode,
            target_width=target_width,
            backend_name=stable_backend,
            slot_profiles=profiles,
            member_job_ids=tuple(job.job_id for job in ordered),
            reason=reason,
        )

    def _observations_match(
        self,
        left: PlacementPatternObservation,
        right: PlacementPatternObservation,
    ) -> bool:
        """Check whether two observations describe the same stable placement."""
        if (
            not self._identities_match(left.identity, right.identity)
            or left.hardware_key != right.hardware_key
            or left.scheduler_mode != right.scheduler_mode
            or left.target_width != right.target_width
            or left.backend_name != right.backend_name
            or len(left.slot_profiles) != len(right.slot_profiles)
        ):
            return False
        return all(
            not self._profile_change(reference, candidate)[0]
            for reference, candidate in zip(
                left.slot_profiles, right.slot_profiles, strict=True
            )
        )

    def _activate_replay_template(self) -> None:
        """Promote repeated matching observations into a replay template."""
        observations = self._placement_replay.observations
        required = (
            self.settings.gpu_scheduler.colocation.decision_replay.min_stable_observations
        )
        if len(observations) < required:
            return
        selected = observations[-required:]
        first = selected[0]
        slot_profiles: list[PlacementProfileSnapshot] = []
        for index in range(first.target_width):
            samples = [observation.slot_profiles[index] for observation in selected]
            confidences = [
                sample.confidence for sample in samples if sample.confidence is not None
            ]
            slot_profiles.append(
                PlacementProfileSnapshot(
                    batch_size=max(
                        1, int(round(median(sample.batch_size for sample in samples)))
                    ),
                    total_training_seconds=float(
                        median(sample.total_training_seconds for sample in samples)
                    ),
                    avg_vram_mb=float(fmean(sample.avg_vram_mb for sample in samples)),
                    source="placement_replay_reference",
                    confidence=min(confidences) if confidences else None,
                )
            )
        self._placement_replay.template = PlacementReplayTemplate(
            identity=first.identity,
            hardware_key=first.hardware_key,
            scheduler_mode=first.scheduler_mode,
            target_width=first.target_width,
            backend_name=first.backend_name,
            slot_profiles=slot_profiles,
            observation_count=len(selected),
        )
        self._placement_replay.pending_observation = None
        self._persist_scheduler_decision_state()
        self.event_logger.emit(
            "placement_replay_activated",
            payload=self._placement_replay.template.to_dict(),
        )

    def _record_pattern_observation(
        self, observation: PlacementPatternObservation | None
    ) -> None:
        """Store a verified observation and reconsider template activation."""
        if observation is None or self._placement_replay.template is not None:
            return
        if any(
            previous.member_fingerprint == observation.member_fingerprint
            for previous in self._placement_replay.observations
        ):
            return
        if self._placement_replay.observations and not self._observations_match(
            self._placement_replay.observations[-1], observation
        ):
            self._placement_replay.observations = []
        self._placement_replay.observations.append(observation)
        required = (
            self.settings.gpu_scheduler.colocation.decision_replay.min_stable_observations
        )
        self._placement_replay.observations = self._placement_replay.observations[
            -required:
        ]
        self._persist_scheduler_decision_state()
        self.event_logger.emit(
            "placement_pattern_observed",
            payload={
                **observation.to_dict(),
                "observation_count": len(self._placement_replay.observations),
                "required_observations": required,
            },
        )
        self._activate_replay_template()

    def _stage_successful_pattern(
        self, jobs: list[TrainingJob], *, backend_name: str
    ) -> None:
        """Stage a successful live placement for post-run verification."""
        observation = self._build_pattern_observation(
            jobs,
            target_width=len(jobs),
            backend_name=backend_name,
            reason="verified_colocation_accepted",
        )
        if observation is None:
            return
        pending = self._placement_replay.pending_observation
        if pending is not None:
            pending_members = set(pending.member_job_ids)
            new_members = set(observation.member_job_ids)
            if (
                self._identities_match(pending.identity, observation.identity)
                and pending.backend_name == observation.backend_name
                and pending_members.issubset(new_members)
                and observation.target_width >= pending.target_width
            ):
                self._placement_replay.pending_observation = observation
            else:
                self._record_pattern_observation(pending)
                self._placement_replay.pending_observation = observation
        else:
            self._placement_replay.pending_observation = observation
        cap = self.settings.gpu_scheduler.parallel_job_cap
        if cap is not None and observation.target_width >= cap:
            self._placement_replay.pending_observation = None
            self._record_pattern_observation(observation)
        else:
            self._persist_scheduler_decision_state()

    def _finalize_pending_pattern(self) -> None:
        """Commit the staged placement after its run finishes cleanly."""
        pending = self._placement_replay.pending_observation
        if pending is None:
            return
        active_ids = set(self._supervisor_active_job_ids())
        if set(pending.member_job_ids).issubset(active_ids):
            return
        self._placement_replay.pending_observation = None
        self._record_pattern_observation(pending)

    def _clear_replay_job_metadata(self, job: TrainingJob) -> None:
        """Remove transient replay assignments from persisted jobs."""
        self.store.update_job(
            job.job_id,
            metadata_updates={
                "placement_replay": False,
                "placement_replay_slot": None,
                "placement_replay_target_width": None,
                "skip_active_scheduler_probes": False,
            },
        )

    def _invalidate_placement_replay(
        self,
        *,
        reason: str,
        job: TrainingJob | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Discard an unsafe template and explain why replay stopped."""
        previous = self._placement_replay.template
        had_learning_state = bool(
            self._placement_replay.observations
            or self._placement_replay.pending_observation
        )
        if previous is None and not had_learning_state:
            return
        self._placement_replay.template = None
        self._placement_replay.observations = []
        self._placement_replay.pending_observation = None
        if job is not None and job.metadata.get("skip_active_scheduler_probes"):
            self._clear_replay_job_metadata(job)
        self._persist_scheduler_decision_state()
        self.event_logger.emit(
            "placement_replay_invalidated",
            job_id=job.job_id if job is not None else None,
            payload={
                "reason": reason,
                "previous_template": previous.to_dict() if previous else None,
                **(details or {}),
            },
        )

    def _choose_placement_replay_plan(
        self,
        runnable: list[TrainingJob],
        *,
        active_jobs: list[TrainingJob],
        backend_available: dict[str, bool],
    ) -> tuple[bool, DispatchPlan | None]:
        """Return a validated learned plan, or release control to the planner.

        The boolean distinguishes "replay owns this decision but must wait"
        from "no usable replay exists; run normal planning".
        """
        if not self.settings.gpu_scheduler.colocation.decision_replay.enabled:
            return False, None
        template = self._placement_replay.template
        ordered = RunnableJobQueue(policy=self.policy, jobs=runnable).ordered()
        if not ordered:
            return (template is not None), None
        candidate = ordered[0]
        candidate_identity = self._workload_identity(candidate)
        if template is None:
            learning_reference = self._placement_replay.pending_observation or (
                self._placement_replay.observations[-1]
                if self._placement_replay.observations
                else None
            )
            if learning_reference is not None and (
                not self._identities_match(
                    learning_reference.identity, candidate_identity
                )
                or learning_reference.hardware_key != self.store.hardware_key()
                or learning_reference.scheduler_mode != self.settings.gpu_scheduler.mode
            ):
                self._invalidate_placement_replay(
                    reason="placement learning scope changed",
                    job=candidate,
                    details={"new_identity": candidate_identity.to_dict()},
                )
            return False, None
        if (
            template.hardware_key != self.store.hardware_key()
            or template.scheduler_mode != self.settings.gpu_scheduler.mode
        ):
            self._invalidate_placement_replay(
                reason="hardware or scheduler mode changed",
                job=candidate,
                details={
                    "hardware_key": self.store.hardware_key(),
                    "scheduler_mode": self.settings.gpu_scheduler.mode,
                },
            )
            return False, None
        if not self._identities_match(template.identity, candidate_identity):
            self._invalidate_placement_replay(
                reason="workload identity changed",
                job=candidate,
                details={"new_identity": candidate_identity.to_dict()},
            )
            return False, None

        if (
            template.target_width < 1
            or len(template.slot_profiles) != template.target_width
        ):
            self._invalidate_placement_replay(
                reason="invalid cached template", job=candidate
            )
            return False, None
        configured_cap = self.settings.gpu_scheduler.parallel_job_cap
        if configured_cap is not None and template.target_width > configured_cap:
            self._invalidate_placement_replay(
                reason="parallel cap changed", job=candidate
            )
            return False, None
        if active_jobs and any(
            not self._identities_match(template.identity, self._workload_identity(job))
            for job in active_jobs
        ):
            return True, None
        if active_jobs and any(
            str(job.metadata.get("placement_backend") or "exclusive")
            != template.backend_name
            for job in active_jobs
        ):
            return True, None
        if len(active_jobs) >= template.target_width:
            return True, None
        if template.target_width > 1 and not self._admission_gate.is_open:
            return True, None
        if not backend_available.get(
            template.backend_name, template.backend_name == "exclusive"
        ):
            self._invalidate_placement_replay(
                reason="cached backend is unavailable",
                job=candidate,
                details={"backend_name": template.backend_name},
            )
            return False, None

        assigned_slots = [
            self._replay_slot(job, template.target_width) for job in active_jobs
        ]
        occupied_slots = {slot for slot in assigned_slots if slot is not None}
        unassigned_members = sum(1 for slot in assigned_slots if slot is None)
        for fallback_slot in (
            index
            for index in range(template.target_width)
            if index not in occupied_slots
        ):
            if unassigned_members <= 0:
                break
            occupied_slots.add(fallback_slot)
            unassigned_members -= 1
        slot_index = next(
            (
                index
                for index in range(template.target_width)
                if index not in occupied_slots
            ),
            None,
        )
        if slot_index is None:
            return True, None
        reference = template.slot_profiles[slot_index]
        try:
            supported_batches = self.planner.candidate_generator.candidate_batch_sizes(
                candidate,
                scheduler_mode=SCHEDULER_MODE_PARALLEL_TIME_AWARE,
            )
        except ValueError:
            supported_batches = []
        if reference.batch_size not in supported_batches:
            self._invalidate_placement_replay(
                reason="cached batch size is unsupported",
                job=candidate,
                details={"slot": slot_index, "batch_size": reference.batch_size},
            )
            return False, None
        profile = self._placement_profile_snapshot(
            candidate,
            backend_name=template.backend_name,
            batch_size=reference.batch_size,
        )
        if profile is None:
            self._invalidate_placement_replay(
                reason="trusted predictor profile unavailable", job=candidate
            )
            return False, None
        significant, ratios = self._profile_change(reference, profile)
        if significant:
            self._invalidate_placement_replay(
                reason="training-time or VRAM profile changed",
                job=candidate,
                details={"slot": slot_index, **ratios},
            )
            return False, None

        if template.target_width > 1:
            if not self.planner.compatibility.pack_eligible(
                candidate, backend_name=template.backend_name
            ):
                self._invalidate_placement_replay(
                    reason="candidate is not pack eligible", job=candidate
                )
                return False, None
            if not self.planner.compatibility.compatible_group(
                [*active_jobs, candidate],
                backend_name=template.backend_name,
            ):
                self._invalidate_placement_replay(
                    reason="known incompatibility or cooldown", job=candidate
                )
                return False, None
        predicted_vram_mb = profile.avg_vram_mb
        for active_index, active_job in enumerate(active_jobs):
            active_slot = self._replay_slot(active_job, template.target_width)
            if active_slot is None:
                active_slot = active_index
            active_slot = min(max(0, active_slot), template.target_width - 1)
            active_profile = self._placement_profile_snapshot(
                active_job,
                backend_name=template.backend_name,
                batch_size=template.slot_profiles[active_slot].batch_size,
            )
            if active_profile is None:
                self._invalidate_placement_replay(
                    reason="active member profile unavailable", job=candidate
                )
                return False, None
            predicted_vram_mb += active_profile.avg_vram_mb
        if predicted_vram_mb > self.planner.estimator.safe_budget_mb() + 1e-9:
            self._invalidate_placement_replay(
                reason="predicted aggregate VRAM exceeds budget",
                job=candidate,
                details={"predicted_vram_mb": predicted_vram_mb},
            )
            return False, None

        return True, DispatchPlan(
            mode=(
                "exclusive"
                if template.target_width == 1
                else "concurrent_group" if active_jobs else "stack_anchor"
            ),
            backend_name=template.backend_name,
            job_ids=(candidate.job_id,),
            reason=f"replayed stable placement width {template.target_width}",
            batch_overrides={candidate.job_id: reference.batch_size},
            objective_breakdown={
                "placement_replay": True,
                "placement_replay_slot": slot_index,
                "placement_replay_target_width": template.target_width,
                "profile_ratios": ratios,
                "predicted_group_vram_mb": predicted_vram_mb,
                "requires_live_trial": False,
            },
            objective_version=self.settings.gpu_scheduler.objective.objective_version,
        )
