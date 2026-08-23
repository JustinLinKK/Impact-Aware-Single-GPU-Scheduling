"""Backend-aware trial enumeration around the measured time objective.

This module only decides which unknown configuration should be measured next.
It does not accept a placement from static analysis.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

from ..config import SchedulerSettings
from ..domain import BatchResolution, TrainingJob, build_colocation_profile_key
from .candidate_generator import CandidateGenerator
from .compatibility import CompatibilityEvaluator
from .planner_types import DispatchPlan, EvaluatedGroup
from .planning_repository import PlanningRepository
from .resource_estimator import BatchOptionEstimate, ResourceEstimator
from .source_fingerprint import StaticJobAnalyzer, StaticJobFingerprint
from .time_objective import (
    EpochRateSet,
    TimeAwareObjectiveScorer,
    project_piecewise_drain,
)
from .trial_candidate import BackendTrialConfig, TrialCandidate
from .trial_priority import TrialPriorityPlanner


class BackendAwarePlacementPlanner:
    """Enumerate bounded pair/newcomer actions and Pareto-rank live trials."""

    def __init__(
        self,
        settings: SchedulerSettings,
        repository: PlanningRepository,
        estimator: ResourceEstimator,
        candidates: CandidateGenerator,
        compatibility: CompatibilityEvaluator,
        time_objective: TimeAwareObjectiveScorer,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.estimator = estimator
        self.candidates = candidates
        self.compatibility = compatibility
        self.time_objective = time_objective
        ranking = settings.gpu_scheduler.source_trial_ranking
        analysis = ranking.source_analysis
        self.analyzer = StaticJobAnalyzer(
            cache_enabled=analysis.cache_enabled,
            max_source_bytes=analysis.max_source_bytes,
            high_unknown_fraction=analysis.max_unknown_operator_fraction_for_high_confidence,
            medium_unknown_fraction=analysis.max_unknown_operator_fraction_for_medium_confidence,
            peak_tflops_by_dtype=analysis.peak_tflops_by_dtype,
            memory_bandwidth_gbps=analysis.memory_bandwidth_gbps,
        )
        self.priority = TrialPriorityPlanner()

    @property
    def config(self):
        return self.settings.gpu_scheduler.source_trial_ranking

    def _backend_configs(
        self,
        backend_name: str,
        *,
        active_config: dict[str, object] | None = None,
    ) -> tuple[BackendTrialConfig, ...]:
        return self.priority.backend_configs(
            backend_name,
            mps_templates=self.config.mps_allocation_templates,
            stream_offsets=self.config.stream_offset_templates_in_steps,
            active_config=active_config,
        )

    def _mode_overhead_mb(self, backend_name: str) -> float:
        return max(
            0.0,
            float(self.config.mode_overhead_mb.get(backend_name, 0.0)),
        )

    def _fingerprint(
        self,
        job: TrainingJob,
        option: BatchOptionEstimate,
    ) -> StaticJobFingerprint:
        return self.analyzer.analyze(
            job,
            option.batch_size,
            predicted_epoch_seconds=option.seconds_per_epoch,
            predicted_vram_bytes=int(option.avg_vram_mb * 1024 * 1024),
        )

    def _execution_signature(self, job: TrainingJob, batch_size: int) -> str:
        return self.analyzer.analyze(job, batch_size).execution_signature

    @staticmethod
    def _descriptors(
        jobs: tuple[TrainingJob, ...],
        fingerprints: tuple[StaticJobFingerprint, ...],
        backend_name: str,
        backend_config: BackendTrialConfig,
    ) -> list[dict[str, object]]:
        descriptors: list[dict[str, object]] = []
        for job, fingerprint in zip(jobs, fingerprints, strict=True):
            descriptors.append(
                TimeAwareObjectiveScorer.member_descriptor(
                    job,
                    backend_name=backend_name,
                    batch_size=fingerprint.batch_size,
                    backend_config=backend_config.to_dict(),
                    execution_signature=fingerprint.execution_signature,
                )
            )
        return descriptors

    def _profile_status(
        self,
        profile: object | None,
    ) -> str:
        if profile is None:
            return "unknown"
        if self.time_objective.profile_rejection_trusted(profile):
            return "bad"
        if self.time_objective.profile_rates_trusted(profile):
            return "good"
        return "unknown"

    def _profile_projection(
        self,
        candidate: TrialCandidate,
        solo_options: tuple[BatchOptionEstimate, ...],
    ) -> tuple[float | None, float | None]:
        profile = candidate.profile
        if profile is None:
            return None, None
        descriptors = self._descriptors(
            candidate.jobs,
            candidate.fingerprints,
            candidate.backend_name,
            candidate.backend_config,
        )
        packed_rates: dict[str, float] = {}
        solo_rates: dict[str, float] = {}
        remaining: dict[str, float] = {}
        for job, descriptor, solo in zip(
            candidate.jobs, descriptors, solo_options, strict=True
        ):
            rate = self.time_objective._profile_epoch_seconds(profile, descriptor)
            if rate is None or rate <= 0:
                return None, None
            packed_rates[job.job_id] = rate
            solo_rates[job.job_id] = solo.seconds_per_epoch
            remaining[job.job_id] = float(solo.remaining_epochs)

        def tails(member_ids: tuple[str, ...]) -> EpochRateSet | None:
            if len(member_ids) != 1:
                return None
            job_id = member_ids[0]
            return EpochRateSet(
                epoch_seconds={job_id: solo_rates[job_id]},
                sources={job_id: "exclusive_profile"},
            )

        projection = project_piecewise_drain(
            remaining,
            packed_rates,
            initial_sources={
                job_id: "exact_colocation_profile" for job_id in packed_rates
            },
            tail_rate_resolver=tails,
        )
        if projection is None:
            return None, None
        sequential = sum(solo.remaining_runtime_seconds for solo in solo_options)
        return projection.total_seconds, sequential

    def _trial_cost(self, fingerprints: tuple[StaticJobFingerprint, ...]) -> float:
        return (
            self.settings.gpu_scheduler.colocation.trial_epochs
            * max(
                (item.predicted_epoch_seconds for item in fingerprints),
                default=0.0,
            )
            + self.config.estimated_setup_seconds
        )

    def _candidate_record_table(
        self, ranked: list[TrialCandidate], selected: TrialCandidate
    ) -> list[dict[str, object]]:
        return [item.to_decision_record(selected=item is selected) for item in ranked]

    def choose_empty(
        self,
        jobs: list[TrainingJob],
        *,
        mandatory: TrainingJob | None,
        backend_available: dict[str, bool],
        effective_priority: Callable[[TrainingJob], int],
    ) -> DispatchPlan | None:
        if len(jobs) < 2:
            return None
        window = jobs[: self.config.ready_window_size]
        pair_candidates: list[TrialCandidate] = []
        generated = rejected_memory = rejected_amortization = 0
        for pair in combinations(window, 2):
            if mandatory is not None and mandatory not in pair:
                continue
            jobs_tuple = tuple(pair)
            for backend_name in self.settings.gpu_scheduler.backend_priority:
                if (
                    backend_name == "exclusive"
                    or not backend_available.get(backend_name, False)
                    or not self.compatibility.compatible_group(
                        list(jobs_tuple), backend_name=backend_name
                    )
                ):
                    continue
                option_sets: list[list[BatchOptionEstimate]] = []
                for job in jobs_tuple:
                    try:
                        batch_sizes = self.candidates.candidate_batch_sizes(job)
                    except ValueError:
                        option_sets = []
                        break
                    options = self.estimator.estimate_batch_options(
                        job, backend_name, batch_sizes
                    )
                    option_sets.append(options)
                if len(option_sets) != 2 or any(not options for options in option_sets):
                    continue
                for left in option_sets[0]:
                    for right in option_sets[1]:
                        backend_configs = self._backend_configs(backend_name)
                        generated += len(backend_configs)
                        used_mb = (
                            left.avg_vram_mb
                            + right.avg_vram_mb
                            + self._mode_overhead_mb(backend_name)
                        )
                        headroom_mb = self.estimator.safe_budget_mb() - used_mb
                        if headroom_mb < -1e-9:
                            rejected_memory += len(backend_configs)
                            continue
                        fingerprints = (
                            self._fingerprint(jobs_tuple[0], left),
                            self._fingerprint(jobs_tuple[1], right),
                        )
                        solo_left = self.estimator.estimate_batch_options(
                            jobs_tuple[0], "exclusive", [left.batch_size]
                        )
                        solo_right = self.estimator.estimate_batch_options(
                            jobs_tuple[1], "exclusive", [right.batch_size]
                        )
                        if not solo_left or not solo_right:
                            continue
                        solo_options = (solo_left[0], solo_right[0])
                        sequential = sum(
                            option.remaining_runtime_seconds for option in solo_options
                        )
                        ideal_packed = max(
                            option.remaining_runtime_seconds for option in solo_options
                        )
                        optimistic_gain = max(0.0, sequential - ideal_packed)
                        trial_cost = self._trial_cost(fingerprints)
                        for backend_config in backend_configs:
                            descriptors = self._descriptors(
                                jobs_tuple,
                                fingerprints,
                                backend_name,
                                backend_config,
                            )
                            profile_key = build_colocation_profile_key(
                                self.repository.hardware_key(), descriptors
                            )
                            profile = self.repository.get_colocation_timing_profile(
                                profile_key
                            )
                            status = self._profile_status(profile)
                            candidate = TrialCandidate(
                                jobs=jobs_tuple,
                                fingerprints=fingerprints,
                                batch_sizes=(left.batch_size, right.batch_size),
                                backend_name=backend_name,
                                backend_config=backend_config,
                                hardware_key=self.repository.hardware_key(),
                                predicted_vram_bytes=int(used_mb * 1024 * 1024),
                                vram_headroom_bytes=max(
                                    0, int(headroom_mb * 1024 * 1024)
                                ),
                                optimistic_makespan_gain_seconds=optimistic_gain,
                                estimated_trial_cost_seconds=trial_cost,
                                exact_profile_status=status,
                                priority_key=(
                                    -max(effective_priority(job) for job in jobs_tuple),
                                    min(job.queue_sequence for job in jobs_tuple),
                                    tuple(sorted(job.job_id for job in jobs_tuple)),
                                ),
                                profile_key=profile_key,
                                profile=profile,
                                extra={
                                    "solo_options": solo_options,
                                    "sequential_seconds": sequential,
                                    "ideal_packed_seconds": ideal_packed,
                                },
                            )
                            if status == "good":
                                packed, measured_sequential = self._profile_projection(
                                    candidate, solo_options
                                )
                                if packed is None or measured_sequential is None:
                                    candidate.exact_profile_status = "unknown"
                                else:
                                    candidate.extra["packed_seconds"] = packed
                                    candidate.extra["sequential_seconds"] = (
                                        measured_sequential
                                    )
                                    candidate.optimistic_makespan_gain_seconds = max(
                                        0.0, measured_sequential - packed
                                    )
                                    measured_gain = measured_sequential / max(
                                        packed, 1e-9
                                    )
                                    candidate.extra["gain"] = measured_gain
                                    if (
                                        measured_gain + 1e-9
                                        < self.settings.gpu_scheduler.colocation.min_gain
                                    ):
                                        candidate.exact_profile_status = "bad"
                            if (
                                candidate.exact_profile_status == "unknown"
                                and optimistic_gain + 1e-9
                                < self.config.amortization_factor * trial_cost
                            ):
                                rejected_amortization += 1
                                continue
                            pair_candidates.append(candidate)

        ranked = self.priority.rank(pair_candidates)
        if not ranked:
            return None
        selected = ranked[0]
        first, second = selected.jobs
        first_fingerprint, second_fingerprint = selected.fingerprints
        requires_trial = bool(
            selected.exact_profile_status == "unknown"
            and self.config.require_live_trial_for_unknown
            and self.settings.gpu_scheduler.colocation.live_trial_enabled
        )
        backend_config = selected.backend_config.to_dict()
        ranking_table = self._candidate_record_table(ranked, selected)
        sequential = float(selected.extra["sequential_seconds"])
        packed = selected.extra.get("packed_seconds")
        gain = selected.extra.get("gain")
        breakdown: dict[str, object] = {
            "score": float(gain if gain is not None else 1.0),
            "gain": gain,
            "gain_threshold": self.settings.gpu_scheduler.colocation.min_gain,
            "sequential_drain_seconds": sequential,
            "packed_drain_seconds": packed,
            "optimistic_packed_drain_seconds": selected.extra["ideal_packed_seconds"],
            "optimistic_makespan_gain_seconds": selected.optimistic_makespan_gain_seconds,
            "estimated_trial_cost_seconds": selected.estimated_trial_cost_seconds,
            "active_pretrial_epoch_seconds": {
                second.job_id: second_fingerprint.predicted_epoch_seconds
            },
            "candidate_solo_epoch_seconds": first_fingerprint.predicted_epoch_seconds,
            "candidate_remaining_epochs": self.time_objective.remaining_epochs(first),
            "candidate_memory_mb": first_fingerprint.predicted_vram_bytes
            / (1024 * 1024),
            "active_vram_mb": second_fingerprint.predicted_vram_bytes / (1024 * 1024),
            "colocation_profile_key": selected.profile_key,
            "known_profile": selected.profile is not None,
            "trusted_profile": selected.exact_profile_status == "good",
            "trusted_rejection_profile": selected.exact_profile_status == "bad",
            "colocation_rejected": False,
            "requires_live_trial": requires_trial,
            "preexisting_job_ids": [second.job_id],
            "backend_config": backend_config,
            "scheduler_decision_mode": "backend_awared",
            "source_trial_ranking": ranking_table,
            "candidate_combinations_generated": generated,
            "candidates_rejected_by_memory": rejected_memory,
            "candidates_rejected_by_amortization": rejected_amortization,
            "objective_version": self.settings.gpu_scheduler.objective.objective_version,
        }
        start_delay: dict[str, float] = {}
        if selected.backend_config.stream_offset_steps not in {None, 0.0}:
            start_delay[second.job_id] = float(
                selected.backend_config.stream_offset_steps
                * (first_fingerprint.step_seconds or 0.0)
            )
        trial_metadata = {
            "requires_live_trial": requires_trial,
            "candidate_job_id": first.job_id,
            "preexisting_job_ids": [second.job_id],
            "profile_key": selected.profile_key,
            "candidate_solo_epoch_seconds": first_fingerprint.predicted_epoch_seconds,
            "pretrial_epoch_seconds": {
                second.job_id: second_fingerprint.predicted_epoch_seconds
            },
            "start_delay_seconds_by_job": start_delay,
            "source_fingerprint_signatures": {
                job.job_id: fingerprint.execution_signature
                for job, fingerprint in zip(
                    selected.jobs, selected.fingerprints, strict=True
                )
            },
        }
        return DispatchPlan(
            mode="concurrent_group",
            backend_name=selected.backend_name,
            job_ids=(first.job_id, second.job_id),
            reason=(
                "exact backend-aware colocation profile reused"
                if selected.exact_profile_status == "good"
                else "backend-aware Pareto-ranked colocation trial"
            ),
            batch_overrides={
                first.job_id: selected.batch_sizes[0],
                second.job_id: selected.batch_sizes[1],
            },
            fallback_order=self.candidates.fallback_order(
                list(selected.jobs), selected.backend_name
            ),
            objective_breakdown=breakdown,
            trial_metadata=trial_metadata,
            backend_config=backend_config,
            mandatory_anchor_job_id=mandatory.job_id if mandatory else None,
            objective_version=self.settings.gpu_scheduler.objective.objective_version,
        )

    def choose_active(
        self,
        jobs: list[TrainingJob],
        *,
        active_jobs: list[TrainingJob],
        active_vram_mb: float,
        mandatory: TrainingJob | None,
        backend_candidates: list[str],
        effective_priority: Callable[[TrainingJob], int],
    ) -> DispatchPlan | None:
        if len(active_jobs) + 1 > self.config.max_group_size:
            return None
        candidates: list[TrialCandidate] = []
        rejected_memory = rejected_amortization = 0
        for job in jobs[: self.config.ready_window_size]:
            for backend_name in backend_candidates:
                active_configs = [
                    active.metadata.get("placement_backend_config")
                    for active in active_jobs
                    if isinstance(active.metadata.get("placement_backend_config"), dict)
                ]
                # MPS client percentages are immutable after context creation.
                if backend_name in {"mps", "mps_process"} and not active_configs:
                    continue
                active_config = dict(active_configs[0]) if active_configs else None
                for backend_config in self._backend_configs(
                    backend_name, active_config=active_config
                ):
                    evaluated = self.time_objective.evaluate_incremental(
                        job,
                        backend_name=backend_name,
                        active_jobs=active_jobs,
                        active_vram_mb=active_vram_mb,
                        mandatory_anchor=mandatory,
                        backend_config=backend_config.to_dict(),
                        execution_signature_for=self._execution_signature,
                    )
                    if evaluated is None:
                        continue
                    if bool(evaluated.score_breakdown.get("colocation_rejected")):
                        continue
                    used_mb = (
                        active_vram_mb
                        + evaluated.estimated_vram_mb
                        + self._mode_overhead_mb(backend_name)
                    )
                    headroom_mb = self.estimator.safe_budget_mb() - used_mb
                    if headroom_mb < -1e-9:
                        rejected_memory += 1
                        continue
                    candidate_option = evaluated.batch_estimates[job.job_id]
                    candidate_fingerprint = self._fingerprint(job, candidate_option)
                    active_fingerprints: list[StaticJobFingerprint] = []
                    for active in active_jobs:
                        batch = BatchResolution.resolved_batch_size(active)
                        options = self.estimator.estimate_batch_options(
                            active, backend_name, [batch]
                        )
                        rate, _ = self.time_objective.current_epoch_seconds(active)
                        if options:
                            active_option = options[0]
                        elif rate is not None:
                            remaining = self.time_objective.remaining_epochs(active)
                            active_option = BatchOptionEstimate(
                                job_id=active.job_id,
                                batch_size=batch,
                                avg_vram_mb=max(
                                    0.0, active_vram_mb / max(1, len(active_jobs))
                                ),
                                seconds_per_epoch=rate,
                                remaining_epochs=remaining or 0,
                                remaining_runtime_seconds=rate * (remaining or 0),
                                source="active_runtime",
                                confidence=None,
                                estimate_version=self.settings.gpu_scheduler.objective.objective_version,
                            )
                        else:
                            active_fingerprints = []
                            break
                        active_fingerprints.append(
                            self._fingerprint(active, active_option)
                        )
                    if len(active_fingerprints) != len(active_jobs):
                        continue
                    fingerprints = tuple([*active_fingerprints, candidate_fingerprint])
                    trusted = bool(evaluated.score_breakdown.get("trusted_profile"))
                    status = "good" if trusted else "unknown"
                    active_rates = {
                        str(key): float(value)
                        for key, value in dict(
                            evaluated.score_breakdown.get(
                                "active_pretrial_epoch_seconds", {}
                            )
                        ).items()
                    }
                    candidate_solo = float(
                        evaluated.score_breakdown["candidate_solo_epoch_seconds"]
                    )
                    packed_ideal = {
                        **active_rates,
                        job.job_id: candidate_solo,
                    }
                    ideal = self.time_objective.estimate_gain(
                        active_jobs,
                        job,
                        backend_name=backend_name,
                        active_epoch_seconds=active_rates,
                        candidate_solo_epoch_seconds=candidate_solo,
                        packed_epoch_seconds=packed_ideal,
                        candidate_batch_size=candidate_option.batch_size,
                    )
                    optimistic_gain = (
                        ideal.sequential_drain_seconds - ideal.packed_drain_seconds
                        if ideal is not None
                        else 0.0
                    )
                    trial_cost = self._trial_cost(fingerprints)
                    if (
                        status == "unknown"
                        and optimistic_gain + 1e-9
                        < self.config.amortization_factor * trial_cost
                    ):
                        rejected_amortization += 1
                        continue
                    trial_candidate = TrialCandidate(
                        jobs=tuple([*active_jobs, job]),
                        fingerprints=fingerprints,
                        batch_sizes=tuple(
                            [
                                *(
                                    BatchResolution.resolved_batch_size(active)
                                    for active in active_jobs
                                ),
                                candidate_option.batch_size,
                            ]
                        ),
                        backend_name=backend_name,
                        backend_config=backend_config,
                        hardware_key=self.repository.hardware_key(),
                        predicted_vram_bytes=int(used_mb * 1024 * 1024),
                        vram_headroom_bytes=max(0, int(headroom_mb * 1024 * 1024)),
                        optimistic_makespan_gain_seconds=max(0.0, optimistic_gain),
                        estimated_trial_cost_seconds=trial_cost,
                        exact_profile_status=status,
                        priority_key=(
                            -effective_priority(job),
                            job.queue_sequence,
                            job.job_id,
                        ),
                        profile_key=str(
                            evaluated.score_breakdown.get("colocation_profile_key")
                            or ""
                        ),
                        extra={"evaluated": evaluated, "candidate_job": job},
                    )
                    candidates.append(trial_candidate)

        ranked = self.priority.rank(candidates)
        if not ranked:
            return None
        selected = ranked[0]
        evaluated = selected.extra["evaluated"]
        if not isinstance(evaluated, EvaluatedGroup):
            return None
        candidate_job = selected.extra["candidate_job"]
        if not isinstance(candidate_job, TrainingJob):
            return None
        requires_trial = bool(
            selected.exact_profile_status == "unknown"
            and self.config.require_live_trial_for_unknown
            and self.settings.gpu_scheduler.colocation.live_trial_enabled
        )
        breakdown = dict(evaluated.score_breakdown)
        breakdown.update(
            {
                "requires_live_trial": requires_trial,
                "backend_config": selected.backend_config.to_dict(),
                "scheduler_decision_mode": "backend_awared",
                "optimistic_makespan_gain_seconds": selected.optimistic_makespan_gain_seconds,
                "estimated_trial_cost_seconds": selected.estimated_trial_cost_seconds,
                "source_trial_ranking": self._candidate_record_table(ranked, selected),
                "candidates_rejected_by_memory": rejected_memory,
                "candidates_rejected_by_amortization": rejected_amortization,
            }
        )
        trial_metadata = {
            "requires_live_trial": requires_trial,
            "candidate_job_id": candidate_job.job_id,
            "preexisting_job_ids": list(breakdown.get("preexisting_job_ids", [])),
            "profile_key": breakdown.get("colocation_profile_key"),
            "candidate_solo_epoch_seconds": breakdown.get(
                "candidate_solo_epoch_seconds"
            ),
            "pretrial_epoch_seconds": dict(
                breakdown.get("active_pretrial_epoch_seconds", {})
            ),
            "source_fingerprint_signatures": {
                job.job_id: fingerprint.execution_signature
                for job, fingerprint in zip(
                    selected.jobs, selected.fingerprints, strict=True
                )
            },
        }
        return DispatchPlan(
            mode="concurrent_group",
            backend_name=selected.backend_name,
            job_ids=(candidate_job.job_id,),
            reason=(
                "exact backend-aware colocation profile reused"
                if selected.exact_profile_status == "good"
                else "backend-aware Pareto-ranked newcomer trial"
            ),
            batch_overrides=evaluated.batch_overrides,
            fallback_order=evaluated.fallback_order,
            objective_breakdown=breakdown,
            trial_metadata=trial_metadata,
            backend_config=selected.backend_config.to_dict(),
            mandatory_anchor_job_id=evaluated.mandatory_anchor_job_id,
            objective_version=evaluated.objective_version,
        )
