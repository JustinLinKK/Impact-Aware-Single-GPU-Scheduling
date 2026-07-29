"""Rolling-horizon completion-time objective for time-aware packing."""

from __future__ import annotations

from itertools import product

from ..config import SchedulerSettings
from ..domain import TrainingJob
from .candidate_generator import CandidateGenerator
from .compatibility import CompatibilityEvaluator
from .planner_types import EvaluatedGroup
from .resource_estimator import BatchOptionEstimate, ResourceEstimator


class TimeAwareObjectiveScorer:
    """Minimize normalized, priority-weighted predicted flow time."""

    def __init__(
        self,
        settings: SchedulerSettings,
        estimator: ResourceEstimator,
        compatibility: CompatibilityEvaluator,
        candidates: CandidateGenerator,
    ) -> None:
        self.settings = settings
        self.estimator = estimator
        self.compatibility = compatibility
        self.candidates = candidates

    def evaluate(
        self,
        jobs: list[TrainingJob],
        *,
        backend_name: str,
        planning_window: list[TrainingJob],
        weights: dict[str, float],
        exclusive_flow_cost: float,
        active_vram_mb: float,
        active_jobs: list[TrainingJob],
        mandatory_anchor: TrainingJob | None,
    ) -> EvaluatedGroup | None:
        combined = [*active_jobs, *jobs]
        if not self.compatibility.compatible_time_aware_group(combined, backend_name=backend_name):
            return None

        active_solo_seconds: dict[str, float] = {}
        active_estimates: dict[str, dict[str, object]] = {}
        for active_job in active_jobs:
            fixed_batch_size = self.estimator.resolved_batch_size(active_job)
            fixed_options = self.estimator.estimate_batch_options(active_job, backend_name, [fixed_batch_size])
            if fixed_options:
                fixed = fixed_options[0]
                remaining_seconds = fixed.remaining_runtime_seconds
                source = fixed.source
                confidence = fixed.confidence
            else:
                fallback = self.estimator.predicted_remaining_runtime_seconds(active_job, backend_name=backend_name)
                if fallback is None:
                    return None
                remaining_seconds = fallback
                source = "runtime_profile"
                confidence = None
            if remaining_seconds <= 0:
                return None
            active_solo_seconds[active_job.job_id] = float(remaining_seconds)
            active_estimates[active_job.job_id] = {
                "batch_size": fixed_batch_size,
                "remaining_runtime_seconds": float(remaining_seconds),
                "source": source,
                "confidence": confidence,
            }

        options_by_job: list[list[BatchOptionEstimate]] = []
        audit_options: dict[str, dict[str, object]] = {}
        for job in jobs:
            try:
                proposals = self.candidates.time_aware_batch_proposals(job)
                batch_sizes = self.candidates.candidate_batch_sizes(
                    job,
                    scheduler_mode=self.settings.gpu_scheduler.mode,
                )
            except ValueError:
                return None
            all_options = self.estimator.estimate_batch_options(job, backend_name, batch_sizes)
            audit_options[job.job_id] = {
                "proposed_batch_sizes": proposals,
                "clipped_batch_sizes": batch_sizes,
                "estimates": [
                    {
                        "batch_size": option.batch_size,
                        "avg_vram_mb": option.avg_vram_mb,
                        "seconds_per_epoch": option.seconds_per_epoch,
                        "remaining_runtime_seconds": option.remaining_runtime_seconds,
                        "source": option.source,
                        "confidence": option.confidence,
                    }
                    for option in all_options
                ],
            }
            pruned = self.estimator.pareto_prune(all_options)
            if not pruned:
                return None
            options_by_job.append(pruned)

        best: EvaluatedGroup | None = None
        for option_vector in self._option_vectors(options_by_job):
            candidate_memory = sum(option.avg_vram_mb for option in option_vector)
            if active_vram_mb + candidate_memory > self.estimator.safe_budget_mb() + 1e-9:
                continue
            completion_offsets = dict(active_solo_seconds)
            completion_offsets.update(
                {option.job_id: option.remaining_runtime_seconds for option in option_vector}
            )
            drain_time = max(completion_offsets.values(), default=0.0)
            selected_ids = set(completion_offsets)
            flow_cost = sum(weights.get(job_id, 1.0) * duration for job_id, duration in completion_offsets.items())
            flow_cost += drain_time * sum(weights[job.job_id] for job in planning_window if job.job_id not in selected_ids)
            normalized_flow = flow_cost / max(1e-9, exclusive_flow_cost)
            score = normalized_flow
            selected = {option.job_id: option for option in option_vector}
            breakdown: dict[str, object] = {
                "score": score,
                "flow_cost": flow_cost,
                "exclusive_flow_cost": exclusive_flow_cost,
                "normalized_flow_cost": normalized_flow,
                "predicted_drain_seconds": drain_time,
                "completion_offsets_seconds": completion_offsets,
                "slowdown_prediction": "disabled",
                "active_fixed_estimates": active_estimates,
                "candidate_estimates": audit_options,
                "candidate_memory_mb": candidate_memory,
                "active_vram_mb": active_vram_mb,
                "objective_version": self.settings.gpu_scheduler.objective.objective_version,
            }
            evaluated = EvaluatedGroup(
                jobs=jobs,
                backend_name=backend_name,
                estimated_vram_mb=candidate_memory,
                estimated_sm_utilization=0.0,
                objective_score=score,
                batch_overrides={job_id: estimate.batch_size for job_id, estimate in selected.items()},
                fallback_order=self.candidates.fallback_order(
                    jobs,
                    {job_id: estimate.batch_size for job_id, estimate in selected.items()},
                    backend_name,
                ),
                reason="minimum rolling-horizon completion-time score",
                batch_estimates=selected,
                score_breakdown=breakdown,
                mandatory_anchor_job_id=(mandatory_anchor.job_id if mandatory_anchor else None),
                objective_version=self.settings.gpu_scheduler.objective.objective_version,
            )
            if best is None or self.tie_key(evaluated) < self.tie_key(best):
                best = evaluated
        return best

    def _option_vectors(self, options_by_job: list[list[BatchOptionEstimate]]) -> list[tuple[BatchOptionEstimate, ...]]:
        if len(options_by_job) <= self.settings.gpu_scheduler.exact_search_max_jobs:
            return list(product(*options_by_job))
        states: list[tuple[BatchOptionEstimate, ...]] = [tuple()]
        for options in options_by_job:
            expanded = [state + (option,) for state in states for option in options]
            states = sorted(
                expanded,
                key=lambda state: (
                    sum(item.remaining_runtime_seconds for item in state),
                    sum(item.avg_vram_mb for item in state),
                    tuple(item.batch_size for item in state),
                ),
            )[: self.settings.gpu_scheduler.beam_width]
        return states

    def tie_key(self, group: EvaluatedGroup) -> tuple[object, ...]:
        oldest_submission = min((job.submitted_at for job in group.jobs), default="")
        try:
            backend_rank = self.settings.gpu_scheduler.backend_priority.index(group.backend_name)
        except ValueError:
            backend_rank = len(self.settings.gpu_scheduler.backend_priority)
        return (
            group.objective_score,
            oldest_submission,
            -len(group.jobs),
            group.estimated_vram_mb,
            tuple(sorted(job.job_id for job in group.jobs)),
            tuple(sorted(group.batch_overrides.items())),
            backend_rank,
            group.backend_name,
        )
