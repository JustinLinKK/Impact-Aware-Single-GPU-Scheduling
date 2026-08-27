"""Batch candidates for incremental time-aware placement."""

from __future__ import annotations

from ..domain import TrainingJob
from ..config import SchedulerSettings
from .resource_estimator import ResourceEstimator


class CandidateGenerator:
    def __init__(
        self,
        settings: SchedulerSettings,
        estimator: ResourceEstimator,
    ):
        self.settings = settings
        self.estimator = estimator

    def candidate_batch_sizes(self, job: TrainingJob) -> list[int]:
        """Return the five time-aware batch proposals after optional clipping."""
        contract = dict(job.metadata.get("training_quality_contract") or {})
        approved_values = contract.get("allowed_physical_batch_sizes") or []
        if approved_values:
            raw_values = sorted(
                {max(1, int(value)) for value in approved_values}
            )
        else:
            raw_values = self.time_aware_batch_proposals(job)
        explicit_cap = job.config.runner_kwargs.get(
            "probe_max_batch_size",
            self.settings.gpu_scheduler.batch_probe_max_batch_size,
        )
        cap = max(1, int(explicit_cap)) if explicit_cap is not None else None
        if approved_values and cap is not None:
            clipped = [value for value in raw_values if value <= cap]
        else:
            clipped = [min(value, cap) if cap is not None else value for value in raw_values]
        if not clipped:
            clipped = [
                int(
                    job.requested_batch_size
                    or self.estimator.resolved_batch_size(job)
                )
            ]
        return list(dict.fromkeys(max(1, int(value)) for value in clipped))

    def time_aware_batch_proposals(self, job: TrainingJob) -> list[int]:
        """Return the five pre-clipping exponent proposals for audit logs."""
        requested = int(job.requested_batch_size or self.estimator.resolved_batch_size(job))
        if self.settings.gpu_scheduler.batch_options.require_power_of_two_original and requested & (requested - 1):
            raise ValueError(f"time-aware scheduling requires a power-of-two requested batch size: {requested}")
        exponent = requested.bit_length() - 1
        return [2 ** max(0, exponent + offset) for offset in self.settings.gpu_scheduler.batch_options.exponent_offsets]

    def fallback_order(
        self,
        jobs: list[TrainingJob],
        backend_name: str,
    ) -> list[str]:
        ranked = sorted(
            jobs,
            key=lambda job: (
                job.priority,
                -(self.estimator.predicted_remaining_runtime_seconds(job, backend_name=backend_name) or 0.0),
                job.queue_sequence,
                job.job_id,
            ),
        )
        return [job.job_id for job in ranked]
