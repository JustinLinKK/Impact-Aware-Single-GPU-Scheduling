"""Mode-specific candidate group sizing for scheduler placement."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import SCHEDULER_MODE_PARALLEL_AUTO_PACK, SchedulerSettings


@dataclass(frozen=True, slots=True)
class CandidateGroupSizing:
    window_size: int
    max_group_size: int
    include_singletons: bool = False


def candidate_group_sizing(
    settings: SchedulerSettings,
    *,
    scheduler_mode: str,
    queued_job_count: int,
) -> CandidateGroupSizing:
    """Resolve how many queued jobs this mode may consider as a group."""

    window_size = min(max(0, int(queued_job_count)), max(1, int(settings.gpu_scheduler.candidate_window_size)))
    if scheduler_mode == SCHEDULER_MODE_PARALLEL_AUTO_PACK:
        return CandidateGroupSizing(
            window_size=window_size,
            max_group_size=window_size,
            include_singletons=True,
        )

    max_group_size = max(1, int(settings.gpu_scheduler.max_packed_jobs_per_gpu))
    if settings.gpu_scheduler.allow_three_way_packing:
        max_group_size = max(max_group_size, 3)
    return CandidateGroupSizing(
        window_size=window_size,
        max_group_size=min(max_group_size, window_size),
        include_singletons=False,
    )
