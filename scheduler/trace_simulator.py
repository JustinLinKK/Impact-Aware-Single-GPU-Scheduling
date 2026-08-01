"""Deterministic small-trace evaluator for scheduler policy validation.

This deliberately models non-preemptive pack drain boundaries and is intended
for fixtures/oracle comparisons, not as the production scheduling loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
import math
import statistics
from typing import Callable, Iterable


@dataclass(frozen=True, slots=True)
class TraceBatchOption:
    batch_size: int
    memory_mb: float
    solo_seconds: float
    actual_memory_mb: float | None = None
    actual_solo_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class TraceJob:
    job_id: str
    release_seconds: float
    priority: int
    options: tuple[TraceBatchOption, ...]
    backend_allowlist: tuple[str, ...] = ("cuda_process",)
    validation_metrics: tuple[float | None, ...] = ()
    planned_epochs: int | None = None


@dataclass(frozen=True, slots=True)
class TraceBackendChange:
    at_seconds: float
    backend_name: str
    available: bool


@dataclass(frozen=True, slots=True)
class TraceMemorySample:
    at_seconds: float
    used_fraction: float


@dataclass(frozen=True, slots=True)
class TracePack:
    members: tuple[TraceJob, ...]
    options: tuple[TraceBatchOption, ...]
    completion_offsets: tuple[float, ...]
    drain_seconds: float
    backend_name: str = "exclusive"
    predicted_completion_offsets: tuple[float, ...] = ()
    predicted_drain_seconds: float = 0.0
    member_slowdowns: tuple[float, ...] = ()

    @property
    def memory_mb(self) -> float:
        return sum(option.memory_mb for option in self.options)

    @property
    def actual_memory_mb(self) -> float:
        return sum(option.actual_memory_mb if option.actual_memory_mb is not None else option.memory_mb for option in self.options)

@dataclass(frozen=True, slots=True)
class TraceMetrics:
    policy: str
    makespan_seconds: float
    total_flow_seconds: float
    mean_flow_seconds: float
    weighted_mean_flow_seconds: float
    median_flow_seconds: float
    p95_flow_seconds: float
    max_wait_seconds: float
    starvation_count: int
    jobs_per_hour: float
    average_slowdown: float = 1.0
    predicted_avg_vram_mb: float = 0.0
    actual_avg_vram_mb: float = 0.0
    actual_memory_over_budget_count: int = 0
    early_stopped_epochs_saved: int = 0
    early_stopped_wall_time_saved_seconds: float = 0.0
    hard_constraint_violations: int = 0
    colocation_trial_epochs: float = 0.0
    rejected_trial_epochs_preserved: float = 0.0
    slowdown_rejections: int = 0
    admission_stalls: int = 0

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "policy": self.policy,
            "makespan_seconds": self.makespan_seconds,
            "total_flow_seconds": self.total_flow_seconds,
            "mean_flow_seconds": self.mean_flow_seconds,
            "weighted_mean_flow_seconds": self.weighted_mean_flow_seconds,
            "median_flow_seconds": self.median_flow_seconds,
            "p95_flow_seconds": self.p95_flow_seconds,
            "max_wait_seconds": self.max_wait_seconds,
            "starvation_count": self.starvation_count,
            "jobs_per_hour": self.jobs_per_hour,
            "average_slowdown": self.average_slowdown,
            "predicted_avg_vram_mb": self.predicted_avg_vram_mb,
            "actual_avg_vram_mb": self.actual_avg_vram_mb,
            "actual_memory_over_budget_count": self.actual_memory_over_budget_count,
            "early_stopped_epochs_saved": self.early_stopped_epochs_saved,
            "early_stopped_wall_time_saved_seconds": self.early_stopped_wall_time_saved_seconds,
            "hard_constraint_violations": self.hard_constraint_violations,
            "colocation_trial_epochs": self.colocation_trial_epochs,
            "rejected_trial_epochs_preserved": self.rejected_trial_epochs_preserved,
            "slowdown_rejections": self.slowdown_rejections,
            "admission_stalls": self.admission_stalls,
        }


@dataclass(frozen=True, slots=True)
class TraceProblem:
    jobs: tuple[TraceJob, ...]
    memory_budget_mb: float
    parallel_cap: int | None = None
    default_slowdown: float = 1.0
    slowdown_by_pair: dict[tuple[str, str], float] | None = None
    compatibility_by_pair: dict[tuple[str, str], bool] | None = None
    initial_backend_availability: dict[str, bool] = field(
        default_factory=lambda: {"exclusive": True, "cuda_process": True}
    )
    backend_changes: tuple[TraceBackendChange, ...] = ()
    live_memory_samples: tuple[TraceMemorySample, ...] = ()
    admission_stop_fraction: float = 0.90
    admission_resume_fraction: float = 0.85
    admission_window_seconds: float = 10.0
    early_stopping_enabled: bool = False
    early_stopping_mode: str = "max"
    early_stopping_patience_epochs: int = 5
    early_stopping_min_delta: float = 0.0
    early_stopping_min_epochs: int = 1
    priority_weight: float = 0.10
    starvation_timeout_seconds: float = 1800.0
    colocation_trial_epochs: int = 2
    colocation_min_gain: float = 1.0

    def pair_slowdown(self, left: TraceJob, right: TraceJob) -> float:
        ordered = sorted((left.job_id, right.job_id))
        key = (ordered[0], ordered[1])
        return float((self.slowdown_by_pair or {}).get(key, self.default_slowdown))

    def pair_compatible(self, left: TraceJob, right: TraceJob) -> bool:
        ordered = sorted((left.job_id, right.job_id))
        return bool((self.compatibility_by_pair or {}).get((ordered[0], ordered[1]), True))


def _backend_availability(problem: TraceProblem, now: float) -> dict[str, bool]:
    availability = dict(problem.initial_backend_availability)
    for change in sorted(problem.backend_changes, key=lambda item: (item.at_seconds, item.backend_name)):
        if change.at_seconds > now + 1e-9:
            break
        availability[change.backend_name] = change.available
    return availability


def _admission_open(problem: TraceProblem, now: float) -> bool:
    is_open = True
    below_resume_since: float | None = None
    window: list[TraceMemorySample] = []
    for sample in sorted(problem.live_memory_samples, key=lambda item: item.at_seconds):
        if sample.at_seconds > now + 1e-9:
            break
        cutoff = sample.at_seconds - problem.admission_window_seconds
        window.append(sample)
        window = [item for item in window if item.at_seconds >= cutoff]
        average = statistics.fmean(item.used_fraction for item in window)
        complete_window = bool(window and sample.at_seconds - window[0].at_seconds >= problem.admission_window_seconds)
        if is_open:
            if complete_window and average >= problem.admission_stop_fraction:
                is_open = False
                below_resume_since = None
        elif average <= problem.admission_resume_fraction:
            below_resume_since = sample.at_seconds if below_resume_since is None else below_resume_since
            if sample.at_seconds - below_resume_since >= problem.admission_window_seconds:
                is_open = True
                below_resume_since = None
        else:
            below_resume_since = None
    return is_open


def _early_stop_epoch(problem: TraceProblem, job: TraceJob) -> int:
    planned_epochs = max(1, int(job.planned_epochs or len(job.validation_metrics) or 1))
    if not problem.early_stopping_enabled or not job.validation_metrics:
        return planned_epochs
    best: float | None = None
    bad_epochs = 0
    for epoch, raw_metric in enumerate(job.validation_metrics[:planned_epochs], start=1):
        if raw_metric is None or not math.isfinite(float(raw_metric)):
            continue
        metric = float(raw_metric)
        improved = best is None
        if best is not None:
            improved = (
                metric > best + problem.early_stopping_min_delta
                if problem.early_stopping_mode == "max"
                else metric < best - problem.early_stopping_min_delta
            )
        if improved:
            best = metric
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch >= problem.early_stopping_min_epochs and bad_epochs >= problem.early_stopping_patience_epochs:
            return epoch
    return planned_epochs


def _actual_solo_seconds(problem: TraceProblem, job: TraceJob, option: TraceBatchOption) -> float:
    full_seconds = option.actual_solo_seconds if option.actual_solo_seconds is not None else option.solo_seconds
    planned_epochs = max(1, int(job.planned_epochs or len(job.validation_metrics) or 1))
    return float(full_seconds) * _early_stop_epoch(problem, job) / planned_epochs


def _backend_for_members(problem: TraceProblem, members: tuple[TraceJob, ...], now: float) -> str | None:
    availability = _backend_availability(problem, now)
    if len(members) == 1 and availability.get("exclusive", False):
        return "exclusive"
    common = set(members[0].backend_allowlist)
    for member in members[1:]:
        common.intersection_update(member.backend_allowlist)
    for backend_name in sorted(common):
        if backend_name != "exclusive" and availability.get(backend_name, False):
            return backend_name
    return None


def feasible_packs(problem: TraceProblem, jobs: Iterable[TraceJob], *, now: float = 0.0) -> list[TracePack]:
    ready = tuple(sorted(jobs, key=lambda job: job.job_id))
    cap = min(len(ready), problem.parallel_cap or len(ready))
    packs: list[TracePack] = []
    for size in range(1, cap + 1):
        for members in combinations(ready, size):
            backend_name = _backend_for_members(problem, members, now)
            if backend_name is None:
                continue
            if size > 1 and (not _admission_open(problem, now) or any(not problem.pair_compatible(left, right) for left, right in combinations(members, 2))):
                continue
            for option_vector in product(*(job.options for job in members)):
                if sum(option.memory_mb for option in option_vector) > problem.memory_budget_mb + 1e-9:
                    continue
                slowdowns: list[float] = []
                for member in members:
                    slowdown = 1.0 + sum(max(0.0, problem.pair_slowdown(member, other) - 1.0) for other in members if other != member)
                    slowdowns.append(slowdown)
                predicted_offsets = tuple(option.solo_seconds for option in option_vector)
                actual_offsets = tuple(
                    _actual_solo_seconds(problem, member, option) * slowdown
                    for member, option, slowdown in zip(members, option_vector, slowdowns, strict=True)
                )
                packs.append(
                    TracePack(
                        members=members,
                        options=option_vector,
                        completion_offsets=actual_offsets,
                        drain_seconds=max(actual_offsets),
                        backend_name=backend_name,
                        predicted_completion_offsets=predicted_offsets,
                        predicted_drain_seconds=max(predicted_offsets),
                        member_slowdowns=tuple(slowdowns),
                    )
                )
    return packs


def _weights(problem: TraceProblem, jobs: Iterable[TraceJob]) -> dict[str, float]:
    materialized = tuple(jobs)
    minimum = min((job.priority for job in materialized), default=0)
    return {job.job_id: 1.0 + problem.priority_weight * (job.priority - minimum) for job in materialized}


def _time_aware_choice(problem: TraceProblem, ready: tuple[TraceJob, ...], now: float) -> TracePack:
    starving = [job for job in ready if now - job.release_seconds >= problem.starvation_timeout_seconds]
    anchor = (
        min(starving, key=lambda job: (job.release_seconds, job.job_id))
        if starving
        else min(
            ready,
            key=lambda job: (
                min(option.solo_seconds for option in job.options if option.memory_mb <= problem.memory_budget_mb),
                -job.priority,
                job.release_seconds,
                job.job_id,
            ),
        )
    )
    return min(
        feasible_packs(problem, (anchor,), now=now),
        key=lambda pack: (pack.predicted_drain_seconds, pack.memory_mb, pack.options[0].batch_size),
    )


def _serial_choice(problem: TraceProblem, ready: tuple[TraceJob, ...], now: float) -> TracePack:
    anchor = sorted(ready, key=lambda job: (-job.priority, job.release_seconds, job.job_id))[0]
    return min(
        feasible_packs(problem, (anchor,), now=now),
        key=lambda pack: (
            pack.predicted_drain_seconds,
            pack.memory_mb,
            pack.options[0].batch_size,
        ),
    )


def _fill_choice(problem: TraceProblem, ready: tuple[TraceJob, ...], now: float) -> TracePack:
    return min(
        feasible_packs(problem, ready, now=now),
        key=lambda pack: (
            -pack.memory_mb,
            -len(pack.members),
            tuple(member.job_id for member in pack.members),
            tuple(-option.batch_size for option in pack.options),
        ),
    )


def simulate_policy(
    problem: TraceProblem,
    policy: str,
    chooser: Callable[[TraceProblem, tuple[TraceJob, ...], float], TracePack],
) -> TraceMetrics:
    remaining = {job.job_id: job for job in problem.jobs}
    completion: dict[str, float] = {}
    first_dispatch: dict[str, float] = {}
    dispatches: list[tuple[float, TracePack]] = []
    now = min((job.release_seconds for job in problem.jobs), default=0.0)
    while remaining:
        ready = tuple(job for job in remaining.values() if job.release_seconds <= now + 1e-9)
        if not ready:
            now = min(job.release_seconds for job in remaining.values())
            continue
        pack = chooser(problem, ready, now)
        dispatches.append((now, pack))
        for job, offset in zip(pack.members, pack.completion_offsets, strict=True):
            first_dispatch[job.job_id] = now
            completion[job.job_id] = now + offset
            remaining.pop(job.job_id)
        now += pack.drain_seconds
    return trace_metrics(problem, policy, completion, first_dispatch, dispatches=dispatches)


def simulate_recursive_time_aware(problem: TraceProblem) -> TraceMetrics:
    """Simulate incremental packing with useful two-epoch admission trials."""

    jobs = {job.job_id: job for job in problem.jobs}
    planned_epochs = {
        job.job_id: max(1, int(job.planned_epochs or len(job.validation_metrics) or 1))
        for job in problem.jobs
    }
    remaining_epochs = {
        job.job_id: float(_early_stop_epoch(problem, job))
        for job in problem.jobs
    }
    selected_options: dict[str, TraceBatchOption] = {}
    active: list[str] = []
    completion: dict[str, float] = {}
    first_dispatch: dict[str, float] = {}
    dispatches: list[tuple[float, TracePack]] = []
    lifecycle: dict[str, float | int] = {
        "trial_epochs": 0.0,
        "rejected_trial_epochs_preserved": 0.0,
        "slowdown_rejections": 0,
        "admission_stalls": 0,
    }
    stalled_members: set[str] | None = None
    active_backend = "exclusive"
    now = min((job.release_seconds for job in problem.jobs), default=0.0)
    epsilon = 1e-9

    def solo_epoch_seconds(job_id: str, *, predicted: bool) -> float:
        option = selected_options[job_id]
        total = option.solo_seconds if predicted or option.actual_solo_seconds is None else option.actual_solo_seconds
        return float(total) / planned_epochs[job_id]

    def slowdown(job_id: str, member_ids: list[str]) -> float:
        member = jobs[job_id]
        return 1.0 + sum(
            max(0.0, problem.pair_slowdown(member, jobs[other_id]) - 1.0)
            for other_id in member_ids
            if other_id != job_id
        )

    def packed_epoch_seconds(job_id: str, member_ids: list[str]) -> float:
        return solo_epoch_seconds(job_id, predicted=False) * slowdown(job_id, member_ids)

    def fastest_option(job_id: str, *, available_memory_mb: float = 0.0) -> TraceBatchOption | None:
        return min(
            (
                option
                for option in jobs[job_id].options
                if available_memory_mb + option.memory_mb <= problem.memory_budget_mb + epsilon
            ),
            key=lambda option: (option.solo_seconds, option.memory_mb, option.batch_size),
            default=None,
        )

    def overlap_backend(member_ids: list[str]) -> str | None:
        availability = _backend_availability(problem, now)
        common: set[str] | None = None
        for job_id in member_ids:
            allowed = set(jobs[job_id].backend_allowlist)
            common = allowed if common is None else common.intersection(allowed)
        for backend_name in sorted(common or ()):
            if backend_name != "exclusive" and availability.get(backend_name, False):
                return backend_name
        return None

    def record_segment(member_ids: list[str], duration: float, rates: dict[str, float]) -> None:
        if duration <= epsilon or not member_ids:
            return
        members = tuple(jobs[job_id] for job_id in member_ids)
        options = tuple(selected_options[job_id] for job_id in member_ids)
        slowdowns = tuple(rates[job_id] / solo_epoch_seconds(job_id, predicted=False) for job_id in member_ids)
        predicted = tuple(solo_epoch_seconds(job_id, predicted=True) * remaining_epochs[job_id] for job_id in member_ids)
        dispatches.append(
            (
                now,
                TracePack(
                    members=members,
                    options=options,
                    completion_offsets=tuple(duration for _ in member_ids),
                    drain_seconds=duration,
                    backend_name=active_backend,
                    predicted_completion_offsets=predicted,
                    predicted_drain_seconds=max(predicted, default=0.0),
                    member_slowdowns=slowdowns,
                ),
            )
        )

    def advance(max_duration: float) -> tuple[float, set[str], dict[str, float]]:
        nonlocal now, active, stalled_members
        if not active or max_duration <= epsilon:
            return 0.0, set(), {}
        member_ids = list(active)
        rates = {job_id: packed_epoch_seconds(job_id, member_ids) for job_id in member_ids}
        duration = min(
            max_duration,
            min(remaining_epochs[job_id] * rates[job_id] for job_id in member_ids),
        )
        record_segment(member_ids, duration, rates)
        for job_id in member_ids:
            remaining_epochs[job_id] = max(0.0, remaining_epochs[job_id] - duration / rates[job_id])
        now += duration
        finished = {job_id for job_id in member_ids if remaining_epochs[job_id] <= epsilon}
        for job_id in finished:
            completion[job_id] = now
        active = [job_id for job_id in active if job_id not in finished]
        if stalled_members is not None and finished.intersection(stalled_members):
            stalled_members = None
        return duration, finished, rates

    def start_anchor() -> bool:
        nonlocal active, active_backend
        ready = [
            job_id
            for job_id, job in jobs.items()
            if job_id not in completion
            and job_id not in active
            and remaining_epochs[job_id] > epsilon
            and job.release_seconds <= now + epsilon
        ]
        if not ready:
            return False
        starving = [job_id for job_id in ready if now - jobs[job_id].release_seconds >= problem.starvation_timeout_seconds]
        eligible = starving or ready
        for job_id in eligible:
            selected_options[job_id] = fastest_option(job_id) or jobs[job_id].options[0]
        anchor = min(
            eligible,
            key=lambda job_id: (
                jobs[job_id].release_seconds if starving else remaining_epochs[job_id] * solo_epoch_seconds(job_id, predicted=True),
                -jobs[job_id].priority,
                jobs[job_id].release_seconds,
                job_id,
            ),
        )
        active = [anchor]
        first_dispatch.setdefault(anchor, now)
        active_backend = overlap_backend(active) or "exclusive"
        return True

    def candidate_order() -> list[str]:
        if stalled_members is not None or active_backend == "exclusive":
            return []
        ready = [
            job_id
            for job_id, job in jobs.items()
            if job_id not in completion
            and job_id not in active
            and remaining_epochs[job_id] > epsilon
            and job.release_seconds <= now + epsilon
        ]
        starving = [job_id for job_id in ready if now - jobs[job_id].release_seconds >= problem.starvation_timeout_seconds]
        if starving:
            return [min(starving, key=lambda job_id: (jobs[job_id].release_seconds, job_id))]
        for job_id in ready:
            if job_id not in selected_options:
                option = fastest_option(job_id)
                if option is not None:
                    selected_options[job_id] = option
        return sorted(
            (job_id for job_id in ready if job_id in selected_options),
            key=lambda job_id: (
                remaining_epochs[job_id] * solo_epoch_seconds(job_id, predicted=True),
                -jobs[job_id].priority,
                jobs[job_id].release_seconds,
                job_id,
            ),
        )

    def feasible_candidate(candidate_id: str) -> bool:
        cap = problem.parallel_cap
        if cap is not None and len(active) >= cap:
            return False
        if not _admission_open(problem, now) or not _backend_availability(problem, now).get(active_backend, False):
            return False
        if active_backend not in jobs[candidate_id].backend_allowlist:
            return False
        if any(not problem.pair_compatible(jobs[candidate_id], jobs[job_id]) for job_id in active):
            return False
        active_memory = sum(selected_options[job_id].memory_mb for job_id in active)
        option = fastest_option(candidate_id, available_memory_mb=active_memory)
        if option is None:
            return False
        selected_options[candidate_id] = option
        return overlap_backend([*active, candidate_id]) == active_backend

    def run_trial(candidate_id: str) -> str:
        nonlocal active, stalled_members
        preexisting = list(active)
        pretrial_rates = {job_id: packed_epoch_seconds(job_id, preexisting) for job_id in preexisting}
        active.append(candidate_id)
        first_dispatch.setdefault(candidate_id, now)
        measured_candidate_epochs = 0.0
        trial_target = min(float(problem.colocation_trial_epochs), remaining_epochs[candidate_id])
        while measured_candidate_epochs + epsilon < trial_target:
            rate = packed_epoch_seconds(candidate_id, active)
            wanted = (trial_target - measured_candidate_epochs) * rate
            elapsed, finished, rates = advance(wanted)
            candidate_progress = elapsed / rates[candidate_id] if candidate_id in rates else 0.0
            measured_candidate_epochs += candidate_progress
            lifecycle["trial_epochs"] = float(lifecycle["trial_epochs"]) + candidate_progress
            if candidate_id in finished:
                return "completed"
            if finished.intersection(preexisting):
                preexisting = list(active)
                if preexisting == [candidate_id] or not [job_id for job_id in preexisting if job_id != candidate_id]:
                    return "accepted"
                preexisting = [job_id for job_id in active if job_id != candidate_id]
                pretrial_rates = {job_id: packed_epoch_seconds(job_id, preexisting) for job_id in preexisting}
                measured_candidate_epochs = 0.0
                trial_target = min(float(problem.colocation_trial_epochs), remaining_epochs[candidate_id])

        packed_rates = {job_id: packed_epoch_seconds(job_id, active) for job_id in active}
        active_drain = max(
            (remaining_epochs[job_id] * pretrial_rates[job_id] for job_id in preexisting),
            default=0.0,
        )
        sequential = active_drain + remaining_epochs[candidate_id] * solo_epoch_seconds(candidate_id, predicted=True)
        packed_drain = max(
            (remaining_epochs[job_id] * packed_rates[job_id] for job_id in active),
            default=0.0,
        )
        gain = sequential / packed_drain if packed_drain > epsilon else float("inf")
        if gain + epsilon >= problem.colocation_min_gain:
            return "accepted"
        active = [job_id for job_id in active if job_id != candidate_id]
        stalled_members = set(preexisting)
        lifecycle["rejected_trial_epochs_preserved"] = (
            float(lifecycle["rejected_trial_epochs_preserved"]) + measured_candidate_epochs
        )
        lifecycle["slowdown_rejections"] = int(lifecycle["slowdown_rejections"]) + 1
        lifecycle["admission_stalls"] = int(lifecycle["admission_stalls"]) + 1
        return "rejected"

    while len(completion) < len(jobs):
        if not active:
            if not start_anchor():
                future = [
                    job.release_seconds
                    for job in problem.jobs
                    if job.job_id not in completion and job.release_seconds > now + epsilon
                ]
                if not future:
                    break
                now = min(future)
                continue

        candidates = candidate_order()
        selected_candidate = next((job_id for job_id in candidates if feasible_candidate(job_id)), None)
        if selected_candidate is not None:
            run_trial(selected_candidate)
            continue

        rates = {job_id: packed_epoch_seconds(job_id, active) for job_id in active}
        next_completion = min(remaining_epochs[job_id] * rates[job_id] for job_id in active)
        future_releases = [
            job.release_seconds - now
            for job in problem.jobs
            if job.job_id not in completion
            and job.job_id not in active
            and job.release_seconds > now + epsilon
        ]
        duration = next_completion
        if stalled_members is None and active_backend != "exclusive" and future_releases:
            duration = min(duration, min(future_releases))
        advance(duration)

    return trace_metrics(
        problem,
        "parallel_time_aware",
        completion,
        first_dispatch,
        dispatches=dispatches,
        colocation_lifecycle=lifecycle,
    )


def trace_metrics(
    problem: TraceProblem,
    policy: str,
    completion: dict[str, float],
    first_dispatch: dict[str, float],
    *,
    dispatches: Iterable[tuple[float, TracePack]] = (),
    colocation_lifecycle: dict[str, float | int] | None = None,
) -> TraceMetrics:
    if not problem.jobs:
        return TraceMetrics(policy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    flows = [completion[job.job_id] - job.release_seconds for job in problem.jobs]
    waits = [first_dispatch[job.job_id] - job.release_seconds for job in problem.jobs]
    ordered_flow = sorted(flows)
    p95_index = max(0, math.ceil(0.95 * len(ordered_flow)) - 1)
    makespan = max(completion.values()) - min(job.release_seconds for job in problem.jobs)
    weights = _weights(problem, problem.jobs)
    weighted_flow = sum(weights[job.job_id] * (completion[job.job_id] - job.release_seconds) for job in problem.jobs) / max(1e-9, sum(weights.values()))
    materialized_dispatches = list(dispatches)
    packs = [pack for _, pack in materialized_dispatches]
    slowdowns = [slowdown for pack in packs for slowdown in pack.member_slowdowns]
    hard_violations = 0
    for started_at, pack in materialized_dispatches:
        availability = _backend_availability(problem, started_at)
        if pack.memory_mb > problem.memory_budget_mb + 1e-9:
            hard_violations += 1
        if problem.parallel_cap is not None and len(pack.members) > problem.parallel_cap:
            hard_violations += 1
        if not availability.get(pack.backend_name, False):
            hard_violations += 1
        if any(member.release_seconds > started_at + 1e-9 for member in pack.members):
            hard_violations += 1
        if any(not problem.pair_compatible(left, right) for left, right in combinations(pack.members, 2)):
            hard_violations += 1
        if len(pack.members) > 1 and not _admission_open(problem, started_at):
            hard_violations += 1
    early_stopped_epochs_saved = 0
    early_stopped_wall_time_saved_seconds = 0.0
    for job in problem.jobs:
        planned_epochs = max(1, int(job.planned_epochs or len(job.validation_metrics) or 1))
        stop_epoch = _early_stop_epoch(problem, job)
        early_stopped_epochs_saved += max(0, planned_epochs - stop_epoch)
        if stop_epoch < planned_epochs:
            selected_pack = next(pack for _, pack in materialized_dispatches if job in pack.members)
            option = selected_pack.options[selected_pack.members.index(job)]
            full_seconds = option.actual_solo_seconds if option.actual_solo_seconds is not None else option.solo_seconds
            early_stopped_wall_time_saved_seconds += float(full_seconds) * (planned_epochs - stop_epoch) / planned_epochs
    lifecycle = colocation_lifecycle or {}
    return TraceMetrics(
        policy=policy,
        makespan_seconds=makespan,
        total_flow_seconds=sum(flows),
        mean_flow_seconds=statistics.fmean(flows),
        weighted_mean_flow_seconds=weighted_flow,
        median_flow_seconds=statistics.median(flows),
        p95_flow_seconds=ordered_flow[p95_index],
        max_wait_seconds=max(waits),
        starvation_count=sum(wait >= problem.starvation_timeout_seconds for wait in waits),
        jobs_per_hour=(3600.0 * len(problem.jobs) / makespan) if makespan > 0 else 0.0,
        average_slowdown=statistics.fmean(slowdowns) if slowdowns else 1.0,
        predicted_avg_vram_mb=statistics.fmean(pack.memory_mb for pack in packs) if packs else 0.0,
        actual_avg_vram_mb=statistics.fmean(pack.actual_memory_mb for pack in packs) if packs else 0.0,
        actual_memory_over_budget_count=sum(pack.actual_memory_mb > problem.memory_budget_mb + 1e-9 for pack in packs),
        early_stopped_epochs_saved=early_stopped_epochs_saved,
        early_stopped_wall_time_saved_seconds=early_stopped_wall_time_saved_seconds,
        hard_constraint_violations=hard_violations,
        colocation_trial_epochs=float(lifecycle.get("trial_epochs", 0.0)),
        rejected_trial_epochs_preserved=float(lifecycle.get("rejected_trial_epochs_preserved", 0.0)),
        slowdown_rejections=int(lifecycle.get("slowdown_rejections", 0)),
        admission_stalls=int(lifecycle.get("admission_stalls", 0)),
    )


def oracle(problem: TraceProblem, *, serial_baseline: TraceMetrics) -> TraceMetrics:
    weights = _weights(problem, problem.jobs)
    best_key: tuple[float, float, float] | None = None
    best_result: tuple[dict[str, float], dict[str, float], tuple[tuple[float, TracePack], ...]] | None = None

    def visit(
        remaining: dict[str, TraceJob],
        now: float,
        completion: dict[str, float],
        first_dispatch: dict[str, float],
        dispatches: tuple[tuple[float, TracePack], ...],
    ) -> None:
        nonlocal best_key, best_result
        if not remaining:
            makespan = max(completion.values()) - min(job.release_seconds for job in problem.jobs)
            weighted_flow = sum(weights[job.job_id] * (completion[job.job_id] - job.release_seconds) for job in problem.jobs) / max(1e-9, sum(weights.values()))
            score = weighted_flow / max(1e-9, serial_baseline.weighted_mean_flow_seconds)
            key = (score, makespan, weighted_flow)
            if best_key is None or key < best_key:
                best_key = key
                best_result = (dict(completion), dict(first_dispatch), dispatches)
            return
        ready = tuple(job for job in remaining.values() if job.release_seconds <= now + 1e-9)
        if not ready:
            visit(
                remaining,
                min(job.release_seconds for job in remaining.values()),
                completion,
                first_dispatch,
                dispatches,
            )
            return
        for pack in feasible_packs(problem, ready, now=now):
            next_remaining = dict(remaining)
            next_completion = dict(completion)
            next_dispatch = dict(first_dispatch)
            for job, offset in zip(pack.members, pack.completion_offsets, strict=True):
                next_remaining.pop(job.job_id)
                next_completion[job.job_id] = now + offset
                next_dispatch[job.job_id] = now
            visit(
                next_remaining,
                now + pack.drain_seconds,
                next_completion,
                next_dispatch,
                (*dispatches, (now, pack)),
            )

    visit(
        {job.job_id: job for job in problem.jobs},
        min(job.release_seconds for job in problem.jobs),
        {},
        {},
        (),
    )
    assert best_result is not None
    completion, first_dispatch, dispatches = best_result
    return trace_metrics(
        problem,
        "small_trace_oracle",
        completion,
        first_dispatch,
        dispatches=dispatches,
    )


def compare_policies(problem: TraceProblem) -> list[TraceMetrics]:
    serial = simulate_policy(problem, "serial_fifo", _serial_choice)
    return [
        serial,
        simulate_policy(problem, "legacy_vram_fill", _fill_choice),
        simulate_recursive_time_aware(problem),
        oracle(problem, serial_baseline=serial),
    ]


def benchmark_fixture() -> TraceProblem:
    def job(job_id: str, release: float, priority: int = 0) -> TraceJob:
        return TraceJob(
            job_id,
            release,
            priority,
            (
                TraceBatchOption(1, 1_500, 13.0, actual_memory_mb=1_550),
                TraceBatchOption(2, 1_800, 11.0, actual_memory_mb=1_850),
                TraceBatchOption(4, 2_000, 10.0, actual_memory_mb=2_050),
                TraceBatchOption(8, 3_500, 12.0, actual_memory_mb=3_600),
                TraceBatchOption(16, 5_000, 15.0, actual_memory_mb=5_100),
            ),
            validation_metrics=((0.50, 0.60, 0.60, 0.59, 0.58, 0.57) if job_id == "d" else ()),
            planned_epochs=6,
        )

    jobs = (job("a", 0), job("b", 0), job("c", 0), job("d", 5, 1))
    slowdowns = {(left, right): 1.10 for left, right in combinations(sorted(member.job_id for member in jobs), 2)}
    return TraceProblem(
        jobs=jobs,
        memory_budget_mb=10_000,
        parallel_cap=2,
        slowdown_by_pair=slowdowns,
        starvation_timeout_seconds=60,
        early_stopping_enabled=True,
        early_stopping_patience_epochs=2,
        early_stopping_min_epochs=2,
    )


def markdown_table(metrics: Iterable[TraceMetrics]) -> str:
    rows = [
        "| Policy | Makespan (s) | Total flow (s) | Mean flow (s) | Weighted flow (s) | Median flow (s) | p95 flow (s) | Max wait (s) | Starved | Jobs/hour | Slowdown | Pred/actual VRAM (MiB) | Actual over-budget packs | Trial/rejected epochs | Rejections/stalls | Early epochs/time saved | Violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        rows.append(
            f"| {item.policy} | {item.makespan_seconds:.2f} | {item.total_flow_seconds:.2f} | "
            f"{item.mean_flow_seconds:.2f} | {item.weighted_mean_flow_seconds:.2f} | "
            f"{item.median_flow_seconds:.2f} | {item.p95_flow_seconds:.2f} | {item.max_wait_seconds:.2f} | "
            f"{item.starvation_count} | {item.jobs_per_hour:.2f} | {item.average_slowdown:.3f} | "
            f"{item.predicted_avg_vram_mb:.1f}/{item.actual_avg_vram_mb:.1f} | "
            f"{item.actual_memory_over_budget_count} | "
            f"{item.colocation_trial_epochs:.1f}/{item.rejected_trial_epochs_preserved:.1f} | "
            f"{item.slowdown_rejections}/{item.admission_stalls} | "
            f"{item.early_stopped_epochs_saved}/{item.early_stopped_wall_time_saved_seconds:.1f}s | "
            f"{item.hard_constraint_violations} |"
        )
    return "\n".join(rows)


def main() -> int:
    print(markdown_table(compare_policies(benchmark_fixture())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
