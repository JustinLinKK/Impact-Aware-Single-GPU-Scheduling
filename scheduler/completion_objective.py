"""Strict measured optimizer-update projections for cooperative admission.

These functions have no predictor input and never infer an unmeasured subgroup
rate. Optimizer-update drain is only one component of whole-job completion;
callers must account for pending validation/finalization before accepting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite


def _integer(value, *, minimum=0):
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
        if number < minimum or float(value) != number:
            return None
        return number
    except (TypeError, ValueError, OverflowError):
        return None


def remaining_optimizer_updates(job, snapshot):
    """Use a validated update budget and exact safe-point progress, not epochs."""
    step = _integer(snapshot.get("global_step"))
    steps_per_epoch = _integer(snapshot.get("steps_per_epoch"), minimum=1)
    epochs = _integer(job.max_epochs or job.config.max_epochs, minimum=1)
    explicit = _integer(job.max_steps, minimum=1) if job.max_steps is not None else None
    if step is None or (job.max_steps is not None and explicit is None):
        return None
    epoch_budget = epochs * steps_per_epoch if epochs is not None and steps_per_epoch is not None else None
    if explicit is not None and epoch_budget is not None and explicit != epoch_budget:
        return None  # The shared runner rejects a truncated/inconsistent recipe.
    budget = explicit if explicit is not None else epoch_budget
    if budget is None or step > budget:
        return None
    if snapshot.get("total_optimizer_updates") is not None and _integer(snapshot["total_optimizer_updates"], minimum=1) != budget:
        return None
    if snapshot.get("planned_epochs") is not None and _integer(snapshot["planned_epochs"], minimum=1) != epochs:
        return None
    return budget - step


@dataclass(frozen=True)
class UpdateDrainPhase:
    member_ids: tuple[str, ...]
    duration_seconds: float
    seconds_per_update: dict[str, float]
    completed_update_budgets: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class UpdateDrainProjection:
    seconds: float | None
    phases: tuple[UpdateDrainPhase, ...] = ()
    missing_membership: tuple[str, ...] | None = None
    error: str | None = None


def project_update_drain(remaining_updates: Mapping[str, float], group_rates: Mapping[tuple[str, ...], Mapping[str, float]]):
    """Change rates at each completion boundary; missing evidence stays missing.

    Initial progress is exact integer optimizer updates. Subsequent progress is
    continuous expected work under the observed seconds/update model. No claim
    about validation, checkpoint or process completion is made here.
    """
    remaining = {}
    for name, value in remaining_updates.items():
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return UpdateDrainProjection(None, error="invalid remaining updates")
        if not isfinite(value) or value < 0:
            return UpdateDrainProjection(None, error="invalid remaining updates")
        if value > 0:
            remaining[name] = value
    elapsed, phases = 0.0, []
    while remaining:
        members = tuple(sorted(remaining))
        vector = group_rates.get(members)
        if vector is None:
            return UpdateDrainProjection(None, tuple(phases), missing_membership=members)
        try:
            rates = {name: float(vector[name]) for name in members}
        except (KeyError, TypeError, ValueError, OverflowError):
            return UpdateDrainProjection(None, tuple(phases), error="incomplete or invalid rate vector")
        if any(not isfinite(rate) or rate <= 0 for rate in rates.values()):
            return UpdateDrainProjection(None, tuple(phases), error="nonpositive or nonfinite update rate")
        finish_times = {name: remaining[name] * rates[name] for name in members}
        duration = min(finish_times.values())
        if not isfinite(duration) or duration <= 0:
            return UpdateDrainProjection(None, tuple(phases), error="invalid projected phase duration")
        completed = tuple(name for name in members if finish_times[name] == duration)
        phases.append(UpdateDrainPhase(members, duration, rates, completed))
        elapsed += duration
        if not isfinite(elapsed):
            return UpdateDrainProjection(None, tuple(phases), error="nonfinite projected drain")
        remaining = {name: max(0.0, remaining[name] - duration / rates[name])
                     for name in members if name not in completed}
        remaining = {name: value for name, value in remaining.items() if value > 0}
    return UpdateDrainProjection(elapsed, tuple(phases))


def project_deferred_candidate(remaining_updates, candidate_id, group_rates):
    """Drain incumbents, then run the selected candidate with its measured solo rate."""
    if candidate_id not in remaining_updates:
        return UpdateDrainProjection(None, error="candidate missing from remaining work")
    active = project_update_drain({name: value for name, value in remaining_updates.items()
                                   if name != candidate_id}, group_rates)
    if active.seconds is None:
        return active
    candidate = project_update_drain({candidate_id: remaining_updates[candidate_id]}, group_rates)
    if candidate.seconds is None:
        return UpdateDrainProjection(None, active.phases + candidate.phases,
                                     candidate.missing_membership, candidate.error)
    return UpdateDrainProjection(active.seconds + candidate.seconds, active.phases + candidate.phases)


def compare_update_completion(initial_remaining, current_remaining, candidate_id, group_half_rates, elapsed_seconds):
    """Compare both sample halves from two exact common progress snapshots.

    The actual measured interval is charged once, with its productive updates
    already reflected in current_remaining. Forward continuation is reported
    separately from whole-attempt amortization. This does not classify admission
    until the caller also resolves non-update completion work.
    """
    try:
        elapsed = float(elapsed_seconds)
        valid = (isfinite(elapsed) and elapsed >= 0
                 and set(initial_remaining) == set(current_remaining)
                 and candidate_id in initial_remaining
                 and all(0 <= float(current_remaining[name]) <= float(initial_remaining[name])
                         for name in initial_remaining))
    except (TypeError, ValueError, KeyError, OverflowError):
        valid = False
    if not valid:
        return {"error": "inconsistent comparison snapshots or elapsed time", "halves": [], "missing_memberships": []}
    comparisons, missing = [], set()
    for half in range(2):
        try:
            vectors = {members: {name: float(values[half]) for name, values in vector.items()}
                       for members, vector in group_half_rates.items()}
        except (KeyError, IndexError, TypeError, ValueError, OverflowError):
            return {"error": "invalid two-half rate evidence", "halves": [], "missing_memberships": []}
        projections = {
            "initial_defer": project_deferred_candidate(initial_remaining, candidate_id, vectors),
            "current_defer": project_deferred_candidate(current_remaining, candidate_id, vectors),
            "current_keep": project_update_drain(current_remaining, vectors),
        }
        for projection in projections.values():
            if projection.error:
                return {"error": projection.error, "halves": [], "missing_memberships": []}
            if projection.missing_membership is not None:
                missing.add(projection.missing_membership)
        comparisons.append({
            **{name + "_seconds": projection.seconds for name, projection in projections.items()},
            **{name + "_phases": [phase.to_dict() for phase in projection.phases]
               for name, projection in projections.items()},
            "elapsed_measurement_seconds": elapsed,
            "measured_attempt_keep_seconds": elapsed + projections["current_keep"].seconds
                if projections["current_keep"].seconds is not None else None,
        })
    return {"error": None, "halves": comparisons, "missing_memberships": sorted(missing)}


def observed_phase_reserve(job, snapshot, *, hardware_key, now, ttl_seconds):
    """Serialize pending phases using recent direct empirical maxima.

    This is an explicit completion proxy, not a guaranteed upper bound. An
    unknown finalizer remains labelled unknown; missing required validation or
    checkpoint observations prevent scoring until productive evidence arrives.
    """
    identity = job.metadata.get("runtime_completion_identity")
    epochs = _integer(snapshot.get("planned_epochs"), minimum=1)
    completed = _integer(snapshot.get("completed_epochs"))
    step = _integer(snapshot.get("global_step"))
    steps = _integer(snapshot.get("steps_per_epoch"), minimum=1)
    checkpoint_epoch = _integer(snapshot.get("checkpoint_completed_epoch"))
    checkpoint_step = _integer(snapshot.get("checkpoint_completed_step"))
    if (not isinstance(identity, str) or not identity or snapshot.get("completion_identity") != identity
            or any(value is None for value in (epochs, completed, step, steps, checkpoint_epoch, checkpoint_step))
            or not checkpoint_epoch <= completed <= epochs or checkpoint_step > step
            or not completed * steps <= step <= min(epochs, completed + 1) * steps):
        return None, {"error": "invalid completion phase identity or progress"}
    sources = {"validation": "shared_training.validation", "checkpoint": "checkpoint_manager.save_checkpoint",
               "finalization": "shared_training.finalization"}
    observations = {phase: [] for phase in sources}
    for sample in job.metadata.get("runtime_phase_timing_history") or []:
        try:
            phase = sample["phase"]
            observed = datetime.fromisoformat(sample["finished_at"])
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            seconds = float(sample["seconds"])
            valid = (sample["identity"] == identity and sample.get("hardware_key") == hardware_key
                     and sample["source"] == sources.get(phase)
                     and isfinite(seconds) and seconds >= 0 and 0 <= now - observed.timestamp() <= ttl_seconds)
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
        if valid and phase in observations:
            observations[phase].append(seconds)
    pending_validation = epochs - completed
    pending_checkpoints = epochs - checkpoint_epoch if job.checkpoint_policy.save_every_epoch else 0
    interval = job.checkpoint_policy.save_every_n_steps
    if interval is not None:
        interval = _integer(interval, minimum=1)
        if interval is None:
            return None, {"error": "invalid step checkpoint policy"}
        pending_checkpoints += (epochs * steps) // interval - checkpoint_step // interval
    pending_checkpoints += not snapshot.get("final_checkpoint_complete", False)
    counts = {"validation": pending_validation, "checkpoint": int(pending_checkpoints),
              "finalization": int(not snapshot.get("finalization_complete", False))}
    maxima = {phase: max(values) if values else None for phase, values in observations.items()}
    missing = [phase for phase in ("validation", "checkpoint") if counts[phase] and maxima[phase] is None]
    known = sum(counts[phase] * value for phase, value in maxima.items() if value is not None)
    diagnostics = {"basis": "sum of pending phases times recent direct empirical maximum; not a guaranteed bound",
                   "pending_phase_counts": counts, "observed_phase_max_seconds": maxima,
                   "missing_required_phases": missing,
                   "unknown_finalization": bool(counts["finalization"] and maxima["finalization"] is None),
                   "known_reserve_seconds": known}
    return (None if missing else known), diagnostics
