"""V4 request latency: one forecast engine, individual and group objectives.

All times are seconds. Forecasts are research estimates, never measurements.
This module has no access to the simulator's performance function or a GPU.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite


FEEDBACK_POLICY = "derived_solo_slope_feedback_v4"


def number(value, *, positive=False):
    if isinstance(value, bool):
        raise ValueError("boolean is not a timing value")
    value = float(value)
    if not isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("expected a finite " + ("positive" if positive else "nonnegative") + " value")
    return value


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return number(parsed.timestamp())


def derived_solo_slope(wall_step_ms):
    """Nominal updates/s from predicted milliseconds/update, not calibration."""
    return number(1000.0 / number(wall_step_ms, positive=True), positive=True)


@dataclass(frozen=True)
class FeedbackRequest:
    request_id: str
    workflow_id: str
    completion_policy: str
    required_job_ids: tuple[str, ...]
    submitted_at: float
    weight: float = 1.0
    completion_condition: str = "successful"

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id is required")
        if not isinstance(self.workflow_id, str):
            raise ValueError("workflow_id must be a string")
        if self.completion_policy not in {"individual", "group"}:
            raise ValueError("completion_policy must be individual or group")
        if (not isinstance(self.required_job_ids, (tuple, list)) or not self.required_job_ids
                or len(set(self.required_job_ids)) != len(self.required_job_ids)
                or any(not isinstance(j, str) or not j for j in self.required_job_ids)):
            raise ValueError("required_job_ids must be nonempty and unique")
        if self.completion_policy == "individual" and len(self.required_job_ids) != 1:
            raise ValueError("an individual request has exactly one job")
        if self.completion_condition not in {"successful", "terminal"}:
            raise ValueError("unsupported completion_condition")
        object.__setattr__(self, "submitted_at", number(self.submitted_at))
        object.__setattr__(self, "weight", number(self.weight, positive=True))
        object.__setattr__(self, "required_job_ids", tuple(self.required_job_ids))

    @property
    def key(self):
        return (self.workflow_id, self.request_id)

    def to_dict(self):
        return asdict(self)


def requests_from_jobs(jobs):
    """Validate sealed membership; omit fully satisfied historical requests.

    One job belongs to one feedback request in v4. Shared dependencies and
    dynamic groups are future extensions, not inferred from a workflow name.
    Missing metadata explicitly defaults to an individual successful result.
    """
    by_id = {j.job_id: j for j in jobs}
    if len(by_id) != len(jobs):
        raise ValueError("duplicate job identity")
    specs, ownership = {}, {}
    for job in jobs:
        raw = job.metadata.get("feedback_request")
        if raw is None:
            spec = FeedbackRequest("job:" + job.job_id, job.workflow_id or "", "individual",
                                   (job.job_id,), timestamp(job.submitted_at))
        else:
            if not isinstance(raw, dict):
                raise ValueError("feedback_request must be an object")
            policy = raw.get("completion_policy", "individual")
            if policy == "group" and raw.get("membership_finalized") is not True:
                raise ValueError("group membership is not finalized")
            members = raw.get("required_job_ids", [job.job_id] if policy == "individual" else None)
            if not isinstance(members, (list, tuple)):
                raise ValueError("required_job_ids are missing")
            spec = FeedbackRequest(raw["request_id"], raw.get("workflow_id", job.workflow_id or ""),
                                   policy, tuple(sorted(members)), timestamp(raw["submitted_at"]),
                                   number(raw.get("weight", 1), positive=True),
                                   raw.get("completion_condition", "successful"))
        if job.job_id not in spec.required_job_ids:
            raise ValueError("job is absent from its request")
        if spec.submitted_at > timestamp(job.submitted_at):
            raise ValueError("request submission cannot follow member submission")
        if spec.key in specs and specs[spec.key] != spec:
            raise ValueError("inconsistent request declarations")
        specs[spec.key] = spec
        ownership[job.job_id] = spec.key
    outstanding, completed = [], {}
    for spec in specs.values():
        if any(j not in by_id or ownership.get(j) != spec.key for j in spec.required_job_ids):
            raise ValueError("missing or inconsistently owned group member")
        members = [by_id[j] for j in spec.required_job_ids]
        if all(j.status.is_terminal for j in members):
            # Historical failed requests remain failures, not successful feedback.
            continue
        for job in members:
            if not job.status.is_terminal:
                continue
            if spec.completion_condition == "successful" and job.status.value != "COMPLETED":
                raise ValueError("required successful group member failed or was cancelled")
            when = timestamp(job.metadata.get("feedback_ready_at") or job.finished_at)
            if when < timestamp(job.submitted_at):
                raise ValueError("completion precedes submission")
            completed[job.job_id] = when
        outstanding.append(spec)
    return tuple(sorted(outstanding, key=lambda s: s.key)), completed


@dataclass(frozen=True)
class CompletionCosts:
    startup: float = 0.0
    finish: float = 0.0
    delivery: float = 0.0
    pause: float = 0.0
    resume: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            object.__setattr__(self, name, number(value))


def project_completions(remaining, initial, initial_rates, solo, queue, evidence, costs,
                        *, initial_delay=0.0, resumed_candidate=None):
    """Drain the current pack, then a frozen serial queue.

    Unknown surviving pack rates inherit their parent's rates, with provenance.
    Finish reserves serialize GPU phases; delivery delays do not hold the GPU.
    This explicit phase approximation is not a bound or a production ETA claim.
    """
    left = {j: number(remaining[j], positive=True) for j in initial}
    rates = {j: number(initial_rates[j], positive=True) for j in initial}
    elapsed = number(initial_delay)
    completion, phases = {}, []
    source = "measured_initial_pack"
    while left:
        durations = {j: left[j] / rates[j] for j in left}
        duration = number(min(durations.values()), positive=True)
        phases.append({"start": elapsed, "end": elapsed + duration, "members": sorted(left),
                       "rates": dict(rates), "source": source})
        elapsed += duration
        finished = sorted(j for j in left if durations[j] == duration)
        left = {j: max(0.0, left[j] - duration * rates[j]) for j in left if j not in finished}
        for j in finished:
            elapsed += costs[j].finish
            completion[j] = number(elapsed + costs[j].delivery)
        if len(left) == 1:
            rates = {j: number(solo[j], positive=True) for j in left}
            source = "predicted_solo_after_membership_change"
        elif left:
            key = tuple(sorted(left))
            vector = evidence.get(key)
            rates = {j: number(vector[j] if vector is not None else rates[j], positive=True) for j in left}
            source = "measured_matching_subgroup" if vector is not None else "assumed_parent_rates"
    for j in queue:
        delay = costs[j].resume if j == resumed_candidate else costs[j].startup
        start = elapsed
        elapsed += delay + number(remaining[j], positive=True) / number(solo[j], positive=True) + costs[j].finish
        completion[j] = number(elapsed + costs[j].delivery)
        phases.append({"start": start, "end": elapsed, "members": [j],
                       "rates": {j: solo[j]}, "source": "predicted_serial_tail"})
    return completion, phases


def score_requests(requests, completion):
    used, keys, details = set(), set(), []
    total = weight_sum = 0.0
    for spec in requests:
        if spec.key in keys or used.intersection(spec.required_job_ids):
            raise ValueError("duplicate request or double-counted job")
        keys.add(spec.key)
        used.update(spec.required_job_ids)
        finish = max(number(completion[j]) for j in spec.required_job_ids)
        latency = number(finish - spec.submitted_at)
        total += spec.weight * latency
        weight_sum += spec.weight
        details.append({**spec.to_dict(), "completion_at": finish, "latency_seconds": latency})
    number(weight_sum, positive=True)
    number(total)
    return {"weighted_latency_seconds": total, "weight_sum": weight_sum,
            "mean_latency_seconds": total / weight_sum, "requests": details}


def compare_feedback_admission(*, remaining, solo, incumbents, candidate, queue,
                               pre_halves, packed_halves, requests, completed, now,
                               costs=None, min_gain_seconds=1.0, bracket_admission_mode="nominal"):
    """Same post-trial state and complete cohort in both alternatives/halves."""
    result = {"admission_policy": FEEDBACK_POLICY, "objective": "request_feedback_latency",
              "decision": "inconclusive", "halves": [], "error": None}
    try:
        now = number(now)
        number(min_gain_seconds, positive=True)
        if bracket_admission_mode not in {"nominal", "recovery_regret"}:
            raise ValueError("unsupported bracket admission mode")
        robust_bracket = (bracket_admission_mode == "recovery_regret"
                          and len(requests) == 1 and requests[0].completion_policy == "group")
        result["decision_basis"] = "bracket_recovery_regret_v1" if robust_bracket else "nominal_request_latency"
        incumbents = tuple(sorted(incumbents))
        queue = tuple(queue)
        if (not incumbents or len(set(incumbents)) != len(incumbents)
                or len(set(queue)) != len(queue) or candidate not in queue
                or set(incumbents).intersection(queue)
                or set(remaining) != set(incumbents) | set(queue)
                or set(completed).intersection(remaining)):
            raise ValueError("invalid outstanding cohort or queue")
        if set(solo) != set(remaining):
            raise ValueError("missing or extra solo predictions")
        for j in remaining:
            number(remaining[j], positive=True)
            number(solo[j], positive=True)
        costs = costs if costs is not None else {j: CompletionCosts() for j in remaining}
        if set(costs) != set(remaining):
            raise ValueError("missing phase costs")
        required = [j for spec in requests for j in spec.required_job_ids]
        if any(spec.submitted_at > now for spec in requests):
            raise ValueError("unarrived request in forecast cohort")
        if len(required) != len(set(required)) or set(required) != set(remaining) | set(completed):
            raise ValueError("request cohort does not cover exactly the forecast jobs")
        if any(number(value) > now for value in completed.values()):
            raise ValueError("observed completion is in the future")
        proposed = (*incumbents, candidate)
        if set(pre_halves) != set(incumbents) or set(packed_halves) != set(proposed):
            raise ValueError("measurement membership mismatch")
        for values in [*pre_halves.values(), *packed_halves.values()]:
            if len(values) != 2:
                raise ValueError("two measurement halves required")
            for value in values:
                number(value, positive=True)
        for half in range(2):
            pre = {j: pre_halves[j][half] for j in incumbents}
            packed = {j: packed_halves[j][half] for j in proposed}
            evidence = {tuple(sorted(incumbents)): pre, tuple(sorted(proposed)): packed}
            defer, defer_phases = project_completions(remaining, incumbents, pre, solo, queue,
                evidence, costs, initial_delay=costs[candidate].pause, resumed_candidate=candidate)
            admit, admit_phases = project_completions(remaining, proposed, packed, solo,
                [j for j in queue if j != candidate], evidence, costs)
            scores = {}
            for name, projection in [("defer", defer), ("admit", admit)]:
                absolute = {**completed, **{j: now + seconds for j, seconds in projection.items()}}
                scores[name] = {**score_requests(requests, absolute), "job_completions": absolute}
            gain = ((scores["defer"]["weighted_latency_seconds"] - scores["admit"]["weighted_latency_seconds"])
                    / scores["defer"]["weight_sum"])
            if not isfinite(gain):
                raise ValueError("nonfinite gain")
            pminus = sum(pre[j] / solo[j] for j in incumbents)
            pplus = sum(packed[j] / solo[j] for j in proposed)
            result["halves"].append({"gain_seconds": gain, **scores,
                                     "normalized_progress_before": pminus,
                                     "normalized_progress_after": pplus,
                                     "normalized_progress_ratio": pplus / pminus,
                                     "incumbent_slowdown_ratios": {j: pre[j] / packed[j] for j in incumbents},
                                     "defer_phases": defer_phases, "admit_phases": admit_phases})
            half_result = result["halves"][-1]
            half_result["decision_score_seconds"] = gain
            if robust_bracket:
                # A slow packed observation cannot distinguish interference from
                # an optimistic solo prediction. Keep both possibilities explicit;
                # do not relabel a packed measurement as a measured solo rate.
                cautious_solo = dict(solo)
                cautious_solo[candidate] = min(solo[candidate], packed[candidate])
                cautious_defer, cautious_defer_phases = project_completions(
                    remaining, incumbents, pre, cautious_solo, queue, evidence, costs,
                    initial_delay=costs[candidate].pause, resumed_candidate=candidate)
                cautious_admit, cautious_admit_phases = project_completions(
                    remaining, proposed, packed, cautious_solo,
                    [j for j in queue if j != candidate], evidence, costs)
                cautious_scores = {}
                for name, projection in [("defer", cautious_defer), ("admit", cautious_admit)]:
                    absolute = {**completed, **{j: now + seconds for j, seconds in projection.items()}}
                    cautious_scores[name] = {**score_requests(requests, absolute), "job_completions": absolute}
                cautious_gain = ((cautious_scores["defer"]["weighted_latency_seconds"]
                                  - cautious_scores["admit"]["weighted_latency_seconds"])
                                 / cautious_scores["defer"]["weight_sum"])
                if not isfinite(cautious_gain):
                    raise ValueError("nonfinite bracket recovery gain")
                admit_regret = max(0.0, -gain, -cautious_gain)
                defer_regret = max(0.0, gain, cautious_gain)
                half_result.update(
                    decision_score_seconds=defer_regret - admit_regret,
                    admit_max_regret_seconds=admit_regret,
                    defer_max_regret_seconds=defer_regret,
                    no_recovery_scenario={
                        "candidate_solo_updates_per_second": cautious_solo[candidate],
                        "source": "hypothesis_from_current_packed_rate_not_a_solo_measurement",
                        "gain_seconds": cautious_gain, **cautious_scores,
                        "defer_phases": cautious_defer_phases, "admit_phases": cautious_admit_phases})
        passes = [h["decision_score_seconds"] >= min_gain_seconds for h in result["halves"]]
        result["decision"] = "accepted" if all(passes) else "rejected" if not any(passes) else "inconclusive"
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, ZeroDivisionError) as exc:
        result["error"] = str(exc)
    return result
