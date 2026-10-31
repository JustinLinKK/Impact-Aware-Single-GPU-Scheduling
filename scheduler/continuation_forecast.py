"""Bounded, pure continuation forecasts for the opt-in latency experiment.

Times are seconds relative to one frozen post-trial snapshot. ``remaining`` is
CURRENT successful-update work: no trial time or completed updates are charged
again. A forecast is an executable reference FIFO continuation, not a claim that
future online replanning will follow this trace.

``phase_state[j]`` optionally specifies phase (starting/training/finishing/
delivering/paused/done), and the *remaining* seconds under startup/finish/delivery.
Finish holds admission memory; delivery does not. Costs describe future phases.
Memory is the conservative admission envelope, not measured allocated VRAM.

Evidence keys are sorted exact tuples of TRAINING job IDs. Values are either
caller-validated fresh matching measured rate vectors, or records with ``rates``,
``identity``, ``observed_at`` and ``max_age_seconds``. ``context`` may supply the
required evidence_identity, membership_generation and snapshot_at; mismatches/staleness are unknown.
Starting/finishing workers reserve memory but are not productive group members;
phase interference is not inferred from a steady training measurement. This
phase approximation is explicit in every trace. No parent-group rate inheritance
or solo-as-packed extrapolation is performed.
"""
from __future__ import annotations

from time import perf_counter

from .feedback_objective import CompletionCosts, number, score_requests

POLICY = "event_continuation_v5"
_PHASES = {"starting", "training", "finishing", "delivering", "paused", "done", "pausing"}
_ERRORS = (ValueError, TypeError, KeyError, IndexError, OverflowError, ZeroDivisionError)


def reference_continuation(queue, residents, *, memory=None, memory_capacity=None,
                           concurrency_limit=None, protected=(), compatible=None,
                           placement_allowed=None):
    """Select legal starts, also reusable by execution fallback.

    Queue order is original-arrival FIFO (caller supplies stable submission/ID
    order). Protected IDs are supplied oldest-first. An eligible protected job
    that cannot join residents reserves the next solo slot: no younger refill.
    Individually oversized/incompatible jobs are reported and skipped, so they
    cannot freeze unrelated work. Without a placement callback only solo starts
    are allowed; a supplied callback must validate matching group evidence as
    well as retry eligibility. Compatibility and retry restrictions must be
    the SAME callbacks as execution; this helper never overrides them.
    """
    queue, residents, protected = tuple(queue), tuple(residents), tuple(protected)
    if len(set(queue)) != len(queue) or set(queue) & set(residents):
        raise ValueError("ambiguous placement queue")
    if concurrency_limit is not None and (isinstance(concurrency_limit, bool) or not isinstance(concurrency_limit, int) or concurrency_limit < 1):
        raise ValueError("invalid concurrency limit")
    memory = memory or {}
    cap = None if memory_capacity is None else number(memory_capacity)
    mem = lambda j: number(memory[j]) if memory else 0.0
    fits_solo = lambda j: cap is None or mem(j) <= cap
    compatible = compatible or (lambda group: True)
    placement_allowed = placement_allowed or (lambda group: len(group) == 1)
    order = [j for j in protected if j in queue] + [j for j in queue if j not in protected]
    chosen, blocked = [], []
    for j in order:
        if not fits_solo(j) or not compatible((j,)):
            blocked.append({"job": j, "reason": "solo_resource_or_compatibility_blocked"})
            continue
        group = tuple(sorted((*residents, *chosen, j)))
        legal = ((concurrency_limit is None or len(group) <= concurrency_limit)
                 and (cap is None or sum(mem(k) for k in group) <= cap)
                 and compatible(group) and placement_allowed(group))
        if legal:
            chosen.append(j)
        elif j in protected:
            blocked.append({"job": j, "reason": "protected_drain_to_legal_slot"})
            break
        else:
            blocked.append({"job": j, "reason": "placement_blocked"})
    return {"selected": tuple(chosen), "blocked": blocked}


def _costs(jobs, costs):
    if costs is None:
        return {j: CompletionCosts() for j in jobs}
    if set(costs) != set(jobs):
        raise ValueError("missing or extra phase costs")
    return {j: c if isinstance(c, CompletionCosts) else CompletionCosts(**c) for j, c in costs.items()}


def _rates(group, solo, evidence, context):
    key = tuple(sorted(group))
    value = evidence.get(key)
    if value is not None:
        provenance = {"source": "caller_validated_exact_group_measurement", "group": list(key)}
        if "rates" in value:
            record = value
            expected = context.get("evidence_identity")
            if expected is not None and record.get("identity") != expected:
                raise ValueError("unknown group: evidence identity mismatch " + repr(key))
            if "membership_generation" in context and record.get("membership_generation") != context["membership_generation"]:
                raise ValueError("unknown group: membership generation mismatch " + repr(key))
            if "snapshot_at" in context and ("observed_at" not in record or "max_age_seconds" not in record):
                raise ValueError("unknown group: evidence freshness missing " + repr(key))
            if "observed_at" in record or "max_age_seconds" in record:
                age = number(context["snapshot_at"]) - number(record["observed_at"])
                if age < 0 or age > number(record["max_age_seconds"]):
                    raise ValueError("unknown group: stale evidence " + repr(key))
            if record.get("valid", True) is not True:
                raise ValueError("unknown group: invalid evidence " + repr(key))
            value = record["rates"]
            provenance.update(source="matching_exact_group_measurement",
                              identity=record.get("identity"), observed_at=record.get("observed_at"),
                              membership_generation=record.get("membership_generation"))
        elif any(k in context for k in ("evidence_identity", "membership_generation", "snapshot_at")):
            raise ValueError("unknown group: evidence identity missing " + repr(key))
        if set(value) != set(key):
            raise ValueError("exact-group rate membership mismatch")
        return {j: number(value[j], positive=True) for j in key}, provenance
    if len(key) == 1:
        j = key[0]
        return {j: number(solo[j], positive=True)}, {"source": "predicted_solo", "group": [j],
            "calibration": "uncalibrated_prior"}
    raise ValueError("unknown exact training group " + repr(key))


def forecast_continuation(remaining, initial, solo, queue, evidence=None, costs=None, *,
                          phase_state=None, memory=None, memory_capacity=None,
                          concurrency_limit=None, protected=(), compatible=None,
                          placement_allowed=None, resumed=(), queue_order=None, context=None,
                          max_events=128, cpu_budget_ms=50, candidate_count=1,
                          _clock=perf_counter, _deadline=None):
    """Return complete relative deliveries or explicit inconclusive, never a tail guess.

    Refills require exact matching evidence or a singleton prediction; unknown
    prospective packed groups are blocked with a trace reason, and drain to solo.
    Unknown already-running/surviving training groups are inconclusive.
    All submitted outstanding jobs must be in initial/queue (delivering jobs are
    initial too). Initial residents keep continuity. Refill occurs at snapshot
    time and legal resource releases, not at training completion while VRAM is
    held. Simultaneous transitions happen before refill. Every rollout includes
    the entire cohort even though only <=6 admission candidates may be inspected.
    ``_clock`` is an inert-test seam; production uses perf_counter.
    """
    started = _clock()
    result = {"status": "inconclusive", "reason": None, "completions": {}, "events": [],
              "phases": [], "event_count": 0, "candidate_count": candidate_count,
              "rate_lookups": 0, "provenance": "frozen_reference_continuation",
              "phase_interference": "not_inferred_from_steady_training_rates"}
    try:
        if (isinstance(candidate_count, bool) or not isinstance(candidate_count, int)
                or not 0 <= candidate_count <= 6):
            raise ValueError("six candidate limit exceeded")
        if isinstance(max_events, bool) or not isinstance(max_events, int) or not 1 <= max_events <= 128:
            raise ValueError("event limit must be within 1..128")
        budget = number(cpu_budget_ms, positive=True)
        if budget > 50:
            raise ValueError("foreground CPU budget exceeds 50 ms")
        deadline = min(started + budget / 1000, _deadline) if _deadline is not None else started + budget / 1000
        def check():
            if _clock() >= deadline:
                raise ValueError("foreground CPU time limit")
        check()
        initial, queue = tuple(initial), list(queue)
        if (len(set(initial)) != len(initial) or len(set(queue)) != len(queue)
                or set(initial) & set(queue) or set(remaining) != set(initial) | set(queue)):
            raise ValueError("incomplete or ambiguous outstanding cohort")
        if memory is not None and set(memory) != set(remaining):
            raise ValueError("memory envelope must cover complete cohort")
        left = {j: number(v) for j, v in remaining.items()}
        solo = dict(solo)
        if set(solo) - set(left):
            raise ValueError("extra solo prediction")
        # Missing solos prevent packed scoring but a valid singleton measurement
        # can still support normal solo execution.
        evidence, context = evidence or {}, context or {}
        for key in evidence:
            if not isinstance(key, tuple) or key != tuple(sorted(set(key))) or not key:
                raise ValueError("evidence requires sorted exact tuple job identities")
        costs = _costs(left, costs)
        states, timers = {}, {}
        supplied = phase_state or {}
        if set(supplied) - set(left):
            raise ValueError("unknown phase identity")
        elapsed = 0.0
        ordered = tuple(queue_order) if queue_order is not None else (*queue, *initial)
        if len(set(ordered)) != len(ordered) or not set(queue).issubset(ordered):
            raise ValueError("invalid original FIFO order")
        order = {j: i for i, j in enumerate(ordered)}
        if any(supplied.get(j, {}).get("phase") == "pausing" and j not in order for j in initial):
            raise ValueError("pausing job missing original FIFO order")
        for j in initial:
            raw = supplied.get(j, {})
            phase = raw.get("phase", "training" if left[j] > 0 else "finishing")
            if phase not in _PHASES or phase == "paused":
                raise ValueError("invalid resident phase")
            if phase in {"finishing", "delivering", "done"} and left[j] != 0:
                raise ValueError("finished phase has remaining training work")
            if phase in {"starting", "training", "pausing"} and left[j] == 0:
                phase = "finishing"
            states[j] = phase
            if phase == "starting": timers[j] = number(raw.get("startup", costs[j].startup))
            if phase == "finishing": timers[j] = number(raw.get("finish", costs[j].finish))
            if phase == "delivering": timers[j] = number(raw.get("delivery", costs[j].delivery))
            if phase == "pausing": timers[j] = number(raw.get("pause", costs[j].pause))
            if phase == "done": result["completions"][j] = 0.0
        for j in queue:
            if supplied.get(j, {}).get("phase", "paused") != "paused":
                raise ValueError("queued job must be paused/not started")
            states[j] = "paused"
        def emit(j, event):
            if result["event_count"] >= max_events:
                raise ValueError("rollout event limit")
            check()
            result["event_count"] += 1
            result["events"].append({"at": elapsed, "job": j, "event": event})
        holders = [j for j in states if states[j] in {"starting", "training", "finishing", "pausing"}]
        if memory is not None and memory_capacity is not None and sum(number(memory[j]) for j in holders) > number(memory_capacity):
            raise ValueError("initial residents exceed admission memory")
        refill = True
        while len(result["completions"]) < len(left):
            check()
            # Resolve zero-duration lifecycle edges before selecting new starts.
            again = True
            while again:
                again = False
                for j in sorted(states):
                    phase = states[j]
                    if phase == "training" and left[j] <= 1e-10:
                        left[j] = 0.0
                        states[j], timers[j] = "finishing", costs[j].finish
                        emit(j, "training_complete")
                        again = True
                    elif phase in {"starting", "finishing", "delivering", "pausing"} and timers[j] <= 1e-10:
                        del timers[j]
                        if phase == "starting":
                            states[j] = "training" if left[j] else "finishing"
                            if not left[j]: timers[j] = costs[j].finish
                            emit(j, "startup_ready")
                        elif phase == "finishing":
                            states[j], timers[j] = "delivering", costs[j].delivery
                            emit(j, "gpu_release")
                            refill = True
                        elif phase == "pausing":
                            states[j] = "paused"
                            # Preserve original FIFO location, not a serial tail.
                            queue.append(j)
                            queue.sort(key=lambda k: order[k])
                            emit(j, "pause_release")
                            refill = True
                        else:
                            states[j] = "done"
                            result["completions"][j] = elapsed
                            emit(j, "result_delivery")
                        again = True
            if len(result["completions"]) == len(left): break
            if refill:
                residents = tuple(j for j in states if states[j] in {"starting", "training", "finishing", "pausing"})
                unknown = []
                def legal_placement(group):
                    # Refill never assumes a future exploratory trial. The same
                    # caller placement restriction is used by execution fallback.
                    if placement_allowed is not None and not placement_allowed(group):
                        return False
                    productive = tuple(j for j in group if left[j] > 0 and states[j] not in {"pausing", "finishing", "delivering", "done"})
                    if productive:
                        try:
                            _rates(productive, solo, evidence, context)
                        except _ERRORS as exc:
                            unknown.append({"group": list(sorted(productive)), "reason": "blocked_unknown_group", "detail": str(exc)})
                            return False
                    return True
                choice = reference_continuation(queue, residents, memory=memory,
                    memory_capacity=memory_capacity, concurrency_limit=concurrency_limit,
                    protected=protected, compatible=compatible, placement_allowed=legal_placement)
                result.setdefault("placement_checks", []).append({"at": elapsed, **choice, "blocked_unknown": unknown})
                for j in choice["selected"]:
                    queue.remove(j)
                    states[j] = "starting" if left[j] else "finishing"
                    timers[j] = (costs[j].resume if j in resumed else costs[j].startup) if left[j] else costs[j].finish
                    emit(j, "resume" if j in resumed else "start")
                refill = False
                if choice["selected"]: continue
            group = tuple(sorted(j for j in states if states[j] == "training"))
            rates, provenance = ({}, {"source": "no_training", "group": []})
            if group:
                rates, provenance = _rates(group, solo, evidence, context)
                result["rate_lookups"] += 1
            durations = [timers[j] for j in timers]
            durations.extend(left[j] / rates[j] for j in group)
            if not durations:
                raise ValueError("no executable legal continuation for remaining cohort")
            duration = number(min(durations), positive=True)
            result["phases"].append({"start": elapsed, "end": elapsed + duration,
                "members": list(group), "rates": dict(rates), **provenance,
                "resource_holders": sorted(j for j in states if states[j] in {"starting", "training", "finishing", "pausing"})})
            for j in group: left[j] = max(0.0, left[j] - duration * rates[j])
            for j in timers: timers[j] = max(0.0, timers[j] - duration)
            elapsed += duration
        check()
        result.update(status="complete", reason="complete_known_cohort", remaining_updates=left)
    except _ERRORS as exc:
        result["reason"] = str(exc)
    result["cpu_ms"] = max(0.0, (_clock() - started) * 1000)
    return result


def compare_continuations(*, remaining, solo, requests, completed, now, alternatives,
                          evidence_halves, costs=None, min_gain_seconds=1.0,
                          candidate_count=1, _clock=perf_counter, **options):
    """Score complete admit/defer continuations; reusable for group bootstrap.

    ``alternatives`` contains admit/defer dictionaries with initial, queue and
    optional phase_state/resumed/placement_allowed overrides. Neither requires
    incumbents or fabricated pre-add measurements. Exactly two evidence maps
    are required. Bootstrap cleanup victims and future costs are explicit here.
    """
    start = _clock()
    result = {"admission_policy": POLICY, "objective": "request_feedback_latency",
              "decision": "inconclusive", "halves": [], "error": None,
              "decision_basis": "nominal_request_latency", "candidate_count": candidate_count}
    try:
        now = number(now)
        number(min_gain_seconds, positive=True)
        if set(alternatives) != {"admit", "defer"} or len(evidence_halves) != 2:
            raise ValueError("two alternatives and two evidence halves required")
        required = [j for spec in requests for j in spec.required_job_ids]
        if (len(required) != len(set(required)) or set(required) != set(remaining) | set(completed)
                or set(remaining) & set(completed) or any(s.submitted_at > now for s in requests)
                or any(number(v) > now for v in completed.values())):
            raise ValueError("request cohort must cover exactly all known arrived jobs")
        budget = number(options.get("cpu_budget_ms", 50), positive=True)
        deadline = start + min(50, budget) / 1000
        for evidence in evidence_halves:
            h = {}
            for name, alternative in alternatives.items():
                args = {**options, **alternative}
                rollout = forecast_continuation(remaining, solo=solo, evidence=evidence,
                    costs=costs, candidate_count=candidate_count, _clock=_clock,
                    _deadline=deadline, **args)
                h[name + "_trace"] = rollout
                if rollout["status"] != "complete":
                    result["failed_trace"] = rollout
                    raise ValueError(name + ": " + rollout["reason"])
                absolute = {**completed, **{j: now + t for j, t in rollout["completions"].items()}}
                h[name] = {**score_requests(requests, absolute), "job_completions": absolute}
                h[name + "_phases"] = rollout["phases"]
            gain = (h["defer"]["weighted_latency_seconds"] - h["admit"]["weighted_latency_seconds"]) / h["defer"]["weight_sum"]
            h.update(gain_seconds=gain, decision_score_seconds=gain)
            result["halves"].append(h)
        passes = [h["decision_score_seconds"] >= min_gain_seconds for h in result["halves"]]
        result["decision"] = "accepted" if all(passes) else "rejected" if not any(passes) else "inconclusive"
    except _ERRORS as exc:
        result["error"] = str(exc)
    result["cpu_ms"] = max(0.0, (_clock() - start) * 1000)
    return result


def compare_feedback_admission(*, remaining, solo, incumbents, candidate, queue,
                               pre_halves, packed_halves, requests, completed, now,
                               costs=None, min_gain_seconds=1.0, bracket_admission_mode="nominal",
                               evidence=None, phase_state=None, context=None, passive=(),
                               memory=None, memory_capacity=None, concurrency_limit=None,
                               protected=(), compatible=None, placement_allowed=None, candidate_count=1,
                               max_events=128, cpu_budget_ms=50, _clock=perf_counter):
    """Drop-in v4 adapter: both halves and original bracket/min-gain rule.

    The deferred candidate preserves its FIFO place and resumes at a legal slot,
    but cannot immediately retry the unchanged rejected pack. Incumbents continue
    productive work during the candidate's future pause. No sunk probe cost.
    Extra evidence is keyed by exact group; rates may be two-half tuples, supplied
    as ``{'halves': [record0, record1]}``, or a frozen shared record/vector.
    """
    start = _clock()
    try:
        incumbents, queue, passive = tuple(sorted(incumbents)), tuple(queue), tuple(passive)
        if (not incumbents or len(set(incumbents)) != len(incumbents)
                or candidate not in queue or len(set(queue)) != len(queue)
                or set(incumbents) & set(queue)
                or len(set(passive)) != len(passive) or set(passive) & (set(incumbents) | set(queue))
                or set(remaining) != set(incumbents) | set(queue) | set(passive)):
            raise ValueError("invalid outstanding cohort or queue")
        if set(solo) != set(remaining):
            raise ValueError("missing or extra solo predictions")
        for j in remaining:
            number(remaining[j])
            number(solo[j], positive=True)
        costs = _costs(remaining, costs)
        proposed = tuple(sorted((*incumbents, candidate)))
        if set(pre_halves) != set(incumbents) or set(packed_halves) != set(proposed):
            raise ValueError("measurement membership mismatch")
        for values in [*pre_halves.values(), *packed_halves.values()]:
            if len(values) != 2: raise ValueError("two measurement halves required")
            for v in values: number(v, positive=True)
        if bracket_admission_mode not in {"nominal", "recovery_regret"}:
            raise ValueError("unsupported bracket admission mode")
        states = dict(phase_state or {})
        if any(remaining[j] != 0 or states.get(j, {}).get('phase') not in {'delivering', 'done'} for j in passive):
            raise ValueError('passive members require resource-free result-delivery proof')
        admit_states = dict(states)
        admit_states[candidate] = states.get(candidate, {"phase": "training" if remaining[candidate] else "finishing"})
        deferred_states = dict(states)
        # Zero remaining means preserve finishing/delivery, never checkpoint it.
        active_candidate = number(remaining[candidate]) > 0
        defer_initial = (*incumbents, *passive)
        defer_queue = list(queue)
        if not active_candidate:
            defer_initial = (*incumbents, candidate, *passive)
            defer_queue.remove(candidate)
            deferred_states[candidate] = admit_states[candidate]
        elif costs[candidate].pause:
            defer_initial = (*incumbents, candidate, *passive)
            defer_queue.remove(candidate)
            deferred_states[candidate] = {"phase": "pausing", "pause": costs[candidate].pause}
        else:
            deferred_states[candidate] = {"phase": "paused"}
        alternatives = {
            "admit": {"initial": (*proposed, *passive), "queue": [j for j in queue if j != candidate], "phase_state": admit_states},
            "defer": {"initial": defer_initial, "queue": defer_queue, "phase_state": deferred_states,
                      "resumed": (candidate,), "queue_order": queue,
                      "placement_allowed": lambda group: group != proposed and (placement_allowed is None or placement_allowed(group))}}
        # Queue priority of a paused candidate is passed explicitly into rollout.
        options = dict(context=context, memory=memory, memory_capacity=memory_capacity,
            concurrency_limit=concurrency_limit, protected=protected, compatible=compatible,
            placement_allowed=placement_allowed, candidate_count=candidate_count, max_events=max_events, cpu_budget_ms=cpu_budget_ms,
            _clock=_clock)
        maps = []
        for half in range(2):
            groups = {}
            for key, value in (evidence or {}).items():
                groups[key] = value["halves"][half] if "halves" in value else value
            # Trial vectors are fresh matching evidence for productive members.
            # Finished members are not rates for the surviving subset.
            for members, values, label in [(incumbents, pre_halves, "pre"), (proposed, packed_halves, "packed")]:
                if all(number(remaining[j]) > 0 for j in members):
                    vector = {j: values[j][half] for j in members}
                    if context:
                        record = {"rates": vector}
                        if "evidence_identity" in context: record["identity"] = context["evidence_identity"]
                        if "membership_generation" in context: record["membership_generation"] = context["membership_generation"]
                        if "snapshot_at" in context:
                            record["observed_at"] = context.get(label + "_observed_at", context.get("trial_observed_at", context["snapshot_at"]))
                            record["max_age_seconds"] = context.get("trial_max_age_seconds", 0.)
                        groups[members] = record
                    else:
                        groups[members] = vector
            maps.append(groups)
        # One foreground deadline across nominal and the optional v4 scenario.
        options["cpu_budget_ms"] = max(1e-9, min(cpu_budget_ms, 50) - (_clock() - start) * 1000)
        result = compare_continuations(remaining=remaining, solo=solo, requests=requests,
            completed=completed, now=now, alternatives=alternatives, evidence_halves=maps,
            costs=costs, min_gain_seconds=min_gain_seconds, **options)
        robust = bracket_admission_mode == "recovery_regret" and len(requests) == 1 and requests[0].completion_policy == "group"
        result["decision_basis"] = "bracket_recovery_regret_v1" if robust else "nominal_request_latency"
        if len(result["halves"]) == 2 and result["error"] is None:
            for half, h in enumerate(result["halves"]):
                pre = {j: pre_halves[j][half] for j in incumbents}
                packed = {j: packed_halves[j][half] for j in proposed}
                pminus = sum(pre[j] / solo[j] for j in incumbents)
                pplus = sum(packed[j] / solo[j] for j in proposed)
                h.update(normalized_progress_before=pminus, normalized_progress_after=pplus,
                    normalized_progress_ratio=pplus / pminus,
                    incumbent_slowdown_ratios={j: pre[j] / packed[j] for j in incumbents})
                if robust:
                    cautious_solo = dict(solo)
                    cautious_solo[candidate] = min(solo[candidate], packed[candidate])
                    options["cpu_budget_ms"] = max(1e-9, min(cpu_budget_ms, 50) - (_clock() - start) * 1000)
                    scenario = compare_continuations(remaining=remaining, solo=cautious_solo,
                        requests=requests, completed=completed, now=now, alternatives=alternatives,
                        evidence_halves=[maps[half], maps[half]], costs=costs,
                        min_gain_seconds=min_gain_seconds, **options)
                    if scenario["error"]:
                        result.update(decision="inconclusive", error=scenario["error"])
                        break
                    cautious = scenario["halves"][0]
                    gain, cgain = h["gain_seconds"], cautious["gain_seconds"]
                    ar, dr = max(0., -gain, -cgain), max(0., gain, cgain)
                    h.update(decision_score_seconds=dr-ar, admit_max_regret_seconds=ar,
                        defer_max_regret_seconds=dr, no_recovery_scenario={
                            "candidate_solo_updates_per_second": cautious_solo[candidate],
                            "source": "hypothesis_from_current_packed_rate_not_a_solo_measurement",
                            **cautious})
            if result["error"] is None:
                passes = [h["decision_score_seconds"] >= min_gain_seconds for h in result["halves"]]
                result["decision"] = "accepted" if all(passes) else "rejected" if not any(passes) else "inconclusive"
        result["cpu_ms"] = max(0., (_clock() - start) * 1000)
        return result
    except _ERRORS as exc:
        return {"admission_policy": POLICY, "objective": "request_feedback_latency",
                "decision": "inconclusive", "halves": [], "error": str(exc),
                "cpu_ms": max(0., (_clock() - start) * 1000)}
