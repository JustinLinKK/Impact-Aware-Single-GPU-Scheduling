"""Non-preemptive, arrived-only resource-aware ready-set scheduling.

A bounded rollout compares dispatch sets, including waiting for a resident to
finish. Estimates are whole-program work, not measured optimizer progress.
Unknown pair interference is explicitly extrapolated from independent history;
no evaluation result or future source is admitted to the model.
"""
from itertools import combinations
import math
import statistics
import time


def equal_counts(keys, total=6):
    keys = list(keys)
    if not keys:
        return {}
    if len(keys) > 3:
        raise ValueError('three-resident envelope')
    q, r = divmod(total, len(keys))
    return {k: q + (i < r) for i, k in enumerate(keys)}


class Costs:
    def __init__(self, history, reports, interference=True, require_measured_triples=False,
                 joint_cpu=False, budget_guard=False, budget_fraction=.9):
        self.reports = reports
        self.interference = interference
        self.require_measured_triples = require_measured_triples
        self.joint_cpu = joint_cpu
        if not math.isfinite(budget_fraction) or not 0 < budget_fraction <= 1:
            raise ValueError('invalid budget prediction margin')
        self.budget_guard = budget_guard
        self.budget_fraction = budget_fraction
        self.native_budgets = {}
        for k, report in reports.items():
            hint = report.get('native_training_budget', {})
            seconds = hint.get('seconds')
            if (hint.get('status') == 'inferred' and
                hint.get('source_sha256') == report.get('source_sha256') and
                type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0):
                self.native_budgets[k] = seconds
        self.profiles = {k: {int(c): p for c, p in r['cpu_profiles'].items()}
                         for k, r in reports.items() if r.get('status') == 'estimated'}
        self.signatures = {k: r['source_sha256'] for k, r in reports.items()}
        self.groups = {}
        solo = {}
        for r in history['profiles']:
            if r['cpu_count'] == 6 and r.get('complete'):
                solo[r['source_sha256']] = r
        coefficients = {}
        for group in history['co_runs']:
            if not group.get('complete'):
                continue
            members = group['members']
            sig = tuple(sorted((m['source_sha256'], m['cpu_count']) for m in members))
            self.groups[sig] = {m['source_sha256']: m['overlap_factor'] for m in members}
            # Only independent pair measurements with equal 3+3 allocations fit
            # the extrapolator; target identities are not task-order preferences.
            if len(members) == 2 and all(m['cpu_count'] == 3 for m in members):
                for m in members:
                    peer = next(x for x in members if x is not m)
                    duty = solo.get(peer['source_sha256'], {}).get('gpu_busy_fraction')
                    if duty is not None and duty > .05:
                        coefficients.setdefault(m['source_sha256'], []).append(
                            max(0., m['overlap_factor'] - 1) / duty)
        self.coefficients = {s: min(2., statistics.median(v)) for s, v in coefficients.items()}
        self.cache = {}
        self.window_cache = {}

    def solo(self, k):
        return self.profiles[k][6]['runtime_seconds']

    def memory(self, k):
        return max(p['planning_memory_mib'] for p in self.profiles[k].values())

    def duty(self, k):
        v = self.profiles[k][6].get('gpu_busy_fraction')
        return 1. if v is None else v

    def cpu(self, k):
        v = self.profiles[k][6].get('cpu_pressure_lower_bound')
        return 6. if v is None else 6 * v

    def allocations(self, keys):
        keys = tuple(keys)
        yield equal_counts(keys)
        if self.joint_cpu and len(keys) == 2:
            yield dict(zip(keys,(2,4)))
            yield dict(zip(keys,(4,2)))

    def admission_budget_ok(self, keys, rates):
        """Risk filter, not a promise of meeting a native training deadline.

        The whole-program forecast under a stationary combination is compared
        with the source's training-only clock. This is intentionally conservative.
        Singleton execution remains possible; estimates never kill a resident.
        """
        return (not self.budget_guard or len(keys) <= 1 or
                all(k not in self.native_budgets or
                    1 / rates[k] <= self.budget_fraction * self.native_budgets[k]
                    for k in keys))

    def rates(self, keys, counts=None):
        keys = tuple(keys)
        counts = equal_counts(keys) if counts is None else dict(counts)
        if set(counts) != set(keys) or (keys and sum(counts.values()) != 6):
            raise ValueError('rates require the actual six-core allocation')
        if any(type(c) is not int or c < 2 for c in counts.values()):
            raise ValueError('invalid CPU count')
        cache_key=(keys,tuple(counts[k] for k in keys))
        if cache_key in self.cache:
            return self.cache[cache_key]
        if any(k not in self.profiles or c not in self.profiles[k] for k, c in counts.items()):
            return None, 'unknown_solo'
        sig = tuple(sorted((self.signatures[k], c) for k, c in counts.items()))
        exact = self.groups.get(sig) if len(set(self.signatures[k] for k in keys)) == len(keys) else None
        if self.require_measured_triples and len(keys) > 2 and not exact:
            return None, 'unmeasured_triple_not_admitted'
        rates = {}
        evidence = 'solo' if len(keys) == 1 else 'interference_disabled'
        for k, c in counts.items():
            factor = 1.
            if self.interference and len(keys) > 1:
                if exact:
                    factor = exact[self.signatures[k]]
                    evidence = 'exact_measured_group'
                else:
                    peers = sum(self.duty(j) for j in keys if j != k)
                    coefficient = self.coefficients.get(self.signatures[k], 1.)
                    factor = 1 + coefficient * peers
                    evidence = 'extrapolated_pair_sensitivity_not_measured_group'
            rates[k] = 1 / (self.profiles[k][c]['runtime_seconds'] * factor)
        self.cache[cache_key] = rates, evidence
        return rates, evidence


def representatives(waiting, costs, limit):
    """Keep FIFO feasible, but look across the entire arrived pool for diversity.

    Similar resource shapes must not hide a complementary job behind the first
    eight entries, including when every job has different source bytes. This
    does not inspect future sources or assign hand-designed task classes.
    Selection preserves original relative order; the optimizer chooses subsets.
    """
    if limit < 3:
        raise ValueError('window must retain the three-slot FIFO reference')
    waiting = list(waiting)
    key = (tuple(waiting), limit)
    if key in costs.window_cache:
        return list(costs.window_cache[key])
    if len(waiting) <= limit:
        return waiting
    raw = {k: (costs.cpu(k), costs.duty(k), math.log1p(costs.solo(k)),
               math.log1p(costs.memory(k))) for k in waiting}
    minima = [min(v[i] for v in raw.values()) for i in range(4)]
    spans = [max(v[i] for v in raw.values()) - minima[i] for i in range(4)]
    features = {k: tuple((v[i]-minima[i])/spans[i] if spans[i] else 0.
                         for i in range(4)) for k, v in raw.items()}
    chosen = waiting[:3]
    while len(chosen) < min(limit, len(waiting)):
        def distance(k):
            return min(sum((a-b)**2 for a,b in zip(features[k],features[j]))
                       for j in chosen)
        # Stable ties keep the older instance. Once all shapes are represented,
        # identical instances simply fill remaining positions in FIFO order.
        chosen.append(max((k for k in waiting if k not in chosen),key=distance))
    result = [k for k in waiting if k in chosen]
    costs.window_cache[key] = tuple(result)
    return result


def actions(waiting, work, costs, capacity, external, *, limit=8, force=None):
    eligible = representatives(waiting, costs, limit)
    if force in waiting and force not in eligible:
        eligible[-1:] = [force]
    for n in range(min(len(eligible), 3 - len(work)) + 1):
        for add in combinations(eligible, n):
            if not work and not add:
                continue
            if force in waiting and n and force not in add:
                continue
            keys = (*work, *add)
            if sum(costs.memory(k) for k in keys) + external > capacity:
                continue
            for counts in costs.allocations(keys):
                rates, evidence = costs.rates(keys,counts)
                if rates is not None and (not add or costs.admission_budget_ok(keys, rates)):
                    yield tuple(add), rates, evidence, counts


def advance(waiting, work, add, rates):
    current = {**work, **{k: 1. for k in add}}
    dt = min(current[k] / rates[k] for k in current)
    rest = {k: v - dt * rates[k] for k, v in current.items() if v - dt * rates[k] > 1e-8}
    return [k for k in waiting if k not in add], rest, dt


def rollout(waiting, work, costs, capacity, external, rule, deadline):
    elapsed = 0.
    while waiting or work:
        if time.perf_counter() > deadline:
            raise TimeoutError('bounded rollout')
        options = list(actions(waiting, work, costs, capacity, external))
        if not options:
            return math.inf
        def score(option):
            add, rates, _, _ = option
            if rule == 'fifo':
                prefix = tuple(waiting[:len(add)])
                return (int(add == prefix), len(add), 0.)
            productivity = sum(rates[k] * costs.solo(k) for k in rates)
            remaining = {**work, **{k: 1. for k in add}}
            tail = sum(remaining[k] * costs.solo(k) for k in rates)
            if rule == 'long':
                productivity *= 1 + .25 * tail / max(1., sum(costs.solo(k) for k in [*waiting, *work]))
            return (productivity, tail, len(add))
        chosen = max(options, key=score)
        waiting, work, dt = advance(waiting, work, chosen[0], chosen[1])
        elapsed += dt
    return elapsed


def choose(ready, remaining, costs, *, accepted, now, used_mib, capacity_mib,
           policy='lookahead', max_wait=180., budget_ms=150., lookahead=8,
           finite_cohort=False, external_floor_mib=0.):
    """No future arrivals, FIFO-head requirement, checkpoint or source rewrites."""
    start = time.perf_counter()
    if not math.isfinite(external_floor_mib) or external_floor_mib < 0:
        raise ValueError('non-job memory reservation must be finite and nonnegative')
    if max_wait is None and not finite_cohort:
        raise ValueError('disabling age protection requires an explicit finite cohort')
    if max_wait is not None and (not math.isfinite(max_wait) or max_wait <= 0):
        raise ValueError('max_wait must be a positive finite duration or None')
    ready = list(ready)
    remaining = dict(remaining)
    if set(ready) & set(remaining) or len(ready) != len(set(ready)):
        raise ValueError('duplicate ready/resident identity')
    if any(not math.isfinite(v) or v <= 0 for v in remaining.values()):
        raise ValueError('invalid work estimate')
    base = dict(admit=[], cpu_counts=equal_counts(remaining), future_arrivals_used=False,
                estimates_are_actual_progress=False, forecasts=[], elapsed_ms=0.,
                non_job_memory_reservation_mib=external_floor_mib,
                fairness_mode='finite_cohort_drain' if max_wait is None else 'queued_age_protection',
                native_budget_risk_guard=costs.budget_guard,
                native_budget_fraction=costs.budget_fraction if costs.budget_guard else None)
    if not ready and not remaining:
        return {**base, 'reason': 'idle'}
    force = next((k for k in ready if max_wait is not None and now - accepted[k] >= max_wait), None)
    unknown_residents = [k for k in remaining if k not in costs.profiles]
    unknown = [k for k in ready if k not in costs.profiles]
    if unknown_residents or unknown:
        # A missing report in the queue tail must not disable useful scheduling
        # among known arrived tasks. Unknown execution is still exclusive; its
        # resources are never guessed from another source. Preserve FIFO modes.
        exclusive = (unknown_residents or force in unknown or
                     (policy in ('fifo','v511') and ready and ready[0] in unknown) or
                     not any(k not in unknown for k in ready))
        if not exclusive:
            if policy in ('fifo','v511'):
                eligible = ready[:next(i for i,k in enumerate(ready) if k in unknown)]
            else:
                eligible = [k for k in ready if k not in unknown]
            answer = choose(eligible,remaining,costs,accepted=accepted,now=now,
                            used_mib=used_mib,capacity_mib=capacity_mib,policy=policy,
                            max_wait=max_wait,budget_ms=budget_ms,lookahead=lookahead,
                            finite_cohort=finite_cohort,external_floor_mib=external_floor_mib)
            return {**answer,'unknown_deferred':unknown}
        add = [ready[0]] if ready and not remaining and used_mib < capacity_mib else []
        if force in unknown and add:
            add = [force]
        return {**base, 'admit': add, 'cpu_counts': equal_counts([*remaining, *add]),
                'reason': 'unknown_exclusive_fallback', 'unknown': unknown + unknown_residents}
    # Peak reservations are not observed current allocations. Subtracting them
    # from live memory cannot justify removing the caller's non-job reservation.
    external = max(external_floor_mib, used_mib - sum(costs.memory(k) for k in remaining))
    if policy == 'lpt':
        # Cost-only control: inspect the full arrived pool, not the diversity
        # window, so three long instances are genuinely eligible together.
        ordered = sorted(ready, key=lambda k: -costs.solo(k))
        if force is not None:
            ordered.remove(force)
            ordered.insert(0, force)
        add = []
        memory = external + sum(costs.memory(k) for k in remaining)
        for k in ordered:
            if len(remaining) + len(add) >= 3:
                break
            if memory + costs.memory(k) > capacity_mib:
                if k == force:
                    break
                continue
            proposed = [*remaining, *add, k]
            rates, _ = costs.rates(proposed)
            if rates is None or not costs.admission_budget_ok(proposed, rates):
                if k == force:
                    break
                continue
            add.append(k)
            memory += costs.memory(k)
        return {**base, 'admit': add, 'cpu_counts': equal_counts([*remaining, *add]),
                'reason': 'longest_predicted_solo_first_no_interference_scoring',
                'age_protected': force, 'selected_interference_evidence': 'not_used_by_lpt_control',
                'fifo_reference_admit': ready[:max(0, 3-len(remaining))],
                'elapsed_ms': 1000*(time.perf_counter()-start)}
    options = list(actions(ready, remaining, costs, capacity_mib, external, limit=lookahead, force=force))
    if not options:
        return {**base, 'reason': 'memory_wait'}
    fifo = [o for o in options if o[0] == tuple(ready[:len(o[0])])]
    if policy in ('fifo', 'v511') and not fifo:
        # An idle GPU and an infeasible head do not authorize a FIFO control
        # to launch a later job. Optimizing policies may still choose it.
        return {**base, 'reason': 'fifo_head_not_feasible',
                'fifo_reference_admit': [], 'age_protected': force,
                'elapsed_ms': 1000*(time.perf_counter()-start)}
    reference = max(fifo or options, key=lambda o: len(o[0]))
    if policy in ('fifo', 'v511'):
        best = reference
        reason = 'fifo_equal_cpu'
    elif policy == 'complement':
        def complement(o):
            keys = [*remaining, *o[0]]
            productivity = sum(o[1][k] * costs.solo(k) for k in keys)
            fit = sum(abs(costs.cpu(a)-costs.cpu(b)) * abs(costs.duty(a)-costs.duty(b))
                      for a,b in combinations(keys,2)
                      if (costs.cpu(a)-costs.cpu(b))*(costs.duty(a)-costs.duty(b)) < 0)
            return productivity + .1 * fit
        best = max(options, key=complement)
        reason = 'complementary_ready_set'
    else:
        # Always complete a cheap reference first, then retain every completed
        # alternative if the budget expires. Never discard a useful result and
        # silently revert to FIFO solely because exhaustive search timed out.
        deadline = start + budget_ms / 1000
        ordered = [reference] + sorted([o for o in options if o is not reference],
            key=lambda o: sum(o[1][k]*costs.solo(k) for k in o[1]), reverse=True)
        best = reference
        best_value = math.inf
        timed_out = False
        for option in ordered:
            wait, work, dt = advance(ready, remaining, option[0], option[1])
            try:
                tail = min(rollout(wait[:], work.copy(), costs, capacity_mib, external, rule, deadline)
                           for rule in ('throughput','long','fifo'))
            except TimeoutError:
                timed_out = True
                break
            value = dt + tail
            base['forecasts'].append(dict(admit=list(option[0]), seconds=value, evidence=option[2],cpu_counts=dict(option[3])))
            if value < best_value - 1e-6:
                best, best_value = option, value
        base['planning_budget_exhausted'] = timed_out
        reason = 'lookahead_ready_set' if best[0] != reference[0] or best[3] != reference[3] else 'forecast_retained_fifo_or_wait'
    add = list(best[0])
    return {**base, 'admit': add, 'cpu_counts': dict(best[3]),
            'reason': reason, 'selected_interference_evidence': best[2], 'age_protected': force,
            'fifo_reference_admit': list(reference[0]),'fifo_reference_cpu_counts':dict(reference[3]),
            'elapsed_ms': 1000*(time.perf_counter()-start)}
