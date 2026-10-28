"""Closed-loop batch and sequential/pipelined submissions using v4's live scorer.

Run from the MLEvolve parent:
python -m localml_scheduler.benchmarks.feedback_v4.submission_scenarios --plot
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
from math import ceil
from pathlib import Path
from statistics import mean

from .simulate import BUDGET, CASES, EPS, JOBS, MEMORY, SOLO, VRAM, hidden_rates
from ...scheduler.feedback_objective import FeedbackRequest, compare_feedback_admission, score_requests

SCENARIOS = ('batch_barrier', 'sequential_pipeline')
POLICIES = ('fixed_three_fifo', 'v4_previous', 'v4_request_feedback')
JOB_IDS = tuple(j + str(round_) for round_ in (1, 2) for j in JOBS)
THINK_MINUTES = 10 / 60
LABELS = {'batch_barrier': 'MLMaster 2-style batch barrier',
          'sequential_pipeline': 'MLEvolve-style sequential pipeline',
          'fixed_three_fifo': 'Fixed-three FIFO', 'v4_previous': 'V4 previous',
          'v4_request_feedback': 'V4 optimized bracket'}


def simulate(scenario, policy, case='reference', *, strict_single=False, workload=None):
    if scenario not in SCENARIOS or policy not in POLICIES or case not in CASES:
        raise ValueError('unknown experiment')
    if strict_single and scenario != 'sequential_pipeline':
        raise ValueError('single-outstanding control applies to the pipeline only')
    actual_solo = dict(SOLO, **({'C': 2.5} if case != 'reference' else {}))
    predicted_solo = dict(actual_solo if case == 'slow_C_accurate_prediction' else SOLO)
    workload = workload or {}
    budget = dict(workload.get('budget', BUDGET))
    actual_solo = dict(workload.get('actual_solo', actual_solo))
    predicted_solo = dict(workload.get('predicted_solo', predicted_solo))
    trial_minutes = float(workload.get('trial_minutes', 1.))
    from ...scheduler.feedback_objective import number
    for values in (budget, actual_solo, predicted_solo):
        if set(values) != set(JOBS):
            raise ValueError('workload must cover exactly the five job types')
        for value in values.values():
            number(value, positive=True)
    number(trial_minutes, positive=True)
    left = {j: budget[j[0]] for j in JOB_IDS}
    arrivals, done, requests = {}, {}, []
    active, pending, attempted = set(), [], set()
    events, decisions, segments = [], [], []
    releases = []
    time, serial, last_release = 0., 0, -THINK_MINUTES
    points = [(0., {j: 0. for j in JOB_IDS})]

    def schedule(when, members):
        nonlocal serial, last_release
        serial += 1
        heapq.heappush(releases, (when, serial, tuple(members)))
        last_release = max(last_release, when)

    if scenario == 'batch_barrier':
        schedule(0., [j + '1' for j in JOBS])
    elif strict_single:
        schedule(0., [JOB_IDS[0]])
    else:
        for i, j in enumerate(JOBS):
            schedule(i * THINK_MINUTES, [j + '1'])

    def release_due():
        changed = False
        while releases and releases[0][0] <= time + EPS:
            when, _, members = heapq.heappop(releases)
            changed = True
            for j in members:
                assert j not in arrivals
                arrivals[j] = when
                pending.append(j)
            events.append(dict(time=when, event='submit', jobs=list(members)))
            mode = 'group' if scenario == 'batch_barrier' else 'individual'
            requests.append(FeedbackRequest('request_' + '_'.join(members), scenario,
                mode, members, when * 60))
        if changed:
            attempted.clear()
        return changed

    def rates(members):
        # There is at most one live job per lineage, so the original five-type
        # interference law applies without inventing duplicate-type behavior.
        assert len({j[0] for j in members}) == len(members)
        by_type = hidden_rates([j[0] for j in members], actual_solo)
        return {j: by_type[j[0]] for j in sorted(members)}

    def advance(duration, members, phase):
        nonlocal time
        assert duration > 0
        vector = rates(members) if members else {}
        before = dict(left)
        for j in members:
            left[j] = max(0., left[j] - duration * vector[j])
        segments.append(dict(start=time, end=time + duration, members=sorted(members),
                             rates=vector, phase=phase, before=before, after=dict(left)))
        time += duration
        points.append((time, {j: budget[j[0]] - left[j] for j in JOB_IDS}))
        finished = sorted(j for j in members if left[j] < EPS)
        for j in finished:
            done[j] = time
            active.discard(j)
            if j in pending:
                pending.remove(j)
            events.append(dict(time=time, event='result_returned', jobs=[j]))
            if scenario == 'sequential_pipeline':
                if strict_single:
                    index = JOB_IDS.index(j) + 1
                    if index < len(JOB_IDS):
                        schedule(time + THINK_MINUTES, [JOB_IDS[index]])
                elif j.endswith('1'):
                    schedule(max(time + THINK_MINUTES, last_release + THINK_MINUTES), [j[0] + '2'])
        if scenario == 'batch_barrier' and finished and all(j + '1' in done for j in JOBS):
            if not any(j.endswith('2') for j in arrivals) and not releases:
                schedule(time + THINK_MINUTES, [j + '2' for j in JOBS])
        if finished:
            attempted.clear()
        return finished

    loops = 0
    while len(done) < len(JOB_IDS):
        loops += 1
        assert loops < 2000
        release_due()
        pending.sort(key=lambda j: (arrivals[j], j))
        if not active:
            if not pending:
                assert releases
                advance(releases[0][0] - time, set(), 'agent_think')
                continue
            anchor = pending.pop(0)
            active.add(anchor)
            decisions.append(dict(time=time, job=anchor, decision='accepted', reason='solo_dispatch'))
        if policy == 'fixed_three_fifo':
            for j in list(pending):
                if len(active) < 3 and sum(MEMORY[k[0]] for k in active) + MEMORY[j[0]] <= VRAM + EPS:
                    active.add(j)
                    pending.remove(j)
                    decisions.append(dict(time=time, job=j, decision='accepted', reason='fixed_three_dispatch'))
        else:
            overdue = [j for j in pending if time - arrivals[j] >= 30.]
            protected = overdue[0] if overdue else None
            candidate = next((j for j in pending if j not in attempted and
                (protected is None or j == protected) and
                sum(MEMORY[k[0]] for k in active) + MEMORY[j[0]] <= VRAM + EPS), None)
            if candidate is not None:
                incumbents = set(active)
                proposed = incumbents | {candidate}
                pre, post = rates(incumbents), rates(proposed)
                deadline = time + trial_minutes
                next_arrival = releases[0][0] if releases else float('inf')
                boundary = min(deadline, next_arrival, time + min(left[j] / post[j] for j in proposed))
                finished = advance(boundary - time, proposed, 'trial')
                arrived = release_due()
                if finished or arrived:
                    decisions.append(dict(time=time, job=candidate, decision='inconclusive',
                        reason='completion or arrival changed forecast cohort', completed=finished))
                    continue
                cohort = {j for j in arrivals if j not in done}
                current_requests = tuple(r for r in requests if not all(j in done for j in r.required_job_ids))
                required = {j for r in current_requests for j in r.required_job_ids}
                evidence = compare_feedback_admission(
                    remaining={j: left[j] for j in sorted(cohort)},
                    solo={j: predicted_solo[j[0]] / 60 for j in sorted(cohort)},
                    incumbents=incumbents, candidate=candidate, queue=pending,
                    pre_halves={j: [pre[j] / 60] * 2 for j in sorted(incumbents)},
                    packed_halves={j: [post[j] / 60] * 2 for j in sorted(proposed)},
                    requests=current_requests, completed={j: done[j] * 60 for j in sorted(done) if j in required},
                    now=time * 60, min_gain_seconds=1.,
                    bracket_admission_mode='recovery_regret' if policy == 'v4_request_feedback' else 'nominal')
                assert evidence['error'] is None, evidence
                decisions.append(dict(time=time, job=candidate, decision=evidence['decision'], evidence=evidence))
                attempted.add(candidate)
                if evidence['decision'] == 'accepted':
                    active.add(candidate)
                    pending.remove(candidate)
                continue
        vector = rates(active)
        boundary = time + min(left[j] / vector[j] for j in active)
        if releases:
            boundary = min(boundary, releases[0][0])
        advance(boundary - time, active.copy(), 'train')

    scores = score_requests(requests, {j: done[j] * 60 for j in JOB_IDS})
    feedback = sorted(r['latency_seconds'] / 60 for r in scores['requests'])
    jct = sorted(done[j] - arrivals[j] for j in JOB_IDS)
    metrics = dict(average_request_feedback_minutes=mean(feedback),
        p95_request_feedback_minutes=feedback[ceil(.95 * len(feedback)) - 1],
        max_request_feedback_minutes=max(feedback), request_count=len(requests),
        average_job_jct_minutes=mean(jct), p95_job_jct_minutes=jct[ceil(.95 * len(jct)) - 1],
        max_job_jct_minutes=max(jct), workflow_makespan_minutes=max(done.values()) - min(arrivals.values()),
        trial_elapsed_minutes=sum(s['end'] - s['start'] for s in segments if s['phase'] == 'trial'),
        finished_jobs=len(done))
    result = dict(scenario=scenario, policy=policy, case=case, strict_single=strict_single,
        budget=budget, trial_minutes=trial_minutes,
        actual_solo=actual_solo, predicted_solo=predicted_solo, arrivals=arrivals, completion=done,
        requests=scores['requests'], metrics=metrics, decisions=decisions, events=events,
        segments=segments, progress=points)
    verify(result)
    return result


def verify(result):
    processed, last = {j: 0. for j in JOB_IDS}, 0.
    for s in result['segments']:
        assert abs(s['start'] - last) < EPS
        assert sum(MEMORY[j[0]] for j in s['members']) <= VRAM + EPS
        if result['policy'] == 'fixed_three_fifo':
            assert len(s['members']) <= 3
        for j in s['members']:
            assert s['start'] + EPS >= result['arrivals'][j]
            processed[j] += (s['end'] - s['start']) * s['rates'][j]
        last = s['end']
    assert set(result['completion']) == set(JOB_IDS)
    for j in JOB_IDS:
        assert abs(processed[j] - result.get('budget', BUDGET)[j[0]]) < 1e-6
        assert result['completion'][j] >= result['arrivals'][j]
    assert abs(last - max(result['completion'].values())) < EPS
    assert all(r['weight'] == 1 for r in result['requests'])
    if result['scenario'] == 'batch_barrier':
        assert len(result['requests']) == 2
        expected = max(result['completion'][j + '1'] for j in JOBS) + THINK_MINUTES
        assert all(abs(result['arrivals'][j + '2'] - expected) < EPS for j in JOBS)
    elif not result['strict_single']:
        assert len(result['requests']) == 10
        assert all(result['arrivals'][j + '2'] + EPS >= result['completion'][j + '1'] + THINK_MINUTES for j in JOBS)
        ordered = sorted(result['arrivals'].values())
        assert all(b - a + EPS >= THINK_MINUTES for a, b in zip(ordered, ordered[1:]))


def report(results, controls):
    text = """# Bracket optimization: previous v4, optimized v4 and fixed-three FIFO

Synthetic experiment, 2026-10-28. MLMaster-style group mode minimizes bracket
makespan, max(required completion timestamps) minus bracket submission time.
Averages below are across two brackets, never across member jobs. MLEvolve-style
individual mode minimizes average submission-to-result JCT. Compare policies
within a submission scenario; those averages count different result events.

## What changed

Previous v4 used a single predicted solo rate for a deferred candidate. If that
rate was optimistic, it postponed a real straggler and lengthened the bracket.
The optimized gate retains the nominal rollout and adds one explicit candidate
recovery scenario: future candidate solo rate = min(prediction, measured packed
rate). This is a hypothesis, **not a measured solo label or a confidence bound**.

For each scenario, regret is an action's predicted bracket duration minus the
better of admit/defer in that same scenario. Choose admit when its maximum regret
is at least one second below defer's maximum regret, in both measurement halves.
This is minimax regret over two discrete scenarios; it is not a guaranteed gain
in actual makespan or a worst-case bound over every possible solo speed.

The change applies only to one outstanding all-required bracket. Individual,
mixed and multi-bracket cohorts retain the nominal objective. FIFO and age
protection are unchanged. Config `feedback_latency.bracket_admission_mode` is
`recovery_regret` by default inside opt-in v4; `nominal` restores previous v4.
The scorer preserves nominal gains and records alternative forecasts and regrets.

## Submission contracts and assumptions

- Batch barrier: A1–E1 arrive together; A2–E2 arrive ten seconds after the last
  first-bracket result. Two group requests, ten jobs total.
- Sequential pipeline: A1–E1 arrive ten seconds apart. Each result enables its
  same-type successor after ten seconds; a single issuer also spaces submissions
  by ten seconds. Ten individual requests; multiple jobs may remain outstanding.

These are illustrative patterns, informed by the
[MLMaster 2 orchestrator](https://github.com/sjtu-sai-agents/EvoMaster/blob/main/playground/ml_master_2/core/playground.py)
and [MLEvolve loop](https://github.com/InternScience/MLEvolve/blob/main/run.py),
not captured framework traffic or claims about default parallelism. Both policies
receive identical jobs and feedback rules; closed-loop arrival timestamps depend
on prior completions, so they are not identical exogenous arrival traces.

Each cycle uses A/B/C/D/E budgets 600/600/200/100/200 updates; memory
4/4/2/2/2 GiB under a 12 GiB budget. Reference solo rates are 12/12/24/6/12
updates/min. Slow-C changes actual C to 2.5; stale predictions still say 24.
The original synthetic interference law is unchanged: all multi-job rates ×5/6,
A/B further multiply .8/.6/.9 with C/D/E, C caps at 2 with A/B, D/E multiply .9
when together. The scheduler never sees this hidden law or future unsubmitted jobs.

Trials last up to one productive minute and are interrupted by arrivals or
completions. All trial updates/time are counted. Rates are noiseless, half-rates
are equal, pre-add rates are observable, and retry/sample-count rules are simplified.
Startup, checkpoint, finalization and delivery cost zero. Fixed-three is the
requested comparator; v4 has no fixed concurrency cap. All runs finish all jobs.
Nearest-rank p95 with two or ten requests equals the maximum, not a reliable
population-tail estimate. These are research illustrations, not production gains.
"""
    for case in CASES:
        text += '\n## ' + case.replace('_', ' ') + '\n\n'
        for scenario in SCENARIOS:
            label = 'bracket makespan' if scenario == 'batch_barrier' else 'job JCT'
            text += f'### {LABELS[scenario]}\n\n'
            text += f'| Policy | Average {label} | p95/max {label} | Workflow makespan |\n|---|---:|---:|---:|\n'
            for r in results:
                if r['case'] != case or r['scenario'] != scenario:
                    continue
                m = r['metrics']
                text += f"| {LABELS[r['policy']]} | {m['average_request_feedback_minutes']:.2f} | {m['p95_request_feedback_minutes']:.2f} | {m['workflow_makespan_minutes']:.2f} |\n"
            text += '\n'
    text += """
All values are minutes. The uncertainty-aware gate improves the severe stale-C
case but regresses the accurate reference case. This tradeoff is explicit;
it does not establish improvement on every bracket. The pipeline schedule is
unchanged between previous and optimized v4 in every case. Strictly one-outstanding
controls also have identical schedules and no packing opportunity.

See [optimization analysis](../optimization_results/report.md) for all development
ablations, the equal-weight three-case average, and twelve prespecified sensitivity
workloads evaluated without further parameter tuning. The same synthetic rate
law limits all these results. Sustained solo calibration and nonzero phase costs
remain necessary for production evaluation.

## Reproduction

```bash
python -m localml_scheduler.benchmarks.feedback_v4.submission_scenarios --plot
python -m localml_scheduler.benchmarks.feedback_v4.bracket_optimization --plot
```

Use the scheduler environment; plotting additionally requires matplotlib.
`results.json` retains every arrival, result, trial, forecast and productive segment.
Simulation units are minutes and updates/min; scorer evidence uses seconds and
updates/s. `summary.csv`, `verification.json`, and `comparison.png` contain metrics,
checks, and the combined Gantt/progress/latency figure. Model artifacts and
prediction records are not changed by this scheduler policy.
"""
    return text


def plot(results, controls, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(JOBS, ['#2563eb', '#e34b55', '#d49a10', '#9b59b6', '#19a785']))
    fig, axes = plt.subplots(12, 6, figsize=(34, 30), gridspec_kw={'height_ratios': [1.2, .9, .7] * 4})
    panels = [(row, col, next(r for r in results if r['case'] == case and r['scenario'] == scenario and r['policy'] == policy))
        for row, case in enumerate(CASES) for si, scenario in enumerate(SCENARIOS)
        for pi, policy in enumerate(POLICIES) for col in [si * len(POLICIES) + pi]]
    panels += [(3, 3 + i, r) for i, r in enumerate(controls)]
    for row, col, r in panels:
        top, progress, metric = axes[row * 3:row * 3 + 3, col]
        for segment in r['segments']:
            for j in segment['members']:
                top.barh(JOB_IDS.index(j), segment['end'] - segment['start'], left=segment['start'],
                    color=colors[j[0]], height=.7, hatch='///' if segment['phase'] == 'trial' else None, edgecolor='white', linewidth=.15)
        for j in JOB_IDS:
            top.scatter(r['arrivals'][j], JOB_IDS.index(j), marker='|', color='black', s=35, zorder=5)
            progress.plot([p[0] for p in r['progress']], [p[1][j] for p in r['progress']],
                color=colors[j[0]], ls='-' if j.endswith('1') else '--', lw=1.4)
        top.set_yticks(range(10), JOB_IDS, fontsize=7)
        top.invert_yaxis()
        title = 'Single-outstanding control' if r['strict_single'] else LABELS[r['scenario']]
        top.set_title(title + '\n' + LABELS[r['policy']] + ' / ' + r['case'].replace('_', ' '), fontsize=10)
        progress.set_ylabel('Completed updates')
        progress.set_xlabel('Minutes since workflow start')
        for ax in (top, progress):
            ax.set_xlim(0, max(z['metrics']['workflow_makespan_minutes'] for z in results + controls) * 1.03)
            ax.grid(axis='x', alpha=.2)
        keys = ('average_request_feedback_minutes', 'p95_request_feedback_minutes', 'workflow_makespan_minutes')
        values = [r['metrics'][key] for key in keys]
        bars = metric.barh([2, 1, 0], values, color=['#2878a8', '#be7144', '#597b60'], height=.65)
        latency_name = 'bracket makespan' if r['scenario'] == 'batch_barrier' else 'job JCT'
        metric.set_yticks([2, 1, 0], ['Avg ' + latency_name, 'p95 ' + latency_name, 'Workflow time'], fontsize=8)
        metric.set_xlim(0, max(z['metrics']['workflow_makespan_minutes'] for z in results + controls) * 1.12)
        for bar, value in zip(bars, values):
            metric.text(value + 2, bar.get_y() + .32, f'{value:.2f}', va='center', fontsize=8)
        metric.set_xlabel('Minutes')
    for row in range(9, 12):
        for col in (0, 1, 2):
            axes[row, col].axis('off')
    axes[9, 0].text(0, .8, 'MLMaster: minimize bracket makespan\nAverage is across 2 brackets\nMLEvolve: average individual JCT\nAverage is across 10 jobs\n\nBlack ticks = submission\nHatched = productive trial\nSolid/dashed = first/second cycle', fontsize=13, va='top')
    fig.suptitle('MLMaster: bracket makespan | MLEvolve: average individual job JCT\nSynthetic; identical jobs and feedback rules; all finish; zero checkpoint/delivery overhead', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, .975])
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('submission_results'))
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [simulate(scenario, policy, case) for case in CASES for scenario in SCENARIOS for policy in POLICIES]
    controls = [simulate('sequential_pipeline', policy, strict_single=True) for policy in POLICIES]
    assert all(r['completion'] == controls[0]['completion'] for r in controls)
    payload = dict(description='Synthetic closed-loop submission scenarios', schema_version=1,
        think_and_issuance_gap_seconds=10, results=results, single_outstanding_controls=controls)
    (args.output / 'results.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    rows = [dict(scenario=r['scenario'], policy=r['policy'], case=r['case'], **r['metrics']) for r in results]
    with (args.output / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'report.md').write_text(report(results, controls))
    verification = dict(main_runs=len(results), control_runs=len(controls), jobs_finished_per_run=10,
        passed=True, checks=['all work conserved including interrupted trials', 'VRAM respected',
        'no work before submission', 'batch barriers respected', 'pipeline dependency and issuer gaps respected',
        'all requests weight one', 'all jobs complete', 'single-outstanding policies identical'])
    (args.output / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    if args.plot:
        plot(results, controls, args.output / 'comparison.png')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
