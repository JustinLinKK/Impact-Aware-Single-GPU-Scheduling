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
POLICIES = ('fixed_three_fifo', 'v4_request_feedback')
JOB_IDS = tuple(j + str(round_) for round_ in (1, 2) for j in JOBS)
THINK_MINUTES = 10 / 60
LABELS = {'batch_barrier': 'MLMaster 2-style batch barrier',
          'sequential_pipeline': 'MLEvolve-style sequential pipeline',
          'fixed_three_fifo': 'Fixed-three FIFO', 'v4_request_feedback': 'V4 request feedback'}


def simulate(scenario, policy, case='reference', *, strict_single=False):
    if scenario not in SCENARIOS or policy not in POLICIES or case not in CASES:
        raise ValueError('unknown experiment')
    if strict_single and scenario != 'sequential_pipeline':
        raise ValueError('single-outstanding control applies to the pipeline only')
    actual_solo = dict(SOLO, **({'C': 2.5} if case != 'reference' else {}))
    predicted_solo = dict(actual_solo if case == 'slow_C_accurate_prediction' else SOLO)
    left = {j: BUDGET[j[0]] for j in JOB_IDS}
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
        points.append((time, {j: BUDGET[j[0]] - left[j] for j in JOB_IDS}))
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
                deadline = time + 1.
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
                    now=time * 60, min_gain_seconds=1.)
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
        assert abs(processed[j] - BUDGET[j[0]]) < 1e-6
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
    text = '''# MLMaster bracket makespan and MLEvolve individual job JCT

Synthetic experiment, 2026-10-28. “Average” here is the arithmetic mean: each
feedback request has weight one. Mean job JCT already weights jobs equally;
changing the word would not change the metric. The substantive change is the
unit being averaged: one batch barrier or one individual returned result.

For Q requests, average feedback = sum(C_q - a_q)/Q. For a batch, C_q is the
last required member's result; for an individual request it is that job's result.
There are two batch requests or ten individual requests in the scenarios below.
Compare policies **within** each scenario; the two averages count different events.

**MLMaster's admission objective is bracket makespan, not average member-job
time.** For bracket b submitted at A_b, M_b = max(C_i for i in bracket b) - A_b.
One outstanding group request makes the shared scorer exactly this maximum.
The admission gain is predicted defer-last-completion minus predicted
admit-last-completion; both measurement halves must pass the one-second margin.
“Average bracket makespan” in the tables averages the two bracket durations.
The next bracket is generated only after the preceding bracket returns all results.
Total workflow time is the sum of bracket makespans plus the inter-bracket think
delay. For MLEvolve's pipeline, average individual job JCT is the primary metric.
This naming clarification leaves the already-correct group scoring unchanged.

The main comparison contains only fixed-three FIFO and v4. V3 is removed from
these tables and the chart; earlier five-job results remain a historical appendix.
V4 selects group/individual completion from the request metadata; it is one engine.

## Submission contracts

- **MLMaster 2-style batch barrier:** submit A1–E1 together at time zero. After
  every result in that batch returns, wait 10 seconds, then submit A2–E2 together.
  Each batch is one all-required feedback request.
- **MLEvolve-style sequential pipeline:** submit A1–E1 one at a time, 10 seconds
  apart. Each returned result enables its same-type successor A2–E2 after a
  10-second think delay. A single issuer spaces submissions by at least 10 seconds.
  Multiple jobs can remain outstanding; each is an individual feedback request.

These are illustrative arrival models, not captured framework traffic. The
[MLMaster 2 orchestrator](https://github.com/sjtu-sai-agents/EvoMaster/blob/main/playground/ml_master_2/core/playground.py)
collects parallel improvement results before processing the direction. The
[MLEvolve loop](https://github.com/InternScience/MLEvolve/blob/main/run.py)
staggers draft dispatch and replenishes work on individual completions. “MLMaster”
does not universally imply a batch barrier, and sequential submission does not
imply a single outstanding job. We choose five initial lineages and two cycles
for this mock; these are not claims about framework defaults or a GPU job cap.

Each policy receives the same ten jobs, recipes and feedback/dependency rules.
Future release timestamps differ because completions differ: this is a closed-loop
comparison, **not an identical fixed arrival trace**. Both real end-to-end workflow
time and request latency are reported. Code generation/learning quality are not
simulated; think time is a fixed assumption. No independent Poisson arrival model
is used.

## Work and measurement assumptions

Each cycle uses the original five-job budgets (600/600/200/100/200 updates),
4/4/2/2/2 GiB memory and a 12 GiB GPU. Reference solo rates are 12/12/24/6/12
updates/min. Slow-C changes true C to 2.5; its stale predictor still says 24.
The same synthetic multi-job interference law is retained: solo ×5/6, A/B
multiply .8/.6/.9 with C/D/E, C capped at 2 beside A/B, and D/E multiply .9
when together. Both cycles finish every required update.

V4 uses the production scorer with FIFO/30-minute age protection, a one-second
average-feedback improvement margin in both measured halves, and no concurrency
cap. Trials consume up to one productive minute; arrivals or completions interrupt
them and require fresh evidence. All trial progress and time are counted. Steady
rates are noise-free, so both half-rates are identical. Pre-add rates are assumed
observable from current execution. Retry and sample-count rules are simplified.
No future unsubmitted job or hidden group rate is exposed to the scorer.

Startup, checkpoint, finalization and delivery costs are explicitly zero. These
are policy illustrations, not GPU or production results. The fixed-three cap is
the requested comparator only. Nearest-rank p95 with just two or ten requests
equals the maximum; there is too little data for a reliable tail estimate.
'''
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
    text += '\nAll table values are minutes. Per-job arrivals, completions and per-request feedback are in `results.json`.\n'
    text += '\n## Reading the comparison\n\n'
    for scenario in SCENARIOS:
        for case in CASES:
            base = next(r for r in results if r['scenario'] == scenario and r['case'] == case and r['policy'] == POLICIES[0])
            v4 = next(r for r in results if r['scenario'] == scenario and r['case'] == case and r['policy'] == POLICIES[1])
            change = v4['metrics']['average_request_feedback_minutes'] - base['metrics']['average_request_feedback_minutes']
            metric_name = 'average bracket makespan' if scenario == 'batch_barrier' else 'average job JCT'
            text += f"- {LABELS[scenario]}, {case.replace('_', ' ')}: v4 {metric_name} is {abs(change):.2f} minutes {'lower' if change < 0 else 'higher'} than fixed-three FIFO.\n"
    text += f'''
The strictly-one-outstanding control produces the same schedule under both
policies: {controls[0]['metrics']['average_request_feedback_minutes']:.2f} minutes
average feedback and {controls[0]['metrics']['workflow_makespan_minutes']:.2f}
minutes workflow time, with no packing trials. This control is not the requested
pipelined scenario; it verifies the absence of concurrency opportunity.
Its lower per-request average also illustrates why averages across different
submission contracts are not directly comparable: delaying submissions can lower
observed request latency while increasing total workflow time. Keep both metrics.

A lower job average does not imply a lower batch feedback average. Predictor bias
can still cause poor choices, and a local continuation score does not guarantee
the best future closed-loop schedule. The chart shows all 12 main runs: submission
markers and Gantt segments, optimizer progress, and request/job/workflow metrics.
Require longer, noisy, overhead-inclusive paired workloads before deployment.

## Reproduction

```bash
python -m localml_scheduler.benchmarks.feedback_v4.submission_scenarios --plot
```

Use the scheduler environment; plotting additionally requires matplotlib.
`results.json` retains every arrival, result, trial, forecast and productive
segment. Simulation units are minutes and updates/min; internal scorer evidence
uses seconds and updates/s. `summary.csv`, `verification.json`, and `comparison.png`
provide the compact metrics, checks and combined figure. `single_outstanding_controls`
in the JSON holds the two control traces, also shown in the chart.
'''
    return text


def plot(results, controls, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(JOBS, ['#2563eb', '#e34b55', '#d49a10', '#9b59b6', '#19a785']))
    fig, axes = plt.subplots(12, 4, figsize=(24, 30), gridspec_kw={'height_ratios': [1.2, .9, .7] * 4})
    panels = [(row, col, next(r for r in results if r['case'] == case and r['scenario'] == scenario and r['policy'] == policy))
        for row, case in enumerate(CASES) for si, scenario in enumerate(SCENARIOS)
        for pi, policy in enumerate(POLICIES) for col in [si * 2 + pi]]
    panels += [(3, 2 + i, r) for i, r in enumerate(controls)]
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
        for col in (0, 1):
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
    assert controls[0]['completion'] == controls[1]['completion']
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
