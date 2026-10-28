"""Five-job fluid experiment using the scheduler's actual v4 scoring engine.

Run from the MLEvolve parent: python -m localml_scheduler.benchmarks.feedback_v4.simulate
Add --plot when matplotlib is installed. Output times are minutes; scoring is seconds.
"""
from __future__ import annotations

import argparse
import csv
import json
from math import ceil
from pathlib import Path
from statistics import mean, median

from ...scheduler.feedback_objective import FeedbackRequest, compare_feedback_admission, score_requests

JOBS = ('A', 'B', 'C', 'D', 'E')
BUDGET = dict(A=600., B=600., C=200., D=100., E=200.)
MEMORY = dict(A=4., B=4., C=2., D=2., E=2.)
SOLO = dict(A=12., B=12., C=24., D=6., E=12.)
VRAM = 12.
EPS = 1e-8
POLICIES = ('fixed_three_fifo', 'v3_slowdown50', 'v4_individual', 'v4_group', 'v4_mixed')
LABELS = dict(zip(POLICIES, ('Fixed-three FIFO', 'V3: 50% slowdown', 'V4: individual', 'V4: group', 'V4: mixed')))
CASES = ('reference', 'slow_C_stale_prediction', 'slow_C_accurate_prediction')


def hidden_rates(members, solo):
    """Synthetic environment only. The forecast engine never calls this function."""
    members = set(members)
    rates = {j: solo[j] * (5 / 6 if len(members) > 1 else 1) for j in sorted(members)}
    for j in members & {'A', 'B'}:
        for other, factor in [('C', .8), ('D', .6), ('E', .9)]:
            if other in members:
                rates[j] *= factor
    if 'C' in members and members & {'A', 'B'}:
        rates['C'] = min(rates['C'], 2.)
    if {'D', 'E'} <= members:
        rates['D'] *= .9
        rates['E'] *= .9
    return rates


def request_partition(mode):
    groups = ([JOBS] if mode == 'group' else [('A', 'B'), ('C', 'D'), ('E',)]
              if mode == 'mixed' else [(j,) for j in JOBS])
    return tuple(FeedbackRequest('request_' + ''.join(group), 'synthetic',
        'individual' if len(group) == 1 else 'group', tuple(group), 0.) for group in groups)


def simulate(policy, case='reference', *, trial_minutes=1., age_minutes=30.):
    if policy not in POLICIES or case not in CASES:
        raise ValueError('unknown experiment')
    actual_solo = dict(SOLO, **({'C': 2.5} if case != 'reference' else {}))
    predicted_solo = dict(actual_solo if case == 'slow_C_accurate_prediction' else SOLO)
    requests = request_partition(policy.removeprefix('v4_'))
    left, done = dict(BUDGET), {}
    active, pending = {'A', 'B'}, ['C', 'D', 'E']
    time, segments, decisions = 0., [], []
    points = [(0., {j: 0. for j in JOBS})]

    def advance(duration, members, phase):
        nonlocal time
        rates = hidden_rates(members, actual_solo)
        duration = min(duration, min(left[j] / rates[j] for j in members))
        assert duration > 0
        before = dict(left)
        for j in members:
            left[j] = max(0., left[j] - duration * rates[j])
        segments.append(dict(start=time, end=time + duration, members=sorted(members),
                             rates=rates, phase=phase, before=before, after=dict(left)))
        time += duration
        points.append((time, {j: BUDGET[j] - left[j] for j in JOBS}))
        finished = {j for j in members if left[j] < EPS}
        for j in sorted(finished):
            done[j] = time
        return finished

    loops = 0
    while len(done) < len(JOBS):
        loops += 1
        assert loops < 500
        restart = False
        # One FIFO pass per committed completion. Rejections are not retried in
        # a tight loop. This is a policy experiment, not the production retry clock.
        for candidate in list(pending):
            if policy.startswith('v4_') and active and time >= age_minutes:
                if candidate != pending[0]:
                    continue  # Oldest waiting request's job prevents backfill.
            if sum(MEMORY[j] for j in active) + MEMORY[candidate] > VRAM + EPS:
                continue
            if policy == 'fixed_three_fifo' and len(active) >= 3:
                continue
            if not active or policy == 'fixed_three_fifo':
                active.add(candidate)
                pending.remove(candidate)
                decisions.append(dict(time=time, job=candidate, decision='accepted', reason='dispatch'))
                continue
            incumbents = set(active)
            pre = hidden_rates(incumbents, actual_solo)  # Already-running measured slope.
            proposed = incumbents | {candidate}
            post = hidden_rates(proposed, actual_solo)  # Observable only in the trial.
            finished = set()
            for _ in range(2):
                finished = advance(trial_minutes / 2, proposed, 'trial')
                if finished:
                    break
            if finished:
                active.difference_update(finished)
                if candidate in finished:
                    pending.remove(candidate)
                decisions.append(dict(time=time, job=candidate, decision='inconclusive',
                                      reason='completion changed trial membership', completed=sorted(finished)))
                restart = True
                break
            if policy == 'v3_slowdown50':
                ratios = {j: pre[j] / post[j] for j in incumbents}
                decision = 'accepted' if max(ratios.values()) <= 1.5 + EPS else 'rejected'
                evidence = dict(slowdown_ratios=ratios)
            else:
                outstanding_requests = tuple(r for r in requests if not all(j in done for j in r.required_job_ids))
                required = {j for r in outstanding_requests for j in r.required_job_ids}
                evidence = compare_feedback_admission(
                    remaining={j: left[j] for j in JOBS if j not in done},
                    solo={j: predicted_solo[j] / 60 for j in JOBS if j not in done},
                    incumbents=incumbents, candidate=candidate, queue=pending,
                    pre_halves={j: [pre[j] / 60] * 2 for j in incumbents},
                    packed_halves={j: [post[j] / 60] * 2 for j in proposed},
                    requests=outstanding_requests, completed={j: done[j] * 60 for j in done if j in required},
                    now=time * 60, min_gain_seconds=1.)
                assert evidence['error'] is None, evidence
                decision = evidence['decision']
            decisions.append(dict(time=time, job=candidate, decision=decision,
                                  incumbents=sorted(incumbents), evidence=evidence))
            if decision == 'accepted':
                active.add(candidate)
                pending.remove(candidate)
        if restart:
            continue
        assert active
        rates = hidden_rates(active, actual_solo)
        finished = advance(min(left[j] / rates[j] for j in active), active, 'train')
        active.difference_update(finished)

    metrics = dict(mean_jct_minutes=mean(done.values()), median_jct_minutes=median(done.values()),
        p95_jct_minutes=sorted(done.values())[ceil(.95 * len(done)) - 1],
        max_jct_minutes=max(done.values()), makespan_minutes=time,
        trial_elapsed_minutes=sum(s['end'] - s['start'] for s in segments if s['phase'] == 'trial'),
        startup_checkpoint_finish_delivery_minutes=0., finished_jobs=len(done))
    for mode in ('individual', 'group', 'mixed'):
        metric = score_requests(request_partition(mode), {j: t * 60 for j, t in done.items()})
        metrics[mode + '_mean_feedback_minutes'] = metric['mean_latency_seconds'] / 60
    result = dict(policy=policy, case=case, actual_solo=actual_solo, predicted_solo=predicted_solo,
                  trial_minutes=trial_minutes, age_minutes=age_minutes, completion=done, metrics=metrics,
                  decisions=decisions, segments=segments, progress=points)
    verify(result)
    return result


def verify(result):
    processed, last = {j: 0. for j in JOBS}, 0.
    for segment in result['segments']:
        assert abs(segment['start'] - last) < EPS
        assert sum(MEMORY[j] for j in segment['members']) <= VRAM + EPS
        if result['policy'] == 'fixed_three_fifo':
            assert len(segment['members']) <= 3
        for j in segment['members']:
            processed[j] += (segment['end'] - segment['start']) * segment['rates'][j]
        last = segment['end']
    assert abs(last - result['metrics']['makespan_minutes']) < EPS
    assert set(result['completion']) == set(JOBS)
    for j in JOBS:
        assert abs(processed[j] - BUDGET[j]) < 1e-6, (j, processed[j])
        assert abs(result['progress'][-1][1][j] - BUDGET[j]) < 1e-6
    assert result['metrics']['makespan_minutes'] == max(result['completion'].values())


def report(results):
    text = '''# V4 feedback-policy experiment

Synthetic CPU simulation, 2026-10-28. No GPU measurements or predictor training.
The simulator calls `scheduler.feedback_objective.compare_feedback_admission`,
the same scorer used by the opt-in live trial controller.

A/B are initially running; C/D/E wait in FIFO order. All submission times are
zero. Required remaining updates are A=600, B=600, C=200, D=100, E=200; VRAM is
4/4/2/2/2 GiB with a 12 GiB budget. Reference solo rates are 12/12/24/6/12
updates/min. Slow-C cases change actual C to 2.5; the stale prediction stays 24.

Each multi-job rate is solo × 5/6. A/B additionally multiply by .8/.6/.9 when
C/D/E are present. C is capped at 2 updates/min with A or B; co-running D/E
each multiply by .9. These are invented, stable, noiseless rates.

Trials take one productive minute, split into two equal halves. All trial work
and time remain in the trace. Startup, checkpoint, finalization and result
delivery cost **zero in this experiment**; production benefits are unverified.
Pre-trial slopes come from already running jobs, with no new baseline cost.
No incumbent is preempted. One FIFO pass occurs per completion, with fresh
evidence after membership changes. V4 suppresses backfill once the oldest
waiting job reaches 30 minutes. This is an event-driven policy comparison,
not an emulation of the controller's sample-count or retry-clock constraints.
The fixed-three cap is solely the user-requested comparator.

Forecasts drain the current pack and then a frozen serial queue. They use only
current pre/post measurements and predicted solo rates, never hidden future
group rates. Unknown surviving subsets retain parent rates as an assumption.
Both evidence halves must improve weighted mean request latency by >=1 second.
Individual requests count five feedback events; group counts the single A–E
barrier; mixed counts AB, CD, and E once each, with equal request weights.
Both baseline schedules are evaluated under every partition for fair comparison.

With five simultaneous arrivals, nearest-rank p95 equals max JCT and makespan.
This example therefore does not independently establish a tail-latency benefit.
Values below are minutes. Every run completes all five jobs.
'''
    for case in CASES:
        text += '\n## ' + case.replace('_', ' ') + '\n\n'
        text += '| Policy | Mean job JCT | p95/max JCT | Makespan | Mean mixed feedback | Trial time |\n|---|---:|---:|---:|---:|---:|\n'
        for r in results:
            if r['case'] != case:
                continue
            m = r['metrics']
            text += f"| {LABELS[r['policy']]} | {m['mean_jct_minutes']:.2f} | {m['p95_jct_minutes']:.2f} | {m['makespan_minutes']:.2f} | {m['mixed_mean_feedback_minutes']:.2f} | {m['trial_elapsed_minutes']:.2f} |\n"
        text += '\n| Policy | A done | B done | C done | D done | E done |\n|---|---:|---:|---:|---:|---:|\n'
        for r in results:
            if r['case'] == case:
                text += '| ' + LABELS[r['policy']] + ' | ' + ' | '.join(f"{r['completion'][j]:.2f}" for j in JOBS) + ' |\n'
    text += '''
## Interpretation and limits

The request type changes which completions matter; it does not make the forecast
accurate. In particular the slow-C stale prediction can favor quick individual
feedback while greatly extending the final completion. Compare each v4 policy
with both baselines under the same request partition. Do not compare a mixed
request average with a five-job average as if they counted the same events.

All three v4 request partitions choose the **same schedule in these five-job
cases**. The example shows a benefit over the baselines in the reference case,
but does not show that group mode improves over individual mode. With accurate
slow-C slopes, mixed feedback is 74.33 minutes versus v3's 66.12, despite better
job mean JCT and makespan. Restricted forecasts are heuristics and can still
mis-rank future schedules. The separate two-job scorer test verifies distinct
behavior: A=100, B=1 updates, solo rates 1/s, packed rates .5/.4/s yield an
individual gain of 48.625 seconds but a group loss of .25 seconds. Individual
accepts and group rejects with the same completion forecasts.

The previous 148.39 → 97.00 minute counterexample is synthetic and prediction-
dependent. This rerun keeps all progress, tests both completion policies and
uses FIFO/age protection; a better slope still cannot identify co-location
interference without measured trials. No result is an optimality guarantee.

Next evaluation must include staggered and feedback-dependent arrivals, noisy
and drifting rates, large/short jobs, calibrated versus biased predictors,
nonzero startup/checkpoint/validation/delivery overhead and failed/cancelled
requests. Record application-consumed results as well as scheduler receipts.
Compare identical workloads/seeds against v3 and fixed-three FIFO, require all
jobs to finish, report job and request mean/p95/max, starvation, coverage,
trial cost and makespan, and assess multiple traces before deployment.

## Reproduction and artifacts

From the MLEvolve parent directory:

```bash
python -m localml_scheduler.benchmarks.feedback_v4.simulate --plot
```

Use the existing scheduler environment (including PyYAML); plotting additionally
requires matplotlib. The simulation itself uses the standard library.
`results.json` saves every segment, update counter, admission decision and both
forecast alternatives. Forecast timestamps/rates are seconds and updates/s;
simulation timestamps/rates are minutes and updates/min. `summary.csv` contains
metrics, `verification.json` records conservation and memory checks, and
`comparison.png` combines every schedule with update curves and outcome metrics.
'''
    return text


def plot(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(JOBS, ['#2563eb', '#e34b55', '#d49a10', '#9b59b6', '#19a785']))
    fig, axes = plt.subplots(9, 5, figsize=(24, 22), gridspec_kw={'height_ratios': [1, 1, .65] * 3})
    for row, case in enumerate(CASES):
        for col, policy in enumerate(POLICIES):
            r = next(r for r in results if r['case'] == case and r['policy'] == policy)
            top, progress, metrics = axes[row * 3:row * 3 + 3, col]
            for s in r['segments']:
                for j in s['members']:
                    top.barh(JOBS.index(j), s['end'] - s['start'], left=s['start'], height=.7,
                        color=colors[j], hatch='///' if s['phase'] == 'trial' else None, edgecolor='white', linewidth=.2)
            top.set_yticks(range(5), JOBS)
            top.invert_yaxis()
            top.set_title(LABELS[policy] + '\n' + case.replace('_', ' '), fontsize=10)
            for j in JOBS:
                progress.plot([p[0] for p in r['progress']], [p[1][j] for p in r['progress']], color=colors[j], label=j)
            progress.set_ylabel('Completed updates')
            progress.set_xlabel('Minutes since submission')
            for ax in (top, progress):
                ax.set_xlim(0, 155)
                ax.grid(axis='x', alpha=.2)
            vals = [r['metrics'][key] for key in ('mean_jct_minutes', 'makespan_minutes', 'mixed_mean_feedback_minutes')]
            bars = metrics.barh([2, 1, 0], vals, color=['#2878a8', '#be7144', '#597b60'], height=.65)
            metrics.set_yticks([2, 1, 0], ['Mean job', 'p95 / makespan', 'Mixed feedback'], fontsize=8)
            metrics.set_xlim(0, 170)
            for bar, value in zip(bars, vals):
                metrics.text(value + 1, bar.get_y() + .32, f'{value:.2f}', va='center', fontsize=8)
            metrics.set_xlabel('Latency (minutes)')
    fig.suptitle('Synthetic five-job experiment: one engine, individual / group / mixed requests\nAll jobs finish; hatched = productive trial; zero startup/checkpoint/delivery cost', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results'))
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [simulate(policy, case) for case in CASES for policy in POLICIES]
    payload = dict(description='Synthetic fluid model, not GPU measurements', schema_version=1,
                   jobs=JOBS, budget=BUDGET, memory_gib=MEMORY, vram_gib=VRAM, results=results)
    (args.output / 'results.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    rows = [dict(case=r['case'], policy=r['policy'], **r['metrics'], **{j: r['completion'][j] for j in JOBS}) for r in results]
    with (args.output / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'report.md').write_text(report(results))
    verification = dict(runs=len(results), jobs_finished_per_run=5, checks=['all work conserved including trials',
        'all segments continuous', 'VRAM within budget', 'all jobs completed', 'fixed-three comparator respects cap'],
        passed=True, gpu_used=False, zero_phase_overheads=True)
    (args.output / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    if args.plot:
        plot(results, args.output / 'comparison.png')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
