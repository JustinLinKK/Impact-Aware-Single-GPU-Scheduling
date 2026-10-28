"""Record all bracket-policy development ablations and prespecified sensitivity runs.

python -m localml_scheduler.benchmarks.feedback_v4.bracket_optimization --plot
"""
from __future__ import annotations

import argparse
import csv
import json
from math import ceil
from pathlib import Path
from statistics import mean
import subprocess

from . import submission_scenarios as simulation
from ...scheduler.feedback_objective import compare_feedback_admission

BASE_COMMIT = '04fedcc5f05a42ac596d341c208ce9bb49d0da11'
VARIANTS = ('fifo_regret', 'longest_nominal', 'longest_regret', 'short_anchor',
            'short_anchor_measured', 'short_anchor_measured_longest', 'short_all_measured')


def development_runs():
    # Freeze the pre-optimization submission/retry model for reproducibility.
    repo = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(['git', 'show', BASE_COMMIT +
        ':benchmarks/feedback_v4/submission_scenarios.py'], cwd=repo, text=True)
    results = []
    for variant in VARIANTS:
        modified = source
        if variant.startswith('longest'):
            modified = modified.replace('pending.sort(key=lambda j: (arrivals[j], j))',
                "pending.sort(key=lambda j: ((-left[j]/predicted_solo[j[0]]) if scenario == 'batch_barrier' and policy != 'fixed_three_fifo' else arrivals[j], arrivals[j], j))")
            modified = modified.replace('protected = overdue[0] if overdue else None',
                'protected = min(overdue,key=lambda j:(arrivals[j],j)) if overdue else None')
        if variant.startswith('short'):
            modified = modified.replace('anchor = pending.pop(0)',
                "anchor = (min(pending,key=lambda j:(left[j]/predicted_solo[j[0]],arrivals[j],j)) if scenario=='batch_barrier' and policy != 'fixed_three_fifo' else pending[0])\n            pending.remove(anchor)")
        if variant.endswith('measured_longest') or variant == 'short_all_measured':
            sign = '-' if variant.endswith('longest') else ''
            modified = modified.replace('candidate = next((j for j in pending',
                "candidate = next((j for j in (sorted(pending,key=lambda j:(" + sign +
                "left[j]/predicted_solo[j[0]],arrivals[j],j)) if scenario=='batch_barrier' else pending)")
        namespace = {'__package__': simulation.__package__, '__name__': 'development_ablation'}
        exec(compile(modified, '<frozen-submission-ablation>', 'exec'), namespace)
        for case in simulation.CASES:
            observed = {}

            def measured(**kwargs):
                if len(kwargs['incumbents']) == 1:
                    job = next(iter(kwargs['incumbents']))
                    observed[job] = kwargs['pre_halves'][job]
                variants = []
                for half in range(2):
                    solo = {j: observed[j][half] if j in observed else rate for j, rate in kwargs['solo'].items()}
                    variants.append(compare_feedback_admission(**{**kwargs, 'solo': solo}))
                result = variants[0]
                assert not any(v['error'] for v in variants)
                result['halves'] = [variants[h]['halves'][h] for h in range(2)]
                passes = [h['gain_seconds'] >= kwargs.get('min_gain_seconds', 1.) for h in result['halves']]
                result['decision'] = 'accepted' if all(passes) else 'rejected' if not any(passes) else 'inconclusive'
                result['development_singleton_evidence'] = dict(observed)
                return result

            if variant.endswith('regret'):
                namespace['compare_feedback_admission'] = lambda **k: compare_feedback_admission(**k, bracket_admission_mode='recovery_regret')
            elif 'measured' in variant:
                namespace['compare_feedback_admission'] = measured
            r = namespace['simulate']('batch_barrier', 'v4_request_feedback', case)
            r.update(experiment='development', variant=variant)
            results.append(r)
    return results


def sensitivity_runs():
    specification = json.loads(Path(__file__).with_name('bracket_validation_cases.json').read_text())
    results = []
    for case in specification['cases']:
        for policy in ('v4_previous', 'v4_request_feedback'):
            r = simulation.simulate('batch_barrier', policy, workload=case['workload'])
            r.update(experiment='sensitivity', case=case['id'], family=case['family'])
            results.append(r)
    return results


def comparison_summary(runs):
    pairs = []
    for case in sorted({r['case'] for r in runs}):
        old = next(r for r in runs if r['case'] == case and r['policy'] == 'v4_previous')
        new = next(r for r in runs if r['case'] == case and r['policy'] == 'v4_request_feedback')
        previous = old['metrics']['average_request_feedback_minutes']
        optimized = new['metrics']['average_request_feedback_minutes']
        pairs.append(dict(case=case, previous=previous, optimized=optimized, delta=optimized - previous))
    return dict(previous_average=mean(p['previous'] for p in pairs),
        optimized_average=mean(p['optimized'] for p in pairs),
        improvements=sum(p['delta'] < -1e-8 for p in pairs),
        regressions=sum(p['delta'] > 1e-8 for p in pairs), pairs=pairs)


def report(main, development, sensitivity):
    core = [r for r in main if r['scenario'] == 'batch_barrier' and r['policy'] != 'fixed_three_fifo']
    summary, additional = comparison_summary(core), comparison_summary(sensitivity)
    relative = 100 * (1 - summary['optimized_average'] / summary['previous_average'])
    text = f'''# Bracket-mode optimization and its tradeoffs

Synthetic CPU experiments, 2026-10-28. Objective: minimize bracket makespan,
then report its arithmetic average across the specified brackets. The individual
MLEvolve policy and FIFO/age candidate ordering are unchanged.

The selected policy reduces the equally weighted three-case development average
from **{summary['previous_average']:.2f} to {summary['optimized_average']:.2f} minutes**
({relative:.2f}% lower). This average describes these fixtures; the scheduler is
not configured with scenario probabilities, and this is not a production estimate.

| Original mock case | Previous v4 | Optimized bracket | Change (min) |
|---|---:|---:|---:|
'''
    for p in summary['pairs']:
        text += f"| {p['case']} | {p['previous']:.2f} | {p['optimized']:.2f} | {p['delta']:+.2f} |\n"
    text += '''
The accurate reference case **regresses**. The gain comes from avoiding the much
larger stale-C deferred tail, not from improving every bracket. Accurate slow-C
predictions keep the previous outcome. The optimized variant outperforms the
fixed-three comparator in these three cases; previous v4 loses to fixed-three
in the stale-C case. Full baseline/pipeline results are in the
[submission report](../submission_results/report.md).

## Selected change: two-scenario recovery regret

For each clean measurement half, retain two candidate future-solo hypotheses:

1. Nominal: use the existing predicted solo slope.
2. No recovery: use min(predicted solo slope, candidate's measured packed slope).

All other predictions, current measured pack rates, remaining work, queue order
and phase costs are identical. The packed slope remains a packed observation;
the second hypothesis is not written into model predictions or solo labels.

Let d_h(x) = predicted bracket completion under defer minus that under admit,
for hypothesis x. The two actions' maximum regrets are:

    R_admit = max(0, -d_h(nominal), -d_h(no_recovery))
    R_defer = max(0,  d_h(nominal),  d_h(no_recovery))

Admit only if R_defer - R_admit >= 1 second in **both** independent halves.
Both failures reject, disagreement/invalid evidence is inconclusive. This is
minimax regret over two discrete hypotheses, not a worst-case bound over a
continuous interval and not a guarantee that actual latency improves. A nominal
regression can be accepted to reduce exposure to a more costly deferral error.

Apply this only to one outstanding all-required bracket. Mixed, multi-bracket
and individual cohorts keep the nominal scorer. Preserve memory constraints,
incumbent continuity, productive trial progress, phase costs and retry controls.
No GPU concurrency cap, new network head, predictor update or incumbent pause
is introduced. The opt-in v4 configuration defaults to
`bracket_admission_mode: recovery_regret`; `nominal` restores the previous gate.

## Development alternatives retained for review

These seven variants were explored on the original three mock cases. They are
development data, not held-out evidence. No hidden solo rate or future group
performance was provided to a scheduler. The chosen FIFO/regret variant avoided
adding a ranking rule that did not consistently help. Productive short anchors
with measured singleton rates did not consistently improve the brackets either.

| Variant | Reference | Stale C | Accurate slow C | Three-case average |
|---|---:|---:|---:|---:|
'''
    for variant in VARIANTS:
        values = [next(r for r in development if r['variant'] == variant and r['case'] == case)['metrics']['average_request_feedback_minutes'] for case in simulation.CASES]
        text += '| ' + variant + ' | ' + ' | '.join(f'{v:.2f}' for v in [*values, mean(values)]) + ' |\n'
    text += f'''
## Prespecified sensitivity workloads

After choosing the gate, twelve additional fixtures were saved in
`bracket_validation_cases.json` before evaluation, using seed 20261028. Four have
nearby predicted/actual rates, four an optimistic C prediction, and four a
pessimistic C prediction. Budgets vary from .5–1.5 times the original and trials
last .5, 1 or 2 minutes. The gate/margin was not changed after these results.

Average bracket time: **{additional['previous_average']:.2f} → {additional['optimized_average']:.2f} minutes**.
Improved fixtures: {additional['improvements']}/12; regressed fixtures: {additional['regressions']}/12;
unchanged: {12-additional['improvements']-additional['regressions']}/12.

| Fixture | Previous v4 | Optimized bracket | Change (min) |
|---|---:|---:|---:|
'''
    for p in additional['pairs']:
        text += f"| {p['case']} | {p['previous']:.2f} | {p['optimized']:.2f} | {p['delta']:+.2f} |\n"
    text += '''
These are sensitivity checks under the **same invented interference law**, not
independent framework traces or broad generalization evidence. Completion-phase
overhead remains zero; trial time and productive interrupted-trial progress are
fully counted. Every required job completes and VRAM remains within 12 GiB.
Longer trials, noisy windows, nonzero checkpoint/delivery costs and changing data
pipelines need separate measured evaluation. Solo-rate calibration is still
valuable: the uncertainty rule manages ambiguity but cannot identify true solo
speed from a packed observation. It can lose when the nominal prediction was right.

## Artifacts and reproduction

`results.json` contains all main, control, development and sensitivity traces,
including both continuation forecasts and regret calculations. `summary.csv`
contains metrics. `all-experiments.png` combines a Gantt chart and optimizer
progress plot for every run, with bracket/JCT and workflow outcomes in each title.
The main comparison also has a more compact chart in `submission_results/`.

```bash
python -m localml_scheduler.benchmarks.feedback_v4.submission_scenarios --plot
python -m localml_scheduler.benchmarks.feedback_v4.bracket_optimization --plot
```

Development ablations use the submission implementation frozen at commit
`04fedcc`; the script loads that source from local Git history. Each uses the
shared scorer plus its explicitly recorded development variation. The live
implementation and main/sensitivity runs use the same current scorer.
'''
    return text


def plot_all(runs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    columns = 6
    blocks = ceil(len(runs) / columns)
    fig, axes = plt.subplots(blocks * 2, columns, figsize=(36, blocks * 4.8), squeeze=False)
    colors = dict(zip(simulation.JOBS, ['#2563eb', '#e34b55', '#d49a10', '#9b59b6', '#19a785']))
    for index, r in enumerate(runs):
        row, col = divmod(index, columns)
        gantt, progress = axes[row * 2:row * 2 + 2, col]
        for segment in r['segments']:
            for j in segment['members']:
                gantt.barh(simulation.JOB_IDS.index(j), segment['end'] - segment['start'],
                    left=segment['start'], height=.7, color=colors[j[0]],
                    hatch='///' if segment['phase'] == 'trial' else None, edgecolor='white', linewidth=.15)
        for j in simulation.JOB_IDS:
            gantt.scatter(r['arrivals'][j], simulation.JOB_IDS.index(j), marker='|', color='black', s=20, zorder=5)
            progress.plot([p[0] for p in r['progress']], [p[1][j] for p in r['progress']],
                color=colors[j[0]], ls='-' if j.endswith('1') else '--', lw=1.)
        gantt.set_yticks(range(10), simulation.JOB_IDS, fontsize=6)
        gantt.invert_yaxis()
        title = r.get('variant', r['policy'])
        m = r['metrics']
        gantt.set_title(f"{r['experiment']} / {title}\n{r['case']} / {r['scenario']}\nAverage {m['average_request_feedback_minutes']:.2f} min; workflow {m['workflow_makespan_minutes']:.2f} min", fontsize=8)
        progress.set_xlabel('Minutes since workflow start', fontsize=7)
        progress.set_ylabel('Completed updates', fontsize=7)
        for ax in (gantt, progress):
            ax.set_xlim(0, m['workflow_makespan_minutes'] * 1.03)
            ax.tick_params(labelsize=6)
            ax.grid(axis='x', alpha=.2)
    for index in range(len(runs), blocks * columns):
        row, col = divmod(index, columns)
        for ax in axes[row * 2:row * 2 + 2, col]: ax.axis('off')
    fig.suptitle('Bracket optimization: every main, control, development and sensitivity run\nSynthetic; all required updates retained; zero phase overhead. Each panel has its own time scale.', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, .985])
    fig.savefig(output, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('optimization_results'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    main_runs = [simulation.simulate(scenario, policy, case) for case in simulation.CASES
                 for scenario in simulation.SCENARIOS for policy in simulation.POLICIES]
    for r in main_runs: r['experiment'] = 'main'
    controls = [simulation.simulate('sequential_pipeline', policy, strict_single=True) for policy in simulation.POLICIES]
    for r in controls: r['experiment'] = 'single_outstanding_control'
    development, sensitivity = development_runs(), sensitivity_runs()
    all_runs = [*main_runs, *controls, *development, *sensitivity]
    for r in all_runs: simulation.verify(r)
    summary = comparison_summary(sensitivity)
    payload = dict(base_commit=BASE_COMMIT, results=all_runs, sensitivity_summary=summary)
    (args.output / 'results.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    rows = [dict(experiment=r['experiment'], variant=r.get('variant', r['policy']), scenario=r['scenario'],
        case=r['case'], **r['metrics']) for r in all_runs]
    with (args.output / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'report.md').write_text(report(main_runs, development, sensitivity))
    (args.output / 'verification.json').write_text(json.dumps(dict(runs=len(all_runs),
        main_runs=len(main_runs), controls=len(controls), development_runs=len(development),
        sensitivity_runs=len(sensitivity), jobs_finished_per_run=10,
        checks_passed=['work conservation', 'retained trial progress', 'VRAM', 'submission dependencies',
            'complete cohorts', 'all jobs finish'], seed=20261028, synthetic=True), indent=2) + '\n')
    if args.plot: plot_all(all_runs, args.output / 'all-experiments.png')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
