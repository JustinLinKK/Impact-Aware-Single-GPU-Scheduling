# V4 refactor validation — 2026-10-28

The live scheduler feature is opt-in. Tests and simulations used CPU execution;
no GPU workload, dataset rewrite, model training, push or deployment was performed.

## Bracket optimization update

- **355 regression tests and four subtests passed**, covering the new recovery
  regret scorer, configuration, live trial adapter, invalid evidence and policy
  changes, complete submission scenarios, memory/placement, cold-start lifecycle
  and recovery. Predictor integration is unchanged from the earlier checked phase.
- Main comparison: 18 runs plus three single-outstanding controls. Optimized and
  previous v4 have identical pipeline schedules in all main cases.
- All seven development variants (21 runs) and twelve prespecified sensitivity
  fixtures (24 paired runs) are retained. Combined with the main/control runs,
  **66 runs each finish all ten jobs** with work, trial-progress, submission
  dependency and memory checks passing.
- Mean bracket time falls from 108.34 to 93.87 minutes across the original three
  fixtures, and 109.94 to 104.58 across the additional fixtures. The reference
  case regresses; five of twelve sensitivity fixtures regress. These outcomes
  are disclosed, not filtered. All share the same synthetic interference law.
- The margin and selected gate were fixed before the sensitivity run. Default
  `recovery_regret` is inside opt-in v4; `nominal` restores the previous gate.
  The underlying group objective remains last required completion, and the
  alternative packed-rate scenario is never recorded as a solo measurement.

Reproduce with `python -m localml_scheduler.benchmarks.feedback_v4.bracket_optimization --plot`.
See `optimization_results/report.md`, the full trace JSON and combined plot.

## Earlier implementation checks

- Broad regression run: **361 passed, 1 skipped, 6 subtests passed**. This covered
  the scorer/adapter/simulation, add-only trial controller, cold-start lifecycle
  and recovery, completion objective, configuration, prediction-first workflow,
  cooperative ordering, GPU placement logic, time-aware scheduler, recovery,
  process cleanup, predictor integration and released-output units.
- After the final request-time guards and two submission scenarios, the focused
  v4 suite completed **79 passed**. These overlap the broad run; the counts must
  not be added as independent test totals.
- The optional skipped check compares rounding with a separately configured
  public release's `portable/v5_objectives.py`; `PERFSEER_RELEASE_ROOT` was unset.
- Predictor integration initially failed because the isolated worktree lacked its
  sibling PerfSeer source/artifacts. Reusing the existing local checkout through
  a temporary sibling link resolved all nine failures; the link was removed.
  No predictor artifact was changed. The test host import path also includes the
  existing MLEvolve host package, which this extracted scheduler depends on.
- The slope-profile contract test independently verifies reciprocal units,
  prediction provenance and preservation of the original six model outputs.
- New submission experiments: **12 main runs + 2 controls**, ten jobs completed
  per run. Work conservation, retained interrupted-trial progress, VRAM bounds,
  arrival/dependency ordering, equal request weights and finish coverage passed.
- Historical five-job rerun: **15 runs**, five jobs completed per run; original
  baseline values and the stale-slope counterexample reproduce numerically.
- Both combined plots were visually inspected. Whitespace/diff checks passed.
- Bracket-metric clarification: the MLMaster group path already minimizes the
  last required completion. Report/chart labels now explicitly distinguish
  bracket makespan from individual job JCT. Sixteen existing objective/submission
  tests passed, and regenerating all 12 main runs plus two controls produced
  exactly unchanged traces, decisions and metrics.

## Commands

Run from the MLEvolve parent of this scheduler checkout with its Python environment.
In an isolated scheduler worktree, include that parent and the MLEvolve host root
in `PYTHONPATH`; predictor integration also needs the normal sibling PerfSeer
checkout. These are existing host dependencies, not new runtime requirements.

```bash
python -m pytest localml_scheduler/tests/test_feedback_objective.py \
  localml_scheduler/tests/test_feedback_trials.py \
  localml_scheduler/tests/test_feedback_simulation.py \
  localml_scheduler/tests/test_feedback_submissions.py -q
python -m localml_scheduler.benchmarks.feedback_v4.submission_scenarios --plot
python -m localml_scheduler.benchmarks.feedback_v4.simulate --plot
```

The broad run additionally selected `test_add_only_trials`, `test_cold_start_trials`,
`test_cold_start_cohort_recovery`, `test_cold_start_liveness`,
`test_completion_objective`, `test_completion_trials`, `test_config_models`,
`test_prediction_first_workflow`, `test_cooperative_work_ranking`,
`test_gpu_scheduler`, `test_time_aware_scheduler`, `test_recovery`,
`test_admission_variants_cleanup`, `test_ml_prediction` and `test_released_outputs`.

## Limits

The new main presentation omits v3 at the user's request. Historical v3 results
remain in `results/`; the batch/pipeline comparison is in `submission_results/`.
The scenarios use the same jobs and feedback rules across policies, with
policy-dependent release timestamps. They are synthetic, not captured framework
traffic. Rates are stable/noiseless and all completion-phase overhead is zero.
The live controller still needs matching phase estimates unless the explicit
research override is used; an automatic collector/calibrator is future work.
Neither good unit-test coverage nor this mock establishes production latency gains.
