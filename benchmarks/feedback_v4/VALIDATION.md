# V4 refactor validation — 2026-10-28

The live scheduler feature is opt-in. Tests and simulations used CPU execution;
no GPU workload, dataset rewrite, model training, push or deployment was performed.

## Checks completed

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
