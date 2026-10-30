# Measured A10 result-return pilot

Campaign v1 is invalidated. Read [REPAIR-V2.md](REPAIR-V2.md) for the confirmed
capture-limit and submission-clock defects, CPU-only repair, and proof. Preserve
its original artifacts as diagnostics; no replacement GPU run is authorized by
this document or repair commit.

Reference scheduler: `95b651d2e2a89599138c790ab4f9c1363f29e65f`.
Initial workload manifest SHA256:
`91c24a3256e10baf34078e893a45ae34535642efc1a571b7cdb9eb9f70e6524b`.
Its 64-update budgets remain provisional until dedicated calibration; any revision
requires a new immutable manifest and runtime signatures before scoring.
The original v4 checkout and dirty main checkout remain unchanged. This isolated
branch supplies measurement and explicit candidate extensions. No import launches
training, cloud instances, or timers.

## Four arms, two workflow modes

1. Reference A10 predictor + exact enabled v4.
2. Improved A10 predictor + exact enabled v4.
3. Same improved A10 predictor + revised admission.
4. FIFO fixed concurrency three, subject to the same preflight memory safety.

Enable `feedback_latency.enabled=true` explicitly; its default is false.
Reference v4 uses `bracket_admission_mode=recovery_regret`. Batch search and
early stopping are disabled. Every arm has identical source, data, batch,
precision, optimizer, successful-update and validation budgets. Use the A10
release; A100 residual models are not compatible evidence.

Freeze six workloads arranged as three chains of two individual requests, or
two finalized all-required brackets of three jobs. Individual parents arrive at
0/2/4 seconds; group parents are submitted together. Submit a dependent request
only after its prerequisite result receipt(s), preserving the same dependency
graph across arms. Workflow mode changes the objective, not training work.
Use two balanced paired repeats (16 bounded arm/mode episodes), with declared
seed/order. Select update budgets from a dedicated calibration split to target
3–4 minute episodes. Total preparation/calibration/evaluation <=2 hours and the
parent's global monetary/time cutoff. Incomplete, failed, budget-mismatched, or
deadline-censored episodes are reported; never rank completed-only subsets.

## Calibration and frozen evidence

Use dedicated calibration configurations before evaluation. Freeze calibration
job identities, source/data hashes, predictor artifact, A10 hardware, batch,
precision/optimizer, fitted correction, uncertainty hypotheses and measured
startup/validation/checkpoint/finalization/exit/delivery costs. Share completion
phase costs across predictor arms. All arms pay the same common preparation;
report that cost plus episode latency separately. No evaluation score tuning.

V4 requires complete matching `measured_phase_model_v1` metadata on every
outstanding job; the supplied scheduler has no automatic collector. Its finish
reserve must cover all remaining required validation/checkpoint/finalization,
not merely the final epoch. Pause/resume costs must come from a real checkpoint
and resume observation. Missing compatible data disables admission; it is not
zero. `allow_zero_overhead` stays false. This pilot supplies an explicit
`calibrated_cross_workload_phase_proxy_v1`: disjoint calibration identities and
original observation timestamps are retained. The required legacy source tag is
compatibility metadata; target-job signatures identify proxy predictions, never
a claim that B4/ConvSmall were directly measured. Phase maxima scale by
`max(1,target_parameter_count/max_calibration_parameter_count)`, a predeclared
model estimate, not a proven bound. The B4 380-pixel transfer remains a limitation.
All arms share the same proxy. TTL is four hours from the oldest calibration
observation, with no restamping.

`policy.conservative_admission` is a small proposal: evaluate the unchanged
nominal request objective across both measured halves under nominal, optional
frozen calibration low/high, and explicit no-recovery hypotheses. Require the
minimum gain to exceed measured reserve plus one second. These are finite
scenario stresses, not a statistical confidence bound over independently mixed
per-job errors. Keep FIFO/age order. `trial_amortizes` limits exploration to one
attempt and skips work shorter than the measured reserve. Remaining update
counters already credit actual trial progress; the reserve is an admission
margin, never double-subtracted from optimizer progress. No measured packed
speed is silently relabeled as solo evidence.

## Agent-consumed receipts

`receipts.Submission(job_id, accepted_monotonic, accepted_unix,
expected_updates, expected_epochs, result_path, execution_path)` freezes the
accepted-request clock. `receive(submission, latest_job,
supervisor.observed_exit_code(job_id))` returns `None` while pending. A valid
receipt requires COMPLETED, observed successful process exit, exact runtime and
returned successful updates/epochs, matching source/data/recipe identity,
finalization complete, parsed training and execution artifacts and checksums.
The receipt clock is sampled after artifact reading/parsing/checksum work.
Persist the receipt before submitting dependent requests.

`request_latency('individual', [job_id], receipts, accepted_monotonic)` measures
per-request JCT. Group mode accepts finalized membership and returns the maximum
required receipt minus bracket acceptance. Missing jobs remain missing. A
COMPLETED status alone is not a result-ready event.

The real shared runner uses FP32 AdamW, checks finite scalar loss/gradients,
increments global_step only after optimizer.step, and resumes exact model,
optimizer, random state and data order. After receipt, audit final checkpoint
global_step and AdamW counters against the frozen update budget. Posthoc audit
must pass before an episode is eligible, but heavyweight checkpoint rereading
is not disguised as required agent result parsing.

Record submission/preparation/first and last successful update, every trial,
validation/checkpoint/finalization phases, child exit observation and actual
receipt; also inference/conversion cost, GPU samples and fallback provenance.
Primary metrics are mean individual JCT and all-required bracket makespan;
report trial/checkpoint/non-training overhead and quality beside them.

## CPU verification

From the isolated candidate parent directory (not the scheduler directory, whose
`engine.py` shadows the MLEvolve namespace):

```
PYTHONPATH=/home/justin/MLEvolve/worktrees/a10_feedback_latency:/home/justin/MLEvolve \
  /home/justin/MLEvolve/.venv/bin/python -m pytest -q --rootdir=localml_scheduler --confcutdir=localml_scheduler \
  localml_scheduler/tests/test_a10_feedback_pilot.py localml_scheduler/tests/test_feedback_objective.py
```

Current result: 93 focused tests passed, including an actual CPU subprocess
worker/shared-training receipt and final optimizer-budget replay. Real A10
calibration and scoring remain pending.

## Remote commands

Run from `/workspace/a10_feedback_pilot/app`, with `PYTHONPATH` set to that
directory and `A10_PILOT_DATA_ROOT` set to the staged real-image dataset root.
The only extra parent source needed is `engine/script_introspection.py`, SHA256
`e988d2d42bb74d3d53a8dddbaddda5698bf7404dc13f6a86849c2811da7cb490`.
The settings JSON points at the published S_A10 bundle. Student SHA256 is pinned
`96d31d84d8d809c1ed9a4154f84389a3cb3b673d75445deb650cee4297398e03`.

```
/venv/main/bin/python -m localml_scheduler.benchmarks.a10_feedback_latency.run calibrate \
  --manifest localml_scheduler/benchmarks/a10_feedback_latency/workloads/manifest.json \
  --settings localml_scheduler/benchmarks/a10_feedback_latency/pilot-settings.json \
  --output /workspace/a10_feedback_pilot/runs/calibration-v1 \
  --max-seconds 900 --deadline-utc 2026-10-30T03:10:00Z

/venv/main/bin/python -m localml_scheduler.benchmarks.a10_feedback_latency.campaign \
  --manifest localml_scheduler/benchmarks/a10_feedback_latency/workloads/manifest.json \
  --settings localml_scheduler/benchmarks/a10_feedback_latency/pilot-settings.json \
  --calibration /workspace/a10_feedback_pilot/runs/calibration-v1/calibration.json \
  --output /workspace/a10_feedback_pilot/runs/campaign-v1 \
  --episode-max-seconds 600 --campaign-max-seconds 7200 --deadline-utc 2026-10-30T03:10:00Z
```

Calibration may revise budgets only before the final evaluation manifest is
frozen. Default v4 measurement windows retain two warmup updates, two clean
4-update halves, and a one-second minimum window. Calibrate rates first and ensure
64 updates suffice; choose a new fixed budget from calibration only if needed.
No GPU-arm concurrency cap is set; fixed-three alone has three slots. Fixed-three
also requires prepared reference predictions and the same 90% live/predicted
memory budget; missing predictions fail rather than silently bypassing safety.
All real commands enforce a single GPU lock, fresh output directories, source
checks, whole-episode deadline, and the 03:10 UTC hard evaluation cutoff. Original
datasets/weights/prior runs are never pruned. Owned episode weights may be pruned
only after usable receipts and optimizer/checkpoint hash audits. Raw event stores,
small artifacts, plots, and one small calibration checkpoint are retained.
