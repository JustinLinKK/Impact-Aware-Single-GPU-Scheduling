# CPU preparation repair after replacement preflight failure

The replacement A10 did not start calibration or evaluation. EfficientNet-B4's CPU source conversion exceeded 180 seconds despite staying below the 16 GiB RSS limit. Its failure and destruction remain recorded separately; no result from that attempt is made eligible by this change.

## Scope

Use four predictor CPU threads in every arm, replacing one. Keep the published S_A10 checkpoint, source-conversion implementation, real FX tracing, eager/joint numerical replay, workloads, seeds, labels, splits, policy thresholds, memory safeguards, 180-second conversion timeout, and 16 GiB RSS limit unchanged. Four threads also apply to the parent TorchScript prediction call. Data-loader and training-worker settings do not change.

Preflight now accepts `--settings` and parses the same prediction settings as evaluation. It validates the explicit S_A10 release/model paths, predictor mode, absence of test overrides, and unchanged conversion limits. It records the actual thread count, parsed settings, and SHA-256 of the exact bytes parsed. Omitting `--settings` retains the original one-thread command solely for reproducing prior evidence; a future four-thread dispatch must reject such an unbound proof.

## Local evidence and its limits

Fresh serial CPU-only comparisons completed all three evaluation architecture/batch pairs within the unchanged limits:

| Architecture / batch | One thread | Four threads | Four-thread peak child RSS |
|---|---:|---:|---:|
| EfficientNet-B4 / 16 | 122.254 s | 82.187 s | 11,418.6 MiB |
| ConvNeXt-Small / 16 | 66.554 s | 43.752 s | 4,608.5 MiB |
| Custom CNN / 64 | 5.101 s | 4.299 s | 1,198.8 MiB |

All six native forward/loss/backward replay checks passed, CUDA remained uninitialized, and peak child RSS stayed below 16 GiB. These ordered pairs are local feasibility observations, not a randomized performance study.

All six raw predictions agreed across thread counts within `rtol=1e-6, atol=1e-5`; rounded VRAM predictions were identical. Parallel floating-point reductions need not be bitwise identical. All 17 input tensors, including graph edges and normalized continuous features, were byte-, shape-, and dtype-identical in every pair. This establishes identity of the complete student input representation for these requests, not unrepresented GraphIR internals. All 271 frozen files checked by the benchmark retained their hashes. The feature-parity summary SHA-256 is `f65f09c09d6fb124db941eab2ec2cbe7dd20f30720df716142546436c903a05b`. Measurements and that receipt belong in `outputs/a10-feedback-latency-pilot-20261029/design/repair-v3/`.

These are local CPU preparation measurements, not A10 GPU throughput, slope-accuracy, or scheduling-latency results. Local and rental Python/PyTorch/timm runtimes differ. The available A10 host must still pass a cold preflight using its actual runtime, available CPU quota, and the exact evaluation settings. The repair must not be accepted merely by extrapolating local timings.

## Required dispatch checks

Run the source check and then CPU preflight with `CUDA_VISIBLE_DEVICES=''`, the new application package, and `--settings` pointing to the same staged settings JSON used for calibration and all four evaluation arms. Require the proof's settings hash to equal the staged file hash, `settings_source=experiment_settings`, and `cpu_threads=4`. Require three complete cases, numerical parity, no CUDA initialization, matching workload and S_A10 identities, and successful capture within 180 seconds and 16 GiB. Do not reuse preflight profiles or the old host's calibration.

After exact-host preflight, run fresh dedicated calibration, verify all four successful 64-update jobs and checkpoint/resume audits, and freeze its artifact before the 16-cohort comparison. Continue to charge preparation, trial, waiting, checkpoint, and result-return costs. Keep unattempted or incomplete cohorts in coverage, and require matched repeats for repeatable-improvement claims. A CPU thread change does not authorize any new rental or extend the user's spending cap or deadline.

## Interpretation of policy comparisons

The v4 scheduler prepares predictions synchronously in its background control thread. This can delay worker reaping and therefore usable-result delivery and feedback-dependent arrivals. The harness includes that delay in latency. Fixed-three already prepares predictions asynchronously, so its results compare whole implementations, including this difference; they must not be attributed solely to admission rules.

Profiles are already reused within each episode when source, recipe, batch, and prediction metadata match. Preflight, calibration, and separate episodes use fresh runtimes and caches. This repair neither introduces a shared precomputed cache nor moves preparation off the v4 control thread. A bounded asynchronous preparation service would be a separate implementation and experiment.
