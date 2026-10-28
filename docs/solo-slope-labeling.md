# Solo-slope labeling: derived labels first, sustained measurements later

Status: development proposal, revised 2026-10-28 (dataset audit: 2026-10-27). This document changes no dataset,
training objective, neural-network head, or deployed artifact. The companion
[v4 scheduler design](oct-proposed-v4-design.md) uses an initially uncalibrated
reciprocal of predicted wall-step time. It replaces the proposed admission
criterion, not the evidence standards for measured performance.

## Three quantities that must remain distinct

| Quantity | Source | Meaning |
|---|---|---|
| `profiled_steps_per_second` | reciprocal of a measured wall-step label | throughput under the existing profiling protocol |
| `derived_solo_slope` | reciprocal of the model's predicted wall-step time | uncalibrated nominal solo-update-rate prior for scheduling |
| `successful_updates_per_second` | counted successful updates / measured solo-window seconds | future sustained runtime label |

The first two use the existing one-profiled-step/one-nominal-update convention.
Neither proves that each profiled iteration successfully updated model state.
The third requires an explicit counter with verified semantics. Do not promote
a prediction to an observation, or call an extrapolated epoch a measured epoch.
The word "solo" requires a supported isolation check, not merely absent group
metadata in a file.

The revised scheduler objective is **average request feedback latency** (one
vote per request by default; optional explicit weights): an
individual result or the last required result of a declared group. Job/request
tail latency, queued age and coverage are reported alongside makespan.
This does not change the solo-slope training target into an end-to-end JCT label.
Queueing and policy decisions determine much of JCT; they are not intrinsic to
one model graph. Keep runtime slope/overhead labels separate from scheduler
outcome records to avoid learning a particular historical queue policy as model
compute cost.

## Stage 1: derive the available label

For each canonical example with positive finite measured
`targets.train_step_wall_ms = t_wall_ms`, compute

\[
\texttt{profiled\_steps\_per\_second}=1000/t_{\mathrm{wall,ms}}.
\]

For example, the A10 profile `model_01_0000::bs4::fp32_ieee::adam` has wall-step
time 28.097898000851274 ms, giving 35.58985088385271 profiled steps/s.
Use wall-step time, not GPU-event time, per-sample time, or whole-epoch time.
If only an epoch label is available, do not silently infer an independent step
measurement; its protocol and step count must first establish that relationship.

Implement this later as an additive derived view/sidecar keyed by canonical
sample identity, leaving the six original targets, source measurements, and
existing tensor layout unchanged. The derived view records the source field,
source unit `milliseconds_per_profiled_step`, target unit `profiled_steps_per_second`,
formula/version, and measurement protocol. Reject nonpositive/nonfinite inputs
and nonfinite reciprocals; report invalid rows without fabricating a fallback.

Preserve hardware, source/workload identity and hashes, physical batch,
precision, optimizer, accumulation, step/epoch semantics, source-file identity,
configuration group, and split. Use the canonical dataset manifest's sample
selection and existing group-isolated splits. Raw label files can include
repeated profile IDs: do not append each repeated row as a new training example,
average conflicting measurements silently, or create a new random split.
Resolve selected rows through the manifest and its existing conversion provenance;
unresolved identity conflicts block the derived export for those rows.

Do not read sealed evaluation keys for fitting, selection, or calibration. A
schema/count audit of source rows is not a new evaluation of a model.

### Dataset audit recorded for this design

The 2026-10-27 read-only audit inspected raw timing labels and the RTX5090
conversion index in the local PerfSeer checkout. Counts below are **raw rows**,
not training-split sizes or counts of independent successful optimizer updates.

| Hardware | Audited rows | Timing/protocol observations |
|---|---:|---|
| A10 | 43,708 | 43,703 distinct profile IDs; five repeated IDs |
| A100 | 47,296 | measured single-step epochs |
| A10G | 48,000 | measured single-step epochs |
| RTX5090 | 4,128 | epoch values derived from step time times explicit count |
| Total | 143,132 | all wall-step values positive and finite |

All checked epoch arithmetic matched wall-step time times steps per epoch, with
zero mismatches at relative tolerance 1e-7 and absolute tolerance 1e-8 ms.
For all 139,004 A10/A100/A10G raw rows:

- Accumulation and steps per epoch are both one.
- There is one warm-up epoch followed by three measured single-step epochs.
- `real_dataloader_backed` is false. Inputs are generated rather than a real
  end-to-end data pipeline.
- The inspected schema has no explicit successful/skipped-update counter or
  co-running-member list. Missing metadata does not prove actual isolation.

The RTX5090 index contains explicit counts from 5 to 1,024 steps per epoch;
its epoch values are derived, not independent sustained multi-step measurements.
Its strict graph-capture inputs also differ from the static forward-only
A10/A100/A10G release inputs. Preserve these distinctions during training and
evaluation rather than pooling all protocols as equivalent evidence.

The shipped A10 bundle has 39,294 training and 2,152 validation rows, illustrating
why the raw-file count must not replace the dataset manifest. Preserve the
corresponding existing partitions for all other hardware as well.

### What the current neural network already supports

The released student has a positive wall/GPU timing branch. Its wall output is
exp(log_time[:,0]); epoch time is wall-step time multiplied by `raw_steps`.
The training objective includes hard and teacher-soft log-time regression plus
ranking terms. Therefore a nominal slope can be exposed as

\[
\widehat s_{\mathrm{profile}}=1000\exp(-\widehat\ell_{\mathrm{wall}}).
\]

Keep this as a derived output rather than introducing a separately learned head
that can disagree with the existing latency prediction. The raw six-column
model output remains unchanged. In symmetric log-error regression,
log(s_hat/s) = -log(t_hat/t), so merely relabeling time as slope supplies no new
information or inherent accuracy improvement. Throughput-relative errors should
still be evaluated directly; they are not numerically identical to latency
percentage errors after reciprocal conversion.

In the scheduler's additive profile field, retain source/artifact identity,
`calibration: uncalibrated_prior`, and `requires_online_validation: true`.
Use only declared compatible hardware and workload semantics. Do not treat
successful schema conversion or a high ranking score as validation of a new
workload's absolute runtime.

## Stage 2: sustained successful-update labels

For a future verified solo execution window [a,b], define

\[
s_{\mathrm{runtime}}^0=\frac{U_{\mathrm{success}}(b)-U_{\mathrm{success}}(a)}{b-a}
\quad[\text{successful optimizer updates/s}].
\]

### Measurement contract

- Run a complete training loop with its real data loader, augmentation, transfer,
  forward, loss, backward, accumulation, and optimizer behavior. Record warm-up
  separately and exclude it from the steady-window target.
- Use a monotonic elapsed clock and ensure GPU work through the window endpoint
  has completed. Synchronize at window boundaries as needed; do not introduce
  a synchronization after every update that changes the normal pipeline.
- Increment the successful-update counter only after the defined logical
  optimizer update succeeds. A backward pass, microbatch, or attempted update
  is not enough. Record skipped updates separately; mixed-precision overflow
  must not inflate successful work.
- Define the logical update for multi-optimizer jobs in the runner contract.
  Keep that convention fixed between the predictor input, labeler, K_i, and u_i.
  A change in batch, accumulation, optimizer, precision, or step semantics starts
  a new labeled configuration.
- Include ordinary input waiting and host/device transfers in elapsed training
  time. Record evaluation, checkpointing, final artifacts, startup, and worker
  teardown separately. Split or flag windows crossing these non-training phases
  rather than mixing them into a stationary training-slope label.
- Check for other GPU workloads before and throughout the window. Record the
  observed process/device evidence and relevant backend state. If supported
  checks cannot establish solo execution, retain the observation as an unverified
  or concurrent window; do not use it as a verified solo target.
- Keep each window's counters and duration. For a pooled rate use total successful
  updates / total measured seconds, not the unweighted average of per-window
  rates when their durations differ. Retain variation and drift diagnostics.

The later collection protocol must predeclare warm-up and window lengths using
training/pilot workloads before held-out evaluation. This document establishes
the measurement semantics, not permission to launch a collection campaign or
an assertion that today's three single-step measurements are sufficient.

### Minimum provenance and coverage

Record sample/workload ID and source hash, hardware and host identity, driver/
framework versions, backend and allocation configuration, batch, precision,
optimizer, accumulation, data-loader settings, dataset/input identity, training
phase, step semantics, warm-up counts/time, window duration, start/end attempted
and successful-update counters, skipped counts, and concurrent-process evidence.
Record termination, invalid-window, or early-stopping reasons rather than
silently treating partial observations as complete planned jobs.

Inputs should expose the runtime factors needed to explain the target, including
the real data pipeline and execution backend where they vary. A static graph
alone cannot distinguish two identical models with different unrepresented
input bottlenecks. Retain hardware-specific validity and explicit out-of-scope
handling instead of asserting zero-shot calibration on another GPU.

### Model development and evaluation

Start by evaluating the existing reciprocal prediction against the new sustained
labels. Calibrate or fine-tune the existing wall-time branch using training data
and validation-only selection. Add runtime-context features where residuals show
missing inputs; add uncertainty estimation only with an explicit calibration
evaluation. Retain the existing workload/configuration group isolation.

Report median and tail slope-ratio/log errors, systematic bias by hardware and
workload family, and uncertainty coverage if intervals are introduced. Preserve
ranking metrics, but do not use them alone to establish ratio or ETA accuracy.
Evaluate full traces with unchanged scientific work against v3 and fixed-three
FIFO, including actual trial, checkpoint, and finalization costs. Rank by mean
time from original submission to successful completion; report median/p95/max
JCT, queued age, completion coverage, and makespan alongside it. Completion
includes required outputs, the final checkpoint, and successful worker exit.

Keep a separate scheduler event record for each logical job: original submission,
first dispatch, queue/pause intervals, trial and retry events, final artifact
completion, and worker exit. Preserve the original submission across retries.
Do not feed observed queue delay, final JCT, or future completion events into
the solo-slope predictor or its labels. These records evaluate scheduling policy;
the isolated training windows evaluate the runtime model.

Co-location is a separate labeling extension. Learning s_i(S) or a slowdown
factor requires labels with group membership, allocation/backend context, and
per-member update rates. Single-job labels alone cannot identify that function.
Until sufficient group evidence exists, keep live measured admission trials.

## Acceptance checks for later development

1. Verify reciprocal values and units against selected source rows, including
   28.097898000851274 ms -> 35.58985088385271 profiled steps/s.
2. Reject zero, negative, missing, and nonfinite timing; distinguish derived
   epoch values and profile-step counts from successful-update observations.
3. Confirm original labels/targets, manifest identities, hardware/recipe metadata,
   split membership, and six-output model compatibility are preserved.
4. Resolve A10 duplicate raw IDs through canonical manifest provenance and detect
   inconsistent duplicates instead of silently inventing more samples.
5. Validate runtime counters for accumulation, skipped updates, resume, early
   stopping, and multi-optimizer contracts; verify non-training-phase separation.
6. Verify GPU completion timing and solo evidence, and retain invalid/contaminated
   windows with explicit reasons. Compare pooled counters/time against the label.
7. Require observed calibration and complete-trace evidence before making a
   sustained-slope accuracy or scheduling-benefit claim.
8. Keep scheduling outcome records separate from runtime targets; validate
   immutable submission timestamps and complete-cohort JCT reporting.

## Inspected source snapshot

PerfSeer paths below are relative to the inspected checkout
`/home/justin/PerfSeer-predictor`; they are references, not files modified here:

- `labels/A10/labels/labels.jsonl`, `labels/v4/A100/labels/labels.jsonl`,
  `labels/A10G/labels.jsonl`, and `labels/RTX5090/ready_for_seerblock/index.json`.
- `paper-release/dataset/README.md` and the hardware dataset receipts describe
  canonical rows, splits, capture differences, and measured/derived protocols.
- `paper-release/labeler/vendor/k8s/profile/run_profile.py`,
  `timed_measured_epoch_phase` and `train_fn`, establish the timed step semantics.
- `paper-release/release_v1/distribution/student_training/runtime/student.py`
  defines the wall/GPU timing outputs and derived epoch column;
  `paper-release/release_v1/distribution/student_training/pretrain_objective.py`
  defines the hard/soft log-time and ranking losses.

Scheduler-side references are [predicted profile construction](../prediction/ml_predictor.py),
[declared optimizer steps per epoch](../prediction/released_student.py), and
[validated remaining updates](../scheduler/completion_objective.py). Preserve
their prediction-versus-measurement distinction when implementing this proposal.
