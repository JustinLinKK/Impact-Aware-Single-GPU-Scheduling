# CPU-only harness repair after invalidated campaign v1

All campaign-v1 results from source `c0851b1e9851441f240f3f91e4fac188c07bd42a`
are diagnostic and cannot support a predictor or admission-policy comparison.
Their original source/configuration and collected result/event artifacts remain
unchanged. The parent stopped the owned campaign and destroyed its instance;
this repair authorizes no replacement rental, GPU launch, or evaluation rerun.

Two confirmed defects invalidated v1:

* EfficientNet-B4 capture exceeded the 8192 MiB CPU conversion limit (observed
  8204–8212 MiB), causing `branch_profile` fallback. The cohort was marked
  predictor-ineligible, but the controller still continued its comparison.
* Jobs were constructed before request acceptance, retaining earlier
  `submitted_at` timestamps. Their later `feedback_request.submitted_at` values
  violated the exact request/member causality contract. Admission forecasts
  consequently failed before reaching the comparator.

The repaired common harness binds `TrainingJob.submitted_at` to the acceptance
timestamp immediately before every `client.submit`, including calibration and
all four arms. Group request clocks remain the first accepted group timestamp;
each member is accepted at or after that instant. Individual request and member
timestamps are identical. CPU conversion remains bounded at 180 seconds, with
the memory limit raised to 16384 MiB (16 GiB). This changes only capture resource
allowance, not workload, batch, seeds, model, optimizer, or prediction output.

Predictor exceptions, fallback sources, missing prepared numeric profiles, and
incomplete exposure now abort diagnostically. The campaign independently refuses
an ineligible cohort. Preparation time remains charged after acceptance. The
unchanged v4/revised admission functions receive the same inputs and retain their
previous decision rules. No calibration is refitted by this repair.

Predictor validity and policy activation are reported separately. Saved summaries
count accepted/rejected forecasts only when error is absent and both independent
halves are present. Inconclusive errors and legitimate revised-policy amortization
skips remain visible. Zero actionable v4 forecasts cannot establish an admission
comparison even with complete predictor coverage.

`tests/test_a10_feedback_repair.py` assembles actual manifest jobs through
`build_job`, acceptance, real SQLite `client.submit`, `requests_from_jobs`,
`freeze_feedback_trial`, and the unmocked `evaluate_feedback_trial` comparator.
Declared synthetic CPU protocol fixtures produce two accepted and two rejected
decisions across individual/group modes, all with error=None and two halves.
These are protocol tests, not measured GPU performance evidence.

`cpu_preflight.py` separately prepares all three unique evaluation architecture/
batch pairs using the unchanged published S_A10 artifact and actual source capture.
It requires CUDA visibility disabled, supplies an explicit A10 hardware identity,
records local runtime versions and measured child RSS, and uses the 16 GiB/180 s
limits. Capture uses synthetic batches for graph extraction and no real-image fit.
Its CPU cache is diagnostic only and must not be reused by a later evaluation;
cold preparation would be charged again. Actual completion/peak-RSS evidence is
saved outside the source tree in `work/a10-latency-pilot/cpu-repair-v2-preflight/`.

Completed local CPU proof (Python3.11.15, Torch2.11.0+cu130,
torchvision0.26.0+cu130, timm1.0.26):

| Architecture / physical batch | Peak child RSS MiB | Capture seconds | Numeric parity |
|---|---:|---:|---|
| EfficientNet-B4 /16 | 11408.000 | 122.748 | true |
| ConvNeXt-Small /16 | 4715.398 | 66.014 | true |
| Custom CNN /64 | 1169.391 | 4.896 | true |

All three produced valid published-A10 profiles without fallback, stayed below
16384 MiB/180 seconds, and left `torch.cuda.is_initialized()` false before and
after. Backward graph counts are1320/1366/57 respectively. This is local runtime
feasibility evidence; a later authorized replacement must repeat CPU preparation
on its exact runtime and perform fresh dedicated calibration. Diagnostic caches
must not warm the measured evaluation.
