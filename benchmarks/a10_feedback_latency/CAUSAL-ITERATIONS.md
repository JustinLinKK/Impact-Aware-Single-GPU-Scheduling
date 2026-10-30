# Three A10 iterations: preparation, admission, and workload properties

This is a new experiment series after the completed 16-cohort pilot. Original
pilot artifacts are immutable. The new branch starts from `9fd46e6`. Original
predictor checkpoints, labels, and dataset splits remain unchanged. The total
budget is $20, conservatively including the prior observed $0.885; billing lag,
storage, and transfer charges count. No local GPU training is needed initially.

## Questions and evidence

The old comparison changed both policy and prediction preparation: fixed-three
prepared in a background thread while v4 could block its scheduler loop. Old
profile publication timestamps cannot identify pure capture time or CPU service
time. In the new experiments record enqueue, preparation start/end, publication,
launch, first/last successful update, worker finish, observed exit, and usable
receipt. Preparation can overlap GPU training; do not sum overlapping intervals
or call every submission-to-launch second prediction overhead.

Primary outcomes are average individual submission-to-usable-receipt latency
and average bracket submission-to-all-required-receipts latency. Report request
p95 and maximum, makespan, every repeat, failures and unattempted denominators.
Six jobs form three causal parent/child chains. Individual children arrive two
seconds after their own parent receipt; bracket children arrive two seconds
after all parent receipts. Child preparation may not start before acceptance.

## Iteration 1: preparation ablation on the original six jobs

Use the existing cancer-image workloads, original budgets (64 successful AdamW
updates and two epochs), published S_A10, and one fresh same-host calibration.
Run five arms in both modes, twice, with reversed arm/mode order in repeat two:

1. Calibrated v4, synchronous preparation (existing policy).
2. Calibrated v4, asynchronous preparation (same policy).
3. Calibrated revised, synchronous preparation (existing conservative policy).
4. Calibrated revised, asynchronous preparation (same conservative policy).
5. Calibrated fixed-three FIFO, asynchronous preparation.

All five use identical calibrated profiles, including fixed-three. The paired
sync/async contrasts estimate the effect of the preparation/lifecycle change
under each policy. Policy contrasts use the asynchronous arms only. These
contrasts remain whole-system effects: preparation readiness changes the
decisions available to a scheduler. They are not an additive decomposition.

The exact treatment is first-demand synchronous preparation versus
acceptance-enqueued serial FIFO preparation. It changes preparation order and
readiness as well as threading. All prediction helper methods use one common
profile cache in both treatments, so legacy forwarding cannot accidentally
read uncalibrated values. This is an explicit common amendment to the prior
pilot, not a byte-identical rerun. Verify per-workload numeric profile equality
across arms before interpreting their differences. There are exactly 20 cohorts
and 120 evaluation jobs in this iteration; the four dedicated calibration jobs
are excluded from evaluation denominators.

The asynchronous worker prepares accepted inputs serially. The controller must
remain responsive to exits and live trial measurements. Unprepared jobs stay
pending; no missing-prediction fallback is allowed. Solo work may launch when
ready, but a packed trial requires compatible profiles for the full outstanding
forecast cohort. Cache keys must bind workload identity, batch, artifact, and
hardware. Cancellation must not reuse a partially prepared profile.

Freeze this iteration's source, settings, manifest, and arm order before GPU
evaluation. Use fresh per-episode runtimes and cold process caches, the same
warm data-cache convention as the original pilot, and charge preparation,
validation, checkpointing and result delivery in request latency.

## Iteration 2: controlled resource and duration mechanisms

Design explicit training jobs to vary persistent memory footprint, active GPU
compute per optimizer update, and update budget. Include light/light,
memory-heavy/compute-light, compute-heavy/memory-light, and heavy/heavy stress
conditions, plus mixed and competing cohorts. Record the actual measured
resource properties; architecture names are not evidence of a resource class.
If synthetic memory ballast or deliberate host delay is used, identify it and
report it separately from learned allocated-VRAM targets and ordinary training.

Before evaluation, run separate solo calibration jobs with disjoint seeds.
Measure sustained successful-update slopes, peak allocated and reserved memory,
device memory, power and sampled GPU busy. GPU busy is not SM occupancy; report
SM/DRAM counters only if actually available. Each evaluation budget and source
must be fixed in its manifest before any arm runs.

Use a common preparation path and common input profiles across arms. A measured
template profile, if required for designed workloads outside predictor coverage,
is an explicitly labeled research control, never claimed as neural prediction
or held-out predictor accuracy. Keep an S_A10 arm on supported workloads to
quantify predictor mismatch separately where feasible.

Compare asynchronous v4, asynchronous fixed-three and one revised policy chosen
from the iteration-1 mechanisms. The policy may change only between iterations;
document the change and freeze it before iteration-2 outcomes. A promising
candidate removes double-counting of already productive trial time in the
post-trial admission margin, and accounts only for future incremental costs.
Existing post-trial forecasts already use current remaining updates and future
defer pause/resume costs. Removing an additional historical reserve is distinct
from the pre-trial exploration-risk safeguard. Productive trial time has an
incremental opportunity cost relative to the no-trial alternative; it is not
entirely wasted and is not free.
Retain memory safeguards, productive update credit, incumbent continuity,
clean two-half evidence and membership invalidation. Evaluate both modes with
paired repeats and explicit complementary versus competing mixtures.

## Iteration 3: frozen validation

Freeze one policy after iteration 2. Evaluate it against asynchronous v4 and
fixed-three on held-out workload configurations, durations, seeds and mixtures
that were not used to select that policy. Use both modes and two balanced paired
repeats. Preserve the same prediction/preparation path within each comparison.
No policy adjustment after observing this iteration's outcomes. Report losses
and gains, uncertainty appropriate to the small repeat count, and limitations
of synthetic workloads. Successful evaluation is not model promotion.

## Validity and operations

Each iteration has a finite operational cutoff, per-episode timeout and bounded
campaign controller. Confirm an idle exact A10 host, source/data/model hashes,
CPU capture correctness and calibration before evaluation. A collector and
independent finite lease/budget guard preserve artifacts and clean up only the
owned instance. The lightweight subagent checks progress every 30 minutes;
there is no agent polling loop. Prior completed results are never mixed with a
failed/repaired protocol without labeling the amendment.

All usable receipts must verify successful update/epoch counts, finite metrics,
successful worker exit, source/runtime identity, causal arrivals and full
coverage. Audit each stage before interpreting rankings. Generate a Gantt chart
above optimizer-update and memory plots for every episode and combined panels
for comparisons; add preparation intervals and receipt markers. Verify exported
artifact hashes before owned-resource cleanup. Keep large checkpoints/caches
out of the report exports. Three iterations are required; this document alone
does not complete the user goal.
