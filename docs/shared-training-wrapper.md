# Shared training wrapper for code agents

An agent supplies the model, loss and data. The shared loop handles optimizer
updates, pause/resume, validation and scheduler communication. Source conversion
alone does not make a training script safe to pause: use this loop or implement
the complete existing safe-point contract.

```python
from pathlib import Path
from localml_scheduler.adapters.shared_training import (
    CandidateWorkload, TrainingRecipe, run_candidate_training,
)

def prepare_training():
    return CandidateWorkload(
        adapter=CandidateAdapter(),  # build_model, build_optimizer, training_step
        adapter_context={"lr": 0.001},
        train_dataset=make_training_dataset(),
        recipe=TrainingRecipe(batch_size=4, steps_per_epoch=8, epochs=3),
        identity=source_and_dataset_fingerprint,
        output_dir=Path("working/shared_training"),
        validate=validate_model,  # (model, device) -> {"rmse": float}
    )

if __name__ == "__main__":
    run_candidate_training(prepare_training)
```

Keep imports and `prepare_training()` free of CUDA allocations. The wrapper
establishes the scheduler context and allocator budget before building the
model. MLEvolve recognizes the direct, unshadowed library call in the main
guard. Merely importing the function or setting a capability flag is insufficient.

Version 1 supports one AdamW optimizer group, FP32, one update per batch, no LR
scheduler, a deterministic map-style dataset, and a metric minimized during
validation. The adapter must optimize every trainable parameter exactly once.
Other optimizers, AMP, accumulation, iterable/stateful datasets and multiworker
loading need an explicitly extended checkpoint contract; do not claim they are
supported by this version.

## One recipe for prediction and execution

Keep the source-bound `candidate.py.perfseer.json` contract. Build its `training`
fields from `recipe.prediction_training()` (and the declared execution backend),
and provide the same adapter context, batch shape, actual steps per epoch and
epoch budget. Rebuild the sidecar after changing a draft. The wrapper rejects a
mismatched batch, precision, optimizer, adapter context or training budget before
training. Predictor capture and execution call the same adapter model and loss.

Submit through `build_mlevolve_job` with the normal script runner, script and
working-directory paths, a byte-file loader, and packing eligibility. Derive
`metadata["cooperative_trial"]` using
`localml_scheduler.adapters.shared_training_contract.supports_shared_training_source(source)`
rather than guessing. Keep each candidate's own working directory and result
path. This detector is usable by any agent framework. Local MLEvolve's existing
`engine.script_introspection.supports_cooperative_trial` now delegates to it as
well as accepting the older handwritten hook contract. Older MLEvolve versions
need that delegation added when adopting this runner; updating only the scheduler
package does not update the host framework's source inspector.
The [MLEvolve integration patch](mlevolve-shared-training-integration.patch)
contains the small host change. From the MLEvolve repository root, check it with
`git apply --check <scheduler-path>/docs/mlevolve-shared-training-integration.patch`
before applying it. The local host change has already been applied here.

Configure the released student for the actual GPU. A10G, A10 and A100 remain
distinct targets. The CPU predictor supplies relative epoch cost and busy-order
hints plus memory reservations rounded up to 100 decimal MB. It does not supply
measured elapsed time or an additive GPU-capacity percentage. The scheduler can
start with no measured profiles, then use live overlap windows to keep or undo
an addition. The device-wide memory budget remains 90%; no fixed concurrency
cap is introduced.

## What pause/resume preserves

Pause is acknowledged after a complete optimizer update. The checkpoint contains
model and optimizer state, Python/NumPy/Torch/CUDA random states, next-batch
position, completed validation epochs, validation history and the best selected
weights. Epoch permutations use a separate deterministic generator. Loading is
synchronous, so unsaved worker prefetch cannot advance augmentation randomness.
Validation pending after the final batch of an epoch is resumed too.

Resume rejects a changed recipe, source/adapter identity or dataset identity.
Supply an identity covering your model dependencies, split and preprocessing, and
keep the underlying dataset immutable. PetFinder's example binds CSV bytes,
split indices, source/helper bytes and image file sizes/timestamps; these are
local change-detection checks, not a cryptographic archive of every image.

`determinism="strict"` requires deterministic PyTorch kernels. The default
`"best_effort"` warns for operators without one, including CUDA adaptive-pooling
backward in these PetFinder drafts. State and sample order are preserved in both
modes; only supported deterministic kernels can promise bitwise GPU replay.

The final `training_state.pt` is the last-update state for recovery.
`selected_model.pt` contains validation-selected weights for inference.
An optional `finalize(model, device)` callback writes predictions/submissions
from the selected model; it should be safe to repeat after an interruption.

## PetFinder adaptation and local verification

The original agent outputs remain in
`verification/mlevolve_petfinder_20261027/raw/`. The corresponding runnable copies
are in `adapted/`. Their model/loss implementations, image preprocessing, AdamW
settings and seeded 90/10 split are preserved. The shared runner replaces their
training loops and two-worker loaders. Candidate 01's prediction context now
also uses the actual TRAIN-only target mean used to initialize its output bias.
Training order is explicitly defined by the wrapper's per-epoch permutation;
this is not intended to reproduce the old loader's sample order.

The adapted defaults retain three epochs and eight batches per epoch. The GPU
smoke harness creates separate source-bound copies with two epochs of 192
batches and 32 validation images to allow startup, pause/resume and overlap
measurement. Those validation scores are integration diagnostics, not model
quality estimates. It uses real local PetFinder images and writes an ID-aligned
test submission from each candidate.

Run the bounded harness from the directory containing this scheduler package:

```bash
python -m localml_scheduler.examples.verify_petfinder_shared_training \
  --output /absolute/path/to/new-smoke-directory \
  --dataset /absolute/path/to/petfinder-pawpularity-score \
  --release-root /absolute/path/to/PerfSeer-paper-release \
  --model-dir /absolute/path/to/checkpoints/S_A10G
```

This harness deliberately tests A10G CPU predictions against local RTX 5090
training via a test-only predictor injection. Production hardware matching is
unchanged. It verifies control and integration, not A10G prediction accuracy or
scheduling speedup. It checks live headroom before launch, uses real telemetry
and allocator guards, and stops its own jobs if the 90% device budget is exceeded.
Other training is left running. Retain failed-attempt receipts when diagnosing
an integration problem.

The [recorded local verification](../verification/shared_training_20261027/README.md)
includes both completed drafts, the manual pause/resume, the overlap decision,
failed-attempt diagnoses, recovery, regression results and the combined chart.
