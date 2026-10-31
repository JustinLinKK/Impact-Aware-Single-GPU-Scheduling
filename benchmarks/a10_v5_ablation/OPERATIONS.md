# Per-host V5 ablation sequence

Preparation only. The parent owns resource acquisition, the original billing
clock, the finite destruction guard, collection, launch authorization and cleanup.
No commands below have been executed against a GPU by the preparer.

1. Freeze the reviewed source/package, released S_A10 files, exact blueprint bytes
   and one explicit host plan. Record the declared source commit together with
   the authoritative complete Python SHA map and engine pin. Use the same global
   configuration-specific update counts on both hosts. The five-condition
   blueprint declares 120 cohorts / 864 jobs per host, six arms in every matched
   condition/mode/repeat block, and two repeats per host. Freeze the numeric
   job/episode bound (at most 3600 seconds), calibration bound (at most 7200),
   campaign bound (at most 43200), and the parent's finite host deadline before
   work starts. These are bounds, not a reset of any billing or lease clock.
   An amended operational allowance must be recorded before the next attempt,
   reserved against the same cumulative budget, and applied equally to all arms.
   Preserve failed development evidence separately; a source repair requires
   fresh qualification, calibration and the complete development suite.
2. Run the reviewed host CPU checks. Create the dedicated calibration manifest
   with `protocol calibration-manifest`; this preserves eight replicas, two
   epochs and 1024 updates per epoch from the current blueprint. On the owned
   idle GPU run the packaged visibility qualifier into a new directory. It owns
   the GPU flock; do not wrap it with a second acquisition of that same flock.
   Retain its raw files and proof. Only a valid fresh packaged qualification
   permits the separate dedicated calibration controller.
3. Run all eight dedicated replicas once using the existing resource calibrator
   and `--visibility-qualification`. Preserve raw calibration evidence, context
   and lifecycle samples, receipt artifacts, phase timing, final job records,
   optimizer audits, snapshots, rows, table and logs. This common measured table
   supplies every measured-profile arm on that host.
4. Run `calibrate_predictor` on CPU against that unchanged completed calibration
   directory. It replays all rows and receipts, reconstructs the measured table,
   and uses only the six no-ballast replicas for fresh released S_A10 CPU
   predictions and the unchanged legacy two-scalar correction. It does not run
   additional calibration jobs, evaluate policies or train weights. The output
   includes the exact GPU and measured-table artifact before the legacy table
   SHA is recomputed. Its temporary reconstructed job directories are owned and
   removed. A failed output is retained and must not be reused or overwritten.
5. Seal evaluation manifests using both calibration files and the host settings.
   They bind plan, blueprint, source/runtime, source map, settings and both table
   hashes before evaluation. Evaluation budgets come directly from the global
   blueprint, independently of each host's rates. All development control keys
   are forbidden in held-out manifests, including keys set to false. Keep any
   development shadow/smoke/cap sweep in separately labeled directories and
   outside held-out denominators.
6. The parent launches the finite campaign once after reviewing its binding.
   Each episode receives the appropriate frozen table: common measured profiles
   or the separate published-predictor correction. The controller verifies pins
   before/after episodes, stops on any failed/invalid episode, and retains every
   failed and unattempted cell/job. Recover all evidence even after failure.
   Summaries and paired contrasts remain within host, condition, mode and
   repeat. Zero mechanism activation is reported and is not fabricated or
   required for receipt validity. The parent performs the predeclared cross-host
   analysis and verifies destruction of only the owned resources.

Example CLI shapes (the parent supplies every actual path, source pin and bound;
these are not launch instructions or a lease amendment):

```text
python -m localml_scheduler.benchmarks.a10_v5_ablation.protocol calibration-manifest --plan HOST-PLAN --output NEW-CALIBRATION-MANIFEST
python APP/localml_scheduler/benchmarks/a10_resource_mechanisms/visibility_qualification.py --gpu-uuid EXACT-GPU --output NEW-QUALIFICATION --max-seconds 120
python -m localml_scheduler.benchmarks.a10_resource_mechanisms.calibrate --manifest CALIBRATION-MANIFEST --settings HOST-SETTINGS --output NEW-MEASURED-DIRECTORY --max-seconds CALIBRATION-BOUND --visibility-qualification QUALIFICATION/qualification-proof.json
python -m localml_scheduler.benchmarks.a10_v5_ablation.calibrate_predictor --manifest CALIBRATION-MANIFEST --settings HOST-SETTINGS --measured-calibration MEASURED-DIRECTORY --output NEW-NN-CORRECTION --source-commit FROZEN-COMMIT
python -m localml_scheduler.benchmarks.a10_v5_ablation.protocol evaluation-manifests --plan HOST-PLAN --settings HOST-SETTINGS --calibration MEASURED-DIRECTORY/calibration.json --neural-calibration NN-CORRECTION/calibration.json --output NEW-SEALED-MANIFESTS
python -m localml_scheduler.benchmarks.a10_v5_ablation.campaign --manifests SEALED-MANIFESTS --settings HOST-SETTINGS --calibration MEASURED-DIRECTORY/calibration.json --neural-calibration NN-CORRECTION/calibration.json --output NEW-CAMPAIGN
```

The released model runtime and source capture are CPU implementations. The bridge
also hides CUDA visibility inside its owned prediction context. Its hardware
selection uses the retained A10 name/VRAM and actual hardware key; optional
compute capability, CUDA and driver values absent from the retained record are
explicitly unknown. No live GPU/hardware probe fills these fields. These are
measured-template mechanism controls plus a separate published-profile pipeline
condition, not a claim of neural prediction accuracy or policy promotion.
