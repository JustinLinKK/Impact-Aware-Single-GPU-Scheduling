# Empirical sampled NVML isolation repair

This source is a new attempt after the bb35d0d calibration attempt failed before producing rows or evaluation outcomes. No failed-attempt values are reused. Resource configurations, data/initialization seeds, two-epoch budgets, measurement-window/warmup rules, policies and64cohort/384job comparison order are unchanged.

This is **not kernel PID mapping**. It conditionally associates a source-pinned owned CUDA process with an NVML PID under the empirically tested completeness of NVML process enumeration. Sampled checks cannot exclude transient unsampled clients. An unknown, extra or changing PID, graphics client, stale/reused process, changed MPS directory, missing proof or violated deadline fails closed.

## Host qualification

`visibility_qualification.py` is standalone stdlib controller plus explicit worker mode. It owns the shared GPU lock through owned cleanup and proof publication, sets an empty private0700nonsymlink MPS pipe and exact CUDA UUID before importing CUDA in workers, and samples0→1→2→1→0with distinct stable NVML PIDs and two independently owned live CUDA contexts. Each source-pinned worker synchronizes a nonce-dependent operation, retains its8MiB probe allocation until source-bound release, and records PID/birth/ancestry/acknowledgment. Repeated queries, rawXML, MPS identity, clocks and exits are retained. Shutdown cleans only owned PID/birth identities; repeated signals cannot strand workers during the bounded cleanup. The absolute bound is120seconds.

```bash
python benchmarks/a10_resource_mechanisms/visibility_qualification.py --gpu-uuid GPU-EXACT-UUID --output /new/runs/visibility-qualification --max-seconds 120
```

The preliminary parent probe established feasibility only. The final frozen package must rerun qualification on the actual host after CPU preflight and before calibration. `audit_qualification` pure-replays the exact qualifier module SHA, GPU UUID, all raw queries, CUDA proofs, runtime and chronology. The live calibration gate also checks the same boot and runtime, and a fresh qualification within600seconds at calibration entry.

## Every calibration launch and resume

Calibrate now requires `--visibility-qualification /path/qualification-proof.json`. It records the proof in its output and every job's raw calibration evidence. Each launch/resume gets a fresh nonce, private empty MPS pipe and bounded deadline including the absolute manifest cutoff. Three idle queries precede dispatch. Executor environment reaches the worker before imports. Before model construction or any update window, `candidate.prepare_training` calls the context challenge and blocks until its synchronized CUDA proof is acknowledged and released. Three singleton queries bracket that live owned descendant's PID, ancestry and start tick before and after each query. Source/spec/runtime/nonce/UUID/MPS identity is checked; pause/resume cannot reuse the previous proof.

The calibration-only context sentinel is exactly one int64 (8logical bytes), retained until worker exit. Actual workload_resources.json records it. Raw allocated/reserved/device measurements are preserved without subtraction; templates conservatively retain any allocator overhead. The8MiB standalone qualification probe allocation is absent from calibration/evaluation. Evaluation has no context-challenge environment and no sentinel; its workload/data/budget recipe is unchanged.

Once training is released, eligible samples must retain the same singleton NVML association and live source-owned process birth in both process snapshots. An empty NVML list while that worker remains live rejects. An empty list is allowed only as explicitly classified terminal-context drain with the source-owned process absent or same-birth zombie in both snapshots. Drain/startup evidence stays raw and **cannot enter any fit peak or update-window aggregate**. Missing eligible evidence rejects instead of inventing measurements.

## Diagnostic timing and replay

Raw launched/runner-ready timestamps and phase histories stay unchanged. Fitted startup/resume subtract only the contained interval from synchronized CUDA proof-ready to the worker's observed source-bound release. Genuine CUDA initialization and small nonce work remain conservatively charged; the whole challenge is never subtracted. Raw duration, diagnostic wait, corrected duration and the shared wall/monotonic conversion are retained per launch in calibration_diagnostic_timing. Prelaunch idle queries occur before launched_unix. The same deterministic correction applies to every calibration job/launch; successful-update windows must start after worker-observed release.

The pure stdlib `context_isolation.audit_raw_evidence(spec, evidence, gpu_uuid=...)` is the local export entry point after saved source/archive/member hashes are verified. It replays qualification, both fresh context associations, raw eligible/drain observations, MPS/process identities and after-release window eligibility without importing local Torch or recomputing host signatures. `extract_row` independently applies those gates alongside all original source/spec/update/window/receipt/allocator checks and retains the raw-evidence digest. Actual source/runtime validation and pooled table replay remain mandatory. Failed/unattempted denominators and original source packages remain separate.

Focused tests use synthetic saved evidence, fake process queries and bounded real CPU-only separate-session/orphan cleanup; they do not run GPU workloads or the expensive six-worker CPU episode. The final exact-host packaged preflight still exercises those existing worker cases before any calibration/evaluation.

Teardown replay correction (separate attempt/source): the first calibration pause
in the preceding attempt saved a known singleton NVML context while its same-birth
owned worker changed from running to zombie across the query. The saved checkpoint
and `job_paused` events preceded this query. This is retained as a terminal
transition, never a usable measurement. A known stale singleton with both worker
snapshots terminal is similarly retained only as a drain, with unchanged launch
root and previous context binding. Unknown/extra NVML PIDs, worker/root PID reuse,
changed parent identity, terminal-to-live resurrection, and live-empty visibility
still reject. No measurement window or peak may use either terminal class, and a
terminal sample cannot satisfy a window's required valid sample coverage. Normal
pause exit-zero/checkpoint and complete-budget receipt audits remain mandatory;
the failed preceding attempt is not retrospectively accepted.

The actual saved query/association are retained unmodified in
`tests/fixtures/a10_teardown_race.json`, with source-file SHA provenance. Focused
CPU replay also injects excessive in-window terminal memory/busy values to ensure
they cannot affect fitted sample indices, means or allocator/device peaks.
