"""Real-run A10 pilot, with a colocated requestor and no cloud management.

Commands: ``python -m localml_scheduler.benchmarks.a10_feedback_latency.run
calibrate|run --manifest ... --settings ... --output ...``. A settings JSON
supplies the staged published S_A10 release/model paths. Every invocation owns a
fresh output directory; existing runs are never resumed or overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
import uuid

from ...adapters.mlevolve import build_mlevolve_job
from ...adapters.shared_training_contract import supports_shared_training_source
from ...client import SchedulerClient
from ...config import SchedulerSettings
from ...domain import CheckpointPolicy, JobStatus, ResourceRequirements, utc_now
from ...execution.control import ControlPlane
from ...execution.executor import SubprocessExecutor
from ...execution.process_utils import terminate_process_tree
from .calibration import CalibratedPredictor, RecordingPredictor, fit_calibration, phase_model, trial_reserve
from .receipts import Submission, receive, request_latency
from .service import RevisedPilotService


ARMS = ("reference_v4", "calibrated_v4", "calibrated_revised", "fixed_three")
CAUSAL_ARMS = ("causal_v4_sync", "causal_v4_async", "causal_revised_sync",
               "causal_revised_async", "causal_fixed_three")
RESOURCE_ARMS = ("resource_v4", "resource_revised", "resource_incremental", "resource_fixed_three")
REFERENCE_COMMIT = "95b651d2e2a89599138c790ab4f9c1363f29e65f"
STUDENT_SHA256 = "96d31d84d8d809c1ed9a4154f84389a3cb3b673d75445deb650cee4297398e03"


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def check_deadline(manifest, elapsed=0., max_seconds=None):
    cutoff = datetime.fromisoformat(manifest["deadline_utc"].replace("Z", "+00:00"))
    if cutoff.tzinfo is None or datetime.now(timezone.utc) >= cutoff:
        raise TimeoutError("global pilot deadline reached")
    if max_seconds is not None and elapsed >= max_seconds:
        raise TimeoutError("predeclared episode deadline reached")


def with_hard_deadline(manifest, deadline_utc):
    result = dict(manifest)
    declared = datetime.fromisoformat(manifest["deadline_utc"].replace("Z", "+00:00"))
    hard = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
    if declared.tzinfo is None or hard.tzinfo is None:
        raise ValueError("deadlines must be timezone-aware")
    result["deadline_utc"] = min(declared, hard).isoformat()
    return result


def source_snapshot(manifest, manifest_path):
    root = Path(__file__).resolve().parents[2]
    sources = {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))
               if not any(part in {"__pycache__", ".venv", "runtime"} for part in p.parts)}
    for spec in [*manifest.get("jobs", []), *manifest.get("calibration", [])]:
        path = (manifest_path.parent / spec["source"]).resolve()
        if file_hash(path) != spec["source_sha256"]:
            raise ValueError("frozen workload source changed: " + spec["name"])
        if not supports_shared_training_source(path.read_text()):
            raise ValueError("workload must directly invoke the real shared runner")
    for item in manifest.get("additional_pins", []):
        path = (manifest_path.parent / item["path"]).resolve()
        if file_hash(path) != item["sha256"]:
            raise ValueError("additional source/data/artifact pin changed: " + str(path))
    import engine.script_introspection as introspection
    expected_engine = "e988d2d42bb74d3d53a8dddbaddda5698bf7404dc13f6a86849c2811da7cb490"
    if file_hash(introspection.__file__) != expected_engine:
        raise ValueError("frozen parent engine/script_introspection.py changed")
    return {"scheduler_reference_commit": REFERENCE_COMMIT,
            "effective_deadline_utc": manifest["deadline_utc"],
            "external_engine_script_introspection_sha256": expected_engine,
            "scheduler_python_sources": sources, "manifest_sha256": file_hash(manifest_path)}


def gpu_sample():
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=5)
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("pilot requires exactly one visible rented GPU")
    name, gpu_uuid, total, used, utilization, memory_busy, power = [part.strip() for part in rows[0].split(",")]
    if name != "NVIDIA A10":
        raise ValueError("pilot is frozen to NVIDIA A10, got " + name)
    optional = lambda x: None if x in {"[N/A]", "N/A", "[Not Supported]"} else float(x)
    return {"name": name, "gpu_uuid": gpu_uuid, "total_mib": float(total), "used_mib": float(used),
            "utilization_percent": float(utilization), "memory_controller_busy_percent": optional(memory_busy),
            "power_watts": optional(power), "occupancy_note": "NVML busy counters are not SM occupancy"}


FEEDBACK_PHASE_MODEL_TTL_SECONDS = 14400.


def make_settings(raw, output, hardware):
    raw = json.loads(json.dumps(raw))
    raw.update(runtime_root=str(output / "runtime"), python_executable=sys.executable,
               scheduler_poll_interval_seconds=.05, early_stopping={"enabled": False},
               graph_db={"enabled": False}, hardware_knowledge_graph={"enabled": False},
               hardware_feature_db={"enabled": False})
    gpu = raw.setdefault("gpu_scheduler", {})
    gpu.update(packing_backend="cuda_process", batch_probe_enabled=False,
               startpoint_probe_enabled=False, parallel_job_cap=None)
    gpu.setdefault("memory", {}).update(gpu_vram_gib=hardware["total_mib"] / 1024.,
                                       predicted_budget_fraction=.9)
    gpu["feedback_latency"] = {"enabled": True, "allow_zero_overhead": False,
        "bracket_admission_mode": "recovery_regret", "phase_model_ttl_seconds": FEEDBACK_PHASE_MODEL_TTL_SECONDS}
    settings = SchedulerSettings.from_dict(raw)
    settings.ensure_runtime_layout()
    atomic_json(settings.runtime_root / "scheduler_settings.json", settings.to_dict())
    return settings


def build_job(spec, *, manifest_path, output, manifest, memory_budget_mib):
    if not spec["name"] or Path(spec["name"]).name != spec["name"]:
        raise ValueError("workload name must be one owned directory component")
    work = output / "jobs" / spec["name"]
    work.mkdir(parents=True, exist_ok=False)
    source = (manifest_path.parent / spec["source"]).resolve()
    metadata = {**spec.get("metadata", {}), "cooperative_trial": True,
        "skip_active_scheduler_probes": True, "placement_backend": "cuda_process",
        "trial_memory_limit_mb": memory_budget_mib, "metric_maximize": False,
        "pilot_parameter_count": spec["parameter_count"],
        "pilot_frozen_workload_sha256": digest(spec)}
    if spec.get("workload_kind") == "synthetic_resource_control":
        from ..a10_resource_mechanisms.workloads import adapter_context, recipe, template_key
        metadata.update(resource_workload_spec=json.loads(json.dumps(spec)),
                        resource_template_key=template_key(spec))
        metadata["perfseer_model"] = {"epochs": spec["epochs"],
            "steps_per_epoch_by_batch": {str(spec["batch"]): spec["steps_per_epoch"]},
            "workload_contract": {"adapter": {"context": adapter_context(spec)},
                                  "training": recipe(spec).prediction_training()}}
    kwargs = {"script_path": str(source), "working_dir": str(work),
              "result_path": str(work / "execution.json"), "batch_size": spec["batch"],
              "timeout": manifest["job_timeout_seconds"]}
    job = build_mlevolve_job(workflow_id="a10-feedback-pilot", baseline_model_id=spec["name"],
        baseline_model_path=str(source), runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
        runner_kwargs=kwargs, loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
        max_epochs=spec["epochs"], max_steps=spec["epochs"] * spec["steps_per_epoch"],
        checkpoint_policy=CheckpointPolicy(save_every_n_steps=0, save_every_epoch=True, keep_last_n=2),
        resource_requirements=ResourceRequirements(requires_gpu=True), packing_eligible=True,
        packing_family=spec["model_family"], packing_backend_allowlist=["cuda_process"], metadata=metadata)
    job.job_id = uuid.uuid5(uuid.NAMESPACE_URL, str(output) + ":" + spec["name"]).hex
    job.config.env.update({"A10_PILOT_SPEC_JSON": json.dumps(spec, sort_keys=True),
        "A10_PILOT_OUTPUT": str(work / "working" / "shared_training"),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        **spec.get("env", {})})
    return job, work


def submission(spec, job, work, accepted_mono, accepted_unix):
    return Submission(job.job_id, accepted_mono, accepted_unix,
        spec["epochs"] * spec["steps_per_epoch"], spec["epochs"],
        str(work / "working" / "shared_training" / "result.json"), str(work / "execution.json"),
        spec["expected_runtime_signature"], digest(spec), spec.get("validation_size", 256))


def accept_job(client, job, accepted_unix):
    """Bind the member timestamp to actual acceptance, after all eager building."""
    if isinstance(accepted_unix, bool) or not math.isfinite(accepted_unix):
        raise ValueError("finite accepted request timestamp required")
    job.submitted_at = datetime.fromtimestamp(accepted_unix, timezone.utc).isoformat()
    return client.submit(job)


def require_predictor_exposure(predictor, jobs, specs, *, require_all=False, required_source="ml_predictor"):
    """A failed or missing prior aborts diagnostically; never continue a comparator."""
    failures = dict(getattr(predictor, "failures", {}))
    sources, errors = dict(predictor.last_sources), dict(predictor.last_errors)
    finished = getattr(predictor, "preparations_finished", set())
    for name, job in jobs.items():
        if job.job_id in failures or job.job_id in errors or sources.get(job.job_id) not in (None, required_source):
            raise RuntimeError("predictor exposure failed for " + name + ":" + str(
                failures.get(job.job_id) or errors.get(job.job_id) or sources.get(job.job_id)))
        if (require_all or job.job_id in finished) and (
                sources.get(job.job_id) != required_source
                or specs[name]["batch"] not in predictor.records.get(job.job_id, {})):
            raise RuntimeError("missing required numeric predictor exposure for " + name)


def events_for(path, job_id):
    if not path.exists(): return []
    return [event for line in path.read_text().splitlines() if line.strip()
            for event in [json.loads(line)] if event.get("job_id") == job_id]


def policy_activation(events):
    """Count observed policy decisions separately from successful predictor exposure."""
    summary = {"accepted_forecasts": 0, "rejected_forecasts": 0,
               "actionable_forecasts": 0, "inconclusive_or_invalid": 0,
               "inconclusive_reasons": {}, "amortization_skips": 0, "skip_reasons": {}}
    for event in events:
        payload = event.get("payload") or {}
        if event.get("event_type") == "pilot_trial_skipped":
            summary["amortization_skips"] += 1
            reason = str(payload.get("reason"))
            summary["skip_reasons"][reason] = summary["skip_reasons"].get(reason, 0) + 1
        if event.get("event_type") != "cold_start_trial_decided": continue
        decision, halves = payload.get("decision"), payload.get("halves")
        has_both_halves = isinstance(halves, list) and len(halves) == 2
        if not has_both_halves and isinstance(payload.get("scenarios"), list) and payload["scenarios"]:
            has_both_halves = all(scenario.get("forecast", {}).get("error") is None
                and len(scenario.get("forecast", {}).get("halves", [])) == 2
                for scenario in payload["scenarios"])
        if payload.get("error") is None and decision in {"accepted", "rejected"} and has_both_halves:
            summary[decision + "_forecasts"] += 1
            summary["actionable_forecasts"] += 1
        else:
            summary["inconclusive_or_invalid"] += 1
            reason = str(payload.get("error") or payload.get("reason") or decision)
            summary["inconclusive_reasons"][reason] = summary["inconclusive_reasons"].get(reason, 0) + 1
    return summary


def timestamp(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def parent_due(start, chain, mode):
    return start if mode == "group" else start + 2. * chain


def fixed_fifo_fits(candidate_profile, running, profiles, *, device_used_mib, budget_mib):
    """Same 90% prediction/live-reservation safety budget; no missing prior fallback."""
    if candidate_profile is None: raise ValueError("fixed-three missing required reference prediction")
    occupied = 0.
    for name, job in running.items():
        profile = profiles.get(name)
        if profile is None: raise ValueError("fixed-three running job missing required reference prediction")
        occupied += max(float(profile["peak_vram_mib"]), float(job.metadata.get("trial_reserved_mb") or 0.))
    peak = float(candidate_profile["peak_vram_mib"])
    if any(not math.isfinite(v) or v < 0 for v in (occupied, peak, device_used_mib, budget_mib)) or peak <= 0:
        raise ValueError("invalid fixed-three memory evidence")
    return max(occupied, device_used_mib) + peak <= budget_mib


def audit_checkpoint(spec, work, job):
    import torch
    path = work / "working" / "shared_training" / "training_state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = spec["epochs"] * spec["steps_per_epoch"]
    if (state["global_step"] != expected or state["completed_epochs"] != spec["epochs"]
            or state["signature"] != spec["expected_runtime_signature"]
            or job.metadata.get("runtime_global_step") != expected):
        raise ValueError("final checkpoint budget/identity audit failed")
    counters = [int(v["step"]) for v in state["optimizer_state"]["state"].values()]
    if not counters or any(v != expected for v in counters):
        raise ValueError("AdamW successful update counters do not match final budget")
    result = {"global_step": expected, "completed_epochs": state["completed_epochs"],
              "optimizer_parameter_states": len(counters), "all_adamw_steps": expected,
              "training_state_sha256": file_hash(path), "training_state_bytes": path.stat().st_size}
    del state
    return result


def prune_owned_weights(output):
    """Only remove owned new-run weights after receipt+checkpoint validity audits."""
    output = output.resolve()
    if not (output / "validity-audits.json").is_file():
        raise ValueError("refuse pruning without completed validity audits")
    removed = []
    for base in (output / "jobs", output / "runtime" / "data" / "checkpoints"):
        if not base.is_dir(): continue
        for path in base.rglob("*.pt"):
            if path.is_symlink() or not path.resolve().is_relative_to(output):
                raise ValueError("refuse weights outside owned episode directory")
            removed.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                            "sha256": file_hash(path)})
            path.unlink()
    atomic_json(output / "pruned-weights.json", removed)


def calibrate(manifest, manifest_path, raw_settings, output, max_seconds):
    hardware = gpu_sample()
    if hardware["used_mib"] > 512: raise RuntimeError("calibration requires an otherwise idle A10")
    settings = make_settings(raw_settings, output, hardware)
    client = SchedulerClient(settings)
    # No scheduler thread competes with the dedicated isolated calibration jobs.
    service = client.create_service()
    predictor = service.planner.estimator.ml_predictor
    if not predictor or not predictor.available: raise ValueError("published A10 predictor unavailable")
    if predictor.artifact_hash != STUDENT_SHA256: raise ValueError("published S_A10 student hash mismatch")
    executor, controls = SubprocessExecutor(settings), ControlPlane(settings)
    start = time.monotonic()
    rows, audits = [], {}
    next_sample = 0.
    for spec in manifest["calibration"]:
        check_deadline(manifest, time.monotonic() - start, max_seconds)
        if spec["role"] != "calibration": raise ValueError("calibration split mismatch")
        job, work = build_job(spec, manifest_path=manifest_path, output=output, manifest=manifest,
                              memory_budget_mib=hardware["total_mib"] * .9)
        job.metadata["trial_step_command"] = {"action": "measure", "owner": "pilot-calibration",
            "kind": "calibration", "token": uuid.uuid4().hex, "window_min_seconds": .2,
            "expires_at": time.time() + max_seconds}
        accepted_mono, accepted_unix = time.monotonic(), time.time()
        job = accept_job(client, job, accepted_unix)
        frozen = submission(spec, job, work, accepted_mono, accepted_unix)
        predicted = service.planner.estimator.predicted_profile(job, spec["batch"])
        if not predicted or predicted["hardware"] != "A10": raise ValueError("A10 prediction missing")
        profile_ready = time.monotonic()
        client.store.set_job_status(job.job_id, JobStatus.READY, reason="isolated dedicated calibration", hold=False)
        handle = executor.start(client.store.get_job(job.job_id))
        launches = [{"launched_unix": time.time(), "launched_monotonic": time.monotonic()}]
        pause_requested, resumed, pause_started, pause_finished = False, False, None, None
        windows, peak_reserved_seen = [], 0.
        try:
            while True:
                check_deadline(manifest, time.monotonic() - start, max_seconds)
                latest = client.store.get_job(job.job_id)
                peak_reserved_seen = max(peak_reserved_seen, float(latest.metadata.get("trial_peak_reserved_mb") or 0.))
                elapsed = time.monotonic() - start
                if elapsed >= next_sample:
                    sample = gpu_sample()
                    if sample["used_mib"] > sample["total_mib"] * .94:
                        raise RuntimeError("calibration GPU live memory emergency cutoff")
                    with (output / "timeline.jsonl").open("a") as stream:
                        stream.write(json.dumps({"elapsed_seconds": elapsed, "gpu": sample,
                            "jobs": {spec["name"]: {"status": latest.status.value,
                                "global_step": latest.metadata.get("runtime_global_step", 0),
                                "reserved_mib": latest.metadata.get("trial_reserved_mb")}}}) + "\n")
                    next_sample = elapsed + 1.
                window = latest.metadata.get("trial_step_window")
                if (isinstance(window, dict) and window.get("complete")
                        and window.get("excludes_nonupdate_phases") is True
                        and window not in windows): windows.append(window)
                exit_code = handle.process.poll()
                if windows and not pause_requested and exit_code is None:
                    pause_started = time.monotonic()
                    controls.request_pause(job.job_id, reason="charged calibration pause/resume", hold=True)
                    pause_requested = True
                if exit_code is not None and latest.status == JobStatus.PAUSED:
                    if exit_code != 0 or resumed: raise RuntimeError("invalid calibration pause exit")
                    pause_finished = time.monotonic()
                    if not latest.latest_checkpoint_path: raise ValueError("pause has no checkpoint")
                    latest = client.store.update_job(job.job_id, status=JobStatus.READY, hold=False,
                        metadata_updates={
                        "trial_step_window": None, "trial_step_command": {
                            "action": "measure", "owner": "pilot-calibration", "kind": "calibration",
                            "token": uuid.uuid4().hex, "window_min_seconds": .2,
                            "expires_at": time.time() + max_seconds}})
                    launches.append({"launched_unix": time.time(), "launched_monotonic": time.monotonic()})
                    handle = executor.start(latest)
                    resumed = True
                    continue
                receipt = receive(frozen, latest, exit_code)
                if receipt is not None: break
                time.sleep(.02)
        finally:
            terminate_process_tree(handle.process)
        if not resumed or not windows: raise ValueError("real pause/resume/window calibration missing")
        events = events_for(settings.events_jsonl_path, job.job_id)
        ready = [timestamp(e["created_at"]) for e in events if e["event_type"] == "trial_runner_ready"]
        if len(ready) != 2: raise ValueError("fresh and resumed runner readiness evidence required")
        startup_seconds = ready[0] - launches[0]["launched_unix"]
        resume_seconds = ready[1] - launches[1]["launched_unix"]
        # The job finished timestamp is persisted in the worker before exit; the
        # extra receipt delay covers exit observation, polling, parse and hash.
        delivery_seconds = max(0., receipt["feedback_ready_unix"] - timestamp(latest.finished_at))
        audits[spec["name"]] = audit_checkpoint(spec, work, latest)
        atomic_json(work / "receipt.json", receipt)
        atomic_json(work / "events.json", events)
        row = {"name": spec["name"], "role": "calibration", "complete": True,
            "source_identity": {"workload_sha256": digest(spec), "source_sha256": spec["source_sha256"]},
            "runtime_completion_identity": spec["expected_runtime_signature"],
            "parameter_count": spec["parameter_count"], "observed_at": utc_now(),
            "predicted_profile": predicted, "windows": windows,
            "isolated_seconds_per_update": statistics.median(
                float(value) for w in windows for value in w["halves"]),
            "peak_reserved_mib": peak_reserved_seen,
            "startup_seconds": startup_seconds, "resume_seconds": resume_seconds,
            "pause_seconds": pause_finished - pause_started,
            "delivery_seconds": delivery_seconds,
            "phase_history": latest.metadata["runtime_phase_timing_history"],
            "prediction_seconds": profile_ready - accepted_mono,
            "receipt": receipt, "launches": launches}
        rows.append(row)
        atomic_json(output / "calibration-rows.json", rows)
        atomic_json(output / "validity-audits.json", audits)
        representative = output / "representative-checkpoint.pt"
        if (not representative.exists() and spec["parameter_count"] == min(
                s["parameter_count"] for s in manifest["calibration"])):
            shutil.copyfile(work / "working" / "shared_training" / "training_state.pt", representative)
            atomic_json(output / "representative-checkpoint.json", {"source_job": spec["name"],
                "sha256": file_hash(representative), "bytes": representative.stat().st_size,
                "runtime_signature": spec["expected_runtime_signature"]})
        if manifest.get("prune_completed_weights", True): prune_owned_weights(output)
    calibration = fit_calibration(rows, hardware_key=client.store.hardware_key(),
                                  predictor_artifact_sha256=predictor.artifact_hash)
    atomic_json(output / "calibration.json", calibration)
    return {"complete": True, "kind": "dedicated_calibration", "calibration": calibration,
            "elapsed_seconds": time.monotonic() - start, "hardware": hardware}


def run_episode(manifest, manifest_path, raw_settings, output, calibration, *, arm, mode, max_seconds):
    setup_started = time.monotonic()
    hardware = gpu_sample()
    if hardware["used_mib"] > 512: raise RuntimeError("episode requires an otherwise idle A10")
    settings = make_settings(raw_settings, output, hardware)
    client = SchedulerClient(settings)
    resource_control = manifest.get("profile_source") == "dedicated_solo_template_proxy"
    if (arm in RESOURCE_ARMS) != resource_control:
        raise ValueError("resource controls and their explicit profile source must be selected together")
    causal = arm in CAUSAL_ARMS + RESOURCE_ARMS
    fixed_three = arm in {"fixed_three", "causal_fixed_three", "resource_fixed_three"}
    revised = arm in {"calibrated_revised", "causal_revised_sync", "causal_revised_async", "resource_revised", "resource_incremental"}
    asynchronous = arm in {"causal_v4_async", "causal_revised_async", "causal_fixed_three", *RESOURCE_ARMS}
    if causal and asynchronous and not fixed_three:
        from .async_preparation import AsyncRevisedService, AsyncV4Service
        if arm == "resource_incremental":
            from ..a10_resource_mechanisms.policy import IncrementalService
            service = IncrementalService(settings, store=client.store)
        else:
            service = (AsyncRevisedService if revised else AsyncV4Service)(settings, store=client.store)
    else:
        service = (RevisedPilotService(settings, store=client.store) if revised else client.create_service())
    supervisor_exit_observations, harness_exit_observations = [], {}
    if causal:
        original_record_exit = service.supervisor._record_process_exit
        def record_exit(job_id, returncode):
            observed = time.monotonic()
            original_record_exit(job_id, returncode)
            supervisor_exit_observations.append({"job_id": job_id, "returncode": returncode,
                "observed_monotonic": observed,
                "source": "supervisor_record_process_exit_entry_after_process_poll"})
        service.supervisor._record_process_exit = record_exit
    source_tag, phase_factory = "ml_predictor", phase_model
    if resource_control:
        from ..a10_resource_mechanisms.profiles import MeasuredTemplatePredictor, SOURCE, phase_model as template_phases
        source_tag, phase_factory = SOURCE, template_phases
        service.planner.estimator.ml_predictor = MeasuredTemplatePredictor(calibration,
            client.store.hardware_profile(), gpu_uuid=hardware["gpu_uuid"])
    else:
        predictor = service.planner.estimator.ml_predictor
        if not predictor or not predictor.available: raise ValueError("published A10 predictor unavailable")
        if predictor.artifact_hash != STUDENT_SHA256: raise ValueError("published S_A10 student hash mismatch")
        if causal or arm in {"calibrated_v4", "calibrated_revised"}:
            service.planner.estimator.ml_predictor = CalibratedPredictor(predictor, calibration, client.store.hardware_key())
        elif predictor.artifact_hash != calibration["predictor_artifact_sha256"]:
            raise ValueError("reference artifact differs from dedicated calibration")
    jobs, works, specs = {}, {}, {}
    for spec in manifest["jobs"]:
        job, work = build_job(spec, manifest_path=manifest_path, output=output, manifest=manifest,
                              memory_budget_mib=hardware["total_mib"] * .9)
        job.metadata["feedback_phase_model"] = phase_factory(calibration, job,
            hardware_key=client.store.hardware_key(), batch_size=spec["batch"])
        if resource_control:
            entry = calibration["entries"][job.metadata["resource_template_key"]]
            c = entry["phase_seconds"]
            reserve = c["startup"] + c["pause"] + c["resume"] + 20 / entry["updates_per_second"]
        else:
            reserve = trial_reserve(calibration)
        job.metadata["feedback_trial_reserve_seconds"] = reserve
        jobs[spec["name"]], works[spec["name"]], specs[spec["name"]] = job, work, spec
    names_by_id = {job.job_id: name for name, job in jobs.items()}
    recording = RecordingPredictor(service.planner.estimator.ml_predictor,
        lambda job_id, profiles: atomic_json(works[names_by_id[job_id]] / "predicted-profiles.json", profiles))
    service.planner.estimator.ml_predictor = recording
    preparation = None
    if causal:
        from .async_preparation import PreparedPredictor
        preparation = PreparedPredictor(recording, asynchronous=asynchronous,
            prepare_on_read=not asynchronous, hardware="A10", profile_source=source_tag)
        service.planner.estimator.ml_predictor = preparation
        if asynchronous and not fixed_three:
            service.predictor_preparation = preparation
    if len(jobs) != 6: raise ValueError("frozen pilot requires exactly six evaluation jobs")
    parents = {s["chain"]: s["name"] for s in manifest["jobs"] if s["role"] == "parent"}
    children = {s["chain"]: s["name"] for s in manifest["jobs"] if s["role"] == "child"}
    if set(parents) != {0, 1, 2} or set(children) != {0, 1, 2}: raise ValueError("three fixed chains required")
    start, start_unix = time.monotonic(), time.time()
    submitted, receipts, request_specs, handles, exit_codes = {}, {}, {}, {}, {}
    arrivals, timeline, audits, first_progress, last_progress = [], [], {}, {}, {}
    executor = SubprocessExecutor(settings)
    prediction_pool = ThreadPoolExecutor(max_workers=1) if arm == "fixed_three" else None
    prediction_futures, fixed_profiles = {}, {}
    if not fixed_three: service.start(background=True)
    next_sample, group_child_release = 0., None
    try:
        while len(receipts) != len(jobs):
            elapsed = time.monotonic() - start
            check_deadline(manifest, elapsed, max_seconds)
            require_predictor_exposure(recording, jobs, specs, required_source=source_tag)
            if preparation is not None and preparation.failures:
                raise RuntimeError("prediction preparation failed: " + str(preparation.failures))
            if mode == "group" and group_child_release is None and all(jobs[n].job_id in receipts for n in parents.values()):
                group_child_release = max(receipts[jobs[n].job_id]["feedback_ready_monotonic"] for n in parents.values()) + 2.
            for name, spec in specs.items():
                if name in submitted: continue
                if spec["role"] == "parent": due = parent_due(start, spec["chain"], mode)
                elif mode == "group": due = group_child_release
                else:
                    parent = jobs[parents[spec["chain"]]].job_id
                    due = receipts[parent]["feedback_ready_monotonic"] + 2. if parent in receipts else None
                if due is None or time.monotonic() < due: continue
                accepted_mono, accepted_unix = time.monotonic(), time.time()
                job = jobs[name]
                required = ([job.job_id] if mode == "individual" else
                    sorted(jobs[n].job_id for n in (parents if spec["role"] == "parent" else children).values()))
                request_id = name if mode == "individual" else spec["role"] + "-bracket"
                if request_id not in request_specs:
                    request_specs[request_id] = {"mode": mode, "required_ids": required,
                        "submitted_monotonic": accepted_mono, "submitted_unix": accepted_unix}
                request = request_specs[request_id]
                job.metadata["feedback_request"] = {"request_id": request_id,
                    "workflow_id": "a10-feedback-pilot", "completion_policy": mode,
                    "required_job_ids": required, "membership_finalized": True,
                    "submitted_at": datetime.fromtimestamp(request["submitted_unix"], timezone.utc).isoformat()}
                job = accept_job(client, job, accepted_unix)
                submitted[name] = submission(spec, job, works[name], accepted_mono, accepted_unix)
                if preparation is not None and asynchronous:
                    preparation.enqueue(job, spec["batch"])
                if prediction_pool is not None:
                    prediction_futures[name] = prediction_pool.submit(service.planner.estimator.predicted_profile,
                                                                      job, spec["batch"])
                arrivals.append({"name": name, "due_seconds": due - start,
                                 "accepted_seconds": accepted_mono - start})
            current = {n: client.store.get_job(jobs[n].job_id) for n in submitted}
            if fixed_three:
                for name, handle in list(handles.items()):
                    code = handle.process.poll()
                    if code is not None:
                        exit_codes[name] = code
                        del handles[name]
                pending = [n for n in submitted if n not in handles and n not in exit_codes]
                for name in pending[:max(0, 3 - len(handles))]:
                    if preparation is not None:
                        if not preparation.ready(jobs[name], specs[name]["batch"]): break
                        fixed_profiles[name] = service.planner.estimator.predicted_profile(jobs[name], specs[name]["batch"])
                    else:
                        future = prediction_futures[name]
                        if not future.done(): break  # Strict FIFO head remains pending during charged conversion.
                        fixed_profiles[name] = future.result()
                    sample = gpu_sample()
                    running = {n: client.store.get_job(jobs[n].job_id) for n in handles}
                    if not fixed_fifo_fits(fixed_profiles[name], running, fixed_profiles,
                            device_used_mib=sample["used_mib"], budget_mib=hardware["total_mib"] * .9):
                        if not handles: raise ValueError("fixed FIFO head does not fit A10 budget even alone")
                        break
                    latest = current[name]
                    client.store.set_job_status(latest.job_id, JobStatus.READY, reason="fixed three FIFO", hold=False)
                    handles[name] = executor.start(client.store.get_job(latest.job_id))
            for name, latest in current.items():
                step = latest.metadata.get("runtime_global_step", 0)
                if step > 0:
                    first_progress.setdefault(name, {"observed_seconds": elapsed, "step": step})
                    if name not in last_progress or step > last_progress[name]["step"]:
                        last_progress[name] = {"observed_seconds": elapsed, "step": step}
                if latest.job_id in receipts: continue
                code = exit_codes.get(name) if fixed_three else service.supervisor.observed_exit_code(latest.job_id)
                if code is not None and latest.status == JobStatus.COMPLETED:
                    harness_exit_observations.setdefault(latest.job_id, {
                        "observed_monotonic": time.monotonic(), "returncode": code,
                        "source": "harness_observation_before_receipt_parse"})
                receipt = receive(submitted[name], latest, code)
                if receipt is not None:
                    receipts[latest.job_id] = receipt
                    atomic_json(works[name] / "receipt.json", receipt)
            if elapsed >= next_sample:
                sample = gpu_sample()
                if sample["used_mib"] > sample["total_mib"] * .94:
                    raise RuntimeError("GPU live memory exceeds predeclared emergency cutoff")
                timeline.append({"elapsed_seconds": elapsed, "gpu": sample,
                    "jobs": {n: {"status": j.status.value, "global_step": j.metadata.get("runtime_global_step", 0),
                        "allocated_mib": j.metadata.get("trial_allocated_mb"),
                        "peak_allocated_mib": j.metadata.get("trial_peak_allocated_mb"),
                        "reserved_mib": j.metadata.get("trial_reserved_mb"),
                        "peak_reserved_mib": j.metadata.get("trial_peak_reserved_mb")} for n, j in current.items()}})
                with (output / "timeline.jsonl").open("a") as stream: stream.write(json.dumps(timeline[-1]) + "\n")
                atomic_json(output / "status.json", {"elapsed_seconds": elapsed, "submitted": len(submitted), "receipts": len(receipts)})
                next_sample = elapsed + 1.
            time.sleep(.02)
        require_predictor_exposure(recording, jobs, specs, require_all=True, required_source=source_tag)
    finally:
        if not fixed_three: service.stop()
        for handle in handles.values(): terminate_process_tree(handle.process)
        if prediction_pool is not None: prediction_pool.shutdown(wait=False, cancel_futures=True)
        if causal:
            atomic_json(output / "exit-observations.json", {
                "episode_started_monotonic": start,
                "supervisor_events_including_pause_exits": supervisor_exit_observations,
                "completed_job_harness_observations": harness_exit_observations,
                "note": "observations bound process exit from above; worker finished_at precedes OS exit"})
        if preparation is not None:
            stopped = preparation.stop(timeout=3.)
            atomic_json(output / "preparation-timeline.json", {
                "episode_started_monotonic": start, "asynchronous": asynchronous,
                "worker_stopped": stopped, "events": preparation.events,
                "failures": preparation.failures})
        atomic_json(output / "predictor-exposure.json", {"failures": recording.failures,
            "prediction_sources": dict(recording.last_sources), "prediction_errors": dict(recording.last_errors),
            "prepared_numeric_profiles": recording.records})
    end = time.monotonic()
    result_ready = max(row["feedback_ready_monotonic"] for row in receipts.values())
    for name, spec in specs.items():
        latest = client.store.get_job(jobs[name].job_id)
        audits[name] = audit_checkpoint(spec, works[name], latest)
        atomic_json(works[name] / "job-final.json", latest.to_dict())
        atomic_json(works[name] / "events.json", events_for(settings.events_jsonl_path, latest.job_id))
    atomic_json(output / "validity-audits.json", audits)
    latencies = {name: request_latency(row["mode"], row["required_ids"], receipts,
                                     row["submitted_monotonic"]) for name, row in request_specs.items()}
    chains = {str(chain): receipts[jobs[children[chain]].job_id]["feedback_ready_monotonic"]
              - submitted[parents[chain]].accepted_monotonic for chain in range(3)}
    result = {"complete": True, "arm": arm, "mode": mode, "requestor": "colocated_python_controller",
        "common_calibrated_profile_access": causal,
        "profile_source": source_tag,
        "evaluation_scope": "scheduler mechanism control; no neural accuracy claim" if resource_control else "released student pipeline",
        "harness_exit_observations": harness_exit_observations,
        "preparation_treatment": ("acceptance_enqueued_serial_async" if asynchronous else
            "first_demand_sync") if causal else "legacy_pilot",
        "episode_started_monotonic": start,
        "transport": "local filesystem JSON read+parse+sha256 after actual successful worker exit",
        "episode_started_unix": start_unix, "episode_duration_seconds": result_ready - start,
        "pre_request_setup_seconds": start - setup_started,
        "scheduler_shutdown_seconds": end - result_ready,
        "request_latencies_seconds": latencies, "mean_request_latency_seconds": statistics.mean(latencies.values()),
        "chain_root_to_child_seconds": chains, "arrivals": arrivals,
        "first_progress_observations": first_progress, "last_progress_observations": last_progress,
        "calibration_sha256": calibration["artifact_sha256"] if resource_control else calibration["calibration_sha256"], "receipts": receipts,
        "source_and_budget_audits": audits,
        "phase_model_role": "dedicated_same_configuration_template_proxy" if resource_control else "calibrated_cross_workload_phase_proxy_v1",
        "group_arrival_note": "group parents submitted together; individual parents at0/2/4s",
        "fixed_reference_memory_profiles": fixed_profiles,
        "prepared_numeric_profiles": recording.records,
        "hardware": hardware, "peak_device_used_mib": max(x["gpu"]["used_mib"] for x in timeline)}
    result["prediction_sources"] = dict(service.planner.estimator.ml_predictor.last_sources)
    result["prediction_errors"] = dict(service.planner.estimator.ml_predictor.last_errors)
    result["policy_activation"] = policy_activation(client.store.list_events())
    result["admission_policy_comparison_eligible"] = (fixed_three
        or result["policy_activation"]["actionable_forecasts"] > 0
        or (revised and result["policy_activation"]["amortization_skips"] > 0))
    result["policy_activation_note"] = "zero actionable v4 forecasts are not evidence of a measured admission-policy comparison; legitimate revised skips are separate observable decisions"
    result["predictor_comparison_eligible"] = all(
        result["prediction_sources"].get(job.job_id) == source_tag
        and specs[name]["batch"] in recording.records.get(job.job_id, {})
        for name, job in jobs.items())
    if not result["predictor_comparison_eligible"]:
        raise RuntimeError("incomplete predictor exposure; episode is diagnostic, not a comparison")
    if manifest.get("prune_completed_weights", True): prune_owned_weights(output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["calibrate", "run", "check"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--arm", choices=ARMS + CAUSAL_ARMS + RESOURCE_ARMS, default="reference_v4")
    parser.add_argument("--mode", choices=["individual", "group"], default="individual")
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--deadline-utc", default="2026-10-30T03:10:00Z")
    args = parser.parse_args(argv)
    if args.command != "check":
        import fcntl
        gpu_lock = open("/tmp/localml-a10-feedback-pilot.lock", "a")
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def interrupted(signum, frame):
            raise InterruptedError("pilot controller terminated; cleaning up owned workers")
        signal.signal(signal.SIGTERM, interrupted)
    manifest_path, output = args.manifest.resolve(), args.output.resolve()
    manifest = with_hard_deadline(json.loads(manifest_path.read_text()), args.deadline_utc)
    raw_settings = json.loads(args.settings.read_text())
    check_deadline(manifest)
    pins = source_snapshot(manifest, manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "source-pins-before.json", pins)
    atomic_json(output / "invocation.json", {**vars(args), "manifest": str(manifest_path),
        "settings": str(args.settings.resolve()), "output": str(output), "calibration": str(args.calibration) if args.calibration else None})
    error = None
    started = time.monotonic()
    try:
        if args.command == "check": result = {"complete": True, "kind": "CPU_source_check_only"}
        elif args.command == "calibrate":
            result = calibrate(manifest, manifest_path, raw_settings, output, args.max_seconds)
        else:
            if args.calibration is None: raise ValueError("--calibration is required for all evaluation arms")
            calibration = json.loads(args.calibration.read_text())
            result = run_episode(manifest, manifest_path, raw_settings, output, calibration,
                                 arm=args.arm, mode=args.mode, max_seconds=args.max_seconds)
        after = source_snapshot(manifest, manifest_path)
        atomic_json(output / "source-pins-after.json", after)
        if after != pins: raise ValueError("source/manifest changed during run")
    except BaseException:
        error = traceback.format_exc()
        result = {"complete": False, "error": error, "elapsed_seconds": time.monotonic() - started,
                  "arm": args.arm, "mode": args.mode}
        try:
            saved = output / "runtime" / "scheduler_settings.json"
            if saved.exists():
                failed_settings = SchedulerSettings.from_dict(json.loads(saved.read_text()))
                failed_client = SchedulerClient(failed_settings)
                atomic_json(output / "failure-jobs.json", [j.to_dict() for j in failed_client.store.list_jobs()])
                atomic_json(output / "failure-events.json", failed_client.store.list_events())
            atomic_json(output / "source-pins-after.json", source_snapshot(manifest, manifest_path))
        except Exception:
            atomic_json(output / "failure-snapshot-error.json", {"error": traceback.format_exc()})
    atomic_json(output / "result.json", result)
    from .plot import episode_plot
    try: episode_plot(output)
    except Exception: atomic_json(output / "plot-error.json", {"error": traceback.format_exc()})
    print(json.dumps({k: v for k, v in result.items() if k not in {"receipts", "source_and_budget_audits", "calibration"}}, indent=2), flush=True)
    if error: raise SystemExit(1)


if __name__ == "__main__": main()
