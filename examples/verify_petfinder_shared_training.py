"""Bounded local integration experiment; CPU A10G prediction, local CUDA control.

Not a hardware-accuracy or scheduling-speedup benchmark. The explicit test-only
target injection leaves normal production hardware matching untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from engine.script_introspection import supports_cooperative_trial
from localml_scheduler.adapters.mlevolve import build_mlevolve_job
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import PredictionSettings, SchedulerSettings
from localml_scheduler.domain import CheckpointPolicy, JobStatus, ResourceRequirements
from localml_scheduler.hardware import HardwareProfile
from localml_scheduler.prediction.ml_predictor import MLVramPredictor


def gpu():
    row = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
    name, total, used = row.split(", ")
    return {"name": name, "total_mib": float(total), "used_mib": float(used)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--steps", type=int, default=192)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ["MLEVOLVE_INPUT_DIR"] = str(args.dataset.resolve())
    initial_gpu = gpu()
    if initial_gpu["used_mib"] + 2400 > 0.90 * initial_gpu["total_mib"]:
        raise RuntimeError("not enough live headroom for this bounded two-model smoke test")
    settings = SchedulerSettings(runtime_root=root / "runtime", scheduler_poll_interval_seconds=0.05,
        graph_db={"enabled": False}, hardware_knowledge_graph={"enabled": False},
        hardware_feature_db={"enabled": False},
        gpu_scheduler={"packing_backend": "cuda_process", "batch_probe_enabled": False,
                       "memory": {"predicted_budget_fraction": 0.9}})
    client = SchedulerClient(settings)
    service = client.create_service()
    prediction = PredictionSettings(mode="ml_predictor", student_release_path=args.release_root,
        student_model_path=args.model_dir, conversion_timeout_seconds=90, conversion_max_rss_mb=4096)
    target = HardwareProfile("test-declared-a10g", "linux", "NVIDIA A10G", 23028, "8.6",
                             "test target, not local GPU", torch.__version__)
    predictor = MLVramPredictor(prediction, target)
    assert predictor.available and predictor.model_id == "release/S_A10G"
    # Explicit, local test injection; actual telemetry and allocator budget use
    # the real GPU. This must never become a production hardware override.
    service.planner.estimator.ml_predictor = predictor
    source_dir = Path(__file__).resolve().parents[1] / "verification/mlevolve_petfinder_20261027/adapted"
    jobs, profiles, audits = [], {}, {}
    for n in ("00", "01"):
        name = f"candidate_{n}"
        work = root / name
        work.mkdir()
        source = work / f"{name}.py"
        text = (source_dir / source.name).read_text().replace("EPOCHS = 3", "EPOCHS = 2")
        text = text.replace("MAX_STEPS_PER_EPOCH = 8", f"MAX_STEPS_PER_EPOCH = {args.steps}")
        text = text.replace(f"variant='{n}')", f"variant='{n}', validation_limit=32)")
        source.write_text(text)
        doc = json.loads((source_dir / f"{source.name}.perfseer.json").read_text())
        doc["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        meta = doc["perfseer_model"]
        meta.update(source_path=str(source), epochs=2, validation_rows=32, steps_per_epoch_by_batch={"4": args.steps})
        source.with_suffix(".py.perfseer.json").write_text(json.dumps(doc, indent=2) + "\n")
        assert supports_cooperative_trial(text)
        job = build_mlevolve_job(workflow_id="petfinder-shared-training-smoke", baseline_model_id=name,
            baseline_model_path=str(source), runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
            runner_kwargs={"script_path": str(source), "working_dir": str(work), "result_path": str(work / "execution.json"),
                           "batch_size": 4, "timeout": 240},
            loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
            resource_requirements=ResourceRequirements(requires_gpu=True), max_epochs=2,
            checkpoint_policy=CheckpointPolicy(save_every_n_steps=32, save_every_epoch=True, keep_last_n=4),
            packing_eligible=True, packing_family=name, packing_backend_allowlist=["cuda_process"],
            metadata={"cooperative_trial": True, "skip_active_scheduler_probes": True,
                      "integration_test_target": "A10G", "integration_test_actual_gpu": initial_gpu["name"]})
        profile = predictor.predict_profile_options(job, [4])[4]
        assert profile["batch_size"] == 4
        profiles[job.job_id] = profile
        audits[job.job_id] = predictor._released.last_capture_audits[4]
        jobs.append(job)
    assert not torch.cuda.is_initialized(), "prediction must stay on CPU"
    ordered = sorted(jobs, key=lambda job: service.planner.estimator.workload_rank(job, prefer_high=True))
    anchor, newcomer = ordered
    (root / "predictions.json").write_text(json.dumps({"student": predictor.model_id,
        "sha256": predictor.artifact_hash, "profiles": profiles, "capture_audits": audits,
        "idle_order": [job.baseline_model_id for job in ordered]}, indent=2) + "\n")
    samples, pause_state = [], None
    pause_sent, resume_sent, joined = False, False, False
    began = time.monotonic()
    client.submit(anchor)
    service.start(background=True)
    failure = None
    try:
        deadline, next_gpu, current_gpu = began + 240, 0, initial_gpu
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_gpu:
                current_gpu, next_gpu = gpu(), now + 1
                if current_gpu["used_mib"] > 0.90 * current_gpu["total_mib"]:
                    raise RuntimeError("live device memory exceeded the 90% test budget")
            current = {job.job_id: job for job in client.store.list_jobs()}
            first = current.get(anchor.job_id)
            if first and first.status == JobStatus.RUNNING and int(first.metadata.get("runtime_global_step", 0)) >= 24 and not pause_sent:
                client.pause(first.job_id)
                pause_sent = True
            if first and first.status == JobStatus.PAUSED and pause_sent and not resume_sent:
                pause_state = torch.load(first.latest_checkpoint_path, map_location="cpu", weights_only=False)["state"]
                assert 24 <= pause_state["global_step"] < args.steps * 2
                (root / "pause-checkpoint.json").write_text(json.dumps({"global_step": pause_state["global_step"],
                    "step_in_epoch": pause_state["step_in_epoch"], "signature": pause_state["signature"],
                    "checkpoint": first.latest_checkpoint_path}, indent=2) + "\n")
                client.resume(first.job_id)
                resume_sent = True
            if first and resume_sent and first.status == JobStatus.RUNNING and int(first.metadata.get("runtime_global_step", 0)) > pause_state["global_step"] and not joined:
                client.submit(newcomer)
                joined = True
            samples.append({"seconds": now - began, "gpu": current_gpu,
                "jobs": {jid: {"status": job.status.value,
                    "steps": job.metadata.get("runtime_global_step", 0),
                    "metrics": job.metadata.get("trial_last_metrics"),
                    "reserved_mib": job.metadata.get("trial_reserved_mb"),
                    "phase": (job.metadata.get("colocation_trial") or {}).get("phase")}
                         for jid, job in current.items()}})
            for job in current.values():
                if job.status in {JobStatus.FAILED, JobStatus.CANCELLED}:
                    raise RuntimeError(f"{job.baseline_model_id}: {job.status.value}; inspect execution.json and job_failed events")
            if joined and len(current) == 2 and all(job.status == JobStatus.COMPLETED for job in current.values()):
                break
            time.sleep(0.05)
        else:
            raise TimeoutError("bounded PetFinder integration exceeded 240 seconds")
    except BaseException as exc:
        failure = repr(exc)
        for job in client.store.list_jobs():
            if not job.status.is_terminal:
                client.cancel(job.job_id)
        cleanup = time.monotonic() + 20
        while time.monotonic() < cleanup and any(not job.status.is_terminal for job in client.store.list_jobs()):
            time.sleep(0.1)
    finally:
        service.stop()
        (root / "timeline.json").write_text(json.dumps(samples, indent=2) + "\n")
        events = client.store.list_events()
        (root / "events.json").write_text(json.dumps(events, indent=2) + "\n")
    if failure:
        (root / "failure.json").write_text(json.dumps({"error": failure}, indent=2) + "\n")
        raise RuntimeError(failure)
    rows = []
    for job in client.store.list_jobs():
        work = root / job.baseline_model_id
        state = torch.load(work / "working/shared_training/training_state.pt", map_location="cpu", weights_only=False)
        assert state["global_step"] == 2 * args.steps
        assert all(int(value["step"]) == 2 * args.steps for value in state["optimizer_state"]["state"].values())
        assert len(state["history"]) == 2
        assert (work / "submission/submission.csv").is_file()
        rows.append({"job_id": job.job_id, "candidate": job.baseline_model_id, "status": job.status.value,
            "steps": state["global_step"], "history": state["history"],
            "allocator_limit_bytes": job.metadata.get("trial_allocator_limit_bytes"),
            "peak_reserved_mib": job.metadata.get("trial_peak_reserved_mb")})
    phases = [event["payload"]["phase"] for event in events if event["event_type"] == "cold_start_phase"]
    decisions = [event["payload"] for event in events if event["event_type"] == "cold_start_trial_decided"]
    concurrent = [row for row in samples if len(row["jobs"]) == 2 and all(j["status"] == "RUNNING" for j in row["jobs"].values())]
    assert pause_state is not None and concurrent and "measure_overlap" in phases
    assert "reference" not in phases and "reference_barrier" not in phases
    assert all(any(row["jobs"][job.job_id]["steps"] > concurrent[0]["jobs"][job.job_id]["steps"] for row in concurrent) for job in jobs)
    result = {"passed": True, "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "control/integration smoke only; A10G CPU prediction on a local RTX 5090; external training remained active",
        "actual_gpu": initial_gpu, "prediction_target": "NVIDIA A10G", "student": predictor.model_id,
        "student_sha256": predictor.artifact_hash, "memory_budget_fraction": 0.9,
        "peak_device_memory_mib": max(row["gpu"]["used_mib"] for row in samples),
        "initial_measured_profiles": 0, "pause_step": pause_state["global_step"],
        "phases": phases, "decisions": decisions, "jobs": rows,
        "concurrent_running_span_seconds": concurrent[-1]["seconds"] - concurrent[0]["seconds"],
        "production_hardware_guard_modified": False, "model_accuracy_measured": False}
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
