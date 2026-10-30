"""Actual CPU subprocess episode; all A10 measurements below are fixtures."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib
import json
import os
from pathlib import Path

import pytest
import torch

from localml_scheduler.adapters import shared_training
from localml_scheduler.benchmarks.a10_feedback_latency import run
from localml_scheduler.benchmarks.a10_resource_mechanisms import workloads
from localml_scheduler.benchmarks.a10_resource_mechanisms.manifest import (
    RESOURCE_CLASSES, make_spec, resource_config, resource_plan,
)
from localml_scheduler.benchmarks.a10_resource_mechanisms.profiles import SOURCE, fit_templates
from localml_scheduler.hardware import HardwareProfile


def spec(config, *, name="cpu", role="parent", chain=0, seed=41, steps=3,
         train_seed=101, validation_seed=211):
    return make_spec(config, name=name, role=role, chain=chain, epochs=2,
        steps_per_epoch=steps, seed=seed, train_pool_seed=train_seed,
        validation_pool_seed=validation_seed, source_path=workloads.HERE / "candidate.py")


def test_resource_class_plans_and_replica_sealing_keep_only_intended_template_reuse():
    rows = resource_plan(["tiny", "resident", "compute", "both"])
    assert [(r["width"], r["repetitions"], r["batch"], r["ballast_mib"]) for r in rows] == [
        (128, 2, 32, 0), (128, 2, 32, 8192), (2048, 8, 256, 0), (2048, 8, 256, 8192)]
    rows[0]["width"] = 9
    assert RESOURCE_CLASSES["tiny"]["width"] == resource_config("tiny")["width"] == 128
    config = resource_config("tiny")
    evaluation = spec(config)
    calibration = spec(config, name="calibration", role="calibration", chain=None,
                       seed=53, steps=11, train_seed=307, validation_seed=401)
    assert evaluation["template_key"] == calibration["template_key"]
    assert evaluation["expected_runtime_signature"] != calibration["expected_runtime_signature"]
    assert evaluation["spec_sha256"] == workloads.spec_digest(evaluation)
    assert evaluation["parameter_count"] == evaluation["trainable_parameter_count"] == workloads.parameter_count(128)
    assert len({spec(resource_config(name))["template_key"] for name in RESOURCE_CLASSES}) == 4
    assert config == resource_config("tiny")
    with pytest.raises(ValueError, match="outside workload"):
        spec(dict(config, target_training_seconds=12))


def calibration_fixture(template_key, hardware):
    """Invented CPU-test values; never calibration evidence for an experiment."""
    rows = []
    for replica in range(2):
        rows.append({"name": f"fixture-calibration-{replica}", "role": "calibration", "complete": True,
            "hardware_key": hardware.hardware_key, "gpu_uuid": "CPU-TEST-NO-GPU",
            "job_signature": str(replica + 1) * 64, "template_key": template_key,
            "solo_process_isolation_verified": True, "successful_updates": 6,
            "windows": [{"start_step": 2, "end_step": 4, "steps": 2, "elapsed_seconds": .02,
                         "excludes_nonupdate_phases": True},
                        {"start_step": 4, "end_step": 6, "steps": 2, "elapsed_seconds": .02,
                         "excludes_nonupdate_phases": True}],
            "peak_allocated_mib": 1., "peak_reserved_mib": 2., "peak_device_used_mib": 3.,
            "steady_allocated_mib": [1., 1.], "steady_gpu_busy_percent": [0., 0.],
            "phase_seconds": dict.fromkeys(("startup", "resume", "pause", "validation", "checkpoint", "finalization", "delivery"), 0.),
            "observed_at": datetime.now(timezone.utc).isoformat()})
    return fit_templates(rows, hardware_key=hardware.hardware_key, gpu_uuid="CPU-TEST-NO-GPU",
                         source_identity={"fixture_only": True, "source": workloads.source_metadata()})


@pytest.mark.parametrize('arm', run.RESOURCE_ARMS)
def test_real_cpu_episode_has_two_causal_brackets_and_exact_audited_receipts(tmp_path, monkeypatch, arm):
    hardware = HardwareProfile(hardware_key="CPU-test-A10-shape-only", os_name="cpu-test",
        gpu_name="NVIDIA A10", total_vram_mb=24000, compute_capability="8.6",
        cuda_runtime=None, torch_version=torch.__version__)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    introspection = importlib.import_module("engine.script_introspection")
    roots = [str(Path(shared_training.__file__).parents[2]), str(Path(introspection.__file__).parents[1])]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(roots))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("localml_scheduler.storage.sqlite_store.detect_hardware_profile", lambda **_: hardware)
    monkeypatch.setattr(run, "gpu_sample", lambda: {"name": "NVIDIA A10", "gpu_uuid": "CPU-TEST-NO-GPU",
        "total_mib": 24000., "used_mib": 0., "utilization_percent": 0.,
        "memory_controller_busy_percent": 0., "power_watts": 0., "fixture_only": True})
    original_build = run.build_job
    def cpu_job(*args, **kwargs):
        job, work = original_build(*args, **kwargs)
        job.resource_requirements.requires_gpu = False
        job.metadata.pop("trial_memory_limit_mb", None)
        return job, work
    monkeypatch.setattr(run, "build_job", cpu_job)
    config = resource_config("tiny", width=8, input_features=8, batch=2,
                             pool_size=16, validation_size=128)
    jobs = [spec(config, name=f"{role}-{chain}", role=role, chain=chain,
                 seed=41 + chain + 10 * (role == "child"),
                 train_seed=101 + 2 * chain, validation_seed=211 + 2 * chain)
            for role in ("parent", "child") for chain in range(3)]
    assert len({job["template_key"] for job in jobs}) == 1
    manifest = {"jobs": jobs, "calibration": [], "profile_source": SOURCE,
        "deadline_utc": (datetime.now(timezone.utc) + timedelta(minutes=3)).isoformat(),
        "job_timeout_seconds": 45, "prune_completed_weights": False,
        "fixture_only": "CPU workflow integration; no measured A10 evidence"}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    snapshot = run.source_snapshot(manifest, manifest_path)
    assert snapshot["manifest_sha256"] == run.file_hash(manifest_path)
    table = calibration_fixture(jobs[0]["template_key"], hardware)
    output = tmp_path / "episode"
    output.mkdir()
    result = run.run_episode(manifest, manifest_path, {}, output, table,
                             arm=arm, mode="group", max_seconds=90)
    assert result["complete"] and result["predictor_comparison_eligible"]
    assert result["profile_source"] == SOURCE
    assert len(result["receipts"]) == len(result["source_and_budget_audits"]) == 6
    assert set(result["request_latencies_seconds"]) == {"parent-bracket", "child-bracket"}
    assert len(result["chain_root_to_child_seconds"]) == 3
    assert set(result["prediction_sources"].values()) == {SOURCE}
    assert not result["prediction_errors"]
    for row in result["receipts"].values():
        assert row["worker_exit_code"] == 0 and row["expected_updates"] == 6
        assert row["returned_result"]["device"] == "cpu"
        assert row["returned_result"]["global_step"] == 6
        assert 0 <= row["selected_auc"] <= 1
        assert row["feedback_ready_monotonic"] >= row["accepted_monotonic"]
        assert row["result_sha256"] and row["execution_sha256"]
    for row in result["source_and_budget_audits"].values():
        assert row["global_step"] == row["all_adamw_steps"] == 6
        assert row["completed_epochs"] == 2 and row["training_state_bytes"] > 0
    parent_ready = max(row["feedback_ready_monotonic"] for row in result["receipts"].values()
                       if row["runtime_completion_identity"] in {s["expected_runtime_signature"] for s in jobs if s["role"] == "parent"})
    children = [row for row in result["arrivals"] if row["name"].startswith("child-")]
    assert len(children) == 3
    assert min(row["accepted_seconds"] + result["episode_started_monotonic"] for row in children) >= parent_ready + 2.
    for job in jobs:
        work = output / "jobs" / job["name"] / "working/shared_training"
        assert json.loads((work / "workload_resources.json").read_text())["allocated_ballast_bytes"] == 0
    assert not torch.cuda.is_initialized()
