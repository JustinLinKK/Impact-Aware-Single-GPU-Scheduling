"""CPU-only evidence audits. Fixture numbers are not A10 measurements."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import signal
from types import SimpleNamespace

import pytest
import torch

from localml_scheduler.benchmarks.a10_resource_mechanisms import calibrate as module
from localml_scheduler.benchmarks.a10_resource_mechanisms import workloads
from localml_scheduler.benchmarks.a10_resource_mechanisms.manifest import make_spec, resource_config
from localml_scheduler.benchmarks.a10_resource_mechanisms.profiles import fit_templates


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def evidence():
    spec = make_spec(resource_config("tiny", width=8, input_features=8, batch=2),
        name="fixture", role="calibration", chain=None, epochs=2, steps_per_epoch=32,
        seed=1001, train_pool_seed=10001, validation_pool_seed=10002,
        source_path=workloads.HERE / "candidate.py")
    signature = spec["expected_runtime_signature"]
    launches = [{"launched_unix": 100., "exited_unix": 105., "resume_step": 0,
                 "pid": 10, "pid_start_ticks": 1000},
                {"launched_unix": 108., "exited_unix": 118., "resume_step": 12,
                 "pid": 20, "pid_start_ticks": 2000}]
    windows = [{"launch": index, "token": str(index), "owner": "test-fixture", "kind": "reference",
        "complete": True, "start_step": start, "end_step": start + 8, "steps": 8,
        "started_at": begin, "finished_at": iso(begin + 2), "elapsed_seconds": 2.,
        "nonupdate_phase_counter": index, "completed_nonupdate_phase_counter": index,
        "completion_identity": signature, "excludes_nonupdate_phases": True,
        "samples": [.25, .25], "halves": [.25, .25]}
        for index, start, begin in [(0, 2, 102.), (1, 14, 110.)]]
    samples = []
    for index, begin, step, allocated, peak_alloc, peak_res, device in [
            (0, 101., 1, 700., 900., 1200., 1300.),
            (0, 103., 6, 100., 900., 1200., 1300.),
            (1, 111., 18, 200., 400., 500., 600.),
            (1, 117., 64, 200., 450., 550., 650.)]:
        root = launches[index]["pid"]
        samples.append({"sample_started_unix": begin, "sample_finished_unix": begin + .1,
            "worker_observed_unix": begin - .1, "launch": index, "root_pid": root,
            "hardware_key": "fixture-hardware", "gpu": {"name": "NVIDIA A10", "gpu_uuid": "fixture-uuid",
                "used_mib": device, "total_mib": 24000., "utilization_percent": 30. + index},
            "compute_apps": [{"gpu_uuid": "fixture-uuid", "pid": root + 1001,
                              "process_name": "fixture-owned-child", "used_gpu_memory_mib": device}],
            "process_snapshot": [
                {"pid": root, "ppid": 1, "nspid": [root + 1000, root],
                 "start_ticks": launches[index]["pid_start_ticks"]},
                {"pid": root + 1, "ppid": root, "nspid": [root + 1001, root + 1], "start_ticks": 2001}],
            "worker": {"runtime_global_step": step, "runtime_completion_identity": signature,
                "trial_allocated_mb": allocated, "trial_peak_allocated_mb": peak_alloc,
                "trial_reserved_mb": peak_res, "trial_peak_reserved_mb": peak_res}})
    phases = [{"phase": name, "seconds": .1, "finished_at": iso(finish),
               "identity": signature, "hardware_key": "fixture-hardware"}
              for name, finish in [("checkpoint", 104.9), ("validation", 114.),
                                   ("checkpoint", 114.5), ("validation", 116.),
                                   ("checkpoint", 116.5), ("finalization", 117.5)]]
    events = [{"event_type": "trial_runner_ready", "created_at": iso(100.5), "payload": {"global_step": 0}},
              {"event_type": "trial_runner_ready", "created_at": iso(108.5), "payload": {"global_step": 12}}]
    events.extend({"event_type": "trial_step_window", "payload": {k:v for k,v in w.items() if k != "launch"}}
                  for w in windows)
    pause = {"requested_unix": 104.2, "finished_unix": 106., "exit_code": 0, "status": "PAUSED",
             "checkpoint": {"global_step": 12, "signature": signature, "sha256": "p" * 64},
             "job_finished_unix": 118.}
    resources = {"signature": signature, "spec_sha256": spec["spec_sha256"],
        "allocated_ballast_bytes": 0, "declared_ballast_bytes": 0,
        "trainable_parameter_bytes": spec["parameter_count"] * 4,
        "ballast_initializations_this_launch": 0, "ballast_checkpoint_persistent": False,
        "precision": "fp32_ieee", "device": "cuda"}
    return spec, {"hardware_key": "fixture-hardware", "gpu_uuid": "fixture-uuid",
        "samples": samples, "windows": windows, "launches": launches, "events": events,
        "phase_history": phases, "pause": pause, "resources": resources,
        "receipt": {"worker_exit_code": 0, "runtime_completion_identity": signature, "feedback_ready_unix": 119.},
        "final_audit": {"all_adamw_steps": 64, "global_step": 64}}


def test_extraction_replays_identically_after_wall_clock_changes(monkeypatch):
    spec, raw = evidence()
    monkeypatch.setattr(module, "utc_now", lambda: iso(200.))
    original = module.extract_row(spec, **raw)
    monkeypatch.setattr(module, "utc_now", lambda: iso(2000.))
    replayed = module.extract_row(spec, **raw)
    assert original == replayed
    assert original["observed_at"] == iso(raw["receipt"]["feedback_ready_unix"])
    source = {"fixture_only": True}
    assert fit_templates([original], hardware_key="fixture-hardware", gpu_uuid="fixture-uuid",
                         source_identity=source) == fit_templates(
        [replayed], hardware_key="fixture-hardware", gpu_uuid="fixture-uuid", source_identity=source)


def test_extraction_binds_raw_evidence_keeps_launch_maxima_and_window_only_samples():
    spec, raw = evidence()
    row = module.extract_row(spec, **raw)
    assert row["solo_process_isolation_verified"]
    assert row["peak_allocated_mib"] == 900. and row["peak_reserved_mib"] == 1200.
    assert row["peak_device_used_mib"] == 1300.
    assert row["steady_allocated_mib"] == [100., 200.]
    assert row["steady_sample_indices"] == [1, 2]
    assert row["steady_gpu_busy_percent"] == [30., 31.]
    assert row["phase_seconds"] == {"validation": .1, "checkpoint": .1, "finalization": .1,
        "startup": .5, "resume": .5, "pause": pytest.approx(1.8), "delivery": 1.}
    assert len(row["raw_evidence_sha256"]) == 64
    table = fit_templates([row], hardware_key="fixture-hardware", gpu_uuid="fixture-uuid",
                          source_identity={"fixture_only": True})
    entry = table["entries"][spec["template_key"]]
    assert entry["updates_per_second"] == 4.
    assert entry["peak_allocated_mib"] == 900.
    assert entry["mean_allocated_mib"] == 150.
    mutated = deepcopy(raw)
    mutated["samples"][0]["gpu"]["used_mib"] = 1400.
    assert module.extract_row(spec, **mutated)["raw_evidence_sha256"] != row["raw_evidence_sha256"]


@pytest.mark.parametrize("change", [
    lambda r:r["samples"][1]["compute_apps"][0].update(pid=99999),
    lambda r:r["samples"][1]["gpu"].update(gpu_uuid="another-device"),
    lambda r:r["samples"][1].update(hardware_key="another-host"),
    lambda r:r["samples"][1]["process_snapshot"][0].update(start_ticks=999),
    lambda r:r["samples"][1].update(worker_observed_unix=101.),
    lambda r:r["samples"][1].update(sample_finished_unix=105.),
    lambda r:r["windows"][0].update(start_step=1, steps=9),
    lambda r:r["windows"][1].update(start_step=9, end_step=17),
    lambda r:r["windows"][1].update(completed_nonupdate_phase_counter=99),
    lambda r:r["windows"][1].update(completion_identity="wrong-identity"),
    lambda r:r["windows"][1].update(elapsed_seconds=float("nan")),
    lambda r:r["phase_history"][0].update(finished_at=iso(103.5)),
    lambda r:r["phase_history"][0].update(hardware_key="wrong-host"),
    lambda r:r["events"].pop(),
    lambda r:r["pause"].update(exit_code=1),
    lambda r:r["pause"]["checkpoint"].update(global_step=13),
    lambda r:r["resources"].update(allocated_ballast_bytes=1024),
    lambda r:r["final_audit"].update(all_adamw_steps=63),
])
def test_unverifiable_or_contaminated_measurements_fail_closed(change):
    spec, raw = evidence()
    change(raw)
    with pytest.raises((ValueError, KeyError)):
        module.extract_row(spec, **raw)


def test_pid_namespace_mapping_requires_unique_owned_descendant_and_idle_is_empty():
    _, raw = evidence()
    sample = raw["samples"][0]
    audit = module.verify_isolation(sample, gpu_uuid="fixture-uuid", root_pid=10)
    assert audit["observed_compute_pids"] == [1011]
    assert audit["owned_pids"] == [10, 11]
    with pytest.raises(ValueError, match="unknown GPU"):
        module.verify_isolation(sample, gpu_uuid="fixture-uuid")
    sample["process_snapshot"].append({"pid": 90, "ppid": 1, "nspid": [1011, 90], "start_ticks": 1})
    with pytest.raises(ValueError, match="ambiguous"):
        module.verify_isolation(sample, gpu_uuid="fixture-uuid", root_pid=10)
    sample["compute_apps"] = []
    assert module.verify_isolation(sample, gpu_uuid="fixture-uuid")["observed_compute_pids"] == []


def test_rejected_process_sample_retains_raw_evidence_before_raising(tmp_path, monkeypatch):
    _, raw = evidence()
    sample = raw["samples"][0]
    monkeypatch.setattr(module.lifecycle, "gpu_sample", lambda: sample["gpu"])
    monkeypatch.setattr(module, "compute_apps", lambda _: sample["compute_apps"])
    monkeypatch.setattr(module, "process_snapshot", lambda: sample["process_snapshot"])
    path = tmp_path / "samples.jsonl"
    with pytest.raises(ValueError, match="unknown GPU"):
        module.collect_sample("fixture-uuid", evidence_path=path)
    retained = json.loads(path.read_text())
    assert retained["compute_apps"] == sample["compute_apps"]
    assert retained["process_snapshot"] == sample["process_snapshot"]
    assert "unknown GPU" in retained["evidence_error"]


def test_pause_checkpoint_audit_checks_partial_real_adamw_counters(tmp_path):
    spec, _ = evidence()
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.ones(1, 1)).square().sum().backward()
        optimizer.step()
    checkpoint = {"state": {"contract": module.shared_contract(), "signature": spec["expected_runtime_signature"],
                           "global_step": 3, "optimizer_state": optimizer.state_dict()}}
    path = tmp_path / "pause.pt"
    torch.save(checkpoint, path)
    assert module.audit_pause_checkpoint(path, spec)["global_step"] == 3
    checkpoint["state"]["global_step"] = 4
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="AdamW"):
        module.audit_pause_checkpoint(path, spec)


def test_calibration_plan_has_disjoint_replica_seeds_and_caller_supplied_budget():
    specs = module.calibration_specs(epochs=2, steps_per_epoch=2048, replicas=2,
                                     source_path=workloads.HERE / "candidate.py")
    assert len(specs) == 8
    assert {s["epochs"] * s["steps_per_epoch"] for s in specs} == {4096}
    seeds = [s[key] for s in specs for key in ("seed", "train_pool_seed", "validation_pool_seed")]
    assert len(set(seeds)) == len(seeds)
    assert len({s["template_key"] for s in specs}) == 4


def test_driver_invalid_manifest_retains_diagnostic_result_without_any_dispatch(tmp_path, monkeypatch):
    output = tmp_path / "output"
    monkeypatch.setattr(module.lifecycle, "source_snapshot", lambda *_: {"fixture_only": True})
    monkeypatch.setattr(module.lifecycle, "gpu_sample", lambda: pytest.fail("GPU probe was not authorized"))
    monkeypatch.setattr(module.SubprocessExecutor, "start", lambda *_: pytest.fail("process dispatch was not authorized"))
    with pytest.raises(ValueError, match="calibration specifications"):
        module.calibrate({"calibration": []}, tmp_path / "manifest.json", {}, output, 10.)
    result = json.loads((output / "result.json").read_text())
    assert result["complete"] is False and result["exception"] == "ValueError"
    assert result["calibration_rows"] == []


def test_full_source_audit_rejects_added_and_deleted_python_files(monkeypatch):
    before = {"scheduler_python_sources": {"original.py": "a"}, "manifest_sha256": "m"}
    for sources in ({"original.py": "a", "added.py": "b"}, {}, {"original.py": "changed"}):
        monkeypatch.setattr(module.lifecycle, "source_snapshot", lambda *_: dict(before, scheduler_python_sources=sources))
        with pytest.raises(ValueError, match="snapshot changed"):
            module._check_sources(before, {}, "unused")


def test_controller_guard_serializes_and_restores_handlers_after_interruption(tmp_path, monkeypatch):
    import fcntl
    path = tmp_path / "gpu.lock"
    monkeypatch.setattr(module, "GPU_LOCK_PATH", str(path))
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    with pytest.raises(InterruptedError):
        with module.controller_guard():
            with path.open("a") as contender:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    assert all(signal.getsignal(number) == handler for number, handler in previous.items())
    with path.open("a") as after:
        fcntl.flock(after, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(after, fcntl.LOCK_UN)


def test_interrupt_cleans_up_owned_handle_and_writes_failure_without_real_dispatch(tmp_path, monkeypatch):
    spec, _ = evidence()
    manifest = {"calibration": [spec], "job_timeout_seconds": 30,
        "deadline_utc": (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()}
    job = SimpleNamespace(job_id="fake-worker", metadata={})
    store = SimpleNamespace(hardware_key=lambda: "fixture-hardware", get_job=lambda _: job,
                            set_job_status=lambda *a, **k: None)
    handle = SimpleNamespace(process=SimpleNamespace(pid=12345))
    cleaned, launches = [], []
    gpu = {"gpu_uuid": "fixture-uuid", "used_mib": 0., "total_mib": 24000.}
    monkeypatch.setattr(module.lifecycle, "source_snapshot", lambda *_: {"fixture_only": True})
    monkeypatch.setattr(module.lifecycle, "gpu_sample", lambda: gpu)
    monkeypatch.setattr(module.lifecycle, "make_settings", lambda *_: object())
    monkeypatch.setattr(module, "SchedulerClient", lambda _: SimpleNamespace(store=store))
    def fake_start(_, **kwargs):
        launches.append(True)
        return handle
    monkeypatch.setattr(module, "SubprocessExecutor", lambda _: SimpleNamespace(start=fake_start))
    monkeypatch.setattr(module, "ControlPlane", lambda _: object())
    monkeypatch.setattr(module, "process_snapshot", lambda: [{"pid": 12345, "start_ticks": 111}])
    monkeypatch.setattr(module.context_isolation, "cleanup_owned", lambda handle, **kw: cleaned.append(handle.process))
    monkeypatch.setattr(module.context_isolation, "qualification_gate", lambda *a, **kw: {})
    monkeypatch.setattr(module.context_isolation, "prepare", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(module.context_isolation, "associate", lambda *a, **kw: {})
    def fake_build(*a, **kwargs):
        work = kwargs["output"] / "fake-work"
        work.mkdir()
        return job, work
    monkeypatch.setattr(module.lifecycle, "build_job", fake_build)
    monkeypatch.setattr(module.lifecycle, "accept_job", lambda client, value, accepted: value)
    monkeypatch.setattr(module.lifecycle, "submission", lambda *a: object())
    def fake_collect(*a, **kwargs):
        if kwargs.get("root_pid") is not None:
            raise InterruptedError("test termination; no real process was launched")
        return {"compute_apps": [], "gpu": gpu}
    monkeypatch.setattr(module, "collect_sample", fake_collect)
    with pytest.raises(InterruptedError):
        module.calibrate(manifest, tmp_path / "manifest.json", {}, tmp_path / "output", 30., visibility_qualification={})
    assert launches == [True] and cleaned == [handle.process]
    assert json.loads((tmp_path / "output/result.json").read_text())["exception"] == "InterruptedError"


def test_cli_refuses_existing_output_without_calling_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
    monkeypatch.setattr(module, "calibrate", lambda *a: pytest.fail("existing output must not dispatch"))
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        module.main(["--manifest", str(tmp_path / "none"), "--settings", str(tmp_path / "none"),
                     "--output", str(output), "--max-seconds", "1", "--visibility-qualification", str(tmp_path / "none")])


def test_expired_deadline_aborts_before_any_gpu_probe(tmp_path, monkeypatch):
    spec, _ = evidence()
    manifest = {"calibration": [spec], "job_timeout_seconds": 30,
                "deadline_utc": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}
    monkeypatch.setattr(module.lifecycle, "source_snapshot", lambda *_: {"fixture_only": True})
    monkeypatch.setattr(module.lifecycle, "gpu_sample", lambda: pytest.fail("expired run must not query GPU"))
    with pytest.raises(TimeoutError, match="deadline"):
        module.calibrate(manifest, tmp_path / "manifest.json", {}, tmp_path / "output", 10.)
