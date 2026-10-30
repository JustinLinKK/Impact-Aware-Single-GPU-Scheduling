"""Prepared-proxy proof failures must invalidate the whole CPU preflight."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from localml_scheduler.benchmarks.a10_feedback_latency import async_preparation
from localml_scheduler.benchmarks.a10_feedback_latency import cpu_preflight
from localml_scheduler.benchmarks.a10_feedback_latency import run
from localml_scheduler import client as client_module
from localml_scheduler.prediction import ml_predictor


@pytest.mark.parametrize("failure", [None, "async_profile_mismatch", "worker_did_not_stop"])
def test_main_requires_successful_prepared_proxy_proof_for_every_case(tmp_path, monkeypatch, failure):
    """Run main and its final aggregation, replacing expensive capture only."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fake_torch = SimpleNamespace(__version__="cpu-test", version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_initialized=lambda: False, is_available=lambda: False),
        set_num_threads=lambda _: None)
    fake_psutil = SimpleNamespace(Process=lambda: SimpleNamespace(children=lambda **_: []),
        NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
        AccessDenied=type("AccessDenied", (Exception,), {}))
    for name, module in {"torch": fake_torch, "psutil": fake_psutil,
                         "timm": SimpleNamespace(__version__="test"),
                         "torchvision": SimpleNamespace(__version__="test")}.items():
        monkeypatch.setitem(sys.modules, name, module)

    manifest_path = tmp_path / "manifest.json"
    specs = [dict(name=f"case{i}", model_family=f"family{i}", batch=4,
                  source_sha256="source-hash", expected_runtime_signature=f"signature-case{i}")
             for i in range(3)]
    manifest_path.write_text(json.dumps({"jobs": specs}))
    output = tmp_path / "proof"
    prediction_calls, stopped = [], []

    class FakePredictor:
        available = True
        artifact_hash = "test-student"
        unavailable_reason = None

        def __init__(self, settings, hardware):
            self.last_sources, self.last_errors = {}, {}
            self._released = SimpleNamespace(last_capture_audits={4: {"cpu_only": True}})

        def predict_profile_options(self, job, batches):
            prediction_calls.append(job.job_id)
            self.last_sources[job.job_id] = "ml_predictor"
            return {batch: dict(kind="predicted_workload", hardware="A10", batch_size=batch,
                artifact_sha256=self.artifact_hash, job_signature=job.packing.signature,
                step_cost=100., epoch_cost=1000., avg_vram_mib=800., peak_vram_mib=1000.,
                busy_indicator=70.) for batch in batches}

    class FakePreparedPredictor:
        """Inject a proof-path failure after the initial real-profile check."""
        def __init__(self, predictor, *, asynchronous, prepare_on_read=False):
            self.predictor, self.asynchronous = predictor, asynchronous
            self.failures, self.events, self.target = {}, [], None

        def enqueue(self, job, batch):
            self.target = job.job_id

        def ready(self, job, batch):
            return True

        def predict_profile_options(self, job, batches):
            self.target = job.job_id
            profiles = deepcopy(self.predictor.predict_profile_options(job, batches))
            if self.asynchronous and self.target == "case0" and failure == "async_profile_mismatch":
                profiles[batches[0]]["step_cost"] += 1.
            return profiles

        def stop(self, timeout):
            stopped.append((self.target, self.asynchronous))
            return not (self.asynchronous and self.target == "case0" and failure == "worker_did_not_stop")

    def build_job(spec, *, output, **_):
        work = output / spec["name"]
        work.mkdir()
        return SimpleNamespace(job_id=spec["name"], packing=SimpleNamespace(
            signature=spec["expected_runtime_signature"])), work

    monkeypatch.setattr(ml_predictor, "MLVramPredictor", FakePredictor)
    monkeypatch.setattr(async_preparation, "PreparedPredictor", FakePreparedPredictor)
    monkeypatch.setattr(run, "STUDENT_SHA256", "test-student")
    monkeypatch.setattr(run, "make_settings", lambda *_: SimpleNamespace())
    monkeypatch.setattr(run, "build_job", build_job)
    monkeypatch.setattr(run, "accept_job", lambda client, job, accepted: job)
    monkeypatch.setattr(client_module, "SchedulerClient", lambda settings: SimpleNamespace(
        store=SimpleNamespace(backend=SimpleNamespace())))
    monkeypatch.setattr(sys, "argv", ["cpu_preflight", "--manifest", str(manifest_path),
        "--release", str(tmp_path / "release"), "--output", str(output), "--prepared-proxy-proof"])

    if failure is None:
        cpu_preflight.main()
    else:
        with pytest.raises(SystemExit) as raised:
            cpu_preflight.main()
        assert raised.value.code == 1

    receipt = json.loads((output / "cpu-preflight.json").read_text())
    assert len(receipt["cases"]) == 3
    assert receipt["complete"] is (failure is None)
    assert all(prediction_calls.count(spec["name"]) == 3 for spec in specs)
    assert stopped == [(name, asynchronous) for name in ("case0", "case1", "case2")
                       for asynchronous in (False, True)]
    assert not receipt["cuda_initialized_before"]
    assert not receipt["cuda_initialized_after"]
    assert all(case["complete"] for case in receipt["cases"][1:])
    first = receipt["cases"][0]
    # The base capture succeeded before the proof failed. Its stale True value
    # must be reset, or all three cases would incorrectly satisfy aggregation.
    assert first["complete"] is (failure is None)
    assert first["profile"]["step_cost"] == 100.
    if failure == "async_profile_mismatch":
        assert "prepared proxy changed the real numeric profile" in first["error"]
    elif failure == "worker_did_not_stop":
        assert "preflight preparation worker did not stop" in first["error"]
    else:
        assert first["prepared_proxy_proof"]["complete"] is True
        assert "error" not in first
