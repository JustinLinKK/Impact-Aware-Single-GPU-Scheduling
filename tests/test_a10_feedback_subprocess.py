"""Tiny CPU-only integration replay of the actual worker/shared-training route."""
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time

from localml_scheduler.adapters import shared_training
from localml_scheduler.benchmarks.a10_feedback_latency.run import (
    audit_checkpoint, build_job, make_settings, submission,
)
from localml_scheduler.benchmarks.a10_feedback_latency.receipts import receive
from localml_scheduler.client import SchedulerClient
from localml_scheduler.domain import JobStatus
from localml_scheduler.execution.executor import SubprocessExecutor
from localml_scheduler.execution.process_utils import terminate_process_tree


SCRIPT = '''import os
from pathlib import Path
import torch
from localml_scheduler.adapters.shared_training import CandidateWorkload, TrainingRecipe, run_candidate_training

class Adapter:
    def build_model(self, context): return torch.nn.Linear(1, 1)
    def build_optimizer(self, model, context): return torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    def training_step(self, model, batch, context):
        return torch.nn.functional.binary_cross_entropy_with_logits(model(batch["x"]), batch["y"])

def prepare_training():
    data = [{"x": torch.tensor([float(i % 2)]), "y": torch.tensor([float(i % 2)])} for i in range(4)]
    def validate(model, device):
        logits = model(torch.tensor([[0.], [1.]], device=device))
        auc = float(logits[1] > logits[0]) + .5 * float(logits[1] == logits[0])
        bce = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[0.], [1.]], device=device)))
        return {"negative_validation_auc": -auc, "validation_auc": auc, "validation_bce": bce,
                "validation_negative_count": 1, "validation_positive_count": 1}
    return CandidateWorkload(Adapter(), {}, data, TrainingRecipe(2,2,2), "cpu-fixture", Path(os.environ["A10_PILOT_OUTPUT"]), validate, selection_metric="negative_validation_auc")

if __name__ == "__main__":
    run_candidate_training(prepare_training)
'''


def test_real_worker_returns_exact_budget_and_usable_auc_without_cuda(tmp_path, monkeypatch):
    script = tmp_path / "candidate.py"
    script.write_text(SCRIPT)
    spec_module = importlib.util.spec_from_file_location("cpu_pilot_fixture", script)
    module = importlib.util.module_from_spec(spec_module)
    monkeypatch.setitem(sys.modules, spec_module.name, module)
    spec_module.loader.exec_module(module)
    expected = sha256(json.dumps({"contract": shared_training.CONTRACT,
        "recipe": asdict(shared_training.TrainingRecipe(2, 2, 2)), "identity": "cpu-fixture",
        "context": {}, "adapter": inspect.getsource(module.Adapter),
        "runner_sha256": sha256(Path(shared_training.__file__).read_bytes()).hexdigest()}, sort_keys=True).encode()).hexdigest()
    spec = dict(name="cpu-unit", source="candidate.py", source_sha256=sha256(script.read_bytes()).hexdigest(),
        model_family="toy", batch=2, epochs=2, steps_per_epoch=2, parameter_count=2,
        expected_runtime_signature=expected, validation_size=2)
    output = tmp_path / "episode"
    output.mkdir()
    settings = make_settings({}, output, {"total_mib": 24000})
    client = SchedulerClient(settings)
    job, work = build_job(spec, manifest_path=tmp_path / "manifest.json", output=output,
                         manifest={"job_timeout_seconds": 30}, memory_budget_mib=20000)
    job.resource_requirements.requires_gpu = False
    job.metadata.pop("trial_memory_limit_mb")
    job = client.submit(job)
    client.store.set_job_status(job.job_id, JobStatus.READY, reason="CPU integration fixture", hold=False)
    frozen = submission(spec, job, work, time.monotonic(), time.time())
    # Parent package roots are explicit, so child imports do not depend on pytest.
    introspection = importlib.import_module("engine.script_introspection")
    parent_roots = [str(Path(shared_training.__file__).parents[2]),
                    str(Path(introspection.__file__).parents[1])]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(parent_roots))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    handle = SubprocessExecutor(settings).start(client.store.get_job(job.job_id))
    try:
        handle.process.wait(timeout=35)
        latest = client.store.get_job(job.job_id)
        receipt = receive(frozen, latest, handle.process.returncode)
        assert receipt["selected_auc"] in (0., .5, 1.)
        assert receipt["auc_maximize"] is True
        audit = audit_checkpoint(spec, work, latest)
        assert audit["all_adamw_steps"] == 4
        assert not shared_training.torch.cuda.is_initialized()
    finally:
        terminate_process_tree(handle.process)
