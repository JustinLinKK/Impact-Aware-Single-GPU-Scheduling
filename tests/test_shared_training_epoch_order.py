"""Cached epoch orders preserve every update's original deterministic samples."""
from hashlib import sha256
import json
from types import SimpleNamespace

import torch

from localml_scheduler.adapters import shared_training
from localml_scheduler.tests.fixtures import shared_training_candidate as fixture


def test_cached_order_preserves_per_update_sample_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = torch.randperm
    draws = []
    def counted(*args, **kwargs):
        value = original(*args, **kwargs)
        draws.append(value.clone())
        return value
    monkeypatch.setattr(torch, "randperm", counted)
    monkeypatch.setattr(shared_training, "_check_runtime_recipe", lambda *_: None)
    context = SimpleNamespace(job=SimpleNamespace(resource_requirements=SimpleNamespace(requires_gpu=False)),
        load_resume_checkpoint=lambda: None, control_hook=SimpleNamespace(safe_point=lambda *_a, **_k: None))
    shared_training.run_candidate_training(fixture.prepare, context=context)
    saved = torch.load(tmp_path/'working/shared_training/training_state.pt', weights_only=False)
    expected = []
    for epoch in range(2):
        order = original(64, generator=torch.Generator().manual_seed(42 + epoch))
        for step in range(12):
            indices = order[step*4:(step+1)*4].tolist()
            expected.append(sha256(json.dumps(indices).encode()).hexdigest())
    assert saved['batch_history'] == expected
    assert saved['global_step'] == 24
    assert len(draws) == 2
