"""Fresh generated workers must not depend on scheduler import order."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("first", ["execution.runner_protocol", "execution.worker_runtime", "profiling"])
def test_fresh_worker_import_order(first, tmp_path):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    code = f"""
import importlib
importlib.import_module('localml_scheduler.{first}')
from localml_scheduler.execution.runner_protocol import RunnerContext, BatchProbeProtocol
from localml_scheduler.profiling import run_batch_probe_preflight, runtime_profile_for_job
assert callable(run_batch_probe_preflight) and callable(runtime_profile_for_job)
import torch
assert not torch.cuda.is_initialized()
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
