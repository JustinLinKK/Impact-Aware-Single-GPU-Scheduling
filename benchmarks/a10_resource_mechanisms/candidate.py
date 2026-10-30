"""Executable SYNTHETIC mechanism-control training workload."""
import os
from pathlib import Path

from localml_scheduler.adapters.shared_training import run_candidate_training
from localml_scheduler.benchmarks.a10_resource_mechanisms.workloads import (
    CandidateAdapter, adapter_context, build_model, expected_signature,
    make_workload, read_spec_env,
)


def prepare_training():
    output = os.environ.get("A10_DESIGNED_OUTPUT", os.environ.get("A10_PILOT_OUTPUT"))
    if output is None:
        raise ValueError("A10_DESIGNED_OUTPUT or A10_PILOT_OUTPUT is required")
    return make_workload(read_spec_env(), Path(output))


if __name__ == "__main__":
    run_candidate_training(prepare_training)
