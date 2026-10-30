"""Executable trace-derived workload; configuration is immutable submission data."""
import os
from pathlib import Path
from localml_scheduler.adapters.shared_training import run_candidate_training
from localml_scheduler.benchmarks.a10_feedback_latency.workloads.core import (
    CandidateAdapter, build_model, adapter_context, expected_signature, make_workload, read_spec_env,
)


def prepare_training():
    return make_workload(read_spec_env(), Path(os.environ["A10_PILOT_OUTPUT"]))


if __name__ == "__main__":
    run_candidate_training(prepare_training)
