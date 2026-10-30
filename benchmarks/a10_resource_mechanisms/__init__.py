"""Explicit SYNTHETIC resource controls for the A10 mechanism experiments."""

from .workloads import (
    CandidateAdapter, ControlledMLP, adapter_context, build_model,
    expected_signature, make_workload, parameter_count, recipe, source_metadata,
    spec_digest, template_key,
)

__all__ = [
    "CandidateAdapter", "ControlledMLP", "adapter_context", "build_model",
    "expected_signature", "make_workload", "parameter_count", "recipe",
    "source_metadata", "spec_digest", "template_key",
]
