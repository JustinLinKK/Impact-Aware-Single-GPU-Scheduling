from __future__ import annotations

import pytest

from localml_scheduler.cuda_mcp_bridge import HardwareFacts, build_query, to_records


def test_cuda_mcp_records_carry_exact_backend_and_runner() -> None:
    records = to_records(
        topic="reduce GPU memory",
        answer="The allocator can reuse inactive blocks; this is plain retrieved context.",
        facts=HardwareFacts(
            gpu_name="Test GPU",
            gpu_architecture="ada",
            compute_capability=(8, 9),
            cuda_version="12.4",
            driver_version="550.54",
            torch_version="2.4.1",
            backend_config_hash="backend-hash",
        ),
        source_refs=[
            {
                "title": "CUDA semantics",
                "url": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
                "source_version": "12.4",
            }
        ],
        verified_date="2026-08-25",
        effective_backend="cuda_process",
    )
    assert records[0]["backend_modes"] == ["cuda_process"]
    assert records[0]["runner_contracts"] == ["subprocess_job_v1"]
    assert records[0]["schema_version"] == "code_doc_chunk_v1"
    assert records[0]["source_refs"][0]["url"].startswith("https://docs.nvidia.com/")
    assert "backend cuda_process" in build_query(
        "reduce GPU memory",
        HardwareFacts(gpu_name="Test GPU"),
        effective_backend="cuda_process",
    )


@pytest.mark.parametrize(
    "answer",
    [
        "- Start the MPS daemon inside the training script.",
        "- Set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 in job code.",
        "- Use torch.cuda.Stream to coordinate multiple jobs.",
    ],
)
def test_cuda_mcp_quarantines_scheduler_owned_or_cross_job_controls(answer: str) -> None:
    assert to_records(
        topic="improve throughput",
        answer=answer,
        facts=HardwareFacts(gpu_name="Test GPU"),
        verified_date="2026-08-25",
        effective_backend="mps_process",
    ) == []


def test_cuda_mcp_rejects_unknown_runner_contract() -> None:
    with pytest.raises(ValueError, match="runner contract"):
        build_query(
            "memory",
            HardwareFacts(),
            effective_backend="cuda_process",
            runner_contract="in_process_v0",
        )
