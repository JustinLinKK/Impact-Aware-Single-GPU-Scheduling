from __future__ import annotations

import pytest

from localml_scheduler.backend_mode import RUNNER_CONTRACT_SUBPROCESS_V1
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import PackingSpec, RuntimeProfile, TrainingJob


@pytest.mark.parametrize("backend", ["cuda_process", "mps_process"])
def test_canonical_packing_backend_round_trips(tmp_path, backend: str) -> None:
    settings = SchedulerSettings(
        runtime_root=tmp_path,
        gpu_scheduler={"packing_backend": backend},
    )
    emitted = settings.to_dict()["gpu_scheduler"]
    assert emitted["packing_backend"] == backend
    assert "backend_priority" not in emitted
    assert "stream" not in emitted


def test_legacy_mps_normalizes_once_and_is_not_emitted(tmp_path) -> None:
    with pytest.warns(DeprecationWarning, match="mps_process"):
        settings = SchedulerSettings(
            runtime_root=tmp_path,
            gpu_scheduler={"packing_backend": "mps"},
        )
    assert settings.gpu_scheduler.packing_backend == "mps_process"
    assert settings.to_dict()["gpu_scheduler"]["packing_backend"] == "mps_process"

    with pytest.warns(DeprecationWarning, match="mps_process"):
        profile = RuntimeProfile.create(
            signature="model-a",
            hardware_key="gpu-a",
            backend_name="mps",
            resolved_batch_size=8,
            strategy="epoch_1",
        )
    assert profile.backend_name == "mps_process"
    assert RUNNER_CONTRACT_SUBPROCESS_V1 == "subprocess_job_v1"


@pytest.mark.parametrize(
    "retired", ["stream", "cuda_stream", "mps_stream", "stream_mps"]
)
def test_retired_backend_aliases_fail_with_runner_explanation(
    tmp_path, retired: str
) -> None:
    with pytest.raises(ValueError, match="child subprocesses"):
        SchedulerSettings(
            runtime_root=tmp_path,
            gpu_scheduler={"packing_backend": retired},
        )
    with pytest.raises(ValueError, match="child subprocesses"):
        PackingSpec(eligible=True, backend_allowlist=[retired])


def test_ambiguous_legacy_backend_priority_fails(tmp_path) -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        SchedulerSettings(
            runtime_root=tmp_path,
            gpu_scheduler={
                "backend_priority": [
                    "cuda_process",
                    "mps_process",
                    "exclusive",
                ]
            },
        )


@pytest.mark.parametrize("backend", ["cuda_process", "mps_process"])
def test_jobs_can_be_submitted_asynchronously_without_group_barrier(
    tmp_path, backend: str
) -> None:
    client = SchedulerClient(
        SchedulerSettings(
            runtime_root=tmp_path,
            gpu_scheduler={"packing_backend": backend},
        )
    )
    first = TrainingJob.create(
        job_id="ready-a",
        runner_target="builtins:dict",
        baseline_model_id="a",
        baseline_model_path="/tmp/a",
        packing=PackingSpec(eligible=True, signature="a"),
    )
    second = TrainingJob.create(
        job_id="later-b",
        runner_target="builtins:dict",
        baseline_model_id="b",
        baseline_model_path="/tmp/b",
        packing=PackingSpec(eligible=True, signature="b"),
    )

    submitted_a = client.submit(first)
    assert client.inspect("ready-a") is not None
    submitted_b = client.submit(second)

    assert [job.job_id for job in client.list_jobs()] == ["ready-a", "later-b"]
    for job in (submitted_a, submitted_b):
        assert job.packing.backend_allowlist == [backend]
        assert job.metadata["effective_backend"] == backend
        assert job.metadata["runner_contract"] == RUNNER_CONTRACT_SUBPROCESS_V1


def test_retired_job_payload_is_readable_only_as_nonselectable_history() -> None:
    payload = TrainingJob.create(
        job_id="historical",
        runner_target="builtins:dict",
        baseline_model_id="old",
        baseline_model_path="/tmp/old",
    ).to_dict()
    payload["packing"] = {
        "eligible": True,
        "signature": "old",
        "backend_allowlist": ["stream"],
    }
    payload["metadata"] = {"placement_backend": "stream"}

    with pytest.raises(ValueError, match="child subprocesses"):
        TrainingJob.from_dict(payload)
    historical = TrainingJob.from_dict(payload, historical_read=True)
    assert historical.packing.eligible is False
    assert historical.packing.backend_allowlist == []
    assert historical.hold is True
    assert historical.metadata["selectable"] is False
    assert historical.metadata["placement_backend"] == "stream"
