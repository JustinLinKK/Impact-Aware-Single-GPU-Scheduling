from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from localml_scheduler.adapters.mlevolve import build_mlevolve_job
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import (
    BatchProbeSpec,
    PackingSpec,
    ResourceRequirements,
    TrainingJob,
    WorkloadIdentity,
)
from localml_scheduler.profiling.batch_probe import _requires_probe
from localml_scheduler.scheduler.policies import PriorityFifoPolicy
from localml_scheduler.scheduler.resource_estimator import BatchOptionEstimate
from localml_scheduler.scheduler.service import (
    PlacementPatternObservation,
    PlacementProfileSnapshot,
    PlacementReplayState,
    PlacementReplayTemplate,
    SchedulerService,
)
from localml_scheduler.scheduler.telemetry import MemoryAdmissionGate


def _settings(tmp_path: Path) -> SchedulerSettings:
    return SchedulerSettings(
        runtime_root=tmp_path,
        gpu_scheduler={
            "mode": "parallel_time_aware",
            "backend_priority": ["cuda_process", "exclusive"],
            "parallel_job_cap": 3,
            "memory": {
                "gpu_vram_gib": 10,
                "predicted_budget_fraction": 0.9,
                "live_admission_stop_fraction": 0.95,
                "live_admission_resume_fraction": 0.85,
            },
            "profiling": {"reuse_profile_if_confidence_ge": 0.8},
            "colocation": {
                "decision_replay": {
                    "enabled": True,
                    "min_stable_observations": 3,
                    "training_time_change_fraction": 0.25,
                    "vram_change_fraction": 0.25,
                }
            },
        },
        graph_db={"enabled": False},
        hardware_feature_db={"enabled": False},
    )


def _job(
    job_id: str,
    *,
    dataset_key: str = "histopathologic-cancer-detection",
    architecture_key: str = "resnet50",
    architecture_family: str = "cnn",
    batch_size: int = 4,
) -> TrainingJob:
    return TrainingJob.create(
        "pkg.runner:train",
        "resnet50",
        "/tmp/resnet50.pt",
        job_id=job_id,
        workflow_id="cancer-training",
        workload_identity=WorkloadIdentity(
            task_key="image-classification",
            dataset_key=dataset_key,
            architecture_key=architecture_key,
            architecture_family=architecture_family,
        ),
        runner_kwargs={"batch_size": batch_size},
        max_epochs=10,
        resource_requirements=ResourceRequirements(estimated_avg_vram_mb=512),
        packing=PackingSpec(
            eligible=True,
            signature="cancer-resnet50",
            family="cnn",
            backend_allowlist=["cuda_process"],
        ),
        batch_probe=BatchProbeSpec(enabled=True),
    )


def _service(tmp_path: Path) -> SchedulerService:
    service = SchedulerService.__new__(SchedulerService)
    service.settings = _settings(tmp_path)
    service.policy = PriorityFifoPolicy(enable_priority_aging=False)
    service.store = Mock()
    service.store.hardware_key.return_value = "gpu-a"
    service.event_logger = Mock()
    service._placement_replay = PlacementReplayState()
    service._admission_gate = SimpleNamespace(is_open=True)
    service._persist_scheduler_decision_state = Mock()

    estimator = Mock()
    estimator.safe_budget_mb.return_value = 9_000.0

    def estimates(job: TrainingJob, backend: str, batch_sizes: list[int]) -> list[BatchOptionEstimate]:
        del backend
        batch_size = int(batch_sizes[0])
        return [
            BatchOptionEstimate(
                job_id=job.job_id,
                batch_size=batch_size,
                avg_vram_mb=float(job.metadata.get("test_vram_mb", 1_000.0)),
                seconds_per_epoch=float(job.metadata.get("test_epoch_seconds", 10.0)),
                remaining_epochs=10,
                remaining_runtime_seconds=100.0,
                source=str(job.metadata.get("test_profile_source", "branch_profile")),
                confidence=job.metadata.get("test_confidence", 0.9),
                estimate_version="test",
            )
        ]

    estimator.estimate_batch_options.side_effect = estimates
    candidate_generator = Mock()
    candidate_generator.candidate_batch_sizes.return_value = [1, 2, 4, 8, 16]
    compatibility = Mock()
    compatibility.pack_eligible.return_value = True
    compatibility.compatible_group.return_value = True
    service.planner = SimpleNamespace(
        estimator=estimator,
        candidate_generator=candidate_generator,
        compatibility=compatibility,
        time_objective=Mock(),
        choose_plan=Mock(),
    )
    return service


def _learn_width_three(service: SchedulerService) -> PlacementReplayTemplate:
    for episode in range(3):
        jobs = [_job(f"episode-{episode}-slot-{slot}") for slot in range(3)]
        service._stage_successful_pattern(jobs, backend_name="cuda_process")
    assert service._placement_replay.template is not None
    return service._placement_replay.template


def test_workload_identity_round_trip_and_normalization() -> None:
    job = _job("identity")
    payload = json.loads(job.to_json())
    restored = TrainingJob.from_dict(payload)

    assert restored.workload_identity == job.workload_identity
    normalized = WorkloadIdentity(
        task_key="Image Classification",
        dataset_key="Histopathologic_Cancer Detection",
        architecture_key="ResNet_50",
        architecture_family="CNN",
    )
    assert normalized.dataset_key == "histopathologic-cancer-detection"
    assert normalized.architecture_key == "resnet-50"
    assert restored.to_job_spec().workload_identity.architecture_key == "resnet50"


def test_mlevolve_adapter_accepts_identity_and_legacy_task_fallback() -> None:
    explicit = build_mlevolve_job(
        workflow_id="workflow-1",
        baseline_model_id="baseline",
        baseline_model_path="/tmp/baseline.pt",
        runner_target="pkg.runner:train",
        task_key="classification",
        dataset_key="cats-vs-dogs",
        architecture_key="vit_b16",
        architecture_family="transformer",
    )
    legacy = build_mlevolve_job(
        workflow_id="legacy-workflow",
        baseline_model_id="baseline",
        baseline_model_path="/tmp/baseline.pt",
        runner_target="pkg.runner:train",
    )

    assert explicit.workload_identity == WorkloadIdentity(
        task_key="classification",
        dataset_key="cats-vs-dogs",
        architecture_key="vit-b16",
        architecture_family="transformer",
    )
    assert legacy.workload_identity.task_key == "legacy-workflow"
    assert not legacy.workload_identity.replay_eligible


def test_three_rejected_width_two_attempts_learn_exclusive(tmp_path: Path) -> None:
    service = _service(tmp_path)

    for episode in range(3):
        anchor = _job(f"anchor-{episode}")
        observation = service._build_pattern_observation(
            [anchor],
            target_width=1,
            backend_name="cuda_process",
            reason="verified_addition_rejected",
        )
        service._record_pattern_observation(observation)

    template = service._placement_replay.template
    assert template is not None
    assert template.target_width == 1
    assert template.backend_name == "exclusive"
    assert template.observation_count == 3


def test_three_successful_width_three_episodes_activate_replay(tmp_path: Path) -> None:
    service = _service(tmp_path)

    template = _learn_width_three(service)

    assert template.target_width == 3
    assert template.backend_name == "cuda_process"
    assert [profile.batch_size for profile in template.slot_profiles] == [4, 4, 4]
    assert len(service._placement_replay.observations) == 3


def test_membership_episode_is_deduplicated(tmp_path: Path) -> None:
    service = _service(tmp_path)
    jobs = [_job(f"slot-{slot}") for slot in range(3)]

    service._stage_successful_pattern(jobs, backend_name="cuda_process")
    service._stage_successful_pattern(jobs, backend_name="cuda_process")

    assert len(service._placement_replay.observations) == 1


def test_width_three_replay_starts_and_fills_first_vacant_slot(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _learn_width_three(service)
    first = _job("replay-0")

    handled, plan = service._choose_placement_replay_plan(
        [first], active_jobs=[], backend_available={"cuda_process": True}
    )
    assert handled and plan is not None
    assert plan.mode == "stack_anchor"
    assert plan.objective_breakdown["placement_replay_slot"] == 0
    assert plan.batch_overrides == {first.job_id: 4}

    active_slot_zero = first.copy(
        metadata={"placement_backend": "cuda_process", "placement_replay_slot": 0}
    )
    active_slot_two = _job("replay-2").copy(
        metadata={"placement_backend": "cuda_process", "placement_replay_slot": 2}
    )
    newcomer = _job("replay-1")
    handled, plan = service._choose_placement_replay_plan(
        [newcomer],
        active_jobs=[active_slot_zero, active_slot_two],
        backend_available={"cuda_process": True},
    )

    assert handled and plan is not None
    assert plan.mode == "concurrent_group"
    assert plan.objective_breakdown["placement_replay_slot"] == 1
    service.planner.time_objective.assert_not_called()
    service.planner.choose_plan.assert_not_called()


@pytest.mark.parametrize(
    ("updates", "expected_family"),
    [
        ({"dataset_key": "cats-vs-dogs"}, "cnn"),
        ({"architecture_key": "resnet101"}, "cnn"),
        ({"architecture_key": "vit-b16", "architecture_family": "transformer"}, "transformer"),
    ],
)
def test_identity_changes_invalidate_before_dispatch(
    tmp_path: Path,
    updates: dict[str, str],
    expected_family: str,
) -> None:
    service = _service(tmp_path)
    _learn_width_three(service)
    candidate = _job("changed", **updates)

    handled, plan = service._choose_placement_replay_plan(
        [candidate], active_jobs=[], backend_available={"cuda_process": True}
    )

    assert not handled and plan is None
    assert service._placement_replay.template is None
    assert candidate.workload_identity.architecture_family == expected_family


def test_profile_change_is_symmetric_and_inclusive_at_boundary(tmp_path: Path) -> None:
    service = _service(tmp_path)
    reference = PlacementProfileSnapshot(4, 100.0, 1_000.0, "test", 0.9)

    assert service._profile_change(reference, PlacementProfileSnapshot(4, 125.0, 1_000.0, "test", 0.9))[0]
    assert service._profile_change(reference, PlacementProfileSnapshot(4, 80.0, 1_000.0, "test", 0.9))[0]
    assert service._profile_change(reference, PlacementProfileSnapshot(4, 100.0, 1_250.0, "test", 0.9))[0]
    assert not service._profile_change(reference, PlacementProfileSnapshot(4, 124.99, 1_249.9, "test", 0.9))[0]


def test_low_confidence_or_missing_profile_cannot_replay(tmp_path: Path) -> None:
    service = _service(tmp_path)
    job = _job("profile")
    job.metadata["test_confidence"] = 0.79

    assert service._placement_profile_snapshot(job, backend_name="exclusive", batch_size=4) is None
    service.planner.estimator.estimate_batch_options.side_effect = None
    service.planner.estimator.estimate_batch_options.return_value = []
    assert service._placement_profile_snapshot(job, backend_name="exclusive", batch_size=4) is None


def test_memory_gate_waits_but_backend_and_vram_failures_invalidate(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _learn_width_three(service)
    candidate = _job("wait")
    service._admission_gate.is_open = False

    handled, plan = service._choose_placement_replay_plan(
        [candidate], active_jobs=[], backend_available={"cuda_process": True}
    )
    assert handled and plan is None
    assert service._placement_replay.template is not None

    service._admission_gate.is_open = True
    handled, plan = service._choose_placement_replay_plan(
        [candidate], active_jobs=[], backend_available={"cuda_process": False}
    )
    assert not handled and plan is None
    assert service._placement_replay.template is None

    _learn_width_three(service)
    service.planner.estimator.safe_budget_mb.return_value = 2_500.0
    active_jobs = [
        _job(f"active-{slot}").copy(
            metadata={"placement_backend": "cuda_process", "placement_replay_slot": slot}
        )
        for slot in range(2)
    ]
    handled, plan = service._choose_placement_replay_plan(
        [candidate], active_jobs=active_jobs, backend_available={"cuda_process": True}
    )
    assert not handled and plan is None
    assert service._placement_replay.template is None


def test_incompatibility_and_unsupported_cached_batch_invalidate(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _learn_width_three(service)
    candidate = _job("incompatible")
    service.planner.compatibility.compatible_group.return_value = False

    handled, plan = service._choose_placement_replay_plan(
        [candidate], active_jobs=[], backend_available={"cuda_process": True}
    )
    assert not handled and plan is None
    assert service._placement_replay.template is None

    service.planner.compatibility.compatible_group.return_value = True
    _learn_width_three(service)
    service.planner.candidate_generator.candidate_batch_sizes.return_value = [8, 16]
    handled, plan = service._choose_placement_replay_plan(
        [candidate], active_jobs=[], backend_available={"cuda_process": True}
    )
    assert not handled and plan is None
    assert service._placement_replay.template is None


def test_identity_change_clears_incomplete_learning_streak(tmp_path: Path) -> None:
    service = _service(tmp_path)
    for episode in range(2):
        observation = service._build_pattern_observation(
            [_job(f"anchor-{episode}")],
            target_width=1,
            backend_name="cuda_process",
            reason="verified_addition_rejected",
        )
        service._record_pattern_observation(observation)
    changed = _job("changed-learning", dataset_key="cats-vs-dogs")

    handled, plan = service._choose_placement_replay_plan(
        [changed], active_jobs=[], backend_available={"cuda_process": True}
    )

    assert not handled and plan is None
    assert service._placement_replay.observations == []


def test_replay_metadata_suppresses_active_batch_probe() -> None:
    job = _job("probe").copy(
        metadata={
            "placement_backend": "exclusive",
            "skip_active_scheduler_probes": True,
        }
    )

    assert not _requires_probe(job)


def test_replay_dispatch_marks_job_and_counts_suppressed_stages(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _learn_width_three(service)
    job = _job("dispatch")
    handled, plan = service._choose_placement_replay_plan(
        [job], active_jobs=[], backend_available={"cuda_process": True}
    )
    assert handled and plan is not None

    saved_jobs: list[TrainingJob] = []
    service.store.get_job.return_value = job
    service.store.save_job.side_effect = saved_jobs.append
    service.supervisor = Mock()
    service.supervisor.dispatch.return_value = SimpleNamespace(can_run=True, group_id="replay-group")
    service._preload_job_baseline = Mock()
    service._prepare_colocation_trial = Mock(return_value=None)
    service._log_run_group_open = Mock()
    service._prediction_metadata = Mock(return_value={})
    service._active_runs = {}
    service._last_telemetry_poll_at = 0.0

    assert service._dispatch_plan(plan)

    persisted = saved_jobs[-1]
    assert persisted.metadata["skip_active_scheduler_probes"] is True
    assert persisted.metadata["placement_replay_slot"] == 0
    assert service._placement_replay.suppressed_probes == 1
    assert service._placement_replay.suppressed_trials == 1
    assert service._placement_replay.suppressed_decisions == 1
    assert any(
        call.args and call.args[0] == "placement_replayed"
        for call in service.event_logger.emit.call_args_list
    )


def test_replayed_backend_failure_invalidates_and_restores_normal_probe_path(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _learn_width_three(service)
    job = _job("backend-failure")
    handled, plan = service._choose_placement_replay_plan(
        [job], active_jobs=[], backend_available={"cuda_process": True}
    )
    assert handled and plan is not None

    service.store.get_job.return_value = job
    service.store.save_job.side_effect = lambda updated: None
    service.supervisor = Mock()
    service.supervisor.dispatch.side_effect = RuntimeError("backend crashed")
    service._preload_job_baseline = Mock()
    service._prepare_colocation_trial = Mock(return_value=None)
    service._active_runs = {}
    service.logger = Mock()

    assert not service._dispatch_plan(plan)
    assert service._placement_replay.template is None
    assert service.supervisor.dispatch.call_count == 1
    assert any(
        call.kwargs.get("metadata_updates", {}).get("skip_active_scheduler_probes") is False
        for call in service.store.update_job.call_args_list
    )


def test_replay_state_round_trip_and_legacy_restore(tmp_path: Path) -> None:
    service = _service(tmp_path)
    template = _learn_width_three(service)
    restored = PlacementReplayState.from_dict(service._placement_replay.to_dict())

    assert restored.template == template
    assert restored.template is not None
    assert restored.template.hardware_key == "gpu-a"

    service._admission_gate = MemoryAdmissionGate(
        stop_fraction=0.95,
        resume_fraction=0.85,
        window_seconds=10,
    )
    service._exclusive_probe_job_id = None
    service._colocation_trial = None
    service._colocation_stall = None
    service._persist_scheduler_decision_state = (
        SchedulerService._persist_scheduler_decision_state.__get__(service)
    )
    service._persist_scheduler_decision_state()
    restarted = _service(tmp_path)
    restarted._restore_scheduler_decision_state()
    assert restarted._placement_replay.template == template

    legacy_service = SchedulerService.__new__(SchedulerService)
    legacy_service.settings = _settings(tmp_path)
    legacy_service.logger = Mock()
    legacy_service._admission_gate = MemoryAdmissionGate(
        stop_fraction=0.95,
        resume_fraction=0.85,
        window_seconds=10,
    )
    legacy_service._exclusive_probe_job_id = None
    legacy_service._colocation_trial = None
    legacy_service._colocation_stall = None
    legacy_service._placement_replay = PlacementReplayState()
    (tmp_path / "scheduler_decision_state.json").write_text(
        json.dumps({"admission_open": True}),
        encoding="utf-8",
    )

    legacy_service._restore_scheduler_decision_state()

    assert legacy_service._placement_replay == PlacementReplayState()


def test_state_objects_accept_files_without_scope_fields() -> None:
    observation = PlacementPatternObservation.from_dict(
        {
            "identity": {
                "task_key": "classification",
                "architecture_key": "resnet50",
            },
            "target_width": 1,
            "backend_name": "exclusive",
            "slot_profiles": [
                {
                    "batch_size": 4,
                    "total_training_seconds": 100,
                    "avg_vram_mb": 1000,
                    "source": "legacy",
                    "confidence": 0.9,
                }
            ],
            "member_job_ids": ["legacy-job"],
        }
    )

    assert observation.hardware_key == ""
    assert observation.scheduler_mode == ""


def test_homogeneous_trace_pays_three_learning_episodes_then_skips_decisions(tmp_path: Path) -> None:
    service = _service(tmp_path)
    full_evaluations = 0
    replayed = 0

    for episode in range(12):
        job = _job(f"trace-{episode}")
        if service._placement_replay.template is None:
            full_evaluations += 1
            observation = service._build_pattern_observation(
                [job],
                target_width=1,
                backend_name="cuda_process",
                reason="verified_addition_rejected",
            )
            service._record_pattern_observation(observation)
            continue
        handled, plan = service._choose_placement_replay_plan(
            [job], active_jobs=[], backend_available={"exclusive": True}
        )
        assert handled and plan is not None
        replayed += 1

    assert full_evaluations == 3
    assert replayed == 9

    heterogeneous = _job("heterogeneous", dataset_key="cats-vs-dogs")
    handled, plan = service._choose_placement_replay_plan(
        [heterogeneous], active_jobs=[], backend_available={"exclusive": True}
    )
    assert not handled and plan is None
