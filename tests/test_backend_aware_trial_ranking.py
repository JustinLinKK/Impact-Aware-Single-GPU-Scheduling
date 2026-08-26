from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
import tempfile

import pytest

from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import (
    BatchSizeObservation,
    ColocationTimingProfile,
    PackingSpec,
    ResourceRequirements,
    RuntimeProfile,
    TrainingJob,
    build_batch_size_observation_key,
    build_colocation_profile_key,
)
from localml_scheduler.execution.backends import MPSBackend
from localml_scheduler.scheduler.backend_compatibility import (
    BackendCompatibilityPolicy,
)
from localml_scheduler.scheduler.pareto import dominates, pareto_fronts
from localml_scheduler.scheduler.placement_planner import PlacementPlanner
from localml_scheduler.scheduler.policies import PriorityFifoPolicy
from localml_scheduler.scheduler.source_fingerprint import StaticJobAnalyzer
from localml_scheduler.scheduler.trial_candidate import BackendTrialConfig
from localml_scheduler.scheduler.trial_priority import TrialPriorityPlanner
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


def _settings(
    runtime_root: str | Path,
    *,
    decision_mode: str = "baseline",
    amortization_factor: float = 3.0,
) -> SchedulerSettings:
    return SchedulerSettings(
        runtime_root=runtime_root,
        graph_db={"enabled": False},
        hardware_feature_db={"enabled": False},
        gpu_scheduler={
            "scheduler_decision_mode": decision_mode,
            "packing_backend": "cuda_process",
            "memory": {
                "gpu_vram_gib": 10,
                "predicted_budget_fraction": 0.85,
                "live_admission_stop_fraction": 0.9,
                "live_admission_resume_fraction": 0.85,
            },
            "source_trial_ranking": {
                "amortization_factor": amortization_factor,
            },
        },
    )


def _job(job_id: str, source_path: Path, *, source_kind: str) -> TrainingJob:
    return TrainingJob.create(
        "pkg.runner:train",
        f"model-{job_id}",
        str(source_path),
        job_id=job_id,
        runner_kwargs={
            "batch_size": 4,
            "precision": "float16",
            "steps_per_epoch": 10,
            "training_step_flops": 8e11 if source_kind == "compute" else 1e11,
            "estimated_bytes_per_step": 1e8 if source_kind == "compute" else 8e8,
        },
        max_epochs=20,
        resource_requirements=ResourceRequirements(estimated_avg_vram_mb=512),
        packing=PackingSpec(
            eligible=True,
            signature=f"signature-{job_id}",
            backend_allowlist=["cuda_process"],
        ),
    )


def _seed_options(
    store: SQLiteStateStore, planner: PlacementPlanner, job: TrainingJob
) -> None:
    for backend in ("exclusive", "cuda_process"):
        for batch_size in (1, 2, 4, 8, 16):
            store.upsert_batch_size_observation(
                BatchSizeObservation(
                    observation_key=build_batch_size_observation_key(
                        job.baseline_model_id,
                        planner.estimator.shape_signature(job),
                        store.hardware_key(),
                        backend,
                        batch_size,
                    ),
                    model_key=job.baseline_model_id,
                    shape_signature=planner.estimator.shape_signature(job),
                    hardware_key=store.hardware_key(),
                    backend_name=backend,
                    batch_param_name="batch_size",
                    batch_size=batch_size,
                    avg_vram_mb=400.0 + batch_size,
                )
            )
            store.upsert_runtime_profile(
                RuntimeProfile.create(
                    signature=job.packing.signature or job.job_id,
                    hardware_key=store.hardware_key(),
                    backend_name=backend,
                    resolved_batch_size=batch_size,
                    strategy="epoch_1",
                    epoch_1_seconds=10.0,
                    estimated_total_runtime_seconds=200.0,
                    confidence=0.9,
                    observations=1,
                    source="branch_profile",
                )
            )


def test_scheduler_decision_mode_defaults_to_baseline_and_validates() -> None:
    default = SchedulerSettings()
    assert default.gpu_scheduler.scheduler_decision_mode == "baseline"
    assert (
        SchedulerSettings(
            gpu_scheduler={"scheduler_decision_mode": "backend-aware"}
        ).gpu_scheduler.scheduler_decision_mode
        == "backend_awared"
    )
    serialized = default.gpu_scheduler.to_dict()
    assert serialized["scheduler_decision_mode"] == "baseline"
    assert serialized["source_trial_ranking"]["policy"] == "pareto"
    assert serialized["source_trial_ranking"]["mode_overhead_mb"]["cuda_process"] > 0
    with pytest.raises(ValueError, match="scheduler_decision_mode"):
        SchedulerSettings(gpu_scheduler={"scheduler_decision_mode": "random"})


def test_static_source_fingerprint_is_normalized_and_execution_scoped() -> None:
    first = TrainingJob.create(
        "pkg:run",
        "model",
        "/tmp/model.py",
        runner_kwargs={
            "architecture_source": "# comment\nx=torch.matmul(a, b)\n",
            "batch_size": 4,
            "precision": "float16",
            "steps_per_epoch": 10,
        },
        max_epochs=2,
    )
    second = first.copy()
    second.config.runner_kwargs["architecture_source"] = "x = torch.matmul(a,b)"
    analyzer = StaticJobAnalyzer()
    left = analyzer.analyze(
        first, 4, predicted_epoch_seconds=5.0, predicted_vram_bytes=1024
    )
    right = analyzer.analyze(
        second, 4, predicted_epoch_seconds=5.0, predicted_vram_bytes=1024
    )
    changed_batch = analyzer.analyze(
        second, 8, predicted_epoch_seconds=5.0, predicted_vram_bytes=1024
    )
    changed_epochs = analyzer.analyze(
        second.copy(max_epochs=99),
        4,
        predicted_epoch_seconds=5.0,
        predicted_vram_bytes=1024,
    )

    assert left.source_hash == right.source_hash
    assert left.graph_hash == right.graph_hash
    assert left.graph_hash != changed_batch.graph_hash
    assert left.execution_signature == changed_epochs.execution_signature
    assert left.operator_histogram["gemm"] == 1
    assert left.step_seconds == 0.5
    assert "compute_pressure_unavailable" in left.analysis_warnings


def test_analyzer_detects_sync_transfers_dataloader_and_dynamic_source() -> None:
    source = """
loader = DataLoader(data, num_workers=4)
while flag:
    x = x.to('cuda', non_blocking=True)
    y = x.cpu().numpy()
    torch.cuda.synchronize()
    torch.save(y, 'checkpoint.pt')
"""
    job = TrainingJob.create(
        "pkg:run",
        "model",
        "/tmp/model.py",
        runner_kwargs={"architecture_source": source, "batch_size": 2},
        max_epochs=2,
    )
    fingerprint = StaticJobAnalyzer().analyze(job, 2)
    assert fingerprint.explicit_sync_count == 1
    assert fingerprint.async_transfer_count == 1
    assert fingerprint.blocking_transfer_count >= 2
    assert fingerprint.dataloader_worker_count == 4
    assert fingerprint.checkpoint_frequency == 1
    assert fingerprint.dynamic_control_flow
    assert fingerprint.confidence == "LOW"


def test_pareto_fronts_are_deterministic_and_do_not_scalarize_risks() -> None:
    risks = {
        "a": MappingProxyType({"compute": 0.1, "memory": 0.7}),
        "b": MappingProxyType({"compute": 0.7, "memory": 0.1}),
        "c": MappingProxyType({"compute": 0.8, "memory": 0.8}),
    }
    assert not dominates(risks["a"], risks["b"])
    assert dominates(risks["a"], risks["c"])
    assert dominates(risks["b"], risks["c"])
    fronts = pareto_fronts(
        ["c", "b", "a"], lambda key: risks[key], stable_key=lambda key: key
    )
    assert fronts == {"a": 0, "b": 0, "c": 1}


def test_mps_policy_prefers_compute_memory_complement() -> None:
    analyzer = StaticJobAnalyzer(
        peak_tflops_by_dtype={"float16": 1.0}, memory_bandwidth_gbps=1.0
    )

    def fingerprint(kind: str):
        source = (
            "layer = torch.nn.Linear(16, 16)"
            if kind == "compute"
            else "layer = torch.nn.Embedding(100, 16)"
        )
        job = TrainingJob.create(
            "pkg:run",
            kind,
            f"/tmp/{kind}.py",
            runner_kwargs={
                "architecture_source": source,
                "batch_size": 1,
                "precision": "float16",
                "steps_per_epoch": 1,
                "training_step_flops": 8e11 if kind == "compute" else 1e11,
                "estimated_bytes_per_step": 1e8 if kind == "compute" else 8e8,
            },
            max_epochs=1,
        )
        return analyzer.analyze(job, 1, predicted_epoch_seconds=1.0)

    compute = fingerprint("compute")
    memory = fingerprint("memory")
    policy = BackendCompatibilityPolicy()
    config = BackendTrialConfig(allocation_percentages=(50, 50))
    complement = policy.evaluate(
        (compute, memory), backend_name="mps_process", backend_config=config
    )
    conflict = policy.evaluate(
        (compute, compute), backend_name="mps_process", backend_config=config
    )
    assert "MPS_COMPUTE_MEMORY_COMPLEMENT" in complement.reason_codes
    assert (
        complement.risk_components["same_resource_conflict"]
        < conflict.risk_components["same_resource_conflict"]
    )


def test_backend_config_is_exact_cache_identity_and_mps_launch_input() -> None:
    members = [
        {
            "signature": "left",
            "batch_size": 4,
            "backend_name": "mps_process",
            "backend_config": {"allocation_percentages": [50, 50]},
        },
        {
            "signature": "right",
            "batch_size": 8,
            "backend_name": "mps_process",
            "backend_config": {"allocation_percentages": [50, 50]},
        },
    ]
    changed = [
        {
            **member,
            "backend_config": {"allocation_percentages": [60, 40]},
        }
        for member in members
    ]
    assert build_colocation_profile_key("gpu", members) != build_colocation_profile_key(
        "gpu", changed
    )

    jobs = [
        TrainingJob.create("pkg:run", "a", "/tmp/a.py", max_epochs=1),
        TrainingJob.create("pkg:run", "b", "/tmp/b.py", max_epochs=1),
    ]
    for job in jobs:
        job.metadata["placement_backend_config"] = {"allocation_percentages": [60, 40]}
    backend = MPSBackend(SchedulerSettings(), executor=object(), mps_binary="mps")
    envs = backend._client_envs(jobs)
    assert [env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] for env in envs] == [
        "60",
        "40",
    ]


def test_active_mps_allocations_are_immutable_and_templates_are_not_reenumerated() -> None:
    planner = TrialPriorityPlanner()
    configs = planner.backend_configs(
        "mps_process",
        mps_templates=[[50, 50], [70, 30]],
        active_config={"allocation_percentages": [60, 40]},
    )
    assert [config.to_dict() for config in configs] == [
        {"allocation_percentages": [60, 40]}
    ]


@pytest.mark.parametrize(
    "retired", ["stream", "cuda_stream", "mps_stream", "stream_mps"]
)
def test_retired_backend_cannot_enter_compatibility_policy(retired: str) -> None:
    job = TrainingJob.create(
        "pkg:run", "model", "/tmp/model.py", runner_kwargs={"batch_size": 1}
    )
    fingerprint = StaticJobAnalyzer().analyze(job, 1)
    with pytest.raises(ValueError, match="retired"):
        BackendCompatibilityPolicy().evaluate(
            (fingerprint, fingerprint),
            backend_name=retired,
            backend_config=BackendTrialConfig(),
        )


def test_backend_aware_mode_jointly_selects_pair_and_baseline_does_not() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        compute_source = root / "compute.py"
        memory_source = root / "memory.py"
        compute_source.write_text("layer = torch.nn.Linear(16, 16)\n", encoding="utf-8")
        memory_source.write_text(
            "layer = torch.nn.Embedding(100, 16)\n", encoding="utf-8"
        )
        jobs = [
            _job("compute", compute_source, source_kind="compute"),
            _job("memory", memory_source, source_kind="memory"),
        ]
        for index, job in enumerate(jobs):
            job.queue_sequence = index + 1

        baseline_settings = _settings(root / "baseline")
        baseline_store = SQLiteStateStore(baseline_settings)
        baseline_planner = PlacementPlanner(
            baseline_settings,
            baseline_store,
            PriorityFifoPolicy(enable_priority_aging=False),
        )
        for job in jobs:
            _seed_options(baseline_store, baseline_planner, job)
        baseline_plan = baseline_planner.choose_plan(
            jobs,
            backend_available={"cuda_process": True, "exclusive": True},
        )
        assert baseline_plan is not None
        assert baseline_plan.mode == "stack_anchor"
        assert len(baseline_plan.job_ids) == 1

        aware_settings = _settings(root / "aware", decision_mode="backend_awared")
        aware_store = SQLiteStateStore(aware_settings)
        aware_planner = PlacementPlanner(
            aware_settings,
            aware_store,
            PriorityFifoPolicy(enable_priority_aging=False),
        )
        for job in jobs:
            _seed_options(aware_store, aware_planner, job)
        aware_plan = aware_planner.choose_plan(
            jobs,
            backend_available={"cuda_process": True, "exclusive": True},
        )
        assert aware_plan is not None
        assert aware_plan.mode == "concurrent_group"
        assert set(aware_plan.job_ids) == {"compute", "memory"}
        assert (
            aware_plan.objective_breakdown["scheduler_decision_mode"]
            == "backend_awared"
        )
        assert aware_plan.objective_breakdown["requires_live_trial"] is True
        assert (
            aware_plan.objective_breakdown["source_trial_ranking"][0]["selected"]
            is True
        )
        assert aware_plan.objective_breakdown["estimated_trial_cost_seconds"] > 0


def test_exact_good_backend_configuration_bypasses_retrial() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        left_source = root / "left.py"
        right_source = root / "right.py"
        left_source.write_text("x = torch.matmul(a, b)\n", encoding="utf-8")
        right_source.write_text("x = torch.embedding(weight, ids)\n", encoding="utf-8")
        jobs = [
            _job("left", left_source, source_kind="compute"),
            _job("right", right_source, source_kind="memory"),
        ]
        settings = _settings(root / "runtime", decision_mode="backend_awared")
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(
            settings, store, PriorityFifoPolicy(enable_priority_aging=False)
        )
        for job in jobs:
            _seed_options(store, planner, job)
        first = planner.choose_plan(
            jobs,
            backend_available={"cuda_process": True, "exclusive": True},
        )
        assert first is not None
        members = []
        timings = []
        source_signatures = first.trial_metadata["source_fingerprint_signatures"]
        for job_id in first.job_ids:
            job = next(job for job in jobs if job.job_id == job_id)
            descriptor = {
                "signature": source_signatures[job_id],
                "batch_size": first.batch_overrides[job_id],
                "backend_name": first.backend_name,
                "backend_config": first.backend_config,
            }
            members.append(descriptor)
            timings.append({**descriptor, "seconds_per_epoch": 6.0, "observations": 2})
        store.upsert_colocation_timing_profile(
            ColocationTimingProfile.create(
                store.hardware_key(),
                members,
                timings,
                observations=2,
                metadata={
                    "evidence_policy": "fresh_member_epochs_v1",
                    "recent_trial_outcomes": [
                        {
                            "trial_id": "accepted-1",
                            "decision": "accepted",
                            "gain": 1.2,
                            "observed_at": first.objective_breakdown.get(
                                "observed_at", "2099-01-01T00:00:00+00:00"
                            ),
                        }
                    ],
                },
            )
        )

        reused = planner.choose_plan(
            jobs,
            backend_available={"cuda_process": True, "exclusive": True},
        )
        assert reused is not None
        assert reused.reason == "exact backend-aware colocation profile reused"
        assert reused.objective_breakdown["requires_live_trial"] is False


def test_non_amortizable_unknown_pair_falls_back_safely() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = root / "model.py"
        source.write_text("x = torch.matmul(a, b)\n", encoding="utf-8")
        jobs = [
            _job("left", source, source_kind="compute"),
            _job("right", source, source_kind="compute"),
        ]
        settings = _settings(
            root / "runtime",
            decision_mode="backend_awared",
            amortization_factor=1_000_000.0,
        )
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(
            settings, store, PriorityFifoPolicy(enable_priority_aging=False)
        )
        for job in jobs:
            _seed_options(store, planner, job)
        plan = planner.choose_plan(
            jobs,
            backend_available={"cuda_process": True, "exclusive": True},
        )
        assert plan is not None
        assert plan.mode == "stack_anchor"
        assert len(plan.job_ids) == 1


def test_backend_aware_active_group_ranks_all_newcomers_before_selection() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        compute_source = root / "compute.py"
        gap_source = root / "gap.py"
        compute_source.write_text("layer = torch.nn.Linear(16, 16)\n", encoding="utf-8")
        gap_source.write_text("sample = torch.relu(sample)\n", encoding="utf-8")
        active = _job("active", compute_source, source_kind="compute")
        active.metadata.update(
            {
                "placement_backend": "cuda_process",
                "runtime_observed_epoch_seconds": 10.0,
            }
        )
        continuous = _job("continuous", compute_source, source_kind="compute")
        gap = _job("gap", gap_source, source_kind="compute")
        gap.config.runner_kwargs["training_step_flops"] = 5e10
        gap.config.runner_kwargs["estimated_bytes_per_step"] = 5e7
        continuous.queue_sequence = 1
        gap.queue_sequence = 2

        settings = _settings(root / "runtime", decision_mode="backend_awared")
        source_analysis = settings.gpu_scheduler.source_trial_ranking.source_analysis
        source_analysis.peak_tflops_by_dtype = {"float16": 1.0}
        source_analysis.memory_bandwidth_gbps = 1.0
        store = SQLiteStateStore(settings)
        planner = PlacementPlanner(
            settings, store, PriorityFifoPolicy(enable_priority_aging=False)
        )
        for job in (active, continuous, gap):
            _seed_options(store, planner, job)

        plan = planner.choose_plan(
            [continuous, gap],
            active_jobs=[active],
            active_vram_mb=404.0,
            backend_available={"cuda_process": True, "exclusive": True},
        )
        assert plan is not None
        assert plan.job_ids == ("gap",)
        assert plan.objective_breakdown["scheduler_decision_mode"] == "backend_awared"
        records = plan.objective_breakdown["source_trial_ranking"]
        assert {record["job_ids"][-1] for record in records} == {
            "continuous",
            "gap",
        }
