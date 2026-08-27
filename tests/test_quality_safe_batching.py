from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import BatchResolution, BatchSizeObservation, TrainingJob
from localml_scheduler.graph_knowledge import SchedulerKnowledgeBase
from localml_scheduler.scheduler.candidate_generator import CandidateGenerator
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


def _job() -> TrainingJob:
    job = TrainingJob.create(
        runner_target="tests.fake:run",
        baseline_model_id="quality-model",
        baseline_model_path="/tmp/model",
        runner_kwargs={"batch_size": 16},
        max_epochs=5,
        metadata={
            "learning_rate": 0.001,
            "training_quality_contract": {
                "proposed_physical_batch_size": 16,
                "allowed_physical_batch_sizes": [8, 16, 32],
                "base_gradient_accumulation_steps": 2,
                "target_effective_batch_size": 32,
                "base_learning_rate": 0.001,
                "learning_rate_scaling_policy": "linear",
                "base_warmup_steps": 20,
                "base_scheduler_total_steps": 200,
                "planned_epochs": 5,
            },
        },
    )
    return job


def test_batch_resolution_is_coupled_and_envelope_guarded() -> None:
    resolved = BatchResolution.apply(_job(), 8)

    assert resolved.metadata["resolved_gradient_accumulation_steps"] == 4
    assert resolved.metadata["resolved_effective_batch_size"] == 32
    assert resolved.metadata["resolved_learning_rate"] == pytest.approx(0.001)
    assert resolved.metadata["resolved_warmup_steps"] == 20
    assert resolved.metadata["resolved_scheduler_total_steps"] == 200
    with pytest.raises(ValueError, match="quality-safe envelope"):
        BatchResolution.apply(_job(), 64)


def test_candidate_generation_uses_only_agent_approved_batches() -> None:
    settings = SchedulerSettings(runtime_root=Path("/tmp/quality-safe-test"))

    class Estimator:
        @staticmethod
        def resolved_batch_size(job: TrainingJob) -> int:
            return BatchResolution.resolved_batch_size(job)

    generator = CandidateGenerator(settings, Estimator())  # type: ignore[arg-type]

    assert generator.candidate_batch_sizes(_job()) == [8, 16, 32]


def test_recommendation_minimizes_time_subject_to_quality() -> None:
    with TemporaryDirectory() as tmpdir:
        settings = SchedulerSettings(runtime_root=Path(tmpdir) / "runtime")
        store = SQLiteStateStore(settings)
        knowledge = SchedulerKnowledgeBase(store)
        hardware_key = store.hardware_key()
        for batch, seconds, metric in ((16, 10.0, 0.90), (32, 4.0, 0.895), (64, 1.0, 0.70)):
            store.upsert_batch_size_observation(
                BatchSizeObservation(
                    observation_key=f"quality-{batch}",
                    model_key="quality-model",
                    shape_signature="quality-shape",
                    hardware_key=hardware_key,
                    backend_name="exclusive",
                    batch_param_name="batch_size",
                    batch_size=batch,
                    effective_batch_size=batch,
                    best_metric=metric,
                    metric_name="validation_score",
                    metric_maximize=True,
                    best_epoch=4,
                    planned_epochs=5,
                    completed_epochs=4,
                    convergence_curve=[{"epoch": 4, "metric": metric}],
                    seed_variance=0.0001,
                    metadata={"seconds_per_epoch": seconds, "fits": True},
                )
            )

        recommendation = knowledge.recommend_batch_size(
            model_or_signature="quality-model",
            shape_signature="quality-shape",
            current_batch_size=16,
            candidate_batch_sizes=[16, 32, 64],
            planned_epochs=5,
            baseline_metric=0.90,
            metric_maximize=True,
            quality_tolerance=0.01,
            max_effective_batch_size=64,
        )

        assert recommendation["recommended_batch_size"] == 32
        assert recommendation["source"] == "constrained_time_minimization"
        rejected_64 = next(
            item
            for item in recommendation["evaluated_candidates"]
            if item["batch_size"] == 64
        )
        assert "quality_below_approved_floor" in rejected_64["rejection_reasons"]
