from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

from localml_scheduler.domain import JobStatus, TrainingJob
from localml_scheduler.hardware import HardwareProfile
from scheduler_benchmark_test.repeat_time_aware_benchmark import (
    aggregate_summaries,
    attach_matched_batch_slowdown,
    hardware_matches,
    markdown_report,
    write_matched_batch_trace,
)
from scheduler_benchmark_test.replay_scheduler import _trace_metrics, build_settings


def _hardware(name: str = "NVIDIA A10") -> HardwareProfile:
    return HardwareProfile(
        hardware_key="hardware-key",
        os_name="linux",
        gpu_name=name,
        total_vram_mb=23028,
        compute_capability="8.6",
        cuda_runtime="12.8",
        torch_version="2.8.0",
    )


def _summary(makespan: float, weighted_flow: float) -> dict[str, object]:
    return {
        "trace_metrics": {
            "complete": True,
            "makespan_seconds": makespan,
            "total_flow_seconds": weighted_flow * 2,
            "mean_flow_seconds": weighted_flow,
            "weighted_mean_flow_seconds": weighted_flow,
            "median_flow_seconds": weighted_flow,
            "p95_flow_seconds": weighted_flow + 1,
            "max_wait_seconds": 1.0,
            "starvation_count": 0,
            "jobs_per_hour": 7200 / makespan,
        },
        "placement_memory_metrics": {
            "predicted_avg_vram_mb": 4000,
            "actual_avg_vram_mb": 4100,
            "actual_memory_over_budget_count": 0,
            "measured_placement_count": 1,
        },
        "execution_metrics": {
            "average_slowdown": 1.1,
            "early_stopped_epochs_saved": 2,
            "early_stopped_wall_time_saved_seconds": 4.0,
        },
        "external_deadline_reached": False,
    }


def test_repeated_report_records_sample_variance_and_hardware_provenance() -> None:
    report = aggregate_summaries(
        {
            "serial_fifo": [_summary(20, 12), _summary(22, 14)],
            "legacy_vram_fill": [_summary(18, 13), _summary(20, 15)],
            "parallel_time_aware": [_summary(10, 8), _summary(12, 10)],
        },
        hardware=_hardware(),
        required_gpu_name=r"NVIDIA A10(?:\s|$)",
    )
    makespan = report["policies"]["parallel_time_aware"]["metrics"]["makespan_seconds"]
    assert makespan["mean"] == pytest.approx(11.0)
    assert makespan["sample_variance"] == pytest.approx(2.0)
    assert report["policies"]["parallel_time_aware"]["speedup_vs_serial"] == pytest.approx(21 / 11)
    assert report["hardware_requirement_met"]
    rendered = markdown_report(report)
    assert "mean ± sample standard deviation" in rendered
    assert "parallel_time_aware" in rendered
    assert "Early epochs/time saved" in rendered


def test_hardware_name_check_does_not_mislabel_an_rtx_run_as_a10() -> None:
    assert hardware_matches(_hardware(), r"NVIDIA A10(?:\s|$)")
    assert not hardware_matches(_hardware("NVIDIA GeForce RTX 5090"), r"NVIDIA A10(?:\s|$)")


def test_replay_time_aware_settings_use_fractional_budget_and_new_cap(tmp_path) -> None:
    settings = build_settings(
        mode="parallel_time_aware",
        backend="cuda_process",
        batch_search="power_of_two",
        max_packed_jobs_per_gpu=3,
        vram_budget_gib=22,
        runtime_root=tmp_path,
        cache_warm_top_k=0,
        cache_warm_policy="top_k",
        cache_entry_capacity=0,
        cache_max_ram_percent=0.0,
        cache_memory_budget_gib=0.0,
        binary_range_up=16,
        binary_range_down=8,
        power_of_two_range_up=2,
        power_of_two_range_down=2,
        target_vram_fraction=0.97,
        predicted_budget_fraction=0.84,
    )
    assert settings.gpu_scheduler.parallel_job_cap == 3
    assert settings.gpu_scheduler.memory.predicted_budget_fraction == pytest.approx(0.84)
    assert settings.gpu_scheduler.memory.live_admission_stop_fraction == pytest.approx(0.90)
    assert settings.gpu_scheduler.thresholds.pack_reject_max_slowdown == pytest.approx(1.30)


def test_replay_metrics_use_release_flow_and_wait_times() -> None:
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    jobs: list[TrainingJob] = []
    for index, (release, start, finish) in enumerate(((0, 2, 8), (3, 5, 13))):
        job = TrainingJob.create(
            "pkg.runner:train",
            f"model-{index}",
            f"/tmp/model-{index}.pt",
            max_epochs=1,
        )
        job.submitted_at = (origin + timedelta(seconds=release)).isoformat()
        job.started_at = (origin + timedelta(seconds=start)).isoformat()
        job.finished_at = (origin + timedelta(seconds=finish)).isoformat()
        job.status = JobStatus.COMPLETED
        jobs.append(job)
    metrics = _trace_metrics(jobs, starvation_timeout_seconds=60)
    assert metrics["complete"]
    assert metrics["makespan_seconds"] == pytest.approx(13.0)
    assert metrics["total_flow_seconds"] == pytest.approx(18.0)
    assert metrics["mean_flow_seconds"] == pytest.approx(9.0)
    assert metrics["max_wait_seconds"] == pytest.approx(2.0)


def test_matched_batch_slowdown_uses_only_equal_selected_batches(tmp_path) -> None:
    packed = {
        "per_job": [
            {"step_idx": 0, "bs": 2, "resolved_batch_size": 4, "elapsed_s": 12.0},
            {"step_idx": 1, "bs": 2, "resolved_batch_size": 8, "elapsed_s": 20.0},
        ],
        "execution_metrics": {},
    }
    solo = {
        "per_job": [
            {"step_idx": 0, "bs": 4, "resolved_batch_size": None, "elapsed_s": 10.0},
            {"step_idx": 1, "bs": 4, "resolved_batch_size": None, "elapsed_s": 10.0},
        ]
    }
    attach_matched_batch_slowdown(packed, solo)
    assert packed["execution_metrics"]["average_slowdown"] == pytest.approx(1.2)
    assert packed["execution_metrics"]["measured_slowdown_members"] == 1

    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"step_idx": 0, "bs": 2, "max_bs": 2}\n'
        '{"step_idx": 1, "bs": 2, "max_bs": 16}\n',
        encoding="utf-8",
    )
    matched = tmp_path / "matched.jsonl"
    write_matched_batch_trace(source, packed, matched)
    rows = [json.loads(line) for line in matched.read_text(encoding="utf-8").splitlines()]
    assert [(row["bs"], row["max_bs"]) for row in rows] == [(4, 4), (8, 16)]
