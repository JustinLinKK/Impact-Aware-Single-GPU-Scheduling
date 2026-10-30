from datetime import datetime
from types import SimpleNamespace

import pytest

from localml_scheduler.benchmarks.a10_feedback_latency.campaign import episode_order
from localml_scheduler.benchmarks.a10_feedback_latency.run import (
    fixed_fifo_fits, make_settings, parent_due, with_hard_deadline,
)


def test_group_members_arrive_together_and_individual_parents_are_staggered():
    assert [parent_due(100., i, "group") for i in range(3)] == [100., 100., 100.]
    assert [parent_due(100., i, "individual") for i in range(3)] == [100., 102., 104.]


def test_same_prediction_and_live_reserved_budget_gates_fixed_fifo():
    current = {"a": SimpleNamespace(metadata={"trial_reserved_mb": 5000.})}
    profiles = {"a": {"peak_vram_mib": 4000.}}
    assert fixed_fifo_fits({"peak_vram_mib": 3000.}, current, profiles,
                          device_used_mib=5500., budget_mib=9000.)
    assert not fixed_fifo_fits({"peak_vram_mib": 4000.}, current, profiles,
                              device_used_mib=5500., budget_mib=9000.)
    with pytest.raises(ValueError):
        fixed_fifo_fits(None, current, profiles, device_used_mib=5000., budget_mib=9000.)
    with pytest.raises(ValueError):
        fixed_fifo_fits({"peak_vram_mib": 4000.}, current, {}, device_used_mib=5000., budget_mib=9000.)


def test_v4_has_no_fixed_concurrency_cap(tmp_path):
    settings = make_settings({"gpu_scheduler": {"parallel_job_cap": 3}}, tmp_path,
                             {"total_mib": 24000.})
    assert settings.gpu_scheduler.parallel_job_cap is None
    assert settings.gpu_scheduler.feedback_latency.enabled
    assert not settings.gpu_scheduler.feedback_latency.allow_zero_overhead


def test_balanced_paired_campaign_declares_all_sixteen_episodes():
    order = episode_order()
    assert len(order) == 16
    for mode in ("individual", "group"):
        first = [r["arm"] for r in order if r["mode"] == mode and r["repeat"] == 1]
        second = [r["arm"] for r in order if r["mode"] == mode and r["repeat"] == 2]
        assert second == list(reversed(first))


def test_hard_cutoff_cannot_extend_manifest_deadline():
    declared = {"deadline_utc": "2026-10-30T03:10:00Z"}
    assert with_hard_deadline(declared, "2026-10-30T04:00:00Z")["deadline_utc"] == "2026-10-30T03:10:00+00:00"
    assert declared["deadline_utc"].endswith("Z")
    with pytest.raises(ValueError): with_hard_deadline(declared, "2026-10-30T03:10:00")
