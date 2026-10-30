"""A passing preflight must describe the same predictor used in the experiment."""
from copy import deepcopy
import hashlib
import json

import pytest

from localml_scheduler.benchmarks.a10_feedback_latency.cpu_preflight import (
    load_preflight_settings, prediction_settings_for_preflight,
)
from localml_scheduler.benchmarks.a10_feedback_latency.run import make_settings


def prediction(release):
    return {"mode": "ml_predictor", "student_release_path": str(release),
        "student_model_path": str(release / "checkpoints/S_A10"),
        "cpu_threads": 4, "conversion_timeout_seconds": 180, "conversion_max_rss_mb": 16384}


@pytest.mark.parametrize("nested", [False, True])
def test_preflight_matches_evaluation_settings(tmp_path, nested):
    release = tmp_path / "release"
    values = prediction(release)
    if nested:
        values = {"mode": values.pop("mode"), "ml": values}
    raw = {"prediction": values}
    before = deepcopy(raw)
    proof = prediction_settings_for_preflight(raw, release)
    evaluation = make_settings(raw, tmp_path / "run", {"total_mib": 22600}).prediction
    assert proof == evaluation
    assert proof.cpu_threads == 4
    assert raw == before


@pytest.mark.parametrize("field,value", [
    ("student_release_path", "/some/other/release"),
    ("student_model_path", "/some/other/student"),
    ("mode", "branch_profile"),
    ("test_override_enabled", True),
    ("conversion_timeout_seconds", 300),
    ("conversion_max_rss_mb", 32768),
])
def test_preflight_refuses_a_different_model_or_relaxed_limits(tmp_path, field, value):
    values = prediction(tmp_path)
    values[field] = value
    with pytest.raises(ValueError):
        prediction_settings_for_preflight({"prediction": values}, tmp_path)


def test_legacy_command_remains_one_thread_and_missing_explicit_settings_fail(tmp_path):
    proof = prediction_settings_for_preflight(None, tmp_path)
    assert proof.cpu_threads == 1 and proof.conversion_timeout_seconds == 180
    with pytest.raises(ValueError):
        prediction_settings_for_preflight({}, tmp_path)


def test_hash_and_configuration_remain_bound_if_file_changes_after_read(tmp_path):
    path = tmp_path / "settings.json"
    original = json.dumps({"prediction": prediction(tmp_path)}).encode()
    path.write_bytes(original)
    raw, receipt_hash = load_preflight_settings(path)
    replacement = prediction(tmp_path)
    replacement["cpu_threads"] = 1
    path.write_text(json.dumps({"prediction": replacement}))
    assert receipt_hash == hashlib.sha256(original).hexdigest()
    assert receipt_hash != hashlib.sha256(path.read_bytes()).hexdigest()
    assert prediction_settings_for_preflight(raw, tmp_path).cpu_threads == 4
