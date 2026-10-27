"""Decimal MB boundaries and unit-independent relative time comparisons."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from localml_scheduler.prediction.released_outputs import (
    MIB_BYTES, VRAM_BUCKET_MIB, ceil_vram_mib, deployment_prediction,
    relative_time_costs,
)


@pytest.mark.parametrize("raw_mb,expected_mb", [
    (0, 100), (0.1, 100), (99.99, 100), (100, 100), (100.000001, 200),
    (470, 500), (500, 500), (500.000001, 600), (24000, 24000),
])
def test_decimal_100mb_ceiling(raw_mb, expected_mb):
    actual = ceil_vram_mib(raw_mb * 1e6 / MIB_BYTES)
    assert actual * MIB_BYTES / 1e6 == expected_mb
    assert ceil_vram_mib(actual) == actual


def test_mib_is_not_decimal_mb():
    assert ceil_vram_mib(100) == 200e6 / MIB_BYTES
    assert ceil_vram_mib(VRAM_BUCKET_MIB) == VRAM_BUCKET_MIB


@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), -float("inf")])
def test_invalid_vram_not_silently_replaced(invalid):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ceil_vram_mib(invalid)


def test_deployment_rounding_preserves_raw_and_time_busy():
    raw = [12.1, 11.8, 96.8, 15.3, 470e6 / MIB_BYTES, 500.000001e6 / MIB_BYTES]
    original = raw.copy()
    rounded = deployment_prediction(raw)
    assert raw == original
    assert rounded[:4] == raw[:4]
    assert rounded[4:] == [500e6 / MIB_BYTES, 600e6 / MIB_BYTES]


def test_relative_time_ratios_independent_of_units_and_input_order():
    milliseconds = {"slower": 113, "fastest": 100, "tie": 100}
    expected = {"slower": 1.13, "fastest": 1, "tie": 1}
    assert relative_time_costs(milliseconds) == pytest.approx(expected)
    seconds = {name: value / 1000 for name, value in reversed(list(milliseconds.items()))}
    assert relative_time_costs(seconds) == pytest.approx(expected)
    assert relative_time_costs({"only": 50}) == {"only": 1}


@pytest.mark.parametrize("values", [{}, {"x": 0}, {"x": -1}, {"x": float("nan")}, {"x": float("inf")}])
def test_relative_time_rejects_missing_or_invalid_duration(values):
    with pytest.raises(ValueError):
        relative_time_costs(values)


def test_rounding_matches_public_release_implementation():
    root = os.environ.get("PERFSEER_RELEASE_ROOT")
    if not root:
        pytest.skip("set PERFSEER_RELEASE_ROOT to verify public objective parity")
    spec = importlib.util.spec_from_file_location("release_objectives", Path(root) / "portable/v5_objectives.py")
    objectives = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(objectives)
    rng = np.random.default_rng(53)
    raw = rng.uniform(1, 30000, (100, 6))
    raw[:, 3] = 50
    raw[0, 4:] = [470e6 / MIB_BYTES, 500e6 / MIB_BYTES]
    actual = np.asarray([deployment_prediction(row.tolist()) for row in raw])
    np.testing.assert_array_equal(actual, objectives.quantize_predictions(raw))
