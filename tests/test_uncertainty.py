import numpy as np
import pytest

from terramind.models import threshold as th
from terramind.uncertainty import calibration as cal


def _calibrated_sample(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2, n)
    t = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(np.float32)
    return z, t


def test_calibrated_predictions_have_low_ece_and_unit_temperature():
    z, t = _calibrated_sample()
    assert cal.summary(z, t)["ece"] < 0.01
    assert cal.fit_temperature(z, t) == pytest.approx(1.0, abs=0.1)


def test_temperature_recovers_overconfidence():
    z, t = _calibrated_sample()
    assert cal.fit_temperature(z * 3, t) == pytest.approx(3.0, rel=0.1)
    assert cal.summary(z * 3, t)["ece"] > cal.summary(z * 3, t, temperature=3.0)["ece"]


def test_abstaining_on_low_confidence_reduces_error():
    z, t = _calibrated_sample()
    s = cal.summary(z, t)
    assert s["error_at_100pct_coverage"] == pytest.approx(1 - s["accuracy"], abs=1e-6)
    assert (
        s["error_at_50pct_coverage"] < s["error_at_80pct_coverage"] < s["error_at_100pct_coverage"]
    )


def test_otsu_separates_two_modes():
    rng = np.random.default_rng(0)
    values = np.clip(
        np.concatenate([rng.normal(0.2, 0.03, 5000), rng.normal(0.6, 0.03, 5000)]), 0, 1
    )
    assert 0.3 < th.otsu_threshold(values) < 0.5


def test_best_global_threshold_on_separable_histograms():
    water, land = np.zeros(th.BINS, np.int64), np.zeros(th.BINS, np.int64)
    water[40:60] = 100
    land[120:200] = 100
    t, iou = th.best_global_threshold(water, land)
    assert iou == pytest.approx(1.0)
    assert 60 / th.BINS <= t <= 120 / th.BINS
