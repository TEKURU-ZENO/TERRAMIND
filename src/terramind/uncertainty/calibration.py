"""Calibration and selective-prediction metrics for binary pixel classifiers.

Everything works on flat arrays: `z` is the water logit (log-odds), `t` is the
0/1 target. Ignored pixels must be removed before calling.

These are pixel-level statistics. Pixels within a chip are strongly correlated,
so do not derive confidence intervals from the pixel count.
"""

from __future__ import annotations

import numpy as np

COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def nll(z: np.ndarray, t: np.ndarray, temperature: float = 1.0) -> float:
    zt = np.clip(z / temperature, -30, 30)
    # log(1 + exp(z)) - t*z, computed stably
    return float(np.mean(np.logaddexp(0.0, zt) - t * zt))


def fit_temperature(z: np.ndarray, t: np.ndarray) -> float:
    """Temperature minimising NLL, by grid search. Fit on validation data only."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 71))
    return float(grid[int(np.argmin([nll(z, t, g) for g in grid]))])


def summary(z: np.ndarray, t: np.ndarray, temperature: float = 1.0, n_bins: int = 15) -> dict:
    p = _sigmoid(z / temperature)
    pred = p >= 0.5
    correct = pred == (t == 1)
    conf = np.where(pred, p, 1.0 - p)

    # Expected calibration error over equal-width confidence bins on [0.5, 1].
    bins = np.minimum(((conf - 0.5) * 2 * n_bins).astype(np.int64), n_bins - 1)
    count = np.bincount(bins, minlength=n_bins).astype(np.float64)
    acc = np.bincount(bins, weights=correct, minlength=n_bins)
    avg_conf = np.bincount(bins, weights=conf, minlength=n_bins)
    nonempty = count > 0
    ece = float(np.sum(np.abs(acc[nonempty] - avg_conf[nonempty])) / len(z))

    # Risk-coverage: keep the most confident fraction, report error rate on it.
    fine = 1000
    fbins = np.minimum(((conf - 0.5) * 2 * fine).astype(np.int64), fine - 1)
    fcount = np.bincount(fbins, minlength=fine)[::-1].cumsum()
    ferr = np.bincount(fbins, weights=~correct, minlength=fine)[::-1].cumsum()
    coverage = fcount / len(z)
    risk = ferr / np.maximum(fcount, 1)
    selective = {}
    for c in COVERAGES:
        i = min(int(np.searchsorted(coverage, c - 1e-9)), fine - 1)
        selective[f"error_at_{int(c * 100)}pct_coverage"] = float(risk[i])
    aurc = float(np.sum(np.diff(coverage, prepend=0.0) * risk))

    return {
        "temperature": temperature,
        "pixels": len(z),
        "accuracy": float(correct.mean()),
        "ece": ece,
        "nll": nll(z, t, temperature),
        "brier": float(np.mean((p - t) ** 2)),
        "aurc": aurc,
        **selective,
    }
