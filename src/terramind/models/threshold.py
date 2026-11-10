"""Non-learned SAR water baselines: threshold the VH band.

Open water is a specular reflector, so it is dark in SAR. Both baselines call a
pixel water when normalised VH backscatter is below a threshold.
"""

from __future__ import annotations

import numpy as np

BINS = 256


def otsu_threshold(values: np.ndarray) -> float:
    """Otsu's threshold for values in [0, 1]."""
    hist = np.bincount(np.minimum((values * BINS).astype(np.int64), BINS - 1), minlength=BINS)
    p = hist / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(BINS))
    denom = omega * (1 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = np.where(denom > 0, (mu[-1] * omega - mu) ** 2 / denom, 0.0)
    return (int(np.argmax(between)) + 1) / BINS


def best_global_threshold(water_hist: np.ndarray, land_hist: np.ndarray) -> tuple[float, float]:
    """Threshold maximising water IoU given VH histograms of water and land pixels.

    Returns (threshold, IoU on the data the histograms came from).
    """
    tp = np.cumsum(water_hist)  # water pixels below each bin edge
    fp = np.cumsum(land_hist)
    fn = water_hist.sum() - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    k = int(np.argmax(iou))
    return (k + 1) / BINS, float(iou[k])


def vh_histograms(vh: np.ndarray, label: np.ndarray, ignore_index: int = 255):
    idx = np.minimum((vh * BINS).astype(np.int64), BINS - 1)
    keep = label != ignore_index
    water = np.bincount(idx[keep & (label == 1)], minlength=BINS)
    land = np.bincount(idx[keep & (label == 0)], minlength=BINS)
    return water, land
