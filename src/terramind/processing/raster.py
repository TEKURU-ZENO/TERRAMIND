"""Raster reading and normalisation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

IGNORE_INDEX = 255

# Sentinel-1 backscatter in dB. Fixed clip range rather than dataset statistics
# so normalisation cannot leak information from test regions.
S1_DB_MIN, S1_DB_MAX = -35.0, 5.0


def read_s1_chip(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (image [2,H,W] float32 in [0,1], valid mask [H,W] bool)."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    valid = np.isfinite(arr).all(axis=0)
    arr = np.nan_to_num(arr, nan=S1_DB_MIN, posinf=S1_DB_MAX, neginf=S1_DB_MIN)
    arr = (np.clip(arr, S1_DB_MIN, S1_DB_MAX) - S1_DB_MIN) / (S1_DB_MAX - S1_DB_MIN)
    return arr, valid


def read_label_chip(path: Path) -> np.ndarray:
    """Return label [H,W] int64 with {0, 1, IGNORE_INDEX}."""
    with rasterio.open(path) as src:
        lab = src.read(1).astype(np.int64)
    lab[(lab != 0) & (lab != 1)] = IGNORE_INDEX
    return lab


def read_s2_chip(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Sentinel-2 L1C chip: (image [13,H,W] float32 in [0,1], valid mask [H,W] bool).

    Values are top-of-atmosphere reflectance scaled by 10000. Pixels that are
    zero in every band are treated as no data.
    """
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    valid = np.isfinite(arr).all(axis=0) & (arr != 0).any(axis=0)
    arr = np.clip(np.nan_to_num(arr, nan=0.0) / 10000.0, 0.0, 1.0)
    return arr, valid
