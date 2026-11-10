"""Evaluate non-learned VH-threshold baselines on Sen1Floods11.

Usage: python experiments/baselines/threshold_baseline.py

- otsu-per-chip: Otsu threshold computed on each chip separately.
- global-threshold: one threshold chosen to maximise water IoU on the training
  split, then applied unchanged to valid, test and Bolivia.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from terramind.ingestion import sen1floods11 as s1f11
from terramind.models import threshold as th
from terramind.processing.raster import IGNORE_INDEX, read_label_chip, read_s1_chip
from terramind.training.metrics import Confusion


def load(s1_path: Path, label_path: Path):
    img, valid = read_s1_chip(s1_path)
    label = read_label_chip(label_path)
    label[~valid] = IGNORE_INDEX
    return img[1], label  # VH


def main() -> None:
    water, land = np.zeros(th.BINS, np.int64), np.zeros(th.BINS, np.int64)
    for s1_path, label_path in s1f11.available_pairs("train"):
        w, l = th.vh_histograms(*load(s1_path, label_path))
        water, land = water + w, land + l
    global_t, train_iou = th.best_global_threshold(water, land)

    report = {"global_threshold_normalised_vh": global_t, "train_iou": train_iou, "results": {}}
    for split in ("valid", "test", "bolivia"):
        conf = {m: defaultdict(Confusion) for m in ("otsu-per-chip", "global-threshold")}
        for s1_path, label_path in s1f11.available_pairs(split):
            vh, label = load(s1_path, label_path)
            event = s1f11.event_of(s1_path.name)
            valid_vh = vh[label != IGNORE_INDEX]
            otsu_t = th.otsu_threshold(valid_vh) if valid_vh.size else 0.0
            conf["otsu-per-chip"][event].update((vh < otsu_t).astype(np.int64), label)
            conf["global-threshold"][event].update((vh < global_t).astype(np.int64), label)
        for method, per_event in conf.items():
            pooled = sum(per_event.values(), Confusion()).summary()
            report["results"].setdefault(method, {})[split] = {
                "pooled": pooled,
                "per_event": {k: v.summary() for k, v in sorted(per_event.items())},
            }
            print(
                f"{method:<17}{split:<9} IoU {pooled['iou']:.3f}  F1 {pooled['f1']:.3f}  "
                f"P {pooled['precision']:.3f}  R {pooled['recall']:.3f}"
            )

    out = Path(__file__).parent / "runs" / "threshold-baselines.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"global threshold {global_t:.3f} (train IoU {train_iou:.3f}) -> {out}")


if __name__ == "__main__":
    main()
