"""E1 completed: late fusion, modality dropout, and fused models with a sensor missing.

All scored on pixels valid in both sensors (common mask), as in E1. Compared:

- radar and optical ensembles alone
- late fusion: a weighted mean of the two ensembles' water probabilities, the
  weight chosen on the validation split
- early fusion (15-channel U-Net), as trained in E1
- early fusion trained with modality dropout (one sensor hidden at random)
- each fused model with the optical bands set to zero ("radar only", the
  cloud-cover case) and with the radar bands set to zero ("optical only")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from terramind.ingestion import sen1floods11 as s1f11
from terramind.models.unet import UNet
from terramind.training.eval_ensemble import _member_probs
from terramind.training.metrics import Confusion
from terramind.training.runs import runs_dir, select_runs
from terramind.training.train_baseline import CHANNELS
from terramind.uncertainty import calibration as cal

SPLITS = ("valid", "test", "bolivia")


def _models(modality, variant, device):
    out = []
    for run_dir, _ in select_runs(modality, variant=variant).values():
        m = UNet(in_channels=CHANNELS[modality]).to(device).eval()
        m.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        out.append(m)
    return out


def _iou(p, t):
    c = Confusion()
    c.update((p >= 0.5).astype(np.int64), t.astype(np.int64))
    return c.summary()["iou"]


def _ece(p, t):
    z = np.log(np.clip(p, 1e-7, 1 - 1e-7)) - np.log1p(-np.clip(p, 1e-7, 1 - 1e-7))
    return cal.summary(z, t)["ece"]


def run() -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets = {
        "radar": ("s1", None, None),
        "optical": ("s2", None, None),
        "early fusion": ("s1s2", None, None),
        "early fusion, radar only": ("s1s2", None, "s2"),
        "early fusion, optical only": ("s1s2", None, "s1"),
        "early fusion + dropout": ("s1s2", "modality-dropout", None),
        "early fusion + dropout, radar only": ("s1s2", "modality-dropout", "s2"),
        "early fusion + dropout, optical only": ("s1s2", "modality-dropout", "s1"),
    }
    loaded = {}
    probs: dict[str, dict[str, np.ndarray]] = {}
    targets: dict[str, np.ndarray] = {}
    for name, (modality, variant, zero) in sets.items():
        key = (modality, variant)
        if key not in loaded:
            loaded[key] = _models(modality, variant, device)
        if not loaded[key]:
            continue
        probs[name] = {}
        for split in SPLITS:
            p, t, _ = _member_probs(
                loaded[key], s1f11.available_pairs(split), modality, device, True, zero
            )
            probs[name][split] = p.mean(0)
            if split in targets and len(targets[split]) != len(t):
                raise RuntimeError("pixel sets differ between models; check the common mask")
            targets[split] = t
    # Late fusion weight on optical, chosen on validation.
    grid = np.linspace(0, 1, 21)
    v_iou = [
        _iou(w * probs["optical"]["valid"] + (1 - w) * probs["radar"]["valid"], targets["valid"])
        for w in grid
    ]
    w_best = float(grid[int(np.argmax(v_iou))])
    for w, label in (
        (0.5, "late fusion (equal weights)"),
        (w_best, "late fusion (weight tuned on valid)"),
    ):
        probs[label] = {s: w * probs["optical"][s] + (1 - w) * probs["radar"][s] for s in SPLITS}
    report = {
        "pixels": "valid in both sensors",
        "late_fusion_optical_weight": w_best,
        "seeds": {f"{m}/{v}": len(ms) for (m, v), ms in loaded.items()},
        "results": {
            name: {
                split: {
                    "iou": _iou(p[split], targets[split]),
                    "ece": _ece(p[split], targets[split]),
                }
                for split in ("test", "bolivia")
            }
            for name, p in probs.items()
        },
    }
    out = runs_dir() / "fusion-eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
