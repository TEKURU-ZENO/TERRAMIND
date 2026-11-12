"""Precompute map overlays and a chip index for the Earth Explorer.

For every Sen1Floods11 chip this writes georeferenced PNG layers (SAR, optical,
label, ensemble prediction, errors, uncertainty) and one index.json holding
bounds, split and per-chip metrics. The API only serves these files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image

from terramind.config import data_dir
from terramind.ingestion import sen1floods11 as s1f11
from terramind.models.unet import UNet
from terramind.processing.raster import IGNORE_INDEX, read_s1_chip, read_s2_chip
from terramind.training.metrics import Confusion
from terramind.training.runs import select_runs
from terramind.training.train_baseline import CHANNELS, load_chip

LAYERS = ("sar", "optical", "label", "prediction", "errors", "uncertainty")


def out_dir() -> Path:
    return data_dir() / "derived" / "explorer"


def _save(rgba: np.ndarray, path: Path) -> None:
    Image.fromarray(rgba, mode="RGBA").save(path, optimize=True)


def _stretch(band: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        return np.zeros(band.shape, np.uint8)
    lo, hi = np.percentile(band[valid], [2, 98])
    return (np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def _overlay(shape, rules: list[tuple[np.ndarray, tuple[int, int, int, int]]]) -> np.ndarray:
    rgba = np.zeros((*shape, 4), np.uint8)
    for mask, colour in rules:
        rgba[mask] = colour
    return rgba


@torch.no_grad()
def build(modality: str = "s1", epochs: int = 30) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = select_runs(modality, epochs)
    if not selected:
        raise RuntimeError(f"no finished runs for modality '{modality}'")
    models = []
    for run_dir, _ in selected.values():
        model = UNet(in_channels=CHANNELS[modality]).to(device).eval()
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        models.append(model)

    chips = []
    for split in s1f11.SPLIT_FILES:
        for s1_path, label_path in s1f11.available_pairs(split):
            chip_id = s1_path.name.replace("_S1Hand.tif", "")
            chip_dir = out_dir() / chip_id
            chip_dir.mkdir(parents=True, exist_ok=True)
            with rasterio.open(s1_path) as src:
                b = src.bounds
            s2_path = s1f11.s2_path_for(s1_path)
            has_s2 = s2_path.exists()

            img, label = load_chip(s1_path, label_path, modality, common_mask=has_s2)
            x = torch.from_numpy(img)[None].to(device)
            probs = torch.stack([m(x).softmax(1)[0, 1] for m in models]).float().cpu().numpy()
            prob, std = probs.mean(0), probs.std(0)
            pred = prob >= 0.5
            known = label != IGNORE_INDEX
            water = label == 1

            s1, s1_valid = read_s1_chip(s1_path)
            vv = _stretch(s1[0], s1_valid)
            _save(
                np.dstack([vv, vv, vv, np.where(s1_valid, 255, 0).astype(np.uint8)]),
                chip_dir / "sar.png",
            )
            layers = ["sar", "label", "prediction", "errors", "uncertainty"]
            if has_s2:
                s2, s2_valid = read_s2_chip(s2_path)
                # True colour: B4, B3, B2 with a fixed reflectance stretch.
                rgb = (np.clip(s2[[3, 2, 1]] / 0.3, 0, 1) ** 0.8 * 255).astype(np.uint8)
                alpha = np.where(s2_valid, 255, 0).astype(np.uint8)
                _save(np.dstack([*rgb, alpha]), chip_dir / "optical.png")
                layers.insert(1, "optical")

            _save(
                _overlay(
                    label.shape, [(water, (56, 189, 248, 220)), (~known, (148, 163, 184, 110))]
                ),
                chip_dir / "label.png",
            )
            _save(_overlay(label.shape, [(pred, (250, 204, 21, 220))]), chip_dir / "prediction.png")
            _save(
                _overlay(
                    label.shape,
                    [
                        (known & pred & water, (255, 255, 255, 90)),
                        (known & pred & ~water, (239, 68, 68, 230)),  # false positive
                        (known & ~pred & water, (59, 130, 246, 230)),  # false negative
                    ],
                ),
                chip_dir / "errors.png",
            )
            unc = np.zeros((*label.shape, 4), np.uint8)
            unc[..., :3] = (217, 70, 239)
            unc[..., 3] = (np.clip(std / 0.25, 0, 1) * 255).astype(np.uint8)
            _save(unc, chip_dir / "uncertainty.png")

            conf = Confusion()
            conf.update(pred.astype(np.int64), label)
            m = conf.summary()
            chips.append(
                {
                    "id": chip_id,
                    "event": s1f11.event_of(s1_path.name),
                    "split": split,
                    "bounds": [b.left, b.bottom, b.right, b.top],
                    "iou": m["iou"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "water_fraction": float(water.sum() / max(known.sum(), 1)),
                    "labelled_fraction": float(known.mean()),
                    "mean_uncertainty": float(std[known].mean()) if known.any() else None,
                    "layers": layers,
                }
            )

    index = {
        "model": {
            "name": f"unet-{modality} ensemble",
            "modality": modality,
            "seeds": list(selected),
            "runs": [d.name for d, _ in selected.values()],
        },
        "note": "Chips in the train split were seen during training; their scores are in-sample.",
        "chips": chips,
    }
    path = out_dir() / "index.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    return path
