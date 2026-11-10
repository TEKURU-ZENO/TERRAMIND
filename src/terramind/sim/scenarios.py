"""Turn benchmark chips into simulation scenarios.

Each 512x512 chip becomes a GRID x GRID map of cells. Per cell:

- truth: share of the cell's labelled pixels that are water (hidden from policies)
- pred:  mean ensemble water probability over the same labelled pixels
- unc:   mean standard deviation across ensemble members, same pixels
- known: fraction of the cell that has a label at all

Only labelled pixels are used, for the prediction as well as the truth, because
a prediction over unlabelled ground cannot be checked. A cell with no labelled
pixels is empty in all three maps, so the label mask leaks into what the
policies see: they are never tempted by water nobody labelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from terramind.config import data_dir

GRID = 16
MIN_WATER = 0.02  # scenarios with less mean water than this are dropped


def path_for(modality: str) -> Path:
    return data_dir() / "derived" / "sim" / f"scenarios-{modality}.npz"


@dataclass
class Scenarios:
    ids: np.ndarray
    events: np.ndarray
    splits: np.ndarray
    truth: np.ndarray  # [N, G, G]
    pred: np.ndarray
    unc: np.ndarray
    known: np.ndarray
    # Alternative views of the same model, for the chain ablation (E8).
    hard: np.ndarray  # share of pixels the ensemble calls water (0.5 threshold)
    single: np.ndarray  # mean probability of one ensemble member
    calibrated: np.ndarray  # ensemble probability after temperature scaling

    def __len__(self) -> int:
        return len(self.ids)

    def select(self, split: str, min_water: float = MIN_WATER) -> np.ndarray:
        """Indices of scenarios in a split that contain enough water to matter."""
        keep = (self.splits == split) & (self.truth.mean(axis=(1, 2)) >= min_water)
        return np.flatnonzero(keep)


def cells(arr: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Mean of a [H, W] array over a grid x grid partition."""
    h, w = arr.shape
    return arr.reshape(grid, h // grid, grid, w // grid).mean(axis=(1, 3))


def build(modality: str = "s1", epochs: int = 30, grid: int = GRID) -> Path:
    import torch

    from terramind.ingestion import sen1floods11 as s1f11
    from terramind.models.unet import UNet
    from terramind.processing.raster import IGNORE_INDEX, read_label_chip
    from terramind.training.runs import select_runs
    from terramind.training.train_baseline import CHANNELS, load_chip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = select_runs(modality, epochs)
    if not selected:
        raise RuntimeError(f"no finished runs for modality '{modality}'")
    models = []
    for run_dir, _ in selected.values():
        model = UNet(in_channels=CHANNELS[modality]).to(device).eval()
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        models.append(model)

    maps = ("truth", "pred", "unc", "known", "hard", "single", "calibrated")
    out: dict[str, list] = {k: [] for k in ("ids", "events", "splits", *maps)}
    temperature = _temperature(modality)
    with torch.no_grad():
        for split in s1f11.SPLIT_FILES:
            for s1_path, label_path in s1f11.available_pairs(split):
                img, _ = load_chip(s1_path, label_path, modality)
                label = read_label_chip(label_path)
                x = torch.from_numpy(img)[None].to(device)
                probs = torch.stack([m(x).softmax(1)[0, 1] for m in models]).float().cpu().numpy()
                out["ids"].append(s1_path.name.replace("_S1Hand.tif", ""))
                out["events"].append(s1f11.event_of(s1_path.name))
                out["splits"].append(split)
                labelled = (label != IGNORE_INDEX).astype(np.float32)
                known = cells(labelled, grid)
                share = np.maximum(known, 1e-6)
                out["known"].append(known)
                out["truth"].append(cells((label == 1).astype(np.float32), grid) / share)
                out["pred"].append(cells(probs.mean(0) * labelled, grid) / share)
                out["unc"].append(cells(probs.std(0) * labelled, grid) / share)
                mean = probs.mean(0)
                out["hard"].append(cells((mean >= 0.5) * labelled, grid) / share)
                out["single"].append(cells(probs[0] * labelled, grid) / share)
                clipped = np.clip(mean, 1e-6, 1 - 1e-6)
                scaled = 1 / (1 + np.exp(-np.log(clipped / (1 - clipped)) / temperature))
                out["calibrated"].append(cells(scaled * labelled, grid) / share)

    path = path_for(modality)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{k: np.array(v) for k, v in out.items() if k in ("ids", "events", "splits")},
        **{k: np.stack(out[k]).astype(np.float32) for k in maps},
    )
    return path


def _temperature(modality: str) -> float:
    """Ensemble temperature fitted on the validation split by `eval-ensemble`."""
    import json

    from terramind.training.runs import runs_dir

    for name in (f"ensemble-{modality}.json", f"ensemble-{modality}-common.json"):
        path = runs_dir() / name
        if path.exists():
            return float(json.loads(path.read_text(encoding="utf-8"))["temperature"]["ensemble"])
    raise RuntimeError(f"no ensemble report for '{modality}'; run `terramind eval-ensemble`")


def load(modality: str = "s1") -> Scenarios:
    with np.load(path_for(modality)) as z:
        return Scenarios(**{k: z[k] for k in z.files})
