"""Deep-ensemble uncertainty evaluation.

Compares single models with the ensemble of all seeds for one modality:
segmentation quality, calibration (ECE, NLL, Brier) and selective prediction
(error rate when the least confident pixels are abstained on). Temperature
scaling is fitted on the validation split only.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.utils.data import DataLoader

from terramind.ingestion import sen1floods11 as s1f11
from terramind.models.unet import UNet
from terramind.processing.raster import IGNORE_INDEX
from terramind.training.metrics import Confusion
from terramind.training.runs import runs_dir, select_runs
from terramind.training.train_baseline import CHANNELS, FloodChips
from terramind.uncertainty import calibration as cal

EVAL_SPLITS = ("valid", "test", "bolivia")


@torch.no_grad()
def _member_probs(models, pairs, modality, device, common_mask):
    """Water probability per member: (probs [M, N] float32, target [N], chip index [N])."""
    dataset = FloodChips(pairs, crop=None, augment=False, modality=modality, common_mask=common_mask)
    probs, targets, chip_ids = [], [], []
    for i, (img, lab, _) in enumerate(DataLoader(dataset, batch_size=1)):
        keep = (lab[0] != IGNORE_INDEX).numpy()
        if not keep.any():
            continue
        x = img.to(device)
        p = torch.stack([m(x).softmax(1)[0, 1] for m in models]).float().cpu().numpy()
        probs.append(p[:, keep])
        targets.append(lab[0].numpy()[keep].astype(np.float32))
        chip_ids.append(np.full(int(keep.sum()), i, dtype=np.int32))
    return np.concatenate(probs, axis=1), np.concatenate(targets), np.concatenate(chip_ids)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def _iou(z: np.ndarray, t: np.ndarray) -> float | None:
    c = Confusion()
    c.update((z >= 0).astype(np.int64), t.astype(np.int64))
    return c.summary()["iou"]


def _per_event_iou(z, t, chip_ids, pairs) -> dict[str, float | None]:
    events = np.array([s1f11.event_of(p.name) for p, _ in pairs])
    per_event: dict[str, Confusion] = defaultdict(Confusion)
    pred = (z >= 0).astype(np.int64)
    for chip in np.unique(chip_ids):
        sel = chip_ids == chip
        per_event[events[chip]].update(pred[sel], t[sel].astype(np.int64))
    return {k: v.summary()["iou"] for k, v in sorted(per_event.items())}


def run(modality: str = "s1", epochs: int = 30, common_mask: bool = False) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = select_runs(modality, epochs)
    if len(selected) < 2:
        raise RuntimeError(f"need at least 2 seeds for modality '{modality}', found {len(selected)}")
    models = []
    for run_dir, _ in selected.values():
        model = UNet(in_channels=CHANNELS[modality]).to(device).eval()
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        models.append(model)

    data = {}
    for split in EVAL_SPLITS:
        pairs = s1f11.available_pairs(split)
        probs, t, chip_ids = _member_probs(models, pairs, modality, device, common_mask)
        data[split] = (probs, t, chip_ids, pairs)

    # Temperatures come from validation only and are then frozen.
    valid_probs, valid_t, _, _ = data["valid"]
    member_temps = [cal.fit_temperature(_logit(p), valid_t) for p in valid_probs]
    ens_temp = cal.fit_temperature(_logit(valid_probs.mean(0)), valid_t)

    report = {
        "modality": modality,
        "common_mask": common_mask,
        "seeds": list(selected),
        "runs": [d.name for d, _ in selected.values()],
        "temperature": {"single_mean": mean(member_temps), "ensemble": ens_temp},
        "note": "Pixel-level statistics; pixels are spatially correlated.",
        "results": {},
    }
    for split, (probs, t, chip_ids, pairs) in data.items():
        z_members = [_logit(p) for p in probs]
        z_ens = _logit(probs.mean(0))

        def avg(summaries: list[dict]) -> dict:
            return {k: mean(s[k] for s in summaries) for k in summaries[0]}

        report["results"][split] = {
            "single_mean": {
                "iou": mean(_iou(z, t) or 0.0 for z in z_members),
                **avg([cal.summary(z, t) for z in z_members]),
            },
            "single_mean_temp_scaled": avg(
                [cal.summary(z, t, temp) for z, temp in zip(z_members, member_temps)]
            ),
            "ensemble": {"iou": _iou(z_ens, t), **cal.summary(z_ens, t)},
            "ensemble_temp_scaled": cal.summary(z_ens, t, ens_temp),
            "ensemble_per_event_iou": _per_event_iou(z_ens, t, chip_ids, pairs),
            # Mean member disagreement (std of water probability) as an epistemic signal.
            "mean_member_std": float(probs.std(0).mean()),
        }

    suffix = "-common" if common_mask else ""
    out = runs_dir() / f"ensemble-{modality}{suffix}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
