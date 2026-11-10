"""Train and evaluate U-Net flood baselines on Sen1Floods11 (SAR, optical or both)."""

from __future__ import annotations

import json
import random
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from terramind.config import repo_root
from terramind.ingestion import sen1floods11 as s1f11
from terramind.models.unet import UNet
from terramind.processing.raster import (
    IGNORE_INDEX,
    read_label_chip,
    read_s1_chip,
    read_s2_chip,
)
from terramind.training.metrics import Confusion

# Input channels per modality. "s1s2" is early fusion: channels concatenated.
CHANNELS = {"s1": 2, "s2": 13, "s1s2": 15}


def load_chip(s1_path: Path, label_path: Path, modality: str, common_mask: bool = False):
    """Return (image [C,H,W], label [H,W]) with unusable pixels set to IGNORE_INDEX.

    By default a pixel is usable when every modality the model reads is valid.
    `common_mask` requires both sensors to be valid whatever the modality, so
    that models with different inputs are scored on exactly the same pixels.
    """
    lab = read_label_chip(label_path)
    parts = []
    if "s1" in modality or common_mask:
        img, valid = read_s1_chip(s1_path)
        lab[~valid] = IGNORE_INDEX
        if "s1" in modality:
            parts.append(img)
    if "s2" in modality or common_mask:
        img, valid = read_s2_chip(s1f11.s2_path_for(s1_path))
        lab[~valid] = IGNORE_INDEX
        if "s2" in modality:
            parts.append(img)
    return np.concatenate(parts, axis=0), lab


# Decoded chips are kept in memory (float16 image, uint8 label) because reading
# 15-band GeoTIFFs every epoch dominates training time. About 3.5 GB for the
# full fused dataset.
_CHIP_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def cached_chip(s1_path: Path, label_path: Path, modality: str, common_mask: bool):
    key = (s1_path, modality, common_mask)
    if key not in _CHIP_CACHE:
        img, lab = load_chip(s1_path, label_path, modality, common_mask)
        _CHIP_CACHE[key] = (img.astype(np.float16), lab.astype(np.uint8))
    img, lab = _CHIP_CACHE[key]
    return img.astype(np.float32), lab.astype(np.int64)


class FloodChips(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        crop: int | None,
        augment: bool,
        modality: str = "s1",
        common_mask: bool = False,
    ):
        self.pairs, self.crop, self.augment = pairs, crop, augment
        self.modality, self.common_mask = modality, common_mask

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        s1_path, label_path = self.pairs[i]
        img, lab = cached_chip(s1_path, label_path, self.modality, self.common_mask)
        if self.crop:
            h, w = lab.shape
            y, x = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
            img = img[:, y : y + self.crop, x : x + self.crop]
            lab = lab[y : y + self.crop, x : x + self.crop]
        if self.augment:
            if random.random() < 0.5:
                img, lab = img[:, :, ::-1], lab[:, ::-1]
            if random.random() < 0.5:
                img, lab = img[:, ::-1, :], lab[::-1, :]
        return (
            torch.from_numpy(np.ascontiguousarray(img)),
            torch.from_numpy(np.ascontiguousarray(lab)),
            s1f11.event_of(s1_path.name),
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    pairs: list[tuple[Path, Path]],
    device: str,
    modality: str = "s1",
    common_mask: bool = False,
) -> dict:
    """Pooled metrics plus a per-event breakdown."""
    model.eval()
    per_event: dict[str, Confusion] = defaultdict(Confusion)
    dataset = FloodChips(pairs, crop=None, augment=False, modality=modality, common_mask=common_mask)
    loader = DataLoader(dataset, batch_size=4)
    for img, lab, events in loader:
        pred = model(img.to(device)).argmax(1).cpu().numpy()
        for p, t, event in zip(pred, lab.numpy(), events):
            per_event[event].update(p, t)
    pooled = sum(per_event.values(), Confusion())
    return {
        "chips": len(pairs),
        "pooled": pooled.summary(),
        "per_event": {k: v.summary() for k, v in sorted(per_event.items())},
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except OSError:
        return None


def run(
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    crop: int = 256,
    seed: int = 0,
    device: str | None = None,
    modality: str = "s1",
    holdout_event: str | None = None,
) -> Path:
    """Train one model and write report.json.

    With `holdout_event`, every chip of that flood event is removed from
    training and validation (leave-one-event-out). Training then uses the
    train, test and Bolivia chips of the remaining events, checkpoints are
    chosen on the remaining validation chips, and the held-out event is scored
    twice: on all of its chips, and on only its official test-split chips so
    the number is comparable with the standard run's per-event test score.
    """
    if modality not in CHANNELS:
        raise ValueError(f"modality must be one of {sorted(CHANNELS)}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    splits = {name: s1f11.available_pairs(name) for name in s1f11.SPLIT_FILES}
    if not splits["train"]:
        raise RuntimeError("no training chips on disk; run `terramind s1f11-download` first")

    if holdout_event:

        def event(pair) -> str:
            return s1f11.event_of(pair[0].name)

        everything = [p for pairs in splits.values() for p in pairs]
        if holdout_event not in {event(p) for p in everything}:
            raise ValueError(f"unknown event '{holdout_event}'")
        original = splits
        splits = {
            "train": [
                p
                for name in ("train", "test", "bolivia")
                for p in original[name]
                if event(p) != holdout_event
            ],
            "valid": [p for p in original["valid"] if event(p) != holdout_event],
            "holdout_all": [p for p in everything if event(p) == holdout_event],
            "holdout_test_chips": [
                p for p in original["test"] + original["bolivia"] if event(p) == holdout_event
            ],
        }

    if "s2" in modality:
        missing = [p for p, _ in splits["train"] if not s1f11.s2_path_for(p).exists()]
        if missing:
            raise RuntimeError(f"{len(missing)} Sentinel-2 chips missing; run s1f11-download --s2")

    # Optical and fused models are scored only where both sensors are valid, which
    # matches the SAR-only mask to within a few dozen pixels, so modalities compare
    # on the same pixels.
    common = "s2" in modality

    model = UNet(in_channels=CHANNELS[modality]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, -(-len(splits["train"]) // batch_size))
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    loader = DataLoader(
        FloodChips(splits["train"], crop=crop, augment=True, modality=modality),
        batch_size=batch_size,
        shuffle=True,
    )

    history, best_iou, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for img, lab, _ in loader:
            img, lab = img.to(device), lab.to(device)
            if (lab != IGNORE_INDEX).sum() == 0:
                continue
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, enabled=device == "cuda"):
                loss = loss_fn(model(img), lab)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, n = total + loss.item(), n + 1
        entry = {"epoch": epoch, "train_loss": total / max(n, 1)}
        if splits["valid"]:
            entry["valid_iou"] = evaluate(
                model, splits["valid"], device, modality, common
            )["pooled"]["iou"]
            # Model selection uses the validation split only, never test or Bolivia.
            if (entry["valid_iou"] or 0.0) > best_iou:
                best_iou = entry["valid_iou"] or 0.0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(entry)
        print(entry)

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = (
        repo_root()
        / "experiments"
        / "baselines"
        / "runs"
        / datetime.now(UTC).strftime(
            (f"loeo-{holdout_event}-{modality}" if holdout_event else f"unet-{modality}")
            + "-%Y%m%dT%H%M%SZ"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    report = {
        "model": f"unet-{modality} (random init)",
        "modality": modality,
        "eval_common_mask": common,
        "holdout_event": holdout_event,
        "dataset": "sen1floods11 hand-labelled",
        "git_commit": _git_commit(),
        "device": device,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "crop": crop,
            "seed": seed,
        },
        "train_chips": len(splits["train"]),
        "history": history,
        "results": {
            name: evaluate(model, pairs, device, modality, common)
            for name, pairs in splits.items()
            if name != "train" and pairs
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_dir
