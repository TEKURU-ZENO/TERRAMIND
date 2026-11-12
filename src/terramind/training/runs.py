"""Locate finished baseline runs."""

from __future__ import annotations

import json
from pathlib import Path

from terramind.config import repo_root

FULL_TRAIN_CHIPS = 252


def runs_dir() -> Path:
    return repo_root() / "experiments" / "baselines" / "runs"


def select_runs(
    modality: str = "s1", epochs: int = 30, variant: str | None = None
) -> dict[int, tuple[Path, dict]]:
    """Full-training-set runs for a modality, one per seed (latest wins).

    Returns {seed: (run_dir, report)}. Smoke runs on partial data are excluded, and
    so are training variants (such as modality dropout) unless `variant` names one.
    """
    by_seed = {}
    for path in sorted(runs_dir().glob("*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            "train_chips" in report
            and report.get("holdout_event") is None
            and report.get("modality", "s1") == modality
            and report.get("variant") == variant
            and report["train_chips"] == FULL_TRAIN_CHIPS
            and report["config"]["epochs"] == epochs
        ):
            by_seed[report["config"]["seed"]] = (path.parent, report)
    return dict(sorted(by_seed.items()))


def select_loeo_runs(modality: str = "s1", epochs: int = 30) -> dict[tuple[str, int], dict]:
    """Leave-one-event-out reports keyed by (held-out event, seed); latest wins."""
    found = {}
    for path in sorted(runs_dir().glob("loeo-*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["modality"] == modality and report["config"]["epochs"] == epochs:
            found[(report["holdout_event"], report["config"]["seed"])] = report
    return dict(sorted(found.items()))
