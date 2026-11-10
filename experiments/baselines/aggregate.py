"""Aggregate baseline runs across seeds: mean, sample std and range.

Usage: python experiments/baselines/aggregate.py [--modality s1|s2|s1s2|all] [--epochs 30] [--events]

Only runs trained on the full training set with the given epoch count are
included, one per seed (latest wins), so smoke runs are ignored.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from statistics import mean, stdev

from terramind.training.runs import select_runs
from terramind.training.train_baseline import CHANNELS


def aggregate(modality: str, epochs: int, events: bool) -> None:
    by_seed = select_runs(modality, epochs)
    print(f"\n== {modality}: {len(by_seed)} seeds {list(by_seed)}")
    if not by_seed:
        return
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _, report in by_seed.values():
        for split, result in report["results"].items():
            for metric in ("iou", "f1", "precision", "recall"):
                values[(split, metric)].append(result["pooled"][metric])
            if events:
                for event, m in result["per_event"].items():
                    if m["iou"] is not None:
                        values[(f"{split}/{event}", "iou")].append(m["iou"])

    print(f"{'split':<20}{'metric':<11}{'mean':>7}{'std':>7}{'min':>7}{'max':>7}")
    for (split, metric), xs in values.items():
        sd = stdev(xs) if len(xs) > 1 else float("nan")
        print(f"{split:<20}{metric:<11}{mean(xs):>7.3f}{sd:>7.3f}{min(xs):>7.3f}{max(xs):>7.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", default="all", choices=[*CHANNELS, "all"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--events", action="store_true", help="include per-event IoU")
    args = parser.parse_args()
    for modality in CHANNELS if args.modality == "all" else [args.modality]:
        aggregate(modality, args.epochs, args.events)


if __name__ == "__main__":
    main()
