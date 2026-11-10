"""Leave-one-event-out transfer summary.

Usage: python experiments/transfer/loeo_summary.py [--modalities s1 s1s2]

For each flood event, compares water IoU on that event's official test-split
chips when the event was (a) part of training ("seen": mean over the standard
seeds) and (b) entirely held out ("unseen": leave-one-event-out). Both columns
score the same chips. Bolivia is never in standard training, so its "seen"
column is already an unseen-region number trained on fewer chips.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from statistics import mean

from terramind.training.runs import select_loeo_runs, select_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", nargs="+", default=["s1", "s1s2"])
    args = parser.parse_args()

    for modality in args.modalities:
        seen: dict[str, list[float]] = defaultdict(list)
        for _, report in select_runs(modality).values():
            for split in ("test", "bolivia"):
                for event, m in report["results"].get(split, {}).get("per_event", {}).items():
                    if m["iou"] is not None:
                        seen[event].append(m["iou"])
        unseen: dict[str, list[float]] = defaultdict(list)
        unseen_all: dict[str, list[float]] = defaultdict(list)
        for (event, _), report in select_loeo_runs(modality).items():
            r = report["results"]
            if r.get("holdout_test_chips", {}).get("pooled", {}).get("iou") is not None:
                unseen[event].append(r["holdout_test_chips"]["pooled"]["iou"])
            unseen_all[event].append(r["holdout_all"]["pooled"]["iou"])

        print(f"\n== {modality}: {len(unseen)} of {len(seen)} events held out so far")
        print(f"{'event':<12}{'seen':>8}{'unseen':>8}{'drop':>8}{'unseen, all chips':>20}")
        drops = []
        for event in sorted(seen):
            s = mean(seen[event])
            if event not in unseen:
                print(f"{event:<12}{s:>8.3f}{'-':>8}{'-':>8}{'-':>20}")
                continue
            u = mean(unseen[event])
            if event != "Bolivia":
                drops.append(s - u)
            print(f"{event:<12}{s:>8.3f}{u:>8.3f}{s - u:>8.3f}{mean(unseen_all[event]):>20.3f}")
        if drops:
            print(
                f"mean drop over {len(drops)} events (excluding Bolivia): {mean(drops):.3f}; "
                f"worst {max(drops):.3f}; events that got worse: {sum(d > 0 for d in drops)}"
            )


if __name__ == "__main__":
    main()
