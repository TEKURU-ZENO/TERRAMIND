"""Binary segmentation metrics for the water class, with an ignore label."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, pred: np.ndarray, target: np.ndarray, ignore_index: int = 255) -> None:
        keep = target != ignore_index
        p, t = pred[keep] == 1, target[keep] == 1
        self.tp += int((p & t).sum())
        self.fp += int((p & ~t).sum())
        self.fn += int((~p & t).sum())
        self.tn += int((~p & ~t).sum())

    def __add__(self, other: Confusion) -> Confusion:
        return Confusion(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn
        )

    def summary(self) -> dict[str, float | int | None]:
        """Metrics for the water class. None where the denominator is zero."""

        def div(a: int, b: int) -> float | None:
            return a / b if b else None

        return {
            "iou": div(self.tp, self.tp + self.fp + self.fn),
            "f1": div(2 * self.tp, 2 * self.tp + self.fp + self.fn),
            "precision": div(self.tp, self.tp + self.fp),
            "recall": div(self.tp, self.tp + self.fn),
            "accuracy": div(self.tp + self.tn, self.tp + self.tn + self.fp + self.fn),
            "water_pixels": self.tp + self.fn,
            "valid_pixels": self.tp + self.tn + self.fp + self.fn,
        }
