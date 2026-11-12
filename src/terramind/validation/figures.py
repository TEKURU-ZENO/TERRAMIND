"""Static figures for the paper, drawn from `results.collect()`.

Colours are the first three slots of a palette checked for colour-vision
separation; each series also has its own marker shape, and every figure has a
table in docs/results.md, so nothing depends on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from terramind.config import repo_root
from terramind.validation.results import MODALITIES

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {"s1": ("#2a78d6", "o"), "s2": ("#eb6834", "s"), "s1s2": ("#1baf7a", "^")}
SERIES = list(STYLE.values())


def out_dir() -> Path:
    return repo_root() / "paper" / "figures"


def _axes(ax, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)


def _save(fig, name: str) -> Path:
    path = out_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def per_event(r: dict) -> Path:
    seg = r["segmentation"]
    ref = seg["s2"]["splits"]["test"]["per_event"]
    events = sorted(ref, key=lambda e: ref[e]["mean"])
    rows = [("test", e) for e in events] + [("bolivia", "Bolivia")]
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE)
    for i, (modality, (colour, marker)) in enumerate(STYLE.items()):
        for y, (split, event) in enumerate(rows):
            m = seg[modality]["splits"][split]["per_event"][event]
            offset = y + (i - 1) * 0.22
            ax.plot([m["min"], m["max"]], [offset, offset], color=colour, linewidth=2)
            ax.plot(
                m["mean"], offset, marker, color=colour, markersize=6, markeredgecolor=SURFACE,
                label=MODALITIES[modality] if y == 0 else None,
            )  # fmt: skip
    ax.axhline(len(events) - 0.5, color=GRID, linewidth=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([e if s == "test" else "Bolivia (unseen region)" for s, e in rows])
    ax.set_xlim(0, 1)
    _axes(ax, "Water IoU (marker: mean of 5 seeds; line: seed range)")
    ax.set_title("Flood mapping accuracy by event", loc="left", color=INK, fontsize=11, pad=26)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=3,
              labelcolor=INK, columnspacing=1.4, handletextpad=0.3)  # fmt: skip
    return _save(fig, "fig1-per-event-iou.png")


def transfer(r: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 4.2), sharey=True, facecolor=SURFACE)
    order = sorted(r["transfer"]["s1"]["events"])
    for ax, (modality, (colour, marker)) in zip(axes, STYLE.items()):
        rows = r["transfer"][modality]["events"]
        for y, event in enumerate(order):
            m = rows[event]
            ax.plot([m["seen"], m["unseen"]], [y, y], color=colour, linewidth=2)
            ax.plot(m["seen"], y, marker, color=SURFACE, markeredgecolor=colour, markersize=7,
                    markeredgewidth=1.6, label="Standard model" if y == 0 else None)  # fmt: skip
            ax.plot(m["unseen"], y, marker, color=colour, markersize=7, markeredgecolor=SURFACE,
                    label="Event held out of training" if y == 0 else None)  # fmt: skip
        ax.set_xlim(0, 1)
        _axes(ax, "Water IoU on the event's test chips")
        ax.set_title(MODALITIES[modality], loc="left", color=INK, fontsize=10)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.07), labelcolor=INK)  # fmt: skip
    fig.suptitle("Holding an event out of training barely changes its score",
                 x=0.07, ha="left", color=INK, fontsize=11)  # fmt: skip
    return _save(fig, "fig2-leave-one-event-out.png")


def policies(r: dict) -> Path:
    names = {
        "blind-lookahead": "Lookahead, ignoring the model",
        "greedy": "Greedy",
        "ppo-nounc": "PPO, no uncertainty input",
        "ppo": "PPO",
        "assignment": "Assignment",
        "lookahead": "Lookahead",
        "lookahead+uncertainty": "Lookahead + uncertainty",
        "oracle-lookahead": "Lookahead on the truth",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, facecolor=SURFACE)
    colour, marker = SERIES[0]
    for ax, split in zip(axes, ("test", "bolivia")):
        res = r["e7"]["results"][split]["policies"]
        for y, key in enumerate(names):
            m = res[key]
            ax.plot(m["auc_ci95"], [y, y], color=colour, linewidth=2)
            ax.plot(m["auc"], y, marker, color=colour, markersize=6, markeredgecolor=SURFACE)
            ax.annotate(f"{m['auc']:.3f}", (m["auc_ci95"][1], y), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=8, color=MUTED)  # fmt: skip
        ax.set_xlim(0.2, 0.7)
        _axes(ax, "Mean share of flooded area reached (95% interval)")
        n = res["lookahead"]["scenarios"]
        label = "Test" if split == "test" else "Bolivia (unseen region)"
        ax.set_title(f"{label}, {n} scenarios", loc="left", color=INK, fontsize=10)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names.values())
    fig.suptitle("Simulated flood response: the learned policy does not beat planning",
                 x=0.07, ha="left", color=INK, fontsize=11)  # fmt: skip
    return _save(fig, "fig3-policies-simulated.png")


def chain(r: dict) -> Path:
    stages = [
        "radar ensemble, thresholded map",
        "radar single model",
        "radar ensemble, calibrated",
        "radar ensemble + uncertainty",
        "optical ensemble",
        "fused ensemble",
        "truth (oracle)",
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    for i, ((setting, res), (colour, marker)) in enumerate(
        zip(r["e8"]["settings"].items(), SERIES)
    ):
        for y, stage in enumerate(stages):
            m = res["splits"]["test"]["stages"][stage]
            offset = y + (i - 1) * 0.22
            ax.plot(m["vs_reference_ci95"], [offset, offset], color=colour, linewidth=2)
            ax.plot(m["vs_reference"], offset, marker, color=colour, markersize=6,
                    markeredgecolor=SURFACE, label=setting.capitalize() if y == 0 else None)  # fmt: skip
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([s[0].upper() + s[1:] for s in stages])
    _axes(ax, "Change in score against the radar ensemble (95% interval, test)")
    none = r["e8"]["settings"]["standard"]["splits"]["test"]["stages"]["no perception"]
    ax.set_title("Simulated: what the planner is given matters little once it has a map",
                 loc="left", color=INK, fontsize=11)  # fmt: skip
    ax.text(0.99, 0.02, f"Not shown: no map at all, {none['vs_reference']:+.2f} (standard)",
            transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)  # fmt: skip
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK, title="Setting",
              title_fontsize=9)  # fmt: skip
    return _save(fig, "fig4-chain-ablation-simulated.png")


def dispatch(r: dict) -> Path:
    """E9: utility as the cost of a wasted deployment rises. Paired differences are in
    docs/results.md; the lines show levels, and the truth is a dashed reference."""
    e9 = r["e9"]
    costs = sorted(e9["settings"], key=float)
    x = [float(c) for c in costs]
    series = [
        ("raw map", "Map at face value", "#2a78d6", "o"),
        ("calibrated", "Calibrated", "#eb6834", "s"),
        ("calibrated + uncertainty", "Calibrated + uncertainty", "#1baf7a", "^"),
        ("optical, calibrated", "Optical model, calibrated", "#eda100", "D"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), sharey=True, facecolor=SURFACE)
    for ax, split in zip(axes, ("test", "bolivia")):
        rows = [e9["settings"][c]["splits"][split]["policies"] for c in costs]
        truth = [p["truth (oracle)"]["utility"] for p in rows]
        ax.plot(x, truth, color=MUTED, linewidth=1.2, linestyle="--",
                label="Truth (oracle)")  # fmt: skip
        for key, label, colour, marker in series:
            ax.plot(x, [p[key]["utility"] for p in rows], color=colour, linewidth=2,
                    marker=marker, markersize=7, markeredgecolor=SURFACE, label=label)  # fmt: skip
        for i, p in enumerate(rows):
            if "ppo" in p:
                ax.plot(x[i], p["ppo"]["utility"], "X", color="#e87ba4", markersize=9,
                        markeredgecolor=SURFACE, label="PPO (3 seeds)")  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:g}" for c in x])
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0, labelsize=9)
        ax.set_xlabel("Cost of a deployment to a dry cell (flooded cells)", color=MUTED,
                      fontsize=9)  # fmt: skip
        n = e9["settings"][costs[0]]["splits"][split]["scenarios"]
        name = "Test" if split == "test" else "Bolivia (unseen region)"
        ax.set_title(f"{name}, {n} scenarios", loc="left", color=INK, fontsize=10)
    axes[0].set_ylabel("Utility", color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.13), labelcolor=INK)  # fmt: skip
    fig.suptitle("Simulated dispatch: calibration pays off in seen regions, not in Bolivia",
                 x=0.07, ha="left", color=INK, fontsize=11)  # fmt: skip
    return _save(fig, "fig5-dispatch-simulated.png")


def build(r: dict) -> list[Path]:
    made = [per_event(r)]
    if len(r["transfer"]) == 3:
        made.append(transfer(r))
    if r["e7"]:
        made.append(policies(r))
    if r["e8"]:
        made.append(chain(r))
    if r.get("e9"):
        made.append(dispatch(r))
    return made
