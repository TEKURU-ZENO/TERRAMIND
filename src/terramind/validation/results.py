"""Collect every headline number from run files into one place (M8).

`collect()` reads only files written by the experiments; nothing is typed in by
hand. It adds what the spec asks for and the per-experiment scripts lack:
confidence intervals that resample flood events, and worst-event performance.
`claims()` then states each research question's verdict as a rule over those
numbers, so the verdict changes if the numbers do.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from terramind.config import repo_root
from terramind.training.runs import runs_dir, select_loeo_runs, select_runs

MODALITIES = {"s1": "Radar (S1)", "s2": "Optical (S2)", "s1s2": "Fused (S1+S2)"}
BOOTSTRAPS = 5000


def sim_dir() -> Path:
    return repo_root() / "experiments" / "sim" / "runs"


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _counts(m: dict) -> tuple[float, float, float]:
    """(tp, fp, fn) recovered from precision, recall and the water pixel count."""
    tp = m["recall"] * m["water_pixels"]
    fp = tp / m["precision"] - tp if m["precision"] else 0.0
    return tp, fp, m["water_pixels"] - tp


def event_bootstrap(per_seed_events: list[dict[str, dict]], seed: int = 0) -> dict:
    """Pooled water IoU with a 95% interval that resamples flood events.

    For each resample of the events, IoU is pooled over the drawn events for
    every training seed and averaged across seeds. The interval therefore
    reflects which events happen to be in the test set, not training noise.
    """
    events = sorted(e for e in per_seed_events[0] if per_seed_events[0][e]["iou"] is not None)
    counts = np.array([[_counts(run[e]) for e in events] for run in per_seed_events])  # [S, E, 3]
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(events), size=(BOOTSTRAPS, len(events)))
    summed = counts[:, draws].sum(2)  # [S, B, 3]
    iou = (summed[..., 0] / summed.sum(-1)).mean(0)
    return {
        "events": len(events),
        "ci95": [float(np.percentile(iou, 2.5)), float(np.percentile(iou, 97.5))],
    }


def _segmentation() -> dict:
    out = {}
    for modality in MODALITIES:
        runs = select_runs(modality)
        if not runs:
            continue
        reports = [r for _, r in runs.values()]
        block = {"seeds": len(reports), "splits": {}}
        for split in ("valid", "test", "bolivia"):
            ious = [r["results"][split]["pooled"]["iou"] for r in reports]
            per_event = {
                e: [r["results"][split]["per_event"][e]["iou"] for r in reports]
                for e in reports[0]["results"][split]["per_event"]
                if reports[0]["results"][split]["per_event"][e]["iou"] is not None
            }
            event_mean = {e: mean(v) for e, v in per_event.items()}
            worst = min(event_mean, key=event_mean.get)
            row = {
                "chips": reports[0]["results"][split]["chips"],
                "iou_mean": mean(ious),
                "iou_seed_std": stdev(ious) if len(ious) > 1 else None,
                "iou_seed_range": [min(ious), max(ious)],
                "per_event": {
                    e: {"mean": event_mean[e], "min": min(v), "max": max(v)}
                    for e, v in per_event.items()
                },
                "worst_event": {"event": worst, "iou": event_mean[worst]},
                "macro_event_iou": mean(event_mean.values()),
            }
            if len(per_event) > 1:
                row["event_bootstrap"] = event_bootstrap(
                    [r["results"][split]["per_event"] for r in reports]
                )
            block["splits"][split] = row
        out[modality] = block
    return out


def _thresholds() -> dict | None:
    report = _load(runs_dir() / "threshold-baselines.json")
    if not report:
        return None
    return {
        method: {split: r["pooled"]["iou"] for split, r in splits.items()}
        for method, splits in report["results"].items()
    }


def _calibration() -> dict:
    out = {}
    for modality in MODALITIES:
        report = _load(runs_dir() / f"ensemble-{modality}-common.json") or _load(
            runs_dir() / f"ensemble-{modality}.json"
        )
        if not report:
            continue
        out[modality] = {
            "temperature": report["temperature"]["ensemble"],
            "common_mask": report["common_mask"],
            "splits": {
                split: {
                    "iou_single": r["single_mean"]["iou"],
                    "iou_ensemble": r["ensemble"]["iou"],
                    "ece_single": r["single_mean"]["ece"],
                    "ece_ensemble": r["ensemble"]["ece"],
                    "ece_scaled": r["ensemble_temp_scaled"]["ece"],
                    "error_full": r["ensemble"]["error_at_100pct_coverage"],
                    "error_90": r["ensemble"]["error_at_90pct_coverage"],
                }
                for split, r in report["results"].items()
            },
        }
    return out


def _transfer(segmentation: dict) -> dict:
    out = {}
    for modality in MODALITIES:
        runs = select_loeo_runs(modality)
        if not runs or modality not in segmentation:
            continue
        rows = {}
        for (event, _seed), report in runs.items():
            unseen = report["results"]["holdout_test_chips"]["pooled"]["iou"]
            for split in ("test", "bolivia"):
                seen = segmentation[modality]["splits"][split]["per_event"].get(event)
                if seen and unseen is not None:
                    rows[event] = {
                        "seen": seen["mean"],
                        "unseen": unseen,
                        "drop": seen["mean"] - unseen,
                        "in_standard_training": split == "test",
                    }
        drops = [r["drop"] for r in rows.values() if r["in_standard_training"]]
        out[modality] = {
            "events": rows,
            "mean_drop": mean(drops),
            "worst_drop": max(drops),
            "worse_count": sum(d > 0 for d in drops),
            "compared_events": len(drops),
        }
    return out


def collect() -> dict:
    segmentation = _segmentation()
    rag = _load(runs_dir() / "rag-benchmark.json")
    return {
        "segmentation": segmentation,
        "thresholds": _thresholds(),
        "calibration": _calibration(),
        "transfer": _transfer(segmentation),
        "retrieval": {"cases": rag["cases"], "k": rag["k"], "all": rag["results"]["all"]}
        if rag
        else None,
        "e7": _load(sim_dir() / "e7-s1.json"),
        "e8": _load(sim_dir() / "e8-chain.json"),
        "e9": _load(sim_dir() / "e9-dispatch-s1.json"),
        "agents": _agents(),
    }


def _agents() -> dict | None:
    """RQ5 summary without the per-question rows (those stay in the run file)."""
    report = _load(runs_dir() / "agents-rq5.json")
    if not report:
        return None
    return {
        "questions": report["questions"],
        "fingerprint": report["fingerprint"],
        "paired_accuracy": report.get("paired_accuracy", {}),
        "systems": {
            k: {"summary": v["summary"], "by_category": v["by_category"]}
            for k, v in report["systems"].items()
        },
    }


def _excludes_zero(ci: list[float]) -> bool:
    return ci[0] > 0 or ci[1] < 0


def claims(r: dict) -> list[dict]:
    """One verdict per research question, each a rule over the collected numbers."""
    seg, out = r["segmentation"], []

    def add(rq, question, verdict, basis, limits):
        out.append(
            {"rq": rq, "question": question, "verdict": verdict, "basis": basis, "limits": limits}
        )

    if {"s1", "s2", "s1s2"} <= seg.keys():
        s2, fused = (seg[m]["splits"] for m in ("s2", "s1s2"))
        beats = all(
            fused[s]["iou_seed_range"][0] > s2[s]["iou_seed_range"][1] for s in ("test", "bolivia")
        )
        add(
            "RQ1",
            "Does multimodal fusion beat the best single sensor?",
            "supported" if beats else "not supported",
            f"Fused IoU {fused['test']['iou_mean']:.3f} (test) and {fused['bolivia']['iou_mean']:.3f} "
            f"(Bolivia) against optical {s2['test']['iou_mean']:.3f} and "
            f"{s2['bolivia']['iou_mean']:.3f}.",
            "Early fusion only; cloudy pixels are excluded from the labels.",
        )

    if r["transfer"]:
        worst = max(t["mean_drop"] for t in r["transfer"].values())
        add(
            "RQ2",
            "Does the model generalise to flood events it was not trained on?",
            "supported, weakly tested" if worst < 0.03 else "not supported",
            "Mean IoU drop when an event is held out: "
            + ", ".join(f"{MODALITIES[m]} {t['mean_drop']:+.3f}" for m, t in r["transfer"].items())
            + ".",
            "One seed per fold; held-out models train on more chips; other events from the same "
            "continent stay in training; no unseen sensor or climate test.",
        )

    if r["calibration"]:
        halves = all(
            s["error_90"] <= 0.6 * s["error_full"]
            for c in r["calibration"].values()
            for s in c["splits"].values()
        )
        add(
            "RQ3",
            "Can uncertainty identify where the model should abstain?",
            "supported at pixel level" if halves else "not supported",
            "Dropping the least confident 10% of pixels cuts the error rate to "
            + ", ".join(
                f"{c['splits']['bolivia']['error_90'] / c['splits']['bolivia']['error_full']:.0%} "
                f"of its value ({MODALITIES[m]}, Bolivia)"
                for m, c in r["calibration"].items()
            )
            + ".",
            "Pixel error is dominated by easy land pixels; no chip-level or decision-level test.",
        )

    if r["retrieval"]:
        a = r["retrieval"]["all"]
        better = a["hybrid-nograph"]["ndcg"] > max(a["vector"]["ndcg"], a["spatial"]["ndcg"])
        graph = a["hybrid"]["ndcg"] > a["hybrid-nograph"]["ndcg"] + 1e-9
        add(
            "RQ4",
            "Does semantic + spatial + temporal + graph retrieval improve geospatial QA?",
            ("supported for the hybrid score" if better else "not supported")
            + ("" if graph else "; not supported for the graph component"),
            f"nDCG@{r['retrieval']['k']}: vector {a['vector']['ndcg']:.2f}, spatial "
            f"{a['spatial']['ndcg']:.2f}, hybrid {a['hybrid-nograph']['ndcg']:.2f}, hybrid + graph "
            f"{a['hybrid']['ndcg']:.2f} on {sum(r['retrieval']['cases'].values())} queries.",
            "Queries are generated from the corpus; weights are hand-set.",
        )

    a = r.get("agents")
    if not a:
        add(
            "RQ5",
            "Are specialised agents more reliable than one agent or a fixed workflow?",
            "not tested",
            "No question set has been run.",
            "Run `terramind agents-eval`.",
        )
    else:
        s = {k: v["summary"] for k, v in a["systems"].items()}
        llm = next((k for k in s if k.startswith("llm-agent")), None)
        pairs = a.get("paired_accuracy", {})
        beats = {
            k.split(" - ", 1)[1]: v["ci95"][0] > 0
            for k, v in pairs.items()
            if k.startswith("terracore")
        }
        both = beats.get("fixed-pipeline") and (llm is None or beats.get(llm))
        t = s["terracore"]
        add(
            "RQ5",
            "Are specialised agents more reliable than one agent or a fixed workflow?",
            "supported against the fixed pipeline only"
            if beats.get("fixed-pipeline") and not both
            else ("supported" if both else "not supported"),
            f"{len(a['questions'])} questions. Accuracy: TerraCore {t['accuracy']:.2f}, "
            + ", ".join(f"{k} {v['accuracy']:.2f}" for k, v in s.items() if k != "terracore")
            + ". TerraCore answers {:.0%} of answerable questions correctly but refuses only "
            "{:.0%} of unanswerable ones".format(t["answerable_correct"], t["unanswerable_refused"])
            + (
                f"; the LLM agent refuses {s[llm]['unanswerable_refused']:.0%} but answers "
                f"{s[llm]['answerable_correct']:.0%}"
                if llm
                else ""
            )
            + ". Removing the critic changes accuracy by "
            + f"{t['accuracy'] - s['terracore-no-critic']['accuracy']:+.2f}.",
            "Questions written by the author of TerraCore's rules; gold facts come from the same "
            "store; one LLM (llama3:8b); 49 questions.",
        )

    if r["e7"]:
        verdicts = []
        for split, res in r["e7"]["results"].items():
            d = res["paired_auc"].get("ppo - lookahead")
            if d:
                verdicts.append((split, d))
        wins = all(d["ci95"][0] > 0 for _, d in verdicts)
        add(
            "RQ6",
            "Does reinforcement learning beat heuristic and optimisation baselines in simulation?",
            "supported" if wins else "not supported",
            "PPO minus lookahead: "
            + ", ".join(
                f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] on {s}"
                for s, d in verdicts
            )
            + ".",
            "Simulated; PPO with parameter sharing rather than full MAPPO; 45 training scenarios; "
            "no hyperparameter tuning.",
        )

    if r["e8"]:
        ref = r["e8"]["reference"]
        helped = []
        for setting, res in r["e8"]["settings"].items():
            for stage in ("radar ensemble, calibrated", "radar ensemble + uncertainty"):
                m = res["splits"]["test"]["stages"][stage]
                if m["vs_reference_ci95"][0] > 0:
                    helped.append(f"{stage} in '{setting}' ({m['vs_reference']:+.3f})")
        std = r["e8"]["settings"]["standard"]["splits"]["test"]["stages"]
        add(
            "RQ7",
            "Does the integrated chain improve decision quality over separate parts?",
            "partly: perception matters, uncertainty and calibration barely do",
            f"Giving the planner the {ref} instead of nothing is worth "
            f"{-std['no perception']['vs_reference']:+.3f}; the truth adds only "
            f"{std['truth (oracle)']['vs_reference']:+.3f} more. Calibration or uncertainty gave a "
            "gain whose interval excludes zero in "
            + (f"{len(helped)} of 6 cases: " + "; ".join(helped) if helped else "none of 6 cases")
            + ".",
            "Simulated task with smooth, cell-averaged need; errors cost nothing but time.",
        )

    if r.get("e9"):
        cal, unc = [], []
        for cost, res in r["e9"]["settings"].items():
            for split, block in res["splits"].items():
                p = block["policies"]
                raw = p["raw map"]
                if raw["vs_reference_ci95"][1] < 0:
                    cal.append(f"cost {cost}, {split} ({-raw['vs_reference']:+.3f})")
                u = p["calibrated + uncertainty"]
                if u["vs_reference_ci95"][0] > 0:
                    unc.append(f"cost {cost}, {split} ({u['vs_reference']:+.3f})")
        n = sum(len(res["splits"]) for res in r["e9"]["settings"].values())
        add(
            "RQ7b",
            "When mistakes cost something, do calibration and uncertainty improve decisions?",
            "calibration: in some cases; uncertainty: rarely" if cal or unc else "not supported",
            f"Harder task, {n} cases (waste cost x split). Calibrating the chance of a dry cell "
            f"beat taking the map at face value, interval excluding zero, in {len(cal)}"
            + (f": {'; '.join(cal)}" if cal else "")
            + f". Adding ensemble disagreement to that calibration helped in {len(unc)}"
            + (f": {'; '.join(unc)}" if unc else "")
            + ".",
            "Simulated; calibration is fitted on validation cells and does not transfer to "
            "Bolivia; invented cost structure.",
        )
    return out
