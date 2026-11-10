"""Evidence records: the only facts agents are allowed to state.

Every record carries where it came from, where and when it applies, and a
provenance score. The corpus is rebuilt from files already in the repository
(run reports, chip index, STAC manifests, registries, docs); nothing here is
written by a language model.

Provenance scores are hand-set placeholders, not calibrated:
  1.0  computed here from versioned runs or read straight from source rasters
  0.9  returned by an external catalogue (STAC)
  0.7  project documents
  0.5  registry entries whose licence text is unverified
  0.4  candidate events whose dates and boxes are unconfirmed
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev

from terramind import registry
from terramind.config import data_dir, repo_root
from terramind.training.runs import runs_dir, select_loeo_runs, select_runs

MODALITY_NAMES = {
    "s1": "radar (Sentinel-1)",
    "s2": "optical (Sentinel-2)",
    "s1s2": "radar + optical early fusion",
}


@dataclass
class Evidence:
    id: str
    kind: str  # chip | scene | event | benchmark | simulation | dataset | document
    title: str
    text: str
    source: dict
    provenance: float
    bbox: list[float] | None = None  # west, south, east, north (EPSG:4326)
    start: str | None = None  # ISO date
    end: str | None = None
    links: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)


def corpus_path() -> Path:
    return data_dir() / "derived" / "evidence" / "corpus.jsonl"


def _union(boxes: list[list[float]]) -> list[float]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _pm(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{mean(values):.3f} ± {stdev(values):.3f}"


def _chips() -> tuple[list[Evidence], dict[str, list[dict]]]:
    index_path = data_dir() / "derived" / "explorer" / "index.json"
    if not index_path.exists():
        return [], {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    by_event: dict[str, list[dict]] = defaultdict(list)
    records = []
    for c in index["chips"]:
        by_event[c["event"]].append(c)
        iou = "not defined" if c["iou"] is None else f"{c['iou']:.3f}"
        records.append(
            Evidence(
                id=f"chip:{c['id']}",
                kind="chip",
                title=f"Sen1Floods11 chip {c['id']}",
                text=(
                    f"Hand-labelled 512x512 Sentinel-1 flood chip {c['id']} from the {c['event']} "
                    f"event, {c['split']} split. Labelled water covers "
                    f"{c['water_fraction'] * 100:.1f}% of labelled pixels. "
                    f"Radar ensemble water IoU on this chip: {iou}."
                ),
                source={"type": "raster", "ref": f"sen1floods11/S1Hand/{c['id']}_S1Hand.tif"},
                provenance=1.0,
                bbox=c["bounds"],
                links=[f"event:s1f11-{c['event']}"],
                data={
                    "event": c["event"],
                    "split": c["split"],
                    "iou": c["iou"],
                    "water_fraction": c["water_fraction"],
                    "mean_uncertainty": c["mean_uncertainty"],
                },
            )
        )
    return records, by_event


def _benchmark_events(by_event: dict[str, list[dict]]) -> list[Evidence]:
    held_out = set(registry_events_cfg()["held_out_events"])
    records = []
    for event, chips in sorted(by_event.items()):
        role = (
            "never used in standard training (geographic hold-out)"
            if event in held_out
            else "part of standard training, validation and test"
        )
        records.append(
            Evidence(
                id=f"event:s1f11-{event}",
                kind="event",
                title=f"Sen1Floods11 flood event: {event}",
                text=(
                    f"{event} is one of the 11 flood events in the Sen1Floods11 benchmark, with "
                    f"{len(chips)} hand-labelled chips. It is {role}. "
                    "The acquisition date of the event is not recorded in this repository."
                ),
                source={"type": "dataset", "ref": "sen1floods11 split files"},
                provenance=1.0,
                bbox=_union([c["bounds"] for c in chips]),
                links=["dataset:sen1floods11"],
                data={"event": event, "chips": len(chips), "held_out": event in held_out},
            )
        )
    return records


def registry_events_cfg() -> dict:
    from terramind.config import load_yaml

    return load_yaml("events.yaml")["sen1floods11"]


def _benchmarks(by_event: dict[str, list[dict]]) -> list[Evidence]:
    records = []
    for modality, name in MODALITY_NAMES.items():
        runs = select_runs(modality)
        if not runs:
            continue
        run_names = [d.name for d, _ in runs.values()]
        source = {"type": "runs", "ref": run_names, "script": "experiments/baselines/aggregate.py"}

        # Pooled results per split.
        for split in ("valid", "test", "bolivia"):
            ious = [r["results"][split]["pooled"]["iou"] for _, r in runs.values()]
            label = "Bolivia (geographically unseen)" if split == "bolivia" else split
            records.append(
                Evidence(
                    id=f"benchmark:{modality}:{split}",
                    kind="benchmark",
                    title=f"U-Net {name}: pooled water IoU, {label} split",
                    text=(
                        f"The U-Net trained on {name} reaches pooled water IoU {_pm(ious)} on the "
                        f"Sen1Floods11 {label} split, across {len(ious)} random seeds."
                    ),
                    source=source,
                    provenance=1.0,
                    links=["dataset:sen1floods11"]
                    + (["event:s1f11-Bolivia"] if split == "bolivia" else []),
                    data={
                        "modality": modality,
                        "split": split,
                        "iou_mean": mean(ious),
                        "iou_std": stdev(ious) if len(ious) > 1 else None,
                        "seeds": len(ious),
                    },
                )
            )

        # Per-event results on the official test chips (Bolivia has its own split).
        per_event: dict[str, list[float]] = defaultdict(list)
        chips_scored: dict[str, int] = {}
        for _, report in runs.values():
            for split in ("test", "bolivia"):
                for event, m in report["results"][split]["per_event"].items():
                    if m["iou"] is not None:
                        per_event[event].append(m["iou"])
        for event, chips in by_event.items():
            chips_scored[event] = sum(c["split"] in ("test", "bolivia") for c in chips)
        for event, ious in sorted(per_event.items()):
            seen = "unseen during training" if event == "Bolivia" else "seen during training"
            records.append(
                Evidence(
                    id=f"benchmark:{modality}:event:{event}",
                    kind="benchmark",
                    title=f"U-Net {name}: water IoU on the {event} flood event",
                    text=(
                        f"On the {event} flood event ({seen}), the U-Net trained on {name} reaches "
                        f"water IoU {_pm(ious)} on {chips_scored.get(event, 0)} test chips, across "
                        f"{len(ious)} seeds."
                    ),
                    source=source,
                    provenance=1.0,
                    bbox=_union([c["bounds"] for c in by_event[event]])
                    if event in by_event
                    else None,
                    links=[f"event:s1f11-{event}"],
                    data={
                        "modality": modality,
                        "event": event,
                        "iou_mean": mean(ious),
                        "iou_std": stdev(ious) if len(ious) > 1 else None,
                        "seeds": len(ious),
                        "test_chips": chips_scored.get(event, 0),
                        "seen_in_training": event != "Bolivia",
                    },
                )
            )

        # Leave-one-event-out transfer, when those runs exist.
        for (event, seed), report in select_loeo_runs(modality).items():
            iou = report["results"]["holdout_all"]["pooled"]["iou"]
            if iou is None:
                continue
            n = report["results"]["holdout_all"]["chips"]
            # Same chips the standard model is scored on, so the two are comparable.
            same = report["results"].get("holdout_test_chips") or {}
            same_iou = (same.get("pooled") or {}).get("iou")
            comparable = (
                f" On the {same['chips']} test chips used to score the standard model it reaches "
                f"{same_iou:.3f}; compare that figure, not the all-chips one, with the seen score."
                if same_iou is not None
                else ""
            )
            records.append(
                Evidence(
                    id=f"benchmark:{modality}:loeo:{event}",
                    kind="benchmark",
                    title=f"U-Net {name}: {event} held out of training (leave-one-event-out)",
                    text=(
                        f"With every {event} chip removed from training, the U-Net trained on {name} "
                        f"reaches water IoU {iou:.3f} on all {n} {event} chips (seed {seed}, one seed)."
                        + comparable
                    ),
                    source={
                        "type": "runs",
                        "ref": "loeo",
                        "script": "experiments/transfer/loeo_summary.py",
                    },
                    provenance=1.0,
                    bbox=_union([c["bounds"] for c in by_event[event]])
                    if event in by_event
                    else None,
                    links=[f"event:s1f11-{event}"],
                    data={
                        "modality": modality,
                        "event": event,
                        "iou_mean": iou,
                        "iou_test_chips": same_iou,
                        "iou_std": None,
                        "seeds": 1,
                        "held_out": True,
                        "chips": n,
                    },
                )
            )

        # Ensemble calibration on the unseen region.
        for path in (
            runs_dir() / f"ensemble-{modality}-common.json",
            runs_dir() / f"ensemble-{modality}.json",
        ):
            if not path.exists():
                continue
            ens = json.loads(path.read_text(encoding="utf-8"))
            for split in ("test", "bolivia"):
                raw, scaled = (
                    ens["results"][split]["ensemble"],
                    ens["results"][split]["ensemble_temp_scaled"],
                )
                records.append(
                    Evidence(
                        id=f"benchmark:{modality}:calibration:{split}",
                        kind="benchmark",
                        title=f"Ensemble calibration, {name}, {split} split",
                        text=(
                            f"The {len(ens['seeds'])}-seed ensemble trained on {name} has pixel-level "
                            f"expected calibration error {raw['ece']:.3f} on the {split} split, "
                            f"{scaled['ece']:.3f} after temperature scaling fitted on validation. "
                            f"Its error rate is {raw['error_at_100pct_coverage'] * 100:.1f}% on all "
                            f"pixels and {raw['error_at_90pct_coverage'] * 100:.1f}% after abstaining "
                            "on the least confident 10%."
                        ),
                        source={
                            "type": "runs",
                            "ref": path.name,
                            "script": "terramind eval-ensemble",
                        },
                        provenance=1.0,
                        links=["dataset:sen1floods11"]
                        + (["event:s1f11-Bolivia"] if split == "bolivia" else []),
                        data={
                            "modality": modality,
                            "split": split,
                            "ece": raw["ece"],
                            "ece_scaled": scaled["ece"],
                            "error_all": raw["error_at_100pct_coverage"],
                            "error_90": raw["error_at_90pct_coverage"],
                        },
                    )
                )
            break
    return records


def _candidate_events_and_scenes() -> list[Evidence]:
    records = []
    for e in registry.load_events().values():
        status = "confirmed" if e.confirmed else "approximate and not yet confirmed"
        records.append(
            Evidence(
                id=f"event:{e.id}",
                kind="event",
                title=f"Candidate flood event: {e.name}",
                text=(
                    f"{e.name} ({e.region}), {e.start} to {e.end}. Candidate event for satellite "
                    f"scene discovery. The date window and bounding box are {status}."
                ),
                source={"type": "config", "ref": "configs/events.yaml"},
                provenance=1.0 if e.confirmed else 0.4,
                bbox=list(e.bbox),
                start=e.start,
                end=e.end,
                data={"event": e.id, "confirmed": e.confirmed},
            )
        )
    for path in sorted((data_dir() / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        event_id = manifest["search"]["event"]
        for s in manifest["scenes"]:
            day = (s["datetime"] or "")[:10]
            cloud = "" if s["cloud_cover"] is None else f" Cloud cover {s['cloud_cover']:.0f}%."
            sensor = "radar" if "sentinel-1" in s["collection"] else "optical"
            records.append(
                Evidence(
                    id=f"scene:{s['id']}",
                    kind="scene",
                    title=f"{s['collection']} scene {day}",
                    text=(
                        f"{s['collection']} {sensor} satellite scene {s['id']} acquired {day} by "
                        f"{s['platform'] or 'unknown platform'}, overlapping the search box for "
                        f"{event_id}.{cloud} Found in the {s['catalog']} STAC catalogue; imagery "
                        "not downloaded."
                    ),
                    source={
                        "type": "stac",
                        "ref": s["catalog"],
                        "searched_at": manifest["searched_at"],
                        "manifest": path.name,
                    },
                    provenance=0.9,
                    bbox=s["bbox"],
                    start=day or None,
                    end=day or None,
                    links=[f"event:{event_id}", f"dataset:{s['collection']}"],
                    data={
                        "collection": s["collection"],
                        "event": event_id,
                        "cloud_cover": s["cloud_cover"],
                        "platform": s["platform"],
                    },
                )
            )
    return records


def _datasets() -> list[Evidence]:
    records = []
    for d in registry.load_datasets().values():
        verified = "verified" if d.licence_verified else "recorded from memory and NOT verified"
        records.append(
            Evidence(
                id=f"dataset:{d.id}",
                kind="dataset",
                title=f"Dataset: {d.name}",
                text=(
                    f"{d.name} from {d.provider}. Modality: {d.modality}; resolution: "
                    f"{d.resolution}. Used for: {', '.join(d.used_for)}. Licence: {d.licence} "
                    f"({verified}). Redistribution: {d.redistribution}. {d.notes or ''}"
                ).strip(),
                source={"type": "config", "ref": "configs/datasets.yaml", "url": d.access_url},
                provenance=1.0 if d.licence_verified else 0.5,
                data={"dataset": d.id, "licence_verified": d.licence_verified},
            )
        )
    return records


def _documents() -> list[Evidence]:
    """Project documents, split at headings and bold lead-ins."""
    records = []
    for rel in ("docs/research-log.md", "docs/specification.md"):
        path = repo_root() / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        parts = re.split(r"\n(?=## |\*\*[^*\n]+\*\*)", text)
        for i, part in enumerate(parts):
            body = part.strip()
            if len(body) < 80:
                continue
            first = body.splitlines()[0].strip("#* ").strip()
            records.append(
                Evidence(
                    id=f"document:{Path(rel).stem}:{i}",
                    kind="document",
                    title=f"{Path(rel).name}: {first[:80]}",
                    text=body[:1500],
                    source={"type": "document", "ref": rel, "section": first[:80]},
                    provenance=0.7,
                )
            )
    return records


POLICY_NAMES = {
    "random": "random walk",
    "blind-lookahead": "lookahead planner that ignores the flood model",
    "greedy": "greedy policy",
    "assignment": "assignment (Hungarian) policy",
    "lookahead": "lookahead planner using the flood model",
    "lookahead+uncertainty": "lookahead planner with an uncertainty bonus",
    "oracle-lookahead": "lookahead planner given the true flood extent (an upper reference)",
    "ppo": "learned PPO policy",
    "ppo-nounc": "learned PPO policy without the uncertainty input",
    "ppo-argmax": "learned PPO policy taking its most likely move",
}
SIM_SPLITS = {"test": "the test split", "bolivia": "the unseen Bolivia region"}
SIM_PROVENANCE = 0.7  # hand-set: real inputs, invented decision problem


def _simulations(by_event: dict[str, list[dict]]) -> list[Evidence]:
    """Results of the simulated flood-response experiments E7 and E8, if they were run."""
    sim_dir = repo_root() / "experiments" / "sim" / "runs"
    bolivia = _union([c["bounds"] for c in by_event["Bolivia"]]) if "Bolivia" in by_event else None
    records = []

    def add(eid, title, text, split, script, data):
        records.append(
            Evidence(
                id=eid,
                kind="simulation",
                title=title,
                text="Simulated: " + text,
                source={"type": "simulation", "script": script},
                provenance=SIM_PROVENANCE,
                bbox=bolivia if split == "bolivia" else None,
                links=["event:s1f11-Bolivia"] if split == "bolivia" else [],
                data={"simulated": True, "split": split, **data},
            )
        )

    e7 = sim_dir / "e7-s1.json"
    if e7.exists():
        report = json.loads(e7.read_text(encoding="utf-8"))
        cfg = report["config"]
        for split, result in report["results"].items():
            for policy, m in result["policies"].items():
                name = POLICY_NAMES.get(policy, policy)
                seeds = f", mean of {m['seeds']} trained seeds" if m.get("seeds") else ""
                add(
                    f"simulation:e7:{split}:{policy}",
                    f"Simulated flood response on {split}: {name}",
                    f"on {SIM_SPLITS[split]} ({m['scenarios']} scenarios, {cfg['teams']} teams, "
                    f"{cfg['horizon']} moves), the {name} reaches on average {m['auc']:.3f} of the "
                    f"flooded area over an episode and {m['served']:.3f} by the end{seeds}.",
                    split,
                    "terramind sim-eval",
                    {"experiment": "e7", "policy": policy, "auc": m["auc"], "served": m["served"]},
                )
            for pair, d in result["paired_auc"].items():
                a, b = pair.split(" - ")
                add(
                    f"simulation:e7:{split}:diff:{a}:{b}",
                    f"Simulated flood response on {split}: {a} versus {b}",
                    f"on {SIM_SPLITS[split]}, the {POLICY_NAMES.get(a, a)} scores {d['mean']:+.3f} "
                    f"relative to the {POLICY_NAMES.get(b, b)} on the same episodes "
                    f"(95% interval {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}).",
                    split,
                    "terramind sim-eval",
                    {
                        "experiment": "e7",
                        "pair": [a, b],
                        "difference": d["mean"],
                        "ci_low": d["ci95"][0],
                        "ci_high": d["ci95"][1],
                    },
                )

    e8 = sim_dir / "e8-chain.json"
    if e8.exists():
        report = json.loads(e8.read_text(encoding="utf-8"))
        for setting, result in report["settings"].items():
            cfg = result["config"]
            slug = setting.replace(", ", "-").replace(" ", "-")
            for split, block in result["splits"].items():
                for stage, m in block["stages"].items():
                    add(
                        f"simulation:e8:{slug}:{split}:{stage.replace(' ', '-').replace(',', '')}",
                        f"Chain ablation ({setting}) on {split}: planner given {stage}",
                        f"in the '{setting}' setting ({cfg['teams']} teams, {cfg['horizon']} moves) on "
                        f"{SIM_SPLITS[split]}, a lookahead planner given {stage} reaches on average "
                        f"{m['auc']:.3f} of the flooded area over an episode, {m['vs_reference']:+.3f} "
                        f"relative to the radar ensemble (95% interval "
                        f"{m['vs_reference_ci95'][0]:+.3f} to {m['vs_reference_ci95'][1]:+.3f}).",
                        split,
                        "terramind chain-ablation",
                        {
                            "experiment": "e8",
                            "setting": setting,
                            "stage": stage,
                            "auc": m["auc"],
                            "difference": m["vs_reference"],
                        },
                    )
    return records


def build_corpus() -> list[Evidence]:
    chips, by_event = _chips()
    records = (
        chips
        + _benchmark_events(by_event)
        + _benchmarks(by_event)
        + _simulations(by_event)
        + _candidate_events_and_scenes()
        + _datasets()
        + _documents()
    )
    ids = [r.id for r in records]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise ValueError(f"duplicate evidence ids: {dupes}")
    return records


def save_corpus(records: list[Evidence]) -> Path:
    path = corpus_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(asdict(r)) + "\n" for r in records)
    return path


def load_corpus() -> list[Evidence]:
    path = corpus_path()
    if not path.exists():
        raise FileNotFoundError("evidence corpus not built; run `terramind build-evidence`")
    with open(path, encoding="utf-8") as f:
        return [Evidence(**json.loads(line)) for line in f if line.strip()]
