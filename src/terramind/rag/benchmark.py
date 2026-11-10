"""Retrieval benchmark (experiment E5, first version).

Queries and their relevant evidence ids are generated from the corpus's own
structure, so the benchmark is synthetic: it checks that retrieval respects
place, time and intent, not that it handles how people actually phrase
questions. Place names in the queries are the same ones the gazetteer knows,
which flatters every method that uses the parser. Treat the numbers as a
regression test and an ablation, not as evidence of real-world quality.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from shapely.geometry import box

from terramind.evidence.store import MODALITY_NAMES, Evidence
from terramind.rag.query import build_gazetteer, parse
from terramind.rag.retriever import Retriever

MODES = ("vector", "spatial", "hybrid-nograph", "hybrid")


@dataclass
class Case:
    group: str
    question: str
    relevant: set[str]


def build_cases(corpus: list[Evidence]) -> list[Case]:
    cases = []
    by_id = {e.id: e for e in corpus}

    # 1. Scenes for a candidate event, by sensor.
    scenes = defaultdict(set)
    for e in corpus:
        if e.kind == "scene":
            scenes[(e.data["event"], e.data["collection"])].add(e.id)
    for (event_id, collection), ids in sorted(scenes.items()):
        event = by_id[f"event:{event_id}"]
        name = event.title.removeprefix("Candidate flood event: ").replace(" floods", "")
        place = {
            "chennai-2015": "Chennai",
            "kerala-2018": "Kerala",
            "ahr-2021": "Ahr",
            "sindh-2022": "Sindh",
        }[event_id]
        sensor = "Sentinel-1 radar" if "sentinel-1" in collection else "Sentinel-2 optical"
        month = f"{event.start[:4]}"
        cases.append(
            Case("scenes", f"Which {sensor} scenes cover the {place} floods in {month}?", ids)
        )
        del name

    # 2. Model performance on a benchmark event.
    for e in corpus:
        if e.kind == "event" and e.id.startswith("event:s1f11-"):
            name = e.data["event"]
            relevant = {f"benchmark:{m}:event:{name}" for m in MODALITY_NAMES} & by_id.keys()
            if relevant:
                cases.append(
                    Case(
                        "performance", f"How well does the flood model perform in {name}?", relevant
                    )
                )

    # 3. Labelled chips in a benchmark region.
    chips = defaultdict(set)
    for e in corpus:
        if e.kind == "chip":
            chips[e.data["event"]].add(e.id)
    for name, ids in sorted(chips.items()):
        cases.append(Case("chips", f"Show the labelled flood chips in {name}.", ids))

    # 4. Dataset licences (no place, no time).
    for dataset, phrase in [
        ("openstreetmap", "OpenStreetMap road data"),
        ("era5", "ERA5 reanalysis weather data"),
        ("worldpop", "WorldPop population grids"),
        ("xbd", "the xBD building damage dataset"),
        ("gpm-imerg", "GPM IMERG precipitation"),
        ("sen1floods11", "the Sen1Floods11 flood benchmark"),
    ]:
        if f"dataset:{dataset}" in by_id:
            cases.append(
                Case("licences", f"What is the licence for {phrase}?", {f"dataset:{dataset}"})
            )
    return cases


def _ndcg(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, eid in enumerate(ranked[:k]) if eid in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def run(retriever: Retriever, k: int = 10) -> dict:
    corpus = retriever.corpus
    gazetteer = build_gazetteer(corpus)
    cases = build_cases(corpus)
    rows: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for case in cases:
        query = parse(case.question, gazetteer)
        for mode in MODES:
            hits = retriever.retrieve(query, k=k, mode=mode)
            ranked = [h.evidence.id for h in hits]
            found = sum(eid in case.relevant for eid in ranked)
            m = rows[(case.group, mode)]
            m["precision"].append(found / k)
            # Recall against what could fit in k results.
            m["recall"].append(found / min(k, len(case.relevant)))
            m["ndcg"].append(_ndcg(ranked, case.relevant, k))
            if query.bboxes:
                located = [h.evidence for h in hits if h.evidence.bbox]
                inside = sum(
                    any(box(*e.bbox).intersects(box(*q)) for q in query.bboxes) for e in located
                )
                # Of the returned records that have a location, how many are in the asked region.
                m["spatial_ok"].append(inside / len(located) if located else 0.0)
    groups = sorted({g for g, _ in rows})
    summary = {
        group: {
            mode: {metric: mean(values) for metric, values in rows[(group, mode)].items()}
            for mode in MODES
        }
        for group in groups
    }
    summary["all"] = {
        mode: {
            metric: mean(v for g in groups for v in rows[(g, mode)].get(metric, []))
            for metric in ("precision", "recall", "ndcg", "spatial_ok")
        }
        for mode in MODES
    }
    return {
        "k": k,
        "embedder": retriever.embedder.name,
        "weights": retriever.weights,
        "cases": {g: sum(c.group == g for c in cases) for g in groups},
        "results": summary,
    }
