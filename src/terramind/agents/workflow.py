"""TerraCore: a tool-grounded agent workflow on LangGraph.

    orchestrator -> evidence -> [earth observation, GIS, prediction, decision] -> critic -> report
                        ^                                                           |
                        +--------------------- retry once, relaxed -----------------+

Design rules:

- Agents never write facts. Each one calls a deterministic tool and emits
  claims; a claim is a sentence plus the evidence ids and the numbers it rests
  on.
- The critic rejects any claim whose evidence ids are missing from the store or
  whose numbers do not appear in the cited evidence, then decides between
  answering, retrying retrieval with relaxed constraints, and abstaining.
- No language model is needed. An optional one may reword the final report; its
  output is used only if every number in it already appears in the verified
  claims, otherwise the templated report is kept.

The decision agent answers only from stored results of the simulated
experiments and labels them as simulated. Forecasting is not implemented; the
orchestrator says so when a question needs it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from shapely.geometry import box

from terramind.evidence.store import MODALITY_NAMES, Evidence
from terramind.rag.query import ParsedQuery, build_gazetteer, parse
from terramind.rag.retriever import Hit, Retriever

MAX_ATTEMPTS = 2
LOW_PROVENANCE = 0.6
MIN_OPEN_SIMILARITY = 0.4  # hand-set; cosine similarity floor for open questions
UNSUPPORTED = ("deploy", "evacuat", "forecast", "next 48", "will flood", "tomorrow")
SIM_CAVEAT = (
    "Simulated outcome: the flood extent and model predictions are real, but the teams and "
    "the scoring are invented. It says nothing about a real response."
)


@dataclass
class Claim:
    agent: str
    text: str
    evidence_ids: list[str]
    # Three kinds of checkable content, each verified differently by the critic:
    count: int | None = None  # must equal the number of cited records
    numbers: list[float] = field(default_factory=list)  # must appear in a cited record
    # (field, scale, value): value must equal scale * mean(field) over cited records
    means: list[tuple[str, float, float]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)


class State(TypedDict, total=False):
    question: str
    query: ParsedQuery
    plan: list[str]
    attempts: int
    relaxed: bool
    hits: list[Hit]
    claims: list[Claim]
    rejected: list[dict]
    verdict: str  # answer | retry | abstain
    reasons: list[str]
    report: dict
    trace: list[dict]


def _log(state: State, agent: str, action: str, detail: Any) -> list[dict]:
    return [*state.get("trace", []), {"agent": agent, "action": action, "detail": detail}]


def _number_in(value: float, evidence: Evidence) -> bool:
    """True if `value` appears among the evidence record's numbers (to 3 decimals)."""
    candidates = [
        v for v in evidence.data.values() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    candidates += [float(x) for x in re.findall(r"-?\d+\.?\d*", evidence.text)]
    return any(abs(value - c) < 5e-4 or abs(value - c * 100) < 5e-2 for c in candidates)


class TerraCore:
    def __init__(self, retriever: Retriever, llm=None):
        self.retriever = retriever
        self.store = retriever.by_id
        self.gazetteer = build_gazetteer(retriever.corpus)
        self.llm = llm
        self.graph = self._build()

    # -- nodes ----------------------------------------------------------------

    def orchestrator(self, state: State) -> State:
        query = parse(state["question"], self.gazetteer)
        kinds = set(query.kinds)
        plan = ["evidence"]
        if "scene" in kinds:
            plan.append("earth_observation")
        if query.regions and ("chip" in kinds or not kinds):
            plan.append("gis")
        if "benchmark" in kinds or (query.regions and not kinds):
            plan.append("prediction")
        if "simulation" in kinds:
            # A question about the simulation is not a question about model accuracy.
            plan = ["evidence", "decision"]
        plan += ["critic", "report"]
        lower = state["question"].lower()
        unsupported = [w for w in UNSUPPORTED if w in lower]
        if unsupported:
            # Answering a forecasting or what-if question from loosely related records
            # would look like an answer while being nothing of the kind. Skip to abstain.
            plan = ["critic", "report"]
        detail = {
            "regions": [r.evidence_id for r in query.regions],
            "time_window": [query.start, query.end],
            "intent": sorted(kinds),
            "plan": plan,
        }
        if unsupported:
            detail["not_available"] = (
                "forecasting and real-world allocation are not built yet"
            )
        return {
            "query": query,
            "plan": plan,
            "attempts": 0,
            "relaxed": False,
            "claims": [],
            "rejected": [],
            "reasons": [
                "This question needs forecasting or a real-world allocation, which are not built yet."
            ]
            if unsupported
            else [],
            "trace": _log(state, "orchestrator", "plan", detail),
        }

    def evidence_agent(self, state: State) -> State:
        query, relaxed = state["query"], state.get("relaxed", False)
        hits = self.retriever.retrieve(query, k=12, mode="hybrid", use_time=not relaxed)
        claims = []
        # Documents and dataset records answer directly; other kinds go through their own agent.
        dataset_question = "dataset" in query.kinds
        wants_direct = "evidence" in state["plan"] and (
            dataset_question or not (query.regions or query.kinds)
        )
        direct = [
            h
            for h in hits
            if h.evidence.kind in (("dataset",) if dataset_question else ("dataset", "document"))
            # An open question with no place or intent is answered only by a close match.
            and (dataset_question or h.components.get("semantic", 0.0) >= MIN_OPEN_SIMILARITY)
        ]
        # Keep only records about as relevant as the best one, so a single-answer
        # question does not drag in loosely related records.
        direct = [h for h in direct if h.score >= direct[0].score - 0.03][:3] if direct else []
        if wants_direct:
            for h in direct:
                e = h.evidence
                caveats = []
                if e.kind == "dataset" and not e.data.get("licence_verified", True):
                    caveats.append(
                        "Licence text is unverified; check the provider's terms before relying on it."
                    )
                claims.append(
                    Claim("evidence", e.text.split("\n")[0][:400], [e.id], caveats=caveats)
                )
        return {
            "hits": hits,
            "attempts": state.get("attempts", 0) + 1,
            "claims": [c for c in state.get("claims", []) if c.agent != "evidence"] + claims,
            "trace": _log(
                state,
                "evidence",
                "hybrid_retrieve",
                {
                    "relaxed_time": relaxed,
                    "top": [
                        {
                            "id": h.evidence.id,
                            "score": round(h.score, 3),
                            "via": h.via,
                            **{k: round(v, 3) for k, v in h.components.items()},
                        }
                        for h in hits[:6]
                    ],
                },
            ),
        }

    def earth_observation_agent(self, state: State) -> State:
        query = state["query"]
        if "earth_observation" not in state["plan"]:
            return {}
        timed = bool(query.start) and not state.get("relaxed", False)
        scenes = [
            e
            for e in self.retriever.corpus
            if e.kind == "scene"
            and (not query.bboxes or any(box(*e.bbox).intersects(box(*q)) for q in query.bboxes))
            and (not timed or (query.start <= e.start <= query.end))
            and (
                not query.modalities
                or ("s1" in query.modalities and "sentinel-1" in e.data["collection"])
                or ("s2" in query.modalities and "sentinel-2" in e.data["collection"])
            )
        ]
        claims = []
        for collection, n in sorted(Counter(e.data["collection"] for e in scenes).items()):
            group = sorted(
                (e for e in scenes if e.data["collection"] == collection), key=lambda e: e.start
            )
            text = (
                f"{n} {collection} scenes overlap the region, acquired between {group[0].start} and "
                f"{group[-1].start}."
            )
            claims.append(
                Claim(
                    "earth_observation",
                    text,
                    [e.id for e in group],
                    count=n,
                    caveats=[
                        "Scene footprints only; the imagery has not been downloaded or inspected."
                    ],
                )
            )
            clear = [
                e for e in group if e.data["cloud_cover"] is not None and e.data["cloud_cover"] < 20
            ]
            if clear:
                claims.append(
                    Claim(
                        "earth_observation",
                        f"{len(clear)} of the {collection} scenes have under 20% cloud cover, "
                        f"the earliest on {clear[0].start}.",
                        [e.id for e in clear],
                        count=len(clear),
                    )
                )
        return {
            "claims": state.get("claims", []) + claims,
            "trace": _log(
                state,
                "earth_observation",
                "scene_filter",
                {"time_filter": timed, "scenes": len(scenes)},
            ),
        }

    def gis_agent(self, state: State) -> State:
        query = state["query"]
        if "gis" not in state["plan"]:
            return {}
        claims = []
        for region in query.regions:
            chips = [
                e
                for e in self.retriever.corpus
                if e.kind == "chip" and box(*e.bbox).intersects(box(*region.bbox))
            ]
            if not chips:
                continue
            water = mean(e.data["water_fraction"] for e in chips) * 100
            splits = Counter(e.data["split"] for e in chips)
            claims.append(
                Claim(
                    "gis",
                    f"{len(chips)} hand-labelled chips fall in the {region.name.title()} region "
                    f"({', '.join(f'{n} {s}' for s, n in sorted(splits.items()))}); on average "
                    f"{water:.1f}% of their labelled pixels are water.",
                    [e.id for e in chips],
                    count=len(chips),
                    means=[("water_fraction", 100.0, water)],
                    caveats=[
                        "Labelled water is what was mapped in the chips, not the full flood extent."
                    ],
                )
            )
        return {
            "claims": state.get("claims", []) + claims,
            "trace": _log(
                state,
                "gis",
                "spatial_intersect",
                {"regions": len(query.regions), "claims": len(claims)},
            ),
        }

    def prediction_agent(self, state: State) -> State:
        query = state["query"]
        if "prediction" not in state["plan"]:
            return {}
        wanted = query.modalities or list(MODALITY_NAMES)
        events = [
            r.evidence_id.removeprefix("event:s1f11-")
            for r in query.regions
            if r.evidence_id.startswith("event:s1f11-")
        ]
        claims = []
        if events:
            ids = [f"benchmark:{m}:event:{ev}" for ev in events for m in wanted]
            ids += [f"benchmark:{m}:loeo:{ev}" for ev in events for m in wanted]
        else:
            ids = [f"benchmark:{m}:{split}" for m in wanted for split in ("test", "bolivia")]
            if any(
                w in query.text.lower() for w in ("calibrat", "uncertain", "confiden", "reliab")
            ):
                ids += [f"benchmark:{m}:calibration:bolivia" for m in wanted]
        for eid in ids:
            e = self.store.get(eid)
            if not e:
                continue
            numbers = [
                e.data[k] for k in ("iou_mean", "ece", "ece_scaled") if e.data.get(k) is not None
            ]
            caveats = []
            if e.data.get("seen_in_training"):
                caveats.append(
                    "This event was in the training data, so the score is in-distribution."
                )
            if e.data.get("seeds") == 1:
                caveats.append("Single seed; no spread available.")
            if (e.data.get("test_chips") or e.data.get("chips") or 99) < 20:
                caveats.append("Fewer than 20 chips; the estimate is noisy.")
            claims.append(Claim("prediction", e.text, [eid], numbers=numbers, caveats=caveats))
        return {
            "claims": state.get("claims", []) + claims,
            "trace": _log(
                state,
                "prediction",
                "benchmark_lookup",
                {"requested": len(ids), "found": len(claims)},
            ),
        }

    def decision_agent(self, state: State) -> State:
        """Answers from stored results of the simulated experiments, never by running one."""
        if "decision" not in state["plan"]:
            return {}
        query = state["query"]
        lower = query.text.lower()
        bolivia = any(r.evidence_id == "event:s1f11-Bolivia" for r in query.regions)
        split = "bolivia" if bolivia else "test"
        prefix = f"simulation:e7:{split}:"
        policies = {
            e.data["policy"]: e
            for i, e in self.store.items()
            if i.startswith(prefix) and "policy" in e.data
        }
        fair = {
            k: e
            for k, e in policies.items()
            if k not in ("oracle-lookahead", "ppo-argmax", "random")
        }
        chosen: list[str] = []
        if fair:
            chosen.append(max(fair.values(), key=lambda e: e.data["auc"]).id)
        chosen += [prefix + k for k in ("ppo", "blind-lookahead", "oracle-lookahead")]
        chosen.append(f"{prefix}diff:ppo:lookahead")
        if any(w in lower for w in ("uncertain", "confiden", "disagree")):
            chosen += [f"{prefix}diff:lookahead+uncertainty:lookahead", f"{prefix}diff:ppo:ppo-nounc"]
        if any(w in lower for w in ("optical", "radar", "fus", "perception", "calibrat", "chain")):
            stage_prefix = f"simulation:e8:standard:{split}:"
            chosen += sorted(i for i in self.store if i.startswith(stage_prefix))

        caveats = [SIM_CAVEAT]
        other = [r.name for r in query.regions if r.evidence_id != "event:s1f11-Bolivia"]
        if other:
            caveats.append(
                f"There is no simulation specific to {', '.join(other)}; these are results over the "
                "test scenarios from several flood events."
            )
        claims = []
        for eid in dict.fromkeys(chosen):
            e = self.store.get(eid)
            if not e:
                continue
            numbers = [e.data[k] for k in ("auc", "difference") if e.data.get(k) is not None]
            claims.append(Claim("decision", e.text, [eid], numbers=numbers, caveats=list(caveats)))
        return {
            "claims": state.get("claims", []) + claims,
            "trace": _log(
                state,
                "decision",
                "simulation_lookup",
                {"split": split, "requested": len(set(chosen)), "found": len(claims)},
            ),
        }

    def critic(self, state: State) -> State:
        accepted, rejected = [], list(state.get("rejected", []))
        for claim in state.get("claims", []):
            missing = [i for i in claim.evidence_ids if i not in self.store]
            if not claim.evidence_ids or missing:
                rejected.append(
                    {"claim": claim.text, "reason": f"cites unknown evidence {missing[:3]}"}
                )
                continue
            cited = [self.store[i] for i in claim.evidence_ids]
            problems = []
            if claim.count is not None and claim.count != len(cited):
                problems.append(f"count {claim.count} but {len(cited)} records cited")
            problems += [
                f"{n} not found in cited evidence"
                for n in claim.numbers
                if not any(_number_in(n, e) for e in cited)
            ]
            for name, scale, value in claim.means:
                actual = scale * mean(e.data[name] for e in cited)
                if abs(actual - value) > 1e-6:
                    problems.append(f"mean {name} is {actual:.4f}, claim says {value:.4f}")
            if problems:
                rejected.append({"claim": claim.text, "reason": "; ".join(problems)})
                continue
            weakest = min(e.provenance for e in cited)
            if weakest < LOW_PROVENANCE:
                claim.caveats.append(
                    f"Rests on evidence with low provenance ({weakest:.1f}): unconfirmed or unverified."
                )
            accepted.append(claim)

        reasons = list(state.get("reasons", []))
        query = state["query"]
        if accepted:
            verdict = "answer"
        elif state["attempts"] < MAX_ATTEMPTS and query.start and not state.get("relaxed", False):
            verdict = "retry"
        else:
            verdict = "abstain"
            if not query.regions and ({"scene", "chip"} & set(query.kinds)):
                reasons.append(
                    "No known region was recognised in the question, so no place could be resolved."
                )
            reasons.append("No claim could be supported by evidence in the store.")
        return {
            "claims": accepted,
            "rejected": rejected,
            "verdict": verdict,
            "relaxed": verdict == "retry" or state.get("relaxed", False),
            "reasons": reasons,
            "trace": _log(
                state,
                "critic",
                "verify",
                {"accepted": len(accepted), "rejected": len(rejected), "verdict": verdict},
            ),
        }

    def reporter(self, state: State) -> State:
        claims = state.get("claims", [])
        order: list[str] = []
        for claim in claims:
            for eid in claim.evidence_ids[:4]:
                if eid not in order:
                    order.append(eid)
        number = {eid: i + 1 for i, eid in enumerate(order)}

        lines, caveats = [], []
        for claim in claims:
            shown = claim.evidence_ids[:4]
            cite = "".join(f"[{number[i]}]" for i in shown)
            extra = len(claim.evidence_ids) - len(shown)
            lines.append(
                f"{claim.text} {cite}" + (f" (+{extra} more records)" if extra > 0 else "")
            )
            caveats += [c for c in claim.caveats if c not in caveats]
        if state.get("relaxed") and claims:
            caveats.append(
                "Nothing matched the stated time window, so it was dropped; dates in the answer may differ from the question."
            )

        if state["verdict"] == "abstain":
            answer = "I can't answer this from the evidence available. " + " ".join(
                state.get("reasons", [])
            )
        else:
            answer = "\n".join(f"- {line}" for line in lines)
            if state.get("reasons"):
                answer += "\n\nNot answered: " + " ".join(state["reasons"])

        polished = None
        if self.llm and state["verdict"] == "answer":
            draft = self.llm.rewrite(state["question"], answer)
            allowed = set(re.findall(r"\d+\.?\d*", answer))
            if draft and set(re.findall(r"\d+\.?\d*", draft)) <= allowed:
                polished = draft

        report = {
            "question": state["question"],
            "verdict": state["verdict"],
            "answer": polished or answer,
            "llm_polished": polished is not None,
            "caveats": caveats,
            "evidence": [
                {
                    "n": number[eid],
                    "id": eid,
                    "kind": self.store[eid].kind,
                    "title": self.store[eid].title,
                    "source": self.store[eid].source,
                    "provenance": self.store[eid].provenance,
                    "bbox": self.store[eid].bbox,
                    "start": self.store[eid].start,
                }
                for eid in order
            ],
            "rejected_claims": state.get("rejected", []),
            "regions": [{"name": r.name, "bbox": r.bbox} for r in state["query"].regions],
        }
        return {
            "report": report,
            "trace": _log(state, "report", "compose", {"claims": len(claims)}),
        }

    # -- graph ----------------------------------------------------------------

    def _build(self):
        g = StateGraph(State)
        g.add_node("orchestrator", self.orchestrator)
        g.add_node("evidence", self.evidence_agent)
        g.add_node("earth_observation", self.earth_observation_agent)
        g.add_node("gis", self.gis_agent)
        g.add_node("prediction", self.prediction_agent)
        g.add_node("decision", self.decision_agent)
        g.add_node("critic", self.critic)
        g.add_node("report", self.reporter)
        g.add_edge(START, "orchestrator")
        g.add_edge("orchestrator", "evidence")
        g.add_edge("evidence", "earth_observation")
        g.add_edge("earth_observation", "gis")
        g.add_edge("gis", "prediction")
        g.add_edge("prediction", "decision")
        g.add_edge("decision", "critic")
        g.add_conditional_edges(
            "critic",
            lambda s: "evidence" if s["verdict"] == "retry" else "report",
            {"evidence": "evidence", "report": "report"},
        )
        g.add_edge("report", END)
        return g.compile()

    def ask(self, question: str) -> dict:
        final = self.graph.invoke({"question": question, "trace": []}, {"recursion_limit": 30})
        return {**final["report"], "trace": final["trace"]}
