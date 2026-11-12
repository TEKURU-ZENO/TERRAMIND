"""RQ5: are specialised agents more reliable than a fixed pipeline or one LLM agent?

A frozen set of questions, each with gold facts taken from the evidence store
(the number, count or licence that a correct answer must contain) and the
records that support it. A fifth of the questions cannot be answered from the
store and should be refused.

Systems compared, all on the same hybrid retriever:

- terracore: the agent workflow with its critic
- terracore-no-critic: the same agents, every claim accepted (critic ablation)
- fixed-pipeline: retrieve, answer with the top record's text, never refuse
- llm-agent: one local LLM given the top retrieved records, told to cite or refuse

The question wording was written before any system was run on it. It was
written by the author of TerraCore's rules, which is a bias in TerraCore's
favour; place names use the corpus's names or listed aliases.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

from terramind.evidence.store import Evidence
from terramind.rag.query import parse

# ---------------------------------------------------------------- questions


@dataclass
class Question:
    id: str
    category: str
    text: str
    answerable: bool = True
    numbers: list[float] = field(default_factory=list)  # each must appear (3 decimals)
    counts: list[int] = field(default_factory=list)  # each must appear as an integer
    strings: list[str] = field(default_factory=list)  # each must appear, case-insensitive
    support: list[str] = field(default_factory=list)  # citing any of these counts as grounded


MODALITY_WORDS = {"s1": "radar", "s2": "optical", "s1s2": "fused"}
PERFORMANCE = [
    ("Spain", "s2", "What water IoU does the {m} model reach on the {e} flood event?"),
    ("Ghana", "s1", "How accurate is flood mapping from {m} imagery for {e}?"),
    ("Pakistan", "s1", "Give the IoU of the {m} U-Net in {e}."),
    ("Pakistan", "s2", "What IoU does the {m} model get on the {e} flood?"),
    ("Mekong", "s1s2", "How well does the {m} model map water for the {e} event?"),
    ("Somalia", "s2", "What is the {m} model's water IoU for {e}?"),
    ("USA", "s1", "How good is the {m} model on the {e} flood event?"),
    ("Paraguay", "s1s2", "Report the IoU of the {m} U-Net on {e}."),
    ("India", "s2", "What water IoU does the {m} model achieve in {e}?"),
    ("Nigeria", "s1", "How accurately does the {m} model detect flood water in {e}?"),
    ("Sri-Lanka", "s1s2", "What IoU does the {m} model reach for Sri Lanka?"),
    ("Mekong", "s1", "What IoU does the {m} model get in Cambodia?"),
    ("Bolivia", "s1", "How well does the {m} model perform on the unseen Bolivia region?"),
    ("Bolivia", "s2", "What water IoU does the {m} model reach in Bolivia?"),
    ("Bolivia", "s1s2", "What IoU does the {m} model get on Bolivia?"),
]
TRANSFER = [
    ("Spain", "s1", "If {e} is left out of training, what IoU does the {m} model get on its test chips?"),
    ("Pakistan", "s2", "With {e} held out of training, how does the {m} model score on it?"),
    ("Ghana", "s1s2", "What IoU does the {m} model reach on {e} when {e} is excluded from training?"),
    ("Somalia", "s2", "How well does the {m} model transfer to {e} if it never saw {e} in training?"),
    ("India", "s1", "Leave-one-event-out: what is the {m} model's IoU on {e}?"),
]  # fmt: skip
SCENES = [
    ("kerala-2018", "sentinel-2-l2a", "2018-08-01", "2018-08-31",
     "How many Sentinel-2 scenes cover the Kerala floods in August 2018?"),
    ("kerala-2018", "sentinel-1-grd", "2018-08-01", "2018-08-31",
     "How many Sentinel-1 radar scenes were acquired over Kerala in August 2018?"),
    ("ahr-2021", "sentinel-1-grd", "2021-07-01", "2021-07-31",
     "How many Sentinel-1 scenes cover the Ahr floods in July 2021?"),
    ("ahr-2021", "sentinel-2-l2a", "2021-07-01", "2021-07-31",
     "How many Sentinel-2 images of the Ahr valley are there from July 2021?"),
    ("chennai-2015", "sentinel-2-l2a", "2015-11-01", "2015-12-31",
     "How many Sentinel-2 scenes cover Chennai between November and December 2015?"),
    ("chennai-2015", "sentinel-1-grd", "2015-11-01", "2015-12-31",
     "How many Sentinel-1 radar passes are there over Chennai in November-December 2015?"),
]  # fmt: skip
LICENCES = [
    ("openstreetmap", "odbl", "What licence applies to OpenStreetMap data?"),
    ("worldpop", "cc by 4.0", "Under what licence is WorldPop released?"),
    ("xbd", "cc by-nc-sa", "What are the licence terms for the xBD dataset?"),
    ("spacenet", "cc by-sa", "Which licence covers the SpaceNet dataset?"),
    ("srtm", "public domain", "What is the licence of the SRTM elevation data?"),
    ("sen1floods11", "none stated", "What licence does the Sen1Floods11 dataset have?"),
]
SIMULATION = [
    ("simulation:e7:test:ppo",
     "In the flood response simulation, what share of flooded area does the PPO policy reach on the test split?"),
    ("simulation:e7:bolivia:lookahead",
     "In the simulation, how does the lookahead planner score on Bolivia?"),
    ("simulation:e7:test:greedy",
     "What does the greedy policy score in the flood response simulation?"),
    ("simulation:e9:cost0.5:test:calibrated",
     "In the dispatch simulation with waste cost 0.5, what utility does the calibrated planner reach on the test split?"),
    ("simulation:e9:cost0.5:test:raw-map",
     "In the dispatch simulation, what utility does a planner that takes the map at face value reach when a wasted deployment costs 0.5?"),
]  # fmt: skip
UNANSWERABLE = [
    "Which roads in Mumbai will flood in the next 48 hours?",
    "Forecast the flood extent in Jakarta for next week.",
    "Will Chennai flood tomorrow?",
    "How well does the flood model perform in Jakarta?",
    "How many Sentinel-2 scenes cover Mumbai in July 2023?",
    "What is the water IoU of the radar model in Bangladesh?",
    "How many people died in the 2018 Kerala floods?",
    "How much rain fell during the 2021 Ahr valley floods?",
    "What was the economic damage of the Sindh 2022 floods?",
    "Which neighbourhoods of Chennai should be evacuated first?",
    "How many people were exposed to flooding in Spain?",
    "Who funded the creation of the Sen1Floods11 dataset?",
]


# A second, held-out set, written after the first set's failures were seen and
# before TerraCore was changed in response. The first set ("dev") may be used to
# develop fixes; this one ("test") is run once afterwards. It asks for different
# targets in different words, and adds calibration and event-fact questions.
TEST_PERFORMANCE = [
    ("Nigeria", "s2", "Which IoU score does the {m} U-Net achieve for the {e} flood?"),
    ("Somalia", "s1", "How reliable is {m} flood mapping in {e}, in terms of IoU?"),
    ("Spain", "s1s2", "For {e}, what is the {m} model's water IoU?"),
    ("Ghana", "s2", "What is the intersection over union of the {m} model on the {e} event?"),
    ("USA", "s2", "Tell me how the {m} model scores on the {e} flood."),
    ("India", "s1s2", "What water IoU is reported for the {m} model in {e}?"),
    ("Paraguay", "s1", "How does the {m} model do on the {e} flood, measured by IoU?"),
    ("Sri-Lanka", "s1", "What is the {m} U-Net's IoU for the Sri Lanka event?"),
    ("Mekong", "s2", "What IoU does the {m} model achieve on the {e} flood?"),
    ("Somalia", "s1s2", "Water IoU of the {m} model for {e}?"),
]
TEST_TRANSFER = [
    ("Nigeria", "s1", "When {e} chips are removed from training, what IoU does the {m} model reach on {e}'s test chips?"),
    ("USA", "s2", "How does the {m} model score on {e} after training without any {e} data?"),
    ("Paraguay", "s1s2", "Held-out evaluation: IoU of the {m} model on {e} when {e} is not in training?"),
    ("Mekong", "s1", "If the {m} model never sees {e} during training, what is its IoU there?"),
]  # fmt: skip
TEST_CALIBRATION = [
    ("s1", "bolivia", "ece_scaled",
     "What is the calibration error of the radar ensemble on Bolivia after temperature scaling?"),
    ("s2", "test", "ece",
     "What expected calibration error does the optical ensemble have on the test split before any scaling?"),
    ("s1s2", "bolivia", "error_90",
     "What error rate does the fused ensemble have on Bolivia when it abstains on its least confident 10% of pixels?"),
]  # fmt: skip
TEST_SCENES = [
    ("sindh-2022", "sentinel-1-grd", "2022-09-01", "2022-09-30",
     "How many Sentinel-1 scenes cover the Sindh floods in September 2022?"),
    ("sindh-2022", "sentinel-2-l2a", "2022-08-01", "2022-08-31",
     "How many Sentinel-2 images of Sindh were acquired in August 2022?"),
    ("kerala-2018", "sentinel-2-l2a", "2018-08-15", "2018-08-31",
     "How many Sentinel-2 scenes of Kerala are there between 15 and 31 August 2018?"),
    ("chennai-2015", "sentinel-2-l2a", "2015-11-01", "2015-11-30",
     "How many optical Sentinel-2 scenes cover Chennai in November 2015?"),
]  # fmt: skip
TEST_LICENCES = [
    ("geoboundaries", "cc by 4.0", "How is geoBoundaries licensed?"),
    (
        "landsat-c2-l2",
        "public domain",
        "Can Landsat Collection 2 data be reused, and under what licence?",
    ),
    ("ghsl", "cc by 4.0", "What licence does the Global Human Settlement Layer use?"),
    ("gpm-imerg", "nasa open data", "What are the licence terms for GPM IMERG precipitation data?"),
    (
        "cop-dem-glo-30",
        "copernicus dem licence",
        "Which licence applies to the Copernicus DEM GLO-30?",
    ),
]
TEST_SIMULATION = [
    ("simulation:e7:test:assignment",
     "In the flood response simulation, what does the assignment policy reach on the test split?"),
    ("simulation:e7:bolivia:oracle-lookahead",
     "In the simulation, what does the lookahead planner that sees the true flood extent score on Bolivia?"),
    ("simulation:e9:cost1.0:test:calibrated-uncertainty",
     "In the dispatch simulation with a waste cost of 1, what utility does the planner using calibration plus uncertainty reach on the test split?"),
    ("simulation:e9:cost0.25:bolivia:optical-calibrated",
     "In the dispatch simulation with waste cost 0.25, what utility does the calibrated optical planner reach on Bolivia?"),
]  # fmt: skip
TEST_EVENTS = [
    ("Ghana", "How many hand-labelled chips does the Ghana flood event have?"),
    ("Somalia", "How many labelled chips are there for Somalia in Sen1Floods11?"),
    ("USA", "How many hand-labelled Sen1Floods11 chips come from the USA event?"),
]
TEST_UNANSWERABLE = [
    "How many fatalities were recorded in the Sindh floods?",
    "How many millimetres of rain fell in Kerala in August 2018?",
    "What were the insured losses from the Ahr floods?",
    "How many people were displaced in Chennai in 2015?",
    "What is the IoU of the radar model in Venice?",
    "How many Sentinel-1 scenes cover Lagos in 2020?",
    "Predict next month's flood risk for Kerala.",
    "Should rescue teams go to the Ahr valley or Sindh first?",
    "Which hospitals in Chennai are inside the flood zone?",
    "What IoU does the optical model reach on the 2024 Valencia floods?",
    "Which satellite will pass over Sindh tomorrow?",
    "How deep was the water in the Ahr valley?",
    "What is the average elevation of the Bolivia chips?",
    "How many buildings were damaged in Kerala?",
    "Who drew the hand labels for the Sen1Floods11 chips?",
    "What was the river discharge of the Mekong during the flood?",
]


def build_questions(corpus: list[Evidence], split: str = "dev") -> list[Question]:
    """The question set: "dev" (the first 49) or "test" (the held-out 49)."""
    if split == "test":
        return _build_test(corpus)
    by = {e.id: e for e in corpus}
    qs: list[Question] = []
    for i, (event, m, tpl) in enumerate(PERFORMANCE):
        e = by[f"benchmark:{m}:event:{event}"]
        qs.append(
            Question(
                f"perf-{i}",
                "performance",
                tpl.format(m=MODALITY_WORDS[m], e=event),
                numbers=[round(e.data["iou_mean"], 3)],
                support=[e.id],
            )
        )
    for i, (event, m, tpl) in enumerate(TRANSFER):
        e = by[f"benchmark:{m}:loeo:{event}"]
        qs.append(
            Question(
                f"transfer-{i}",
                "transfer",
                tpl.format(m=MODALITY_WORDS[m], e=event),
                numbers=[round(e.data["iou_test_chips"], 3)],
                support=[e.id],
            )
        )
    for i, (event, coll, start, end, text) in enumerate(SCENES):
        ids = [
            e.id
            for e in corpus
            if e.kind == "scene"
            and e.data.get("event") == event
            and e.data.get("collection") == coll
            and e.start
            and start <= e.start <= end
        ]
        qs.append(Question(f"scenes-{i}", "scenes", text, counts=[len(ids)], support=ids))
    for i, (ds, phrase, text) in enumerate(LICENCES):
        qs.append(
            Question(f"licence-{i}", "licence", text, strings=[phrase], support=[f"dataset:{ds}"])
        )
    for i, (eid, text) in enumerate(SIMULATION):
        e = by[eid]
        value = e.data.get("utility", e.data.get("auc"))
        qs.append(
            Question(f"sim-{i}", "simulation", text, numbers=[round(value, 3)], support=[eid])
        )
    for i, text in enumerate(UNANSWERABLE):
        qs.append(Question(f"none-{i}", "unanswerable", text, answerable=False))
    return qs


def _scene_ids(corpus, event, coll, start, end) -> list[str]:
    return [
        e.id
        for e in corpus
        if e.kind == "scene"
        and e.data.get("event") == event
        and e.data.get("collection") == coll
        and e.start
        and start <= e.start <= end
    ]


def _build_test(corpus: list[Evidence]) -> list[Question]:
    by = {e.id: e for e in corpus}
    qs: list[Question] = []
    for i, (event, m, tpl) in enumerate(TEST_PERFORMANCE):
        e = by[f"benchmark:{m}:event:{event}"]
        text = tpl.format(m=MODALITY_WORDS[m], e=event)
        qs.append(
            Question(
                f"t-perf-{i}",
                "performance",
                text,
                numbers=[round(e.data["iou_mean"], 3)],
                support=[e.id],
            )
        )
    for i, (event, m, tpl) in enumerate(TEST_TRANSFER):
        e = by[f"benchmark:{m}:loeo:{event}"]
        text = tpl.format(m=MODALITY_WORDS[m], e=event)
        qs.append(
            Question(
                f"t-transfer-{i}",
                "transfer",
                text,
                numbers=[round(e.data["iou_test_chips"], 3)],
                support=[e.id],
            )
        )
    for i, (m, split, key, text) in enumerate(TEST_CALIBRATION):
        e = by[f"benchmark:{m}:calibration:{split}"]
        qs.append(
            Question(
                f"t-calib-{i}", "calibration", text, numbers=[round(e.data[key], 3)], support=[e.id]
            )
        )
    for i, (event, coll, start, end, text) in enumerate(TEST_SCENES):
        ids = _scene_ids(corpus, event, coll, start, end)
        qs.append(Question(f"t-scenes-{i}", "scenes", text, counts=[len(ids)], support=ids))
    for i, (ds, phrase, text) in enumerate(TEST_LICENCES):
        qs.append(
            Question(f"t-licence-{i}", "licence", text, strings=[phrase], support=[f"dataset:{ds}"])
        )
    for i, (eid, text) in enumerate(TEST_SIMULATION):
        e = by[eid]
        value = e.data.get("utility", e.data.get("auc"))
        qs.append(
            Question(f"t-sim-{i}", "simulation", text, numbers=[round(value, 3)], support=[eid])
        )
    for i, (event, text) in enumerate(TEST_EVENTS):
        e = by[f"event:s1f11-{event}"]
        chips = int(re.search(r"with (\d+) hand-labelled chips", e.text).group(1))
        qs.append(Question(f"t-event-{i}", "event facts", text, counts=[chips], support=[e.id]))
    for i, text in enumerate(TEST_UNANSWERABLE):
        qs.append(Question(f"t-none-{i}", "unanswerable", text, answerable=False))
    return qs


def fingerprint(questions: list[Question]) -> str:
    blob = json.dumps([q.text for q in questions]).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


# ------------------------------------------------------------------ grading

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")
BRACKETS = re.compile(r"\[[^\]]*\]")


def _numbers(text: str) -> list[str]:
    return NUMBER.findall(BRACKETS.sub(" ", text))


def _has_number(text: str, value: float) -> bool:
    for tok in _numbers(text):
        pct = tok.endswith("%")
        v = float(tok.rstrip("%"))
        if abs(v - value) < 6e-4 or (pct and abs(v / 100 - value) < 6e-4):
            return True
    return False


def _supported_decimal(tok: str, records: list[Evidence]) -> bool:
    """True if the value appears in a cited record, as a field or written in its text.

    The record text is tokenised the same way as the answer, so "0.3-0.5" in both
    reads as 0.3 and 0.5. A percentage matches a fraction (14.1% = 0.141). A mean
    the critic computed over several records is not in any one of them and counts
    as unsupported here, so TerraCore's rate is slightly overstated.
    """
    v = float(tok.rstrip("%"))
    candidates = {v, v / 100} if tok.endswith("%") else {v}
    for e in records:
        values = [
            x for x in e.data.values() if isinstance(x, (int, float)) and not isinstance(x, bool)
        ]
        for t in _numbers(e.text):
            x = float(t.rstrip("%"))
            values += [x, x / 100] if t.endswith("%") else [x]
        if any(abs(c - x) < 6e-4 for c in candidates for x in values):
            return True
    return False


def grade(q: Question, out: dict, store: dict[str, Evidence]) -> dict:
    text = out["text"] or ""
    cited = [i for i in out["cited"] if i in store]
    if not q.answerable:
        correct = out["abstained"]
    else:
        correct = (
            not out["abstained"]
            and all(_has_number(text, v) for v in q.numbers)
            and all(re.search(rf"(?<!\d){c}(?!\d)", BRACKETS.sub(" ", text)) for c in q.counts)
            and all(s in text.lower() for s in q.strings)
        )
    decimals = [t for t in _numbers(text) if "." in t]
    unsupported = [t for t in decimals if not _supported_decimal(t, [store[i] for i in cited])]
    return {
        "correct": bool(correct),
        "abstained": out["abstained"],
        "grounded": bool(set(cited) & set(q.support)) if q.answerable else None,
        "invalid_citations": len(out["cited"]) - len(cited),
        "citations": len(cited),
        "decimals": len(decimals),
        "unsupported_decimals": len(unsupported),
    }


# ------------------------------------------------------------------ systems


class TerraCoreSystem:
    name = "terracore"

    def __init__(self, retriever):
        from terramind.agents.workflow import TerraCore

        self.core = TerraCore(retriever)

    def answer(self, question: str) -> dict:
        r = self.core.ask(question)
        return {
            "text": "" if r["verdict"] == "abstain" else r["answer"],
            "cited": [e["id"] for e in r["evidence"]],
            "abstained": r["verdict"] == "abstain",
        }


class FirstVersionSystem(TerraCoreSystem):
    """TerraCore as it was before the relevance fixes (relevance=False)."""

    name = "terracore-v1"

    def __init__(self, retriever):
        from terramind.agents.workflow import TerraCore

        self.core = TerraCore(retriever, relevance=False)


class NoCriticSystem(TerraCoreSystem):
    """Same agents and reporter; the critic accepts every claim."""

    name = "terracore-no-critic"

    def __init__(self, retriever):
        from terramind.agents.workflow import TerraCore

        class Trusting(TerraCore):
            def critic(self, state):
                claims = state.get("claims", [])
                return {
                    "claims": claims,
                    "rejected": [],
                    "verdict": "answer" if claims else "abstain",
                    "reasons": list(state.get("reasons", []))
                    + ([] if claims else ["No claims were produced."]),
                    "relaxed": state.get("relaxed", False),
                }

        self.core = Trusting(retriever)


class FixedPipeline:
    """Retrieve once and answer with the best record's text. Never refuses."""

    name = "fixed-pipeline"

    def __init__(self, retriever):
        self.retriever = retriever
        from terramind.rag.query import build_gazetteer

        self.gazetteer = build_gazetteer(retriever.corpus)

    def answer(self, question: str) -> dict:
        hits = self.retriever.retrieve(parse(question, self.gazetteer), k=5, mode="hybrid")
        if not hits:
            return {"text": "", "cited": [], "abstained": True}
        top = hits[0].evidence
        return {"text": top.text, "cited": [top.id], "abstained": False}


LLM_PROMPT = """You answer questions about a flood-mapping research project using ONLY the
evidence records below. Rules:
- Use only facts stated in the records. Do not use outside knowledge.
- Cite every record you use by writing its id in square brackets, e.g. [benchmark:s1:test].
- Copy numbers exactly as they appear in the records.
- If the records do not contain the answer, reply exactly: INSUFFICIENT EVIDENCE
- Answer in at most three sentences.

Records:
{records}

Question: {question}
Answer:"""


class LLMAgent:
    """One local language model with the top retrieved records as context."""

    def __init__(self, retriever, model: str = "llama3:8b", host: str = "http://127.0.0.1:11434"):
        from terramind.rag.query import build_gazetteer

        self.retriever, self.model, self.host = retriever, model, host
        self.gazetteer = build_gazetteer(retriever.corpus)
        self.name = f"llm-agent ({model})"

    def answer(self, question: str) -> dict:
        hits = self.retriever.retrieve(parse(question, self.gazetteer), k=8, mode="hybrid")
        records = "\n".join(f"[{h.evidence.id}] {h.evidence.text[:500]}" for h in hits)
        res = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": LLM_PROMPT.format(records=records, question=question),
                "stream": False,
                "options": {"temperature": 0, "seed": 0, "num_ctx": 4096},
            },
            timeout=300,
        )
        res.raise_for_status()
        text = res.json().get("response", "").strip()
        abstained = "INSUFFICIENT EVIDENCE" in text.upper() and len(text) < 60
        cited = re.findall(r"\[([a-z0-9]+:[^\]\s]+)\]", text)
        return {"text": "" if abstained else text, "cited": cited, "abstained": abstained}


# --------------------------------------------------------------------- run


def _summary(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if not r["abstained"]]
    decimals = sum(r["decimals"] for r in rows)
    return {
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "answerable_correct": sum(r["correct"] for r in ans) / len(ans),
        "unanswerable_refused": sum(r["correct"] for r in neg) / len(neg) if neg else None,
        "over_refusal": sum(r["abstained"] for r in ans) / len(ans),
        "grounded_when_answering": (
            sum(bool(r["grounded"]) for r in answered) / len(answered) if answered else None
        ),
        "unsupported_decimal_rate": (
            sum(r["unsupported_decimals"] for r in rows) / decimals if decimals else 0.0
        ),
        "invalid_citations": sum(r["invalid_citations"] for r in rows),
        "mean_citations": sum(r["citations"] for r in rows) / len(rows),
        "seconds_per_question": sum(r["seconds"] for r in rows) / len(rows),
    }


def paired_differences(systems: dict, reference: str = "terracore", n: int = 10000) -> dict:
    """Accuracy of the reference minus each other system, 95% bootstrap over questions."""
    import numpy as np

    correct = {k: np.array([r["correct"] for r in s["rows"]], float) for k, s in systems.items()}
    if reference not in correct:
        return {}
    draws = np.random.default_rng(0).integers(
        len(correct[reference]), size=(n, len(correct[reference]))
    )
    out = {}
    for name, c in correct.items():
        if name == reference:
            continue
        d = correct[reference] - c
        means = d[draws].mean(1)
        out[f"{reference} - {name}"] = {
            "mean": float(d.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        }
    return out


def run(llm_model: str | None = "llama3:8b", split: str = "dev", out: Path | None = None) -> Path:
    from terramind.evidence import store
    from terramind.rag.retriever import Retriever
    from terramind.training.runs import runs_dir

    corpus = store.load_corpus()
    by = {e.id: e for e in corpus}
    retriever = Retriever(corpus)
    questions = build_questions(corpus, split)
    systems = [
        TerraCoreSystem(retriever),
        FirstVersionSystem(retriever),
        NoCriticSystem(retriever),
        FixedPipeline(retriever),
    ]
    if llm_model:
        systems.append(LLMAgent(retriever, llm_model))

    report = {
        "questions": [asdict(q) for q in questions],
        "split": split,
        "fingerprint": fingerprint(questions),
        "systems": {},
    }
    for system in systems:
        rows = []
        for q in questions:
            started = time.time()
            try:
                out_ = system.answer(q.text)
                error = None
            except Exception as err:  # noqa: BLE001 - a failed call counts as a wrong answer
                out_, error = {"text": "", "cited": [], "abstained": False}, repr(err)[:200]
            g = grade(q, out_, by)
            rows.append(
                {
                    "id": q.id,
                    "category": q.category,
                    "answerable": q.answerable,
                    **g,
                    "seconds": time.time() - started,
                    "answer": out_["text"][:600],
                    "cited": out_["cited"][:12],
                    "error": error,
                }
            )
        by_cat = {}
        for cat in dict.fromkeys(q.category for q in questions):
            sub = [r for r in rows if r["category"] == cat]
            by_cat[cat] = sum(r["correct"] for r in sub) / len(sub)
        report["systems"][system.name] = {
            "summary": _summary(rows),
            "by_category": by_cat,
            "rows": rows,
        }
        print(f"{system.name}: {report['systems'][system.name]['summary']}", flush=True)

    report["paired_accuracy"] = paired_differences(report["systems"])
    path = out or runs_dir() / f"agents-rq5-{split}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
