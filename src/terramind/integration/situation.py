"""One report per chip that runs the whole chain.

    perception -> uncertainty -> evidence on reliability -> simulated response

Each statement carries the ids of the evidence it rests on, or "computed" when
it was calculated here from the model outputs. Statements about the response
are simulated and marked so. Thresholds are hand-set and listed in the report.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from terramind.evidence import store
from terramind.sim import scenarios as sc
from terramind.sim.env import SimConfig
from terramind.sim.evaluate import make_policy, replay

LOW_IOU = 0.40  # below this the event's map is treated as unreliable
GOOD_IOU = 0.65
MIN_WATER = sc.MIN_WATER
THRESHOLDS = {
    "low_reliability_event_iou": LOW_IOU,
    "good_reliability_event_iou": GOOD_IOU,
    "min_water_share_for_a_response": MIN_WATER,
}


@lru_cache(maxsize=1)
def _evidence() -> dict[str, store.Evidence]:
    return {e.id: e for e in store.load_corpus()}


@lru_cache(maxsize=1)
def _scenarios() -> sc.Scenarios:
    return sc.load("s1")


def _statement(text: str, evidence: list[str], simulated: bool = False) -> dict:
    return {"text": text, "evidence": evidence, "simulated": simulated}


def build(chip_id: str, depot: int = 0) -> dict:
    data, evidence = _scenarios(), _evidence()
    matches = np.flatnonzero(data.ids == chip_id)
    if len(matches) == 0:
        raise LookupError(f"unknown chip '{chip_id}'")
    i = int(matches[0])
    event, split = str(data.events[i]), str(data.splits[i])
    chip_ref = [f"chip:{chip_id}"] if f"chip:{chip_id}" in evidence else ["computed"]

    # 1. Perception and uncertainty, computed from the ensemble output.
    weight = data.known[i] / max(float(data.known[i].sum()), 1e-6)
    predicted = float((data.pred[i] * weight).sum())
    calibrated = float((data.calibrated[i] * weight).sum())
    disagreement = float((data.unc[i] * weight).sum())
    all_disagreement = (data.unc * data.known).sum((1, 2)) / np.maximum(
        data.known.sum((1, 2)), 1e-6
    )
    rank = float((all_disagreement < disagreement).mean())
    perception = [
        _statement(
            f"The radar ensemble estimates water over {predicted:.1%} of the labelled area of this "
            f"chip ({calibrated:.1%} after temperature scaling).",
            ["computed", *chip_ref],
        ),
        _statement(
            f"Ensemble members disagree by {disagreement:.3f} on average, more than on "
            f"{rank:.0%} of all chips.",
            ["computed"],
        ),
    ]

    # 2. Reliability, from stored benchmark evidence for this flood event.
    trust, event_iou, flags = [], None, []
    seen = evidence.get(f"benchmark:s1:event:{event}")
    held_out = evidence.get(f"benchmark:s1:loeo:{event}")
    for record in (seen, held_out):
        if record:
            trust.append(_statement(record.text, [record.id]))
    if held_out and held_out.data.get("iou_test_chips") is not None:
        event_iou = held_out.data["iou_test_chips"]
    elif seen:
        event_iou = seen.data.get("iou_mean")
    if event_iou is None:
        reliability = "unknown"
        flags.append("No benchmark result exists for this flood event, so reliability is unknown.")
    elif event_iou < LOW_IOU:
        reliability = "low"
        flags.append(
            f"The radar model scores IoU {event_iou:.2f} on this event, below {LOW_IOU:.2f}: "
            "treat the map as indicative only and confirm with optical imagery or reports."
        )
    elif event_iou < GOOD_IOU:
        reliability = "moderate"
    else:
        reliability = "good"
    if split == "train":
        flags.append(
            "This chip was in the model's training data, so its map is better than it would be on new ground."
        )
    if rank >= 0.9:
        flags.append("Ensemble disagreement on this chip is in the top 10% of all chips.")

    # 3. Response, simulated on this chip.
    response, abstain = [], False
    if float(data.truth[i].mean()) < MIN_WATER and predicted < MIN_WATER:
        abstain = True
        response.append(
            _statement(
                f"Less than {MIN_WATER:.0%} of the chip is predicted or labelled as water, so no "
                "response is simulated.",
                ["computed"],
            )
        )
    else:
        config = SimConfig()
        informed = replay(data, chip_id, make_policy("lookahead"), depot, config)
        blind = replay(data, chip_id, make_policy("blind-lookahead"), depot, config)
        response.append(
            _statement(
                f"On this chip, {config.teams} simulated teams following a lookahead plan on the "
                f"model's map reach on average {informed['auc']:.1%} of the flooded area over "
                f"{config.horizon} moves ({informed['served'][-1]:.1%} by the end), against "
                f"{blind['auc']:.1%} for the same plan ignoring the model.",
                ["computed"],
                simulated=True,
            )
        )
        general = "bolivia" if split == "bolivia" else "test"
        for name in ("lookahead", "ppo"):
            record = evidence.get(f"simulation:e7:{general}:{name}")
            if record:
                response.append(_statement(record.text, [record.id], simulated=True))
        if reliability == "low":
            abstain = True

    return {
        "chip": chip_id,
        "event": event,
        "split": split,
        "reliability": reliability,
        "event_iou": event_iou,
        "abstain": abstain,
        "recommendation": (
            "No allocation is recommended from this map alone."
            if abstain
            else "The model's map is usable for planning in simulation; see the caveats."
        ),
        "flags": flags,
        "sections": [
            {"title": "What the model sees", "statements": perception},
            {"title": "How far to trust it", "statements": trust},
            {"title": "Simulated response", "statements": response},
        ],
        "caveats": [
            (
                "The response is simulated: teams and scoring are invented, and need is flooded "
                "area, not people or assets."
            ),
            "Labels include permanent water; no population, road or elevation data is used.",
            "Reliability thresholds are hand-set, not calibrated.",
        ],
        "thresholds": THRESHOLDS,
    }


NARRATE_PROMPT = """Write a short plain-language summary (at most five sentences) of this flood
situation report for a non-specialist. Rules: use only the statements below; keep
every number exactly as written; say clearly which parts are simulated; add no new
facts, numbers, places or dates; no preamble.

Recommendation: {recommendation}
Flags:
{flags}
Statements:
{statements}
"""


def narrate(report: dict, llm=None) -> dict:
    """A local LLM's plain-language summary, kept only if it adds no numbers.

    Returns {"text", "accepted", "reason"}. The structured report stays the source
    of truth; the summary is a convenience and is dropped when it cannot be checked.
    """
    import re

    from terramind.agents.llm import OllamaLLM

    llm = llm or OllamaLLM()
    statements = [
        ("[simulated] " if st["simulated"] else "") + st["text"]
        for sec in report["sections"]
        for st in sec["statements"]
    ]
    source = "\n".join([report["recommendation"], *report["flags"], *statements])
    prompt = NARRATE_PROMPT.format(
        recommendation=report["recommendation"],
        flags="\n".join(f"- {f}" for f in report["flags"]) or "- none",
        statements="\n".join(f"- {s}" for s in statements),
    )
    draft = llm.generate(prompt) if hasattr(llm, "generate") else llm.rewrite("", prompt)
    if not draft:
        return {"text": None, "accepted": False, "reason": "the local model gave no answer"}
    allowed = set(re.findall(r"\d+\.?\d*", source))
    extra = sorted(set(re.findall(r"\d+\.?\d*", draft)) - allowed)
    if extra:
        return {"text": None, "accepted": False, "reason": f"draft added numbers {extra[:5]}"}
    return {"text": draft.strip(), "accepted": True, "reason": None}
