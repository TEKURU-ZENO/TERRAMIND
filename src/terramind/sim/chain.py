"""E8: how much each link of the chain is worth to the decision (simulated).

The planner is fixed (lookahead). What changes is what it is given, from
nothing, through cruder and better perception, to the hidden truth, under
three settings of increasing difficulty. This measures decision quality as a
function of the perception and uncertainty stages that feed it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from terramind.sim import scenarios as sc
from terramind.sim.env import BatchedFloodResponse, SimConfig
from terramind.sim.evaluate import EVAL_SPLITS, TUNE_SPLIT, bootstrap_ci, episodes, per_scenario
from terramind.sim.policies import Lookahead, rollout

SETTINGS = {
    "standard": SimConfig(teams=3, horizon=48, sight=1),
    "scarce": SimConfig(teams=2, horizon=24, sight=1),
    # sight 0: a team learns the truth of a cell only by entering it, so the
    # planner can never correct the model's picture of cells it has not visited.
    "scarce, no reveal": SimConfig(teams=2, horizon=24, sight=0),
}
# Positive weights seek out uncertain cells, negative weights avoid them.
BETAS = (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0)
REFERENCE = "radar ensemble"


def _play(truth, pred, unc, policy, scenario, depot, config) -> np.ndarray:
    env = BatchedFloodResponse(truth, pred, unc, config)
    return per_scenario(rollout(env, policy, scenario, depot)["auc"])


def run() -> Path:
    from terramind.sim.ppo import runs_dir

    data = {m: sc.load(m) for m in ("s1", "s2", "s1s2")}
    s1 = data["s1"]
    for other in ("s2", "s1s2"):
        assert (data[other].ids == s1.ids).all() and np.allclose(data[other].truth, s1.truth)
    zeros = np.zeros_like(s1.unc)

    report = {"simulated": True, "planner": "lookahead", "reference": REFERENCE, "settings": {}}
    for setting, config in SETTINGS.items():
        scenario, depot = episodes(s1, TUNE_SPLIT)
        tuning = {
            beta: float(
                _play(
                    s1.truth, s1.pred, s1.unc, Lookahead(beta=beta), scenario, depot, config
                ).mean()
            )
            for beta in BETAS
        }
        beta = max(tuning, key=tuning.get)

        # name -> (what the planner sees, uncertainty map, planner)
        stages = {
            "no perception": (s1.pred, zeros, Lookahead(value="blind")),
            "radar ensemble, thresholded map": (s1.hard, zeros, Lookahead()),
            "radar single model": (s1.single, zeros, Lookahead()),
            REFERENCE: (s1.pred, zeros, Lookahead()),
            "radar ensemble, calibrated": (s1.calibrated, zeros, Lookahead()),
            "radar ensemble + uncertainty": (s1.pred, s1.unc, Lookahead(beta=beta)),
            "optical ensemble": (data["s2"].pred, zeros, Lookahead()),
            "fused ensemble": (data["s1s2"].pred, zeros, Lookahead()),
            "truth (oracle)": (s1.pred, zeros, Lookahead(value="oracle")),
        }
        result = {
            "config": {"teams": config.teams, "horizon": config.horizon, "sight": config.sight},
            "uncertainty_weight": {"tuned_on": TUNE_SPLIT, "auc_by_weight": tuning, "chosen": beta},
            "splits": {},
        }
        for split in EVAL_SPLITS:
            scenario, depot = episodes(s1, split)
            scores = {
                name: _play(s1.truth, pred, unc, planner, scenario, depot, config)
                for name, (pred, unc, planner) in stages.items()
            }
            rows = {}
            for name, auc in scores.items():
                diff = auc - scores[REFERENCE]
                rows[name] = {
                    "auc": float(auc.mean()),
                    "auc_ci95": list(bootstrap_ci(auc)),
                    "vs_reference": float(diff.mean()),
                    "vs_reference_ci95": list(bootstrap_ci(diff)),
                }
            result["splits"][split] = {"scenarios": len(scores[REFERENCE]), "stages": rows}
        report["settings"][setting] = result

    out = runs_dir() / "e8-chain.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
