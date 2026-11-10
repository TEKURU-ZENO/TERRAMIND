"""Compare policies on fixed simulated episodes (experiment E7).

Episodes are fixed: every scenario in a split is played from the same
`DEPOTS_PER_SCENARIO` depots by every policy, so differences are paired.
Scores are averaged per scenario first; confidence intervals resample scenarios.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from terramind.sim import scenarios as sc
from terramind.sim.env import BatchedFloodResponse, SimConfig, border_cells
from terramind.sim.policies import Assignment, Greedy, Lookahead, RandomWalk, rollout

DEPOTS_PER_SCENARIO = 4
BETAS = (0.5, 1.0, 2.0, 4.0)
EVAL_SPLITS = ("test", "bolivia")
TUNE_SPLIT = "valid"


def episodes(data: sc.Scenarios, split: str, seed: int = 0):
    """(scenario indices, depot indices) for a split; the same for every policy."""
    idx = data.select(split)
    rng = np.random.default_rng(seed)
    n_border = len(border_cells(data.truth.shape[-1]))
    depots = np.stack([rng.choice(n_border, DEPOTS_PER_SCENARIO, replace=False) for _ in idx])
    return np.repeat(idx, DEPOTS_PER_SCENARIO), depots.reshape(-1)


def per_scenario(values: np.ndarray) -> np.ndarray:
    return values.reshape(-1, DEPOTS_PER_SCENARIO).mean(1)


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(n, len(values)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def play(data, policy, split: str, config: SimConfig) -> dict[str, np.ndarray]:
    env = BatchedFloodResponse(data.truth, data.pred, data.unc, config)
    scenario, depot = episodes(data, split)
    return {k: per_scenario(v) for k, v in rollout(env, policy, scenario, depot).items()}


def _summary(scores: dict[str, np.ndarray]) -> dict:
    lo, hi = bootstrap_ci(scores["auc"])
    return {
        "auc": float(scores["auc"].mean()),
        "auc_ci95": [lo, hi],
        "served": float(scores["served"].mean()),
        "scenarios": len(scores["auc"]),
    }


def ppo_runs(modality: str, tag: str) -> list[Path]:
    """Latest finished run per seed for a tag ('ppo' or 'ppo-nounc')."""
    from terramind.sim.ppo import runs_dir

    by_seed = {}
    for path in sorted(runs_dir().glob(f"{tag}-{modality}-seed*/report.json")):
        by_seed[json.loads(path.read_text(encoding="utf-8"))["config"]["seed"]] = path.parent
    return [by_seed[s] for s in sorted(by_seed)]


def run(modality: str = "s1", config: SimConfig | None = None) -> Path:
    from terramind.sim.ppo import load_policy, runs_dir

    config = config or SimConfig()
    data = sc.load(modality)

    # The exploration weight is chosen on the tuning split only.
    tuning = {}
    for beta in BETAS:
        scores = play(data, Lookahead(beta=beta), TUNE_SPLIT, config)
        tuning[beta] = float(scores["auc"].mean())
    tuning[0.0] = float(play(data, Lookahead(), TUNE_SPLIT, config)["auc"].mean())
    best_beta = max((b for b in BETAS), key=lambda b: tuning[b])

    policies = {
        "random": RandomWalk(),
        "blind-lookahead": Lookahead(value="blind"),
        "greedy": Greedy(),
        "assignment": Assignment(),
        "lookahead": Lookahead(),
        "lookahead+uncertainty": Lookahead(beta=best_beta),
        "oracle-lookahead": Lookahead(value="oracle"),
    }
    learned = {
        "ppo": [load_policy(p) for p in ppo_runs(modality, "ppo")],
        "ppo-nounc": [load_policy(p) for p in ppo_runs(modality, "ppo-nounc")],
        # Most-likely-move execution, kept to show why moves are sampled instead.
        "ppo-argmax": [load_policy(p, sample=False) for p in ppo_runs(modality, "ppo")],
    }

    report = {
        "simulated": True,
        "modality": modality,
        "config": {"teams": config.teams, "horizon": config.horizon, "sight": config.sight},
        "grid": int(data.truth.shape[-1]),
        "depots_per_scenario": DEPOTS_PER_SCENARIO,
        "beta_tuning": {"split": TUNE_SPLIT, "auc_by_beta": tuning, "chosen": best_beta},
        "results": {},
    }
    for split in EVAL_SPLITS:
        raw = {name: play(data, p, split, config) for name, p in policies.items()}
        rows = {name: _summary(s) for name, s in raw.items()}
        for name, members in learned.items():
            if not members:
                continue
            per_seed = [play(data, p, split, config) for p in members]
            mean_scores = {k: np.mean([s[k] for s in per_seed], axis=0) for k in ("auc", "served")}
            raw[name] = mean_scores
            rows[name] = _summary(mean_scores)
            seed_auc = [float(s["auc"].mean()) for s in per_seed]
            rows[name]["seeds"] = len(members)
            rows[name]["auc_seed_std"] = float(np.std(seed_auc, ddof=1)) if len(members) > 1 else None
            rows[name]["auc_by_seed"] = seed_auc

        # Paired differences on the same episodes.
        pairs = [
            ("ppo", "lookahead"),
            ("ppo", "assignment"),
            ("ppo", "ppo-nounc"),
            ("lookahead+uncertainty", "lookahead"),
            ("lookahead", "blind-lookahead"),
            ("oracle-lookahead", "lookahead"),
        ]
        diffs = {}
        for a, b in pairs:
            if a in raw and b in raw:
                d = raw[a]["auc"] - raw[b]["auc"]
                lo, hi = bootstrap_ci(d)
                diffs[f"{a} - {b}"] = {"mean": float(d.mean()), "ci95": [lo, hi]}

        events = data.events[data.select(split)]
        by_event = {
            str(ev): {name: float(s["auc"][events == ev].mean()) for name, s in raw.items()}
            for ev in np.unique(events)
        }
        report["results"][split] = {"policies": rows, "paired_auc": diffs, "by_event": by_event}

    out = runs_dir() / f"e7-{modality}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


REPLAY_POLICIES = ("blind-lookahead", "greedy", "assignment", "lookahead", "oracle-lookahead", "ppo")


def make_policy(name: str, modality: str = "s1"):
    """A policy by the name used in the E7 table. 'ppo' loads the lowest trained seed."""
    if name == "ppo":
        from terramind.sim.ppo import load_policy

        runs = ppo_runs(modality, "ppo")
        if not runs:
            raise LookupError("no trained PPO policy; run `terramind sim-train`")
        return load_policy(runs[0])
    table = {
        "blind-lookahead": lambda: Lookahead(value="blind"),
        "greedy": Greedy,
        "assignment": Assignment,
        "lookahead": Lookahead,
        "oracle-lookahead": lambda: Lookahead(value="oracle"),
    }
    if name not in table:
        raise LookupError(f"unknown policy '{name}'")
    return table[name]()


def replay(data: sc.Scenarios, chip_id: str, policy, depot: int = 0, config: SimConfig | None = None):
    """Play one episode and return every step, for display."""
    matches = np.flatnonzero(data.ids == chip_id)
    if len(matches) == 0:
        raise LookupError(f"unknown chip '{chip_id}'")
    i = int(matches[0])
    env = BatchedFloodResponse(data.truth, data.pred, data.unc, config or SimConfig())
    env.reset([i], [depot % len(env.borders)])
    positions, served = [env.pos[0].tolist()], [float(env.cumulative[0])]
    while not env.done:
        env.step(policy.act(env))
        positions.append(env.pos[0].tolist())
        served.append(float(env.cumulative[0]))
    m = env.metrics()
    return {
        "simulated": True,
        "chip": chip_id,
        "split": str(data.splits[i]),
        "grid": env.grid,
        "teams": env.cfg.teams,
        "horizon": env.cfg.horizon,
        "truth": np.round(data.truth[i], 3).tolist(),
        "prediction": np.round(data.pred[i], 3).tolist(),
        "positions": positions,
        "served": served,
        "auc": float(m["auc"][0]),
        "total_water_share": float(data.truth[i].mean()),
    }
