"""A harder flood-response task in which mistakes cost something (simulated).

The first environment forgave perception error: entering a cell served it for
free, so a wrong guess cost one step. Here moving and deploying are separate.
A deployment takes a step, serves the cell, and is *wasted* if the cell turns
out to be dry, at a cost of `waste_cost` fully flooded cells. Nothing is
revealed until a team deploys, and then only around that cell. A policy must
therefore decide, cell by cell, whether the chance of water is worth the risk,
which is the kind of decision calibrated probabilities and uncertainty are
supposed to help with.

Score ("utility") = (flooded area served - waste_cost x wasted deployments)
/ total flooded area. It can be negative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from terramind.sim import scenarios as sc
from terramind.sim.env import MOVES, BatchedFloodResponse, SimConfig
from terramind.sim.evaluate import (
    EVAL_SPLITS,
    TUNE_SPLIT,
    bootstrap_ci,
    episodes,
    per_scenario,
)

WAIT, DEPLOY = 0, len(MOVES)  # actions: 0 wait, 1-4 move, 5 deploy
N_ACTIONS = len(MOVES) + 1
WASTE_COSTS = (0.25, 0.5, 1.0)
# Learned policies by run-directory tag: PPO from scratch, imitation of the
# calibrated planner, imitation then PPO, and imitation then MAPPO (global critic).
LEARNED = (
    ("dispatch", "ppo"),
    ("dispatch-nounc", "ppo, no uncertainty input"),
    ("dispatch-bc", "imitation"),
    ("dispatch-bcppo", "imitation + ppo"),
    ("dispatch-bcppo-nounc", "imitation + ppo, no uncertainty input"),
    ("dispatch-mappo", "imitation + mappo"),
)
REFERENCE = "calibrated"


@dataclass(frozen=True)
class DispatchConfig:
    teams: int = 3
    horizon: int = 48
    waste_cost: float = 0.5
    dry: float = 0.05  # a cell with less water than this counts as dry
    reveal: int = 1  # cells around a deployment whose truth becomes known


class DispatchEnv(BatchedFloodResponse):
    actor_channels = 9
    critic_channels = 10
    n_actions = N_ACTIONS

    def __init__(self, truth, pred, unc, config: DispatchConfig | None = None):
        self.dcfg = config or DispatchConfig()
        super().__init__(truth, pred, unc, SimConfig(self.dcfg.teams, self.dcfg.horizon, 0))

    def _arrive(self) -> np.ndarray:
        """Arriving at a cell does nothing here; only deploying serves it."""
        return np.zeros(self.batch, np.float32)

    def reset(self, scenario, depot, transform=None) -> None:
        super().reset(scenario, depot, transform)
        self.wasted = np.zeros(self.batch, np.float32)
        self.deployed = np.zeros(self.batch, np.float32)

    def action_mask(self) -> np.ndarray:
        """[B, K, 6]: moves must stay on the map; deploy needs an unserved cell."""
        moves = super().action_mask()
        rows = np.arange(self.batch)[:, None]
        fresh = ~self.served[rows, self.pos[..., 0], self.pos[..., 1]]
        return np.concatenate([moves, fresh[..., None]], axis=-1)

    def step(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions)
        deploy = actions == DEPLOY
        nxt = self.pos + MOVES[np.where(deploy, WAIT, actions)]
        ok = ((nxt >= 0) & (nxt < self.grid)).all(-1, keepdims=True)
        self.pos = np.where(ok, nxt, self.pos)

        rows = np.arange(self.batch)
        reward = np.zeros(self.batch, np.float32)
        r = self.dcfg.reveal
        for k in range(self.cfg.teams):
            row, col = self.pos[:, k, 0], self.pos[:, k, 1]
            act = deploy[:, k] & ~self.served[rows, row, col]
            water = self.truth[rows, row, col]
            waste = act & (water < self.dcfg.dry)
            reward += np.where(act, water, 0.0) - self.dcfg.waste_cost * waste
            self.cumulative += np.where(act, water, 0.0) / self.total
            self.wasted += waste
            self.deployed += act
            self.served[rows[act], row[act], col[act]] = True
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    rr = np.clip(row[act] + dr, 0, self.grid - 1)
                    cc = np.clip(col[act] + dc, 0, self.grid - 1)
                    self.seen[rows[act], rr, cc] = True
        self.t += 1
        return reward / self.total

    def metrics(self) -> dict[str, np.ndarray]:
        waste = self.dcfg.waste_cost * self.wasted / self.total
        return {
            "utility": self.cumulative - waste,
            "served": self.cumulative.copy(),
            "wasted": self.wasted.copy(),
            "deployed": self.deployed.copy(),
        }

    global_channels = 10

    def global_state(self, use_uncertainty: bool = True) -> np.ndarray:
        """[B, K, 10, G, G]: the whole map from each team's seat, for a MAPPO critic.

        Channels: belief, uncertainty, served, seen, every team, this team, time,
        team id, waste cost, and the hidden truth (training only).
        """
        b, k, g = self.batch, self.cfg.teams, self.grid
        rows = np.arange(b)
        everyone = np.zeros((b, g, g), np.float32)
        own = np.zeros((b, k, g, g), np.float32)
        for i in range(k):
            np.add.at(everyone, (rows, self.pos[:, i, 0], self.pos[:, i, 1]), 1.0)
            own[rows, i, self.pos[:, i, 0], self.pos[:, i, 1]] = 1.0
        unc = self.uncertainty() * 4.0 if use_uncertainty else np.zeros_like(self.pred)
        shared = np.stack(
            [
                self.belief(),
                unc,
                self.served.astype(np.float32),
                self.seen.astype(np.float32),
                everyone,
            ],
            axis=1,
        ).astype(np.float32)
        out = np.empty((b, k, self.global_channels, g, g), np.float32)
        out[:, :, :5] = shared[:, None]
        out[:, :, 5] = own
        out[:, :, 6] = self.t / self.cfg.horizon
        out[:, :, 7] = (np.arange(k, dtype=np.float32) / max(k - 1, 1))[None, :, None, None]
        out[:, :, 8] = self.dcfg.waste_cost
        out[:, :, 9] = self.truth[:, None]
        return out

    def observe(self, use_uncertainty: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """The base view plus a team-id plane and the waste cost, then truth for the critic.

        The team id breaks the symmetry that made teams sharing one actor move as one.
        """
        _, base = super().observe(use_uncertainty)
        b, k, _, s, _ = base.shape
        ident = np.broadcast_to(
            (np.arange(k, dtype=np.float32) / max(k - 1, 1))[None, :, None, None, None],
            (b, k, 1, s, s),
        )
        cost = np.full((b, k, 1, s, s), self.dcfg.waste_cost, np.float32)
        critic = np.concatenate([base[:, :, :7], ident, cost, base[:, :, 7:]], axis=2)
        return critic[:, :, : self.actor_channels], critic


# -- what a planner believes about each cell ---------------------------------


class Belief:
    """Maps the model's cell outputs to (expected water, chance the cell is dry)."""

    name = "base"

    def fit(self, pred, unc, truth, dry: float) -> Belief:
        return self

    def __call__(self, pred, unc, dry: float):
        raise NotImplementedError


class Raw(Belief):
    """Takes the model's map at face value: dry exactly when it says so."""

    name = "raw map"

    def __call__(self, pred, unc, dry):
        return pred, (pred < dry).astype(np.float32)


class Calibrated(Belief):
    """Chance of a dry cell from a logistic model fitted on validation cells.

    With `use_uncertainty` the ensemble disagreement is an extra input, so the
    difference between the two variants is exactly what uncertainty adds.
    """

    def __init__(self, use_uncertainty: bool = False):
        self.use_uncertainty = use_uncertainty
        self.name = "calibrated + uncertainty" if use_uncertainty else "calibrated"

    def _features(self, pred, unc):
        cols = [pred, np.sqrt(pred), np.log(pred + 1e-3)]
        if self.use_uncertainty:
            cols += [unc, np.log(unc + 1e-4), unc / (pred + 0.05)]
        return np.stack([c.reshape(-1) for c in cols], axis=1)

    def fit(self, pred, unc, truth, dry):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=2000))
        self.model.fit(self._features(pred, unc), (truth < dry).reshape(-1))
        return self

    def __call__(self, pred, unc, dry):
        p_dry = self.model.predict_proba(self._features(pred, unc))[:, 1]
        return pred, p_dry.reshape(pred.shape).astype(np.float32)


class Dispatcher:
    """Planner: deploy where expected utility is positive, nearest and best first.

    Expected utility of deploying to a cell = expected water - waste_cost x
    chance it is dry. Cells already revealed use their true values. Teams claim
    targets in turn, move toward them, and deploy on arrival; with no cell worth
    the risk, a team waits.
    """

    def __init__(self, belief: Belief | None, decay: float = 0.85, oracle: bool = False):
        self.belief, self.decay, self.oracle = belief, decay, oracle
        self._grid = None

    def _prepare(self, grid: int) -> None:
        if self._grid != grid:
            self._grid = grid
            self.cells = np.stack(np.unravel_index(np.arange(grid * grid), (grid, grid)), 1)
            self.reach = self.decay ** np.abs(self.cells[:, None] - self.cells[None]).sum(-1)

    def utilities(self, env: DispatchEnv) -> np.ndarray:
        cost, dry = env.dcfg.waste_cost, env.dcfg.dry
        known = env.truth - cost * (env.truth < dry)
        if self.oracle:
            u = known
        else:
            if getattr(self, "_cached_for", None) is not env.pred:
                self._cached = self.belief(env.pred, env.unc, dry)
                self._cached_for = env.pred
            expected, p_dry = self._cached
            u = np.where(env.seen, known, expected - cost * p_dry)
        return np.where(env.served, -1.0, u).reshape(env.batch, -1)

    def act(self, env: DispatchEnv) -> np.ndarray:
        self._prepare(env.grid)
        g = env.grid
        utilities = self.utilities(env)
        actions = np.zeros((env.batch, env.cfg.teams), int)
        for b in range(env.batch):
            u = utilities[b].copy()
            for k, p in enumerate(env.pos[b]):
                here = p[0] * g + p[1]
                score = np.where(u > 0, u * self.reach[here], -1.0)
                target = int(score.argmax())
                if score[target] <= 0:
                    continue  # nothing worth the risk: wait
                u[target] = -1.0
                if target == here:
                    actions[b, k] = DEPLOY
                    continue
                delta = self.cells[target] - p
                axis = 0 if abs(delta[0]) >= abs(delta[1]) else 1
                step = np.zeros(2, int)
                step[axis] = np.sign(delta[axis])
                actions[b, k] = int(np.flatnonzero((MOVES == step).all(1))[0])
        return actions


def rollout(env: DispatchEnv, policy, scenario, depot) -> dict[str, np.ndarray]:
    env.reset(scenario, depot)
    while not env.done:
        env.step(policy.act(env))
    return env.metrics()


def cell_skill(belief: Belief, data: sc.Scenarios, split: str, dry: float) -> dict:
    """How well a belief predicts dry cells on a split (lower Brier is better)."""
    from sklearn.metrics import roc_auc_score

    idx = data.select(split)
    _, p = belief(data.pred[idx], data.unc[idx], dry)
    y = (data.truth[idx] < dry).reshape(-1)
    p = p.reshape(-1)
    return {"brier": float(np.mean((p - y) ** 2)), "auc": float(roc_auc_score(y, p))}


def run(modality: str = "s1") -> Path:
    from terramind.sim.evaluate import ppo_runs
    from terramind.sim.ppo import load_policy, runs_dir

    data = sc.load(modality)
    optical = sc.load("s2")
    report = {
        "simulated": True,
        "modality": modality,
        "reference": REFERENCE,
        "config": {"teams": 3, "horizon": 48, "dry": 0.05, "reveal": 1},
        "cell_skill": {},
        "settings": {},
    }
    fit = data.select(TUNE_SPLIT)
    for cost in WASTE_COSTS:
        config = DispatchConfig(waste_cost=cost)

        def fitted(belief, source=data, config=config):
            i = source.select(TUNE_SPLIT)
            return belief.fit(source.pred[i], source.unc[i], source.truth[i], config.dry)

        beliefs = {
            "raw map": Raw(),
            "calibrated": fitted(Calibrated()),
            "calibrated + uncertainty": fitted(Calibrated(use_uncertainty=True)),
        }
        if not report["cell_skill"]:
            report["cell_skill"] = {
                split: {name: cell_skill(b, data, split, config.dry) for name, b in beliefs.items()}
                for split in EVAL_SPLITS
            }
            report["fit_cells"] = int(fit.size * data.truth.shape[-1] ** 2)

        # name -> (policy, scenarios providing the model outputs)
        policies = {name: (Dispatcher(b), data) for name, b in beliefs.items()}
        policies["optical, calibrated"] = (Dispatcher(fitted(Calibrated(), optical)), optical)
        policies["truth (oracle)"] = (Dispatcher(None, oracle=True), data)
        learned = {}
        for tag, label in LEARNED:
            runs = [
                p
                for p in ppo_runs(modality, tag)
                if json.loads((p / "report.json").read_text(encoding="utf-8"))["config"][
                    "waste_cost"
                ]
                == cost
            ]
            if runs:
                learned[label] = [load_policy(p) for p in runs]

        result = {"splits": {}}
        for split in EVAL_SPLITS:
            scenario, depot = episodes(data, split)

            def play(policy, source, config=config, scenario=scenario, depot=depot):
                env = DispatchEnv(source.truth, source.pred, source.unc, config)
                return {
                    k: per_scenario(v) for k, v in rollout(env, policy, scenario, depot).items()
                }

            raw = {name: play(p, source) for name, (p, source) in policies.items()}
            seeds = {}
            for label, members in learned.items():
                per_seed = [play(p, data) for p in members]
                raw[label] = {k: np.mean([s[k] for s in per_seed], axis=0) for k in per_seed[0]}
                seeds[label] = [float(s["utility"].mean()) for s in per_seed]
            rows = {}
            for name, m in raw.items():
                diff = m["utility"] - raw[REFERENCE]["utility"]
                rows[name] = {
                    "utility": float(m["utility"].mean()),
                    "utility_ci95": list(bootstrap_ci(m["utility"])),
                    "served": float(m["served"].mean()),
                    "wasted_deployments": float(m["wasted"].mean()),
                    "deployments": float(m["deployed"].mean()),
                    "vs_reference": float(diff.mean()),
                    "vs_reference_ci95": list(bootstrap_ci(diff)),
                }
                if name in seeds:
                    rows[name]["utility_by_seed"] = seeds[name]
            result["splits"][split] = {
                "scenarios": len(raw[REFERENCE]["utility"]),
                "policies": rows,
            }
        report["settings"][str(cost)] = result

    out = runs_dir() / f"e9-dispatch-{modality}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
