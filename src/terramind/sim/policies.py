"""Non-learned policies for the flood-response simulation.

Every policy scores unserved cells with a value map and differs in how it turns
that map into moves:

- value "belief": the model's prediction, replaced by truth where seen
- value "belief" with beta > 0: adds beta * ensemble disagreement (explore)
- value "blind": every unserved cell is worth the same (ignores perception)
- value "oracle": the hidden truth (an upper reference, not a fair policy)
"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.optimize import linear_sum_assignment

from terramind.sim.env import MOVES, BatchedFloodResponse


def _coords(grid: int) -> np.ndarray:
    return np.stack(np.unravel_index(np.arange(grid * grid), (grid, grid)), 1)


class Heuristic:
    """Base: builds the value map and the distance tables."""

    def __init__(self, value: str = "belief", beta: float = 0.0, decay: float = 0.85):
        self.value, self.beta, self.decay = value, beta, decay
        self._grid = None

    def _prepare(self, grid: int) -> None:
        if self._grid == grid:
            return
        self._grid = grid
        self.cells = _coords(grid)
        dist = np.abs(self.cells[:, None] - self.cells[None]).sum(-1)
        self.dist = dist
        self.reach = self.decay**dist  # [cells, cells]

    def values(self, env: BatchedFloodResponse) -> np.ndarray:
        """[B, cells] value of serving each cell now; served cells are worth 0."""
        if self.value == "oracle":
            v = env.truth
        elif self.value == "blind":
            v = np.ones_like(env.pred)
        else:
            v = env.belief() + self.beta * env.uncertainty()
        return np.where(env.served, 0.0, v).reshape(env.batch, -1)

    def act(self, env: BatchedFloodResponse) -> np.ndarray:
        self._prepare(env.grid)
        values = self.values(env)
        actions = np.zeros((env.batch, env.cfg.teams), int)
        for b in range(env.batch):
            actions[b] = self.plan(env.pos[b], values[b].copy(), env.grid)
        return actions

    def plan(self, pos: np.ndarray, value: np.ndarray, grid: int) -> np.ndarray:
        raise NotImplementedError

    def _step_toward(self, pos, target, value, grid) -> int:
        """Move one cell toward target, preferring the more valuable next cell."""
        best, best_value = 0, -1.0
        here = np.abs(target - pos).sum()
        for a in range(1, len(MOVES)):
            nxt = pos + MOVES[a]
            if (nxt < 0).any() or (nxt >= grid).any():
                continue
            if np.abs(target - nxt).sum() < here and value[nxt[0] * grid + nxt[1]] > best_value:
                best, best_value = a, value[nxt[0] * grid + nxt[1]]
        return best


class Greedy(Heuristic):
    """Each team in turn heads for the best value-over-distance cell not yet claimed."""

    def plan(self, pos, value, grid):
        actions = np.zeros(len(pos), int)
        for k, p in enumerate(pos):
            score = value * self.reach[p[0] * grid + p[1]]
            target = int(score.argmax())
            if score[target] <= 0:
                continue
            actions[k] = self._step_toward(p, self.cells[target], value, grid)
            value[target] = 0.0
        return actions


class Assignment(Heuristic):
    """Optimisation baseline: teams are matched to target cells jointly
    (Hungarian algorithm) to maximise total value discounted by distance."""

    def __init__(self, candidates: int = 24, **kw):
        super().__init__(**kw)
        self.candidates = candidates

    def plan(self, pos, value, grid):
        actions = np.zeros(len(pos), int)
        top = np.argsort(value)[::-1][: self.candidates]
        top = top[value[top] > 0]
        if len(top) == 0:
            return actions
        start = pos[:, 0] * grid + pos[:, 1]
        gain = value[top][None] * self.reach[start][:, top]
        teams, targets = linear_sum_assignment(gain, maximize=True)
        for k, j in zip(teams, targets):
            actions[k] = self._step_toward(pos[k], self.cells[top[j]], value, grid)
        return actions


class Lookahead(Heuristic):
    """Planning baseline: each team in turn evaluates every path of `depth` moves,
    scoring discounted value collected plus the best distance-discounted cell
    left at the end, then takes the first move. Replanned every step."""

    def __init__(self, depth: int = 5, gamma: float = 0.95, **kw):
        super().__init__(**kw)
        self.depth, self.gamma = depth, gamma
        seqs = np.array(list(product(range(1, len(MOVES)), repeat=depth)))
        self.seqs = seqs
        self.offsets = MOVES[seqs].cumsum(1)  # [S, depth, 2]
        self.weights = gamma ** np.arange(depth)

    def plan(self, pos, value, grid):
        actions = np.zeros(len(pos), int)
        for k, p in enumerate(pos):
            path = p[None, None] + self.offsets
            ok = ((path >= 0) & (path < grid)).all((1, 2))
            if not ok.any() or value.max() <= 0:
                continue
            seqs, path = self.seqs[ok], path[ok]
            flat = path[..., 0] * grid + path[..., 1]  # [S, depth]
            fresh = np.ones(flat.shape, bool)
            for i in range(1, self.depth):
                fresh[:, i] = (flat[:, i : i + 1] != flat[:, :i]).all(1)
            collected = (value[flat] * fresh * self.weights).sum(1)
            # Value left after the path, as seen from where the path ends.
            ends, inverse = np.unique(flat[:, -1], return_inverse=True)
            # The end cell itself is excluded: the path has already collected it.
            reach = self.reach[ends].copy()
            reach[np.arange(len(ends)), ends] = 0.0
            tail = (value[None] * reach).max(1)[inverse]
            best = int((collected + self.gamma**self.depth * tail).argmax())
            actions[k] = seqs[best, 0]
            value[flat[best]] = 0.0  # claimed; later teams plan around it
        return actions


class RandomWalk:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env: BatchedFloodResponse) -> np.ndarray:
        scores = self.rng.random((env.batch, env.cfg.teams, len(MOVES)))
        mask = env.action_mask()
        mask[..., 0] = False
        return np.where(mask, scores, -1.0).argmax(-1)


def rollout(env: BatchedFloodResponse, policy, scenario, depot) -> dict[str, np.ndarray]:
    """Play one episode per (scenario, depot) pair and return per-episode metrics."""
    env.reset(scenario, depot)
    while not env.done:
        env.step(policy.act(env))
    return env.metrics()
