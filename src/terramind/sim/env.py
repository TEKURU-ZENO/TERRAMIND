"""Flood-response environment (simulated).

K teams start together at a depot on the edge of a G x G map and move one cell
per step for T steps. Entering a cell serves it: the team reward is that
cell's true water fraction, divided by the scenario total so an episode's
return lies in [0, 1]. Teams see the model's prediction and its uncertainty;
the truth of a cell is revealed only once a team is within `sight` cells.

The environment is batched: B independent episodes advance together.
`FloodResponseEnv` wraps one episode in the Gymnasium API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])  # stay, N, S, W, E
ACTOR_CHANNELS = 7
CRITIC_CHANNELS = 8


@dataclass(frozen=True)
class SimConfig:
    teams: int = 3
    horizon: int = 48
    sight: int = 1


def border_cells(grid: int) -> np.ndarray:
    idx = np.arange(grid)
    top = np.stack([np.zeros(grid, int), idx], 1)
    bottom = np.stack([np.full(grid, grid - 1), idx], 1)
    left = np.stack([idx[1:-1], np.zeros(grid - 2, int)], 1)
    right = np.stack([idx[1:-1], np.full(grid - 2, grid - 1)], 1)
    return np.concatenate([top, bottom, left, right])


def dihedral(a: np.ndarray, k: int) -> np.ndarray:
    """One of the 8 rotations/reflections of a square map."""
    a = np.rot90(a, k % 4, axes=(-2, -1))
    return np.flip(a, axis=-1) if k >= 4 else a


class BatchedFloodResponse:
    def __init__(self, truth, pred, unc, config: SimConfig | None = None):
        self.pool = (np.asarray(truth), np.asarray(pred), np.asarray(unc))
        self.cfg = config or SimConfig()
        self.grid = self.pool[0].shape[-1]
        self.borders = border_cells(self.grid)

    def reset(self, scenario, depot, transform=None) -> None:
        """Start len(scenario) episodes.

        scenario: indices into the pool; depot: index into `self.borders` per
        episode; transform: optional dihedral index (0-7) per episode.
        """
        scenario = np.asarray(scenario)
        maps = [m[scenario].astype(np.float32) for m in self.pool]
        if transform is not None:
            maps = [np.stack([dihedral(a, k) for a, k in zip(m, transform)]) for m in maps]
        self.truth, self.pred, self.unc = maps
        b = len(scenario)
        self.total = np.maximum(self.truth.sum(axis=(1, 2)), 1e-6)
        self.pos = np.repeat(self.borders[np.asarray(depot)][:, None, :], self.cfg.teams, axis=1)
        self.served = np.zeros((b, self.grid, self.grid), bool)
        self.seen = np.zeros_like(self.served)
        self.t = 0
        self.cumulative = np.zeros(b, np.float32)  # fraction of need served so far
        self.auc = np.zeros(b, np.float32)  # running sum of `cumulative` per step
        self._arrive()

    @property
    def batch(self) -> int:
        return len(self.truth)

    @property
    def done(self) -> bool:
        return self.t >= self.cfg.horizon

    def _arrive(self) -> np.ndarray:
        """Serve and reveal cells at the teams' positions; return need newly served."""
        rows = np.arange(self.batch)
        gained = np.zeros(self.batch, np.float32)
        s = self.cfg.sight
        for k in range(self.cfg.teams):
            r, c = self.pos[:, k, 0], self.pos[:, k, 1]
            gained += np.where(self.served[rows, r, c], 0.0, self.truth[rows, r, c])
            self.served[rows, r, c] = True
            for dr in range(-s, s + 1):
                for dc in range(-s, s + 1):
                    rr = np.clip(r + dr, 0, self.grid - 1)
                    cc = np.clip(c + dc, 0, self.grid - 1)
                    self.seen[rows, rr, cc] = True
        gained /= self.total
        self.cumulative += gained
        return gained

    def action_mask(self) -> np.ndarray:
        """[B, K, 5] True where the move stays on the map."""
        nxt = self.pos[:, :, None, :] + MOVES[None, None]
        return ((nxt >= 0) & (nxt < self.grid)).all(-1)

    def step(self, actions: np.ndarray) -> np.ndarray:
        """Move every team; off-map moves become 'stay'. Returns reward [B]."""
        nxt = self.pos + MOVES[np.asarray(actions)]
        ok = ((nxt >= 0) & (nxt < self.grid)).all(-1, keepdims=True)
        self.pos = np.where(ok, nxt, self.pos)
        reward = self._arrive()
        self.t += 1
        self.auc += self.cumulative
        return reward

    # What the teams know.
    def belief(self) -> np.ndarray:
        return np.where(self.seen, self.truth, self.pred)

    def uncertainty(self) -> np.ndarray:
        return np.where(self.seen, 0.0, self.unc)

    def metrics(self) -> dict[str, np.ndarray]:
        """served: fraction of need served by the end. auc: its mean over time,
        which also rewards serving early."""
        return {"served": self.cumulative.copy(), "auc": self.auc / max(self.t, 1)}

    def observe(self, use_uncertainty: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Agent-centred observations.

        Returns (actor [B, K, 7, S, S], critic [B, K, 8, S, S]) with S = 2G - 1 and
        the agent at the centre. Channels: belief, uncertainty, served, seen,
        other teams, on-map, time. The critic also gets the hidden truth; it is
        used only during training.
        """
        b, g, k = self.batch, self.grid, self.cfg.teams
        rows = np.arange(b)
        occupancy = np.zeros((b, k, g, g), np.float32)
        for i in range(k):
            np.add.at(occupancy[:, i], (rows, self.pos[:, i, 0], self.pos[:, i, 1]), 1.0)
        everyone = occupancy.sum(1)
        unc = self.uncertainty() * 4.0 if use_uncertainty else np.zeros_like(self.pred)
        shared = np.stack(
            [
                self.belief(),
                unc,
                self.served.astype(np.float32),
                self.seen.astype(np.float32),
                np.zeros_like(self.pred),  # others, filled per agent
                np.ones_like(self.pred),
                np.full_like(self.pred, self.t / self.cfg.horizon),
                self.truth,
            ],
            axis=1,
        )
        pad = g - 1
        size = 2 * g - 1
        out = np.zeros((b, k, CRITIC_CHANNELS, size, size), np.float32)
        span = np.arange(size)
        for i in range(k):
            shared[:, 4] = everyone - occupancy[:, i]
            padded = np.pad(shared, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
            # The time plane is a scalar, so it should not vanish off the map.
            padded[:, 6] = self.t / self.cfg.horizon
            rr = self.pos[:, i, 0][:, None] + span[None]
            cc = self.pos[:, i, 1][:, None] + span[None]
            out[:, i] = padded[rows[:, None, None], :, rr[:, :, None], cc[:, None, :]].transpose(
                0, 3, 1, 2
            )
        return out[:, :, :ACTOR_CHANNELS], out


def make_gym_env(truth, pred, unc, config: SimConfig | None = None, seed: int = 0):
    """One episode at a time in the Gymnasium API (global observation)."""
    import gymnasium as gym

    class FloodResponseEnv(gym.Env):
        metadata: ClassVar[dict] = {"render_modes": []}

        def __init__(self):
            self.sim = BatchedFloodResponse(truth, pred, unc, config)
            g, k = self.sim.grid, self.sim.cfg.teams
            self.action_space = gym.spaces.MultiDiscrete([len(MOVES)] * k)
            self.observation_space = gym.spaces.Box(0.0, float(k), (6, g, g), np.float32)
            self._rng = np.random.default_rng(seed)

        def _obs(self):
            s = self.sim
            teams = np.zeros((s.grid, s.grid), np.float32)
            np.add.at(teams, (s.pos[0, :, 0], s.pos[0, :, 1]), 1.0)
            return np.stack(
                [
                    s.belief()[0],
                    s.uncertainty()[0],
                    s.served[0].astype(np.float32),
                    s.seen[0].astype(np.float32),
                    teams,
                    np.full((s.grid, s.grid), s.t / s.cfg.horizon, np.float32),
                ]
            ).astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            options = options or {}
            scenario = options.get("scenario", self._rng.integers(len(self.sim.pool[0])))
            depot = options.get("depot", self._rng.integers(len(self.sim.borders)))
            self.sim.reset([scenario], [depot])
            return self._obs(), {"scenario": int(scenario), "depot": int(depot)}

        def step(self, action):
            reward = float(self.sim.step(np.asarray(action)[None])[0])
            info = {k: float(v[0]) for k, v in self.sim.metrics().items()}
            return self._obs(), reward, False, self.sim.done, info

    return FloodResponseEnv()
