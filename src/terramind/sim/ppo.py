"""PPO for the flood-response simulation.

One actor is shared by all teams (parameter sharing) and each team acts on its
own agent-centred view. Teams share a single team reward. The critic is used
only in training and also sees the hidden truth (centralised training,
decentralised execution), which is the MAPPO recipe applied to a small
cooperative problem.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from terramind.config import repo_root
from terramind.sim import scenarios as sc
from terramind.sim.env import (
    ACTOR_CHANNELS,
    CRITIC_CHANNELS,
    MOVES,
    BatchedFloodResponse,
    SimConfig,
)

REWARD_SCALE = 10.0


def runs_dir() -> Path:
    return repo_root() / "experiments" / "sim" / "runs"


@dataclass
class PPOConfig:
    modality: str = "s1"
    train_split: str = "valid"
    use_uncertainty: bool = True
    seed: int = 0
    updates: int = 400
    envs: int = 64
    gamma: float = 0.97
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 1024
    entropy: float = 0.01
    value_coef: float = 0.5
    teams: int = 3
    horizon: int = 48


class Net(nn.Module):
    def __init__(self, channels: int, outputs: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, outputs),
        )

    def forward(self, x):
        return self.body(x)


class PPOPolicy:
    """Runs a trained actor.

    Moves are sampled, as in training. Teams share one actor and start on the
    same cell with identical views, so taking the most likely move
    (`sample=False`) makes them move as one team and roughly halves the score.
    Sampling is seeded at the start of every episode batch, so results repeat.
    """

    def __init__(
        self,
        actor: Net,
        device: str,
        use_uncertainty: bool = True,
        sample: bool = True,
        seed: int = 0,
    ):
        self.actor, self.device, self.use_uncertainty = actor.eval(), device, use_uncertainty
        self.sample, self.seed = sample, seed
        self._rng = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def act(self, env: BatchedFloodResponse) -> np.ndarray:
        if env.t == 0:
            self._rng.manual_seed(self.seed)
        obs, _ = env.observe(self.use_uncertainty)
        b, k = obs.shape[:2]
        logits = self.actor(torch.from_numpy(obs).flatten(0, 1).to(self.device))
        mask = torch.from_numpy(env.action_mask()).flatten(0, 1).to(self.device)
        logits = logits.masked_fill(~mask, -1e9).float().cpu()
        if self.sample:
            moves = torch.multinomial(logits.softmax(-1), 1, generator=self._rng).squeeze(-1)
        else:
            moves = logits.argmax(-1)
        return moves.view(b, k).numpy()


def load_policy(run_dir: Path, device: str | None = None, sample: bool = True) -> PPOPolicy:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    actor = Net(ACTOR_CHANNELS, len(MOVES)).to(device)
    size = 2 * report["grid"] - 1
    actor(torch.zeros(1, ACTOR_CHANNELS, size, size, device=device))  # materialise LazyLinear
    actor.load_state_dict(torch.load(run_dir / "actor.pt", map_location=device))
    return PPOPolicy(actor, device, report["config"]["use_uncertainty"], sample=sample)


def train(cfg: PPOConfig, device: str | None = None) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    data = sc.load(cfg.modality)
    pool = data.select(cfg.train_split)
    env = BatchedFloodResponse(
        data.truth[pool], data.pred[pool], data.unc[pool], SimConfig(cfg.teams, cfg.horizon)
    )
    size = 2 * env.grid - 1
    actor = Net(ACTOR_CHANNELS, len(MOVES)).to(device)
    critic = Net(CRITIC_CHANNELS, 1).to(device)
    actor(torch.zeros(1, ACTOR_CHANNELS, size, size, device=device))
    critic(torch.zeros(1, CRITIC_CHANNELS, size, size, device=device))
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)

    b, k, t_max = cfg.envs, cfg.teams, cfg.horizon
    n = b * k
    history = []
    started = time.time()
    for update in range(cfg.updates):
        for group in opt.param_groups:
            group["lr"] = cfg.lr * (1 - update / cfg.updates)
        env.reset(
            rng.integers(len(pool), size=b),
            rng.integers(len(env.borders), size=b),
            rng.integers(8, size=b),
        )
        # The actor's view is the critic's view without the last (truth) channel,
        # so only the critic's is stored. Buffers live on the training device.
        obs = torch.zeros((t_max, n, CRITIC_CHANNELS, size, size), device=device)
        masks = torch.zeros((t_max, n, len(MOVES)), dtype=torch.bool, device=device)
        actions = torch.zeros((t_max, n), dtype=torch.long, device=device)
        logps = torch.zeros((t_max, n), device=device)
        values = torch.zeros((t_max + 1, n), device=device)
        rewards = torch.zeros((t_max, n), device=device)
        with torch.no_grad():
            for t in range(t_max):
                _, c = env.observe(cfg.use_uncertainty)
                obs[t] = torch.from_numpy(c).flatten(0, 1).to(device)
                masks[t] = torch.from_numpy(env.action_mask()).flatten(0, 1).to(device)
                logits = actor(obs[t, :, :ACTOR_CHANNELS]).masked_fill(~masks[t], -1e9)
                dist = torch.distributions.Categorical(logits=logits, validate_args=False)
                act = dist.sample()
                actions[t], logps[t] = act, dist.log_prob(act)
                values[t] = critic(obs[t]).squeeze(-1)
                reward = env.step(act.view(b, k).cpu().numpy())
                rewards[t] = torch.from_numpy(reward).to(device).repeat_interleave(k) * REWARD_SCALE
            # Episodes end at the horizon, so the value after the last step is 0.
            advantages = torch.zeros((t_max, n), device=device)
            last = torch.zeros(n, device=device)
            for t in reversed(range(t_max)):
                delta = rewards[t] + cfg.gamma * values[t + 1] - values[t]
                last = delta + cfg.gamma * cfg.lam * last
                advantages[t] = last
            returns = advantages + values[:-1]

        flat = [x.flatten(0, 1) for x in (obs, masks, actions, logps, advantages, returns)]
        total = t_max * n
        for _ in range(cfg.epochs):
            for idx in torch.randperm(total, device=device).split(cfg.minibatch):
                oc, mk, ac, lp, adv, ret = (x[idx] for x in flat)
                oa = oc[:, :ACTOR_CHANNELS]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                with torch.autocast(device, enabled=device == "cuda"):
                    logits, value = actor(oa), critic(oc).squeeze(-1)
                dist = torch.distributions.Categorical(
                    logits=logits.float().masked_fill(~mk, -1e9), validate_args=False
                )
                ratio = (dist.log_prob(ac) - lp).exp()
                policy_loss = -torch.min(
                    ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv
                ).mean()
                value_loss = (value.float() - ret).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()

        m = env.metrics()
        history.append(
            {"update": update, "served": float(m["served"].mean()), "auc": float(m["auc"].mean())}
        )
        if update % 20 == 0 or update == cfg.updates - 1:
            print(
                f"update {update:4d}  served {history[-1]['served']:.3f}  "
                f"auc {history[-1]['auc']:.3f}  {time.time() - started:.0f}s",
                flush=True,
            )

    tag = "ppo" if cfg.use_uncertainty else "ppo-nounc"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = runs_dir() / f"{tag}-{cfg.modality}-seed{cfg.seed}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), out / "actor.pt")
    (out / "report.json").write_text(
        json.dumps(
            {
                "simulated": True,
                "config": asdict(cfg),
                "grid": env.grid,
                "train_scenarios": len(pool),
                "episodes": cfg.updates * cfg.envs,
                "seconds": round(time.time() - started),
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out
