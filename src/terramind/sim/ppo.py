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
    task: str = "response"  # response (entering a cell serves it) or dispatch
    waste_cost: float = 1.0  # dispatch only
    # Dispatch only: imitate the calibrated planner first (DAgger), for this many
    # batches of `envs` episodes, then run `updates` PPO updates from there.
    imitate_batches: int = 0
    critic_warmup: int = 0  # PPO updates that train only the critic
    holdout: int = 0  # training-split scenarios kept out for tuning
    # "local": the critic sees each team's own view plus the truth. "global" (MAPPO):
    # the critic sees the whole map, every team and the truth, from each team's seat.
    critic: str = "local"
    name: str = ""  # run-directory tag; default from task and inputs


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
    channels = report.get("actor_channels", ACTOR_CHANNELS)
    actor = Net(channels, report.get("actions", len(MOVES))).to(device)
    size = 2 * report["grid"] - 1
    actor(torch.zeros(1, channels, size, size, device=device))  # materialise LazyLinear
    actor.load_state_dict(torch.load(run_dir / "actor.pt", map_location=device))
    return PPOPolicy(actor, device, report["config"]["use_uncertainty"], sample=sample)


def holdout_indices(data: sc.Scenarios, split: str, n: int) -> np.ndarray:
    """A fixed set of scenarios from the training split, kept out for tuning."""
    if n <= 0:
        return np.array([], int)
    pool = data.select(split)
    return np.sort(np.random.default_rng(2026).choice(pool, n, replace=False))


def _dispatch_score(actor, cfg, data, idx, device, rng_seed=0) -> dict:
    """Utility of the actor and of the calibrated planner on given scenarios."""
    from terramind.sim.dispatch import (
        Calibrated,
        DispatchConfig,
        DispatchEnv,
        Dispatcher,
        rollout,
    )
    from terramind.sim.env import border_cells

    rng = np.random.default_rng(rng_seed)
    depots = rng.integers(len(border_cells(data.truth.shape[-1])), size=(len(idx), 4))
    scenario, depot = np.repeat(idx, 4), depots.reshape(-1)
    config = DispatchConfig(cfg.teams, cfg.horizon, cfg.waste_cost)
    fit = np.setdiff1d(data.select(cfg.train_split), idx)
    belief = Calibrated().fit(data.pred[fit], data.unc[fit], data.truth[fit], config.dry)

    def play(policy):
        env = DispatchEnv(data.truth, data.pred, data.unc, config)
        return float(rollout(env, policy, scenario, depot)["utility"].mean())

    return {
        "learned": play(PPOPolicy(actor, device, cfg.use_uncertainty)),
        "planner": play(Dispatcher(belief)),
    }


def _imitate(actor, env, cfg, data, pool, device, rng, a_ch, n_act) -> list[dict]:
    """DAgger: label every visited state with the calibrated planner's action.

    The planner drives for the first half of the batches, then the learner
    drives and the planner only labels, so the learner sees its own mistakes.
    """
    from terramind.sim.dispatch import Calibrated, Dispatcher

    belief = Calibrated().fit(data.pred[pool], data.unc[pool], data.truth[pool], env.dcfg.dry)
    expert = Dispatcher(belief)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    b, k, t_max = cfg.envs, cfg.teams, cfg.horizon
    size = 2 * env.grid - 1
    history = []
    for i in range(cfg.imitate_batches):
        beta = max(0.0, 1.0 - i / max(1, cfg.imitate_batches // 2))
        env.reset(
            rng.integers(len(pool), size=b),
            rng.integers(len(env.borders), size=b),
            rng.integers(8, size=b),
        )
        obs = torch.zeros((t_max, b * k, a_ch, size, size), dtype=torch.float16, device=device)
        masks = torch.zeros((t_max, b * k, n_act), dtype=torch.bool, device=device)
        labels = torch.zeros((t_max, b * k), dtype=torch.long, device=device)
        with torch.no_grad():
            for t in range(t_max):
                a, _ = env.observe(cfg.use_uncertainty)
                oa = torch.from_numpy(a).flatten(0, 1).to(device)
                mk = torch.from_numpy(env.action_mask()).flatten(0, 1).to(device)
                target = expert.act(env)
                own = actor(oa).masked_fill(~mk, -1e9).argmax(-1).view(b, k).cpu().numpy()
                drive = np.where((rng.random(b) < beta)[:, None], target, own)
                obs[t], masks[t] = oa.half(), mk
                labels[t] = torch.from_numpy(target).flatten().to(device)
                env.step(drive)
        x, mk, y = obs.flatten(0, 1), masks.flatten(0, 1), labels.flatten(0, 1)
        correct = 0
        for _ in range(2):
            for idx in torch.randperm(len(y), device=device).split(cfg.minibatch):
                logits = actor(x[idx].float()).masked_fill(~mk[idx], -1e9)
                loss = nn.functional.cross_entropy(logits, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                correct += int((logits.argmax(-1) == y[idx]).sum())
        history.append(
            {
                "batch": i,
                "expert_share": beta,
                "agreement": correct / (2 * len(y)),
                "utility": float(env.metrics()["utility"].mean()),
            }
        )
        if i % 10 == 0 or i == cfg.imitate_batches - 1:
            h = history[-1]
            print(
                f"imitate {i:3d}  planner drives {beta:.2f}  agreement {h['agreement']:.3f}  "
                f"utility {h['utility']:.3f}",
                flush=True,
            )
    return history


def train(cfg: PPOConfig, device: str | None = None) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    data = sc.load(cfg.modality)
    held = holdout_indices(data, cfg.train_split, cfg.holdout)
    pool = np.setdiff1d(data.select(cfg.train_split), held)
    maps = (data.truth[pool], data.pred[pool], data.unc[pool])
    if cfg.task == "dispatch":
        from terramind.sim.dispatch import DispatchConfig, DispatchEnv

        env = DispatchEnv(*maps, DispatchConfig(cfg.teams, cfg.horizon, cfg.waste_cost))
        a_ch, c_ch, n_act, score = env.actor_channels, env.critic_channels, env.n_actions, "utility"
    else:
        env = BatchedFloodResponse(*maps, SimConfig(cfg.teams, cfg.horizon))
        a_ch, c_ch, n_act, score = ACTOR_CHANNELS, CRITIC_CHANNELS, len(MOVES), "auc"
    size = 2 * env.grid - 1
    actor = Net(a_ch, n_act).to(device)
    global_critic = cfg.critic == "global"
    if global_critic and cfg.task != "dispatch":
        raise ValueError("the global critic is implemented for the dispatch task only")
    g_ch, g_size = (env.global_channels, env.grid) if global_critic else (c_ch, size)
    critic = Net(g_ch, 1).to(device)
    actor(torch.zeros(1, a_ch, size, size, device=device))
    critic(torch.zeros(1, g_ch, g_size, g_size, device=device))
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)

    b, k, t_max = cfg.envs, cfg.teams, cfg.horizon
    n = b * k
    history = []
    started = time.time()
    imitation = []
    if cfg.imitate_batches:
        if cfg.task != "dispatch":
            raise ValueError("imitation is implemented for the dispatch task only")
        imitation = _imitate(actor, env, cfg, data, pool, device, rng, a_ch, n_act)
    for update in range(cfg.updates):
        train_actor = update >= cfg.critic_warmup
        for group in opt.param_groups:
            group["lr"] = cfg.lr * (1 - update / cfg.updates)
        env.reset(
            rng.integers(len(pool), size=b),
            rng.integers(len(env.borders), size=b),
            rng.integers(8, size=b),
        )
        # The actor's view is the critic's view without the last (truth) channel,
        # so only the critic's is stored. Buffers live on the training device.
        obs = torch.zeros((t_max, n, c_ch, size, size), device=device)
        gobs = (
            torch.zeros((t_max, n, g_ch, g_size, g_size), device=device) if global_critic else obs
        )
        masks = torch.zeros((t_max, n, n_act), dtype=torch.bool, device=device)
        actions = torch.zeros((t_max, n), dtype=torch.long, device=device)
        logps = torch.zeros((t_max, n), device=device)
        values = torch.zeros((t_max + 1, n), device=device)
        rewards = torch.zeros((t_max, n), device=device)
        with torch.no_grad():
            for t in range(t_max):
                _, c = env.observe(cfg.use_uncertainty)
                obs[t] = torch.from_numpy(c).flatten(0, 1).to(device)
                masks[t] = torch.from_numpy(env.action_mask()).flatten(0, 1).to(device)
                logits = actor(obs[t, :, :a_ch]).masked_fill(~masks[t], -1e9)
                dist = torch.distributions.Categorical(logits=logits, validate_args=False)
                act = dist.sample()
                actions[t], logps[t] = act, dist.log_prob(act)
                if global_critic:
                    g = env.global_state(cfg.use_uncertainty)
                    gobs[t] = torch.from_numpy(g).flatten(0, 1).to(device)
                values[t] = critic(gobs[t]).squeeze(-1)
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

        flat = [x.flatten(0, 1) for x in (obs, gobs, masks, actions, logps, advantages, returns)]
        total = t_max * n
        for _ in range(cfg.epochs):
            for idx in torch.randperm(total, device=device).split(cfg.minibatch):
                oc, gc, mk, ac, lp, adv, ret = (x[idx] for x in flat)
                oa = oc[:, :a_ch]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                with torch.autocast(device, enabled=device == "cuda"):
                    logits, value = actor(oa), critic(gc).squeeze(-1)
                dist = torch.distributions.Categorical(
                    logits=logits.float().masked_fill(~mk, -1e9), validate_args=False
                )
                ratio = (dist.log_prob(ac) - lp).exp()
                policy_loss = -torch.min(
                    ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv
                ).mean()
                value_loss = (value.float() - ret).pow(2).mean()
                loss = cfg.value_coef * value_loss
                if train_actor:
                    loss = loss + policy_loss - cfg.entropy * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()

        m = env.metrics()
        history.append(
            {"update": update, "served": float(m["served"].mean()), score: float(m[score].mean())}
        )
        if update % 20 == 0 or update == cfg.updates - 1:
            print(
                f"update {update:4d}  served {history[-1]['served']:.3f}  "
                f"{score} {history[-1][score]:.3f}  {time.time() - started:.0f}s",
                flush=True,
            )

    tag = cfg.name or ("dispatch" if cfg.task == "dispatch" else "ppo")
    tag += "" if cfg.use_uncertainty else "-nounc"
    tuning = None
    if len(held) and cfg.task == "dispatch":
        tuning = _dispatch_score(actor, cfg, data, held, device)
        tuning["scenarios"] = len(held)
        print(f"held-out tuning scenarios: {tuning}", flush=True)
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
                "actor_channels": a_ch,
                "actions": n_act,
                "train_scenarios": len(pool),
                "holdout_ids": [str(i) for i in data.ids[held]],
                "tuning": tuning,
                "imitation": imitation,
                "episodes": cfg.updates * cfg.envs,
                "seconds": round(time.time() - started),
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out
