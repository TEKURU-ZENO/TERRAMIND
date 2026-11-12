"""Which road points are flooded? A graph network against simpler baselines.

Every model sees the same per-node features (see roads.graph). The question is
whether passing information along the road network helps:

- threshold: the leave-one-event-out radar model's water probability >= 0.5
- mlp: a small network on each node's features alone
- gnn: GraphSAGE (3 layers, mean aggregation) over the road graph

Trained on train-split chips, early-stopped on valid, scored on test and Bolivia
(F1 and IoU of the "flooded" class), 3 seeds each.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

from terramind.config import repo_root
from terramind.roads.graph import out_dir


def runs_dir() -> Path:
    return repo_root() / "experiments" / "roads" / "runs"


def _batch(graphs, device):
    """Concatenate graphs into one disconnected graph with symmetric edges."""
    xs, ys, src, dst, off = [], [], [], [], 0
    for g in graphs:
        xs.append(g["x"])
        ys.append(g["y"])
        if len(g["edges"]):
            e = g["edges"] + off
            src += [e[:, 0], e[:, 1]]
            dst += [e[:, 1], e[:, 0]]
        off += len(g["y"])
    x = torch.from_numpy(np.concatenate(xs)).to(device)
    y = torch.from_numpy(np.concatenate(ys)).to(device)
    if src:
        s = torch.from_numpy(np.concatenate(src)).to(device)
        d = torch.from_numpy(np.concatenate(dst)).to(device)
    else:
        s = d = torch.zeros(0, dtype=torch.long, device=device)
    return x, y, s, d


class SAGE(nn.Module):
    def __init__(self, f, hidden=64, layers=3):
        super().__init__()
        self.inp = nn.Linear(f, hidden)
        self.self_w = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.nbr_w = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, s, d):
        h = torch.relu(self.inp(x))
        for ws, wn in zip(self.self_w, self.nbr_w):
            agg = torch.zeros_like(h).index_add_(0, d, h[s])
            deg = torch.zeros(len(h), device=h.device).index_add_(
                0, d, torch.ones_like(d, dtype=h.dtype)
            )
            agg = agg / deg.clamp(min=1).unsqueeze(1)
            h = torch.relu(ws(h) + wn(agg))
        return self.out(h).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, f, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, s, d):
        return self.net(x).squeeze(-1)


def _scores(pred, y):
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    return {
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "nodes": len(y),
        "flooded": int(y.sum()),
    }


def _train(kind, train, valid, f, seed, device, epochs=150):
    torch.manual_seed(seed)
    model = (SAGE(f) if kind == "gnn" else MLP(f)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    xt, yt, st, dt = train
    pos = yt.float().mean().clamp(1e-3, 0.999)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos)
    best, best_state = -1.0, None
    xv, yv, sv, dv = valid
    for _ in range(epochs):
        model.train()
        loss = loss_fn(model(xt, st, dt), yt.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            f1 = (
                _scores((model(xv, sv, dv) > 0).cpu().numpy(), yv.cpu().numpy().astype(bool))["f1"]
                or 0
            )
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


def run(seeds=(0, 1, 2)) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with (out_dir() / "graphs.pkl").open("rb") as fh:
        data = pickle.load(fh)
    graphs, features = data["graphs"], data["features"]
    by = {s: [g for g in graphs if g["split"] == s] for s in ("train", "valid", "test", "bolivia")}
    # Standardise with training statistics.
    xt = np.concatenate([g["x"] for g in by["train"]])
    mu, sd = xt.mean(0), xt.std(0) + 1e-6
    for g in graphs:
        g["xs"] = ((g["x"] - mu) / sd).astype(np.float32)

    def batch(split):
        return _batch([{**g, "x": g["xs"]} for g in by[split]], device)

    data_b = {s: batch(s) for s in by}
    f = len(features)
    report = {
        "features": list(features),
        "splits": {
            s: {
                "chips": len(by[s]),
                "nodes": int(sum(len(g["y"]) for g in by[s])),
                "flooded": int(sum(int(g["y"].sum()) for g in by[s])),
            }
            for s in by
        },
        "results": {},
    }
    prob_col = features.index("water_prob")
    for split in ("test", "bolivia"):
        y = data_b[split][1].cpu().numpy().astype(bool)
        raw = np.concatenate([g["x"][:, prob_col] for g in by[split]])
        report["results"].setdefault(split, {})["threshold"] = _scores(raw >= 0.5, y)
    for kind in ("mlp", "gnn"):
        per_seed = {"test": [], "bolivia": []}
        for seed in seeds:
            model = _train(kind, data_b["train"], data_b["valid"], f, seed, device)
            with torch.no_grad():
                for split, rows in per_seed.items():
                    x, y, s, d = data_b[split]
                    rows.append(
                        _scores((model(x, s, d) > 0).cpu().numpy(), y.cpu().numpy().astype(bool))
                    )
        for split, rows in per_seed.items():
            f1s = [r["f1"] for r in rows]
            report["results"][split][kind] = {
                "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s, ddof=1)),
                "iou_mean": float(np.mean([r["iou"] for r in rows])), "seeds": rows,
            }  # fmt: skip
    out = runs_dir() / "e10-roads.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
