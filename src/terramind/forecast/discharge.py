"""E3: forecasting river discharge and flood-threshold exceedance at event sites.

Target: GloFAS v4 reanalysis discharge (a model river, not a gauge) at the
largest river cell near each of 15 flood event sites, 1, 3 and 7 days ahead.
Inputs: the last 30 days of discharge and ERA5 precipitation at the site; no
future rainfall, so this is a forecast from what is known on the issue day.

Splits are by time: train 1991-2012, valid 2013-2016, test 2017-2024. Discharge
is log-transformed and standardised per site with training-period statistics so
one model can be pooled across sites. A leave-one-site-out run checks whether
the pooled model works at a site it never saw.

Flood threshold: each site's 95th percentile of training-period daily discharge.
For each event we also check whether GloFAS itself exceeds that threshold within
7 days of the satellite-observed flood: if the model river does not flood, no
forecast of it can.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from terramind.config import repo_root
from terramind.ingestion import openmeteo

LEADS = (1, 3, 7)
WINDOW = 30
TRAIN, VALID, TEST = (
    ("1991-01-01", "2012-12-31"),
    ("2013-01-01", "2016-12-31"),
    ("2017-01-01", "2024-12-31"),
)
EVENT_WINDOW = 7  # days either side of the observed flood


@dataclass
class SiteSeries:
    id: str
    name: str
    event_date: str
    dates: np.ndarray  # datetime64[D]
    q: np.ndarray  # m3/s
    p: np.ndarray  # mm/day
    z: np.ndarray  # standardised log discharge
    mu: float
    sd: float
    threshold: float  # training-period 95th percentile, m3/s


def runs_dir() -> Path:
    return repo_root() / "experiments" / "forecast" / "runs"


def _mask(dates, span):
    return (dates >= np.datetime64(span[0])) & (dates <= np.datetime64(span[1]))


def load_sites() -> list[SiteSeries]:
    out = []
    for site in openmeteo.sites():
        try:
            rec = openmeteo.load(site.id)
        except FileNotFoundError:
            continue
        dates = np.array(rec["time"], dtype="datetime64[D]")
        q = np.array([np.nan if v is None else v for v in rec["discharge"]], float)
        p = np.array([rec["precipitation"].get(t) for t in rec["time"]], dtype=object)
        p = np.array([np.nan if v is None else float(v) for v in p], float)
        train = _mask(dates, TRAIN)
        lq = np.log1p(np.clip(q, 0, None))
        mu, sd = np.nanmean(lq[train]), np.nanstd(lq[train]) + 1e-6
        thr = float(np.nanpercentile(q[train], 95))
        out.append(
            SiteSeries(
                site.id,
                site.name,
                site.event_date,
                dates,
                q,
                np.nan_to_num(p),
                (lq - mu) / sd,
                mu,
                sd,
                thr,
            )
        )
    return out


def to_q(z: np.ndarray, s: SiteSeries) -> np.ndarray:
    return np.expm1(z * s.sd + s.mu)


def features(s: SiteSeries, lead: int):
    """(X [N, F], seq [N, WINDOW, 2], target change [N], issue index [N]) for every day."""
    n = len(s.z)
    idx = np.arange(WINDOW, n - lead)
    lags = np.stack([s.z[idx - k] for k in range(7)], 1)
    zmean = np.stack([s.z[i - WINDOW + 1 : i + 1].mean() for i in idx])
    lp = np.log1p(s.p)
    csum = np.concatenate([[0.0], np.cumsum(lp)])
    psums = np.stack([(csum[idx + 1] - csum[idx + 1 - w]) / w for w in (1, 3, 7, 14, 30)], 1)
    doy = (s.dates[idx] - s.dates[idx].astype("datetime64[Y]")).astype(int)
    season = np.stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], 1)
    X = np.column_stack([lags, zmean, psums, season])
    seq = np.stack(
        [np.stack([s.z[i - WINDOW + 1 : i + 1], lp[i - WINDOW + 1 : i + 1]], 1) for i in idx]
    )
    y = s.z[idx + lead] - s.z[idx]
    ok = np.isfinite(X).all(1) & np.isfinite(y) & np.isfinite(seq).all((1, 2))
    return X[ok], seq[ok].astype(np.float32), y[ok], idx[ok]


def _nse(obs, sim) -> float:
    ok = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[ok], sim[ok]
    return float(1 - ((obs - sim) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())


def _gru_fit(seq, y, seed=0, epochs=12):
    import torch
    from torch import nn

    torch.manual_seed(seed)

    class GRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(2, 32, batch_first=True)
            self.out = nn.Linear(32, 1)

        def forward(self, x):
            return self.out(self.rnn(x)[0][:, -1]).squeeze(-1)

    model = GRU()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    x, t = torch.from_numpy(seq), torch.from_numpy(y.astype(np.float32))
    for _ in range(epochs):
        for b in torch.randperm(len(x)).split(512):
            loss = (model(x[b]) - t[b]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()

    def predict(s):
        with torch.no_grad():
            return np.concatenate(
                [
                    model(torch.from_numpy(c)).numpy()
                    for c in np.array_split(s, max(1, len(s) // 4096))
                ]
            )

    return predict


def _fit_models(train_rows, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    X = np.concatenate([r[0] for r in train_rows])
    seq = np.concatenate([r[1] for r in train_rows])
    y = np.concatenate([r[2] for r in train_rows])
    ridge = Ridge(alpha=1.0).fit(X, y)
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed).fit(
        X, y
    )
    gru = _gru_fit(seq, y, seed)
    return {
        "ridge": lambda X_, s_: ridge.predict(X_),
        "gbm": lambda X_, s_: gbm.predict(X_),
        "gru": lambda X_, s_: gru(s_),
    }


def _climatology(s: SiteSeries) -> np.ndarray:
    """Day-of-year mean of z over training years, smoothed over +-7 days."""
    train = _mask(s.dates, TRAIN)
    doy = (s.dates - s.dates.astype("datetime64[Y]")).astype(int) % 365
    mean = np.array([np.nanmean(s.z[train & (np.abs(doy - d) % 365 <= 7)]) for d in range(365)])
    return mean[doy]


def _exceedance(obs_q, pred_q, thr):
    o, p = obs_q > thr, pred_q > thr
    tp, fp, fn = int((o & p).sum()), int((~o & p).sum()), int((o & ~p).sum())
    return {"tp": tp, "fp": fp, "fn": fn}


def _scores(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "csi": tp / (tp + fp + fn) if tp + fp + fn else None,
    }


def run() -> Path:
    sites = load_sites()
    if len(sites) < 3:
        raise RuntimeError("need discharge series; run `terramind forecast-fetch`")
    report = {
        "target": "GloFAS v4 reanalysis river discharge (a model river), Open-Meteo",
        "splits": {"train": TRAIN, "valid": VALID, "test": TEST},
        "sites": [
            {"id": s.id, "name": s.name, "event_date": s.event_date, "threshold_q95": s.threshold}
            for s in sites
        ],
        "leads": {},
        "events": [],
    }
    METHODS = ("persistence", "climatology", "ridge", "gbm", "gru")
    for lead in LEADS:
        rows = {s.id: features(s, lead) for s in sites}
        train_rows = [
            tuple(a[_mask(s.dates[r[3]], TRAIN)] for a in r) for s in sites for r in [rows[s.id]]
        ]
        models = _fit_models(train_rows)
        per_site, pooled = {}, {m: {"tp": 0, "fp": 0, "fn": 0} for m in METHODS}
        sq_err = {m: [] for m in METHODS}
        preds_by_site = {}
        for s in sites:
            X, seq, y, idx = rows[s.id]
            test = _mask(s.dates[idx], TEST)
            X, seq, y, idx = X[test], seq[test], y[test], idx[test]
            z0 = s.z[idx]
            clim = _climatology(s)
            pred_z = {
                "persistence": z0,
                "climatology": clim[idx + lead],
                **{m: z0 + f(X, seq) for m, f in models.items()},
            }
            obs_z = s.z[idx + lead]
            obs_q = s.q[idx + lead]
            per_site[s.id] = {}
            preds_by_site[s.id] = (idx, {m: to_q(z, s) for m, z in pred_z.items()})
            for m, z in pred_z.items():
                q = to_q(z, s)
                per_site[s.id][m] = {"nse": _nse(obs_q, q)}
                sq_err[m].append((obs_z - z) ** 2)
                c = _exceedance(obs_q, q, s.threshold)
                for k in c:
                    pooled[m][k] += c[k]
        mse = {m: float(np.concatenate(v).mean()) for m, v in sq_err.items()}
        report["leads"][str(lead)] = {
            "median_nse": {
                m: float(np.median([per_site[s][m]["nse"] for s in per_site])) for m in METHODS
            },
            "skill_vs_persistence": {m: 1 - mse[m] / mse["persistence"] for m in METHODS},
            "exceedance": {m: _scores(pooled[m]) for m in METHODS},
            "per_site_nse": per_site,
        }
        # Did the model river flood at the satellite-observed event, and was it forecast?
        for s in sites:
            d = np.datetime64(s.event_date)
            if not (np.datetime64(TEST[0]) <= d <= np.datetime64(TEST[1])):
                continue
            idx, preds = preds_by_site[s.id]
            target_days = s.dates[idx + lead]
            win = np.abs((target_days - d).astype(int)) <= EVENT_WINDOW
            glofas_floods = (
                bool(np.nanmax(s.q[idx + lead][win]) > s.threshold) if win.any() else None
            )
            entry = next((e for e in report["events"] if e["site"] == s.id), None)
            if entry is None:
                entry = {
                    "site": s.id,
                    "event_date": s.event_date,
                    "glofas_exceeds_q95": glofas_floods,
                    "peak_over_threshold": float(np.nanmax(s.q[idx + lead][win]) / s.threshold)
                    if win.any()
                    else None,
                    "forecast_hit": {},
                }
                report["events"].append(entry)
            entry["forecast_hit"][str(lead)] = {
                m: bool((p[win] > s.threshold).any()) if win.any() else None
                for m, p in preds.items()
            }
    # Leave one site out, gradient boosting, 3-day lead: does the pooled model transfer?
    lead = 3
    rows = {s.id: features(s, lead) for s in sites}
    loso = {}
    for held in sites:
        train_rows = [
            tuple(a[_mask(s.dates[rows[s.id][3]], TRAIN)] for a in rows[s.id])
            for s in sites
            if s.id != held.id
        ]
        from sklearn.ensemble import HistGradientBoostingRegressor

        X = np.concatenate([r[0] for r in train_rows])
        y = np.concatenate([r[2] for r in train_rows])
        gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(
            X, y
        )
        Xh, _, _, idx = rows[held.id]
        test = _mask(held.dates[idx], TEST)
        q = to_q(held.z[idx[test]] + gbm.predict(Xh[test]), held)
        loso[held.id] = {
            "nse_unseen_site": _nse(held.q[idx[test] + lead], q),
            "nse_pooled": report["leads"][str(lead)]["per_site_nse"][held.id]["gbm"]["nse"],
        }
    report["leave_one_site_out_gbm_lead3"] = loso
    out = runs_dir() / "e3-discharge.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
