"""River discharge (GloFAS v4) and precipitation (ERA5) at the flood event sites.

Both come from Open-Meteo's free, non-commercial API (data CC BY 4.0; GloFAS from
the Copernicus Emergency Management Service, ERA5 from the Copernicus Climate
Change Service, both CC BY 4.0 since 2 July 2025). No account is needed.

Each site is one flood event: the 11 Sen1Floods11 events (date from the
dataset's own metadata file) and the 4 candidate events in configs/events.yaml
(dates approximate). GloFAS is a 0.05-degree model grid, so the cell nearest a
site may not be on the flooded river. We pick, among a 5 x 5 block of cells
around the site, the one with the largest mean discharge in one sample year.

Note: GloFAS discharge is itself a simulation (a hydrological model driven by
reanalysis), not a gauge record. Forecasting it measures how predictable the
model river is, not the real one.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import requests

from terramind import registry
from terramind.config import data_dir

FLOOD_API = "https://flood-api.open-meteo.com/v1/flood"
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
START, END = "1991-01-01", "2024-12-31"
GRID_STEP = 0.05
MIN_RIVER = 2.0  # m3/s; below this the nearby cells are treated as missing the river
USER_AGENT = "terramind-research/0.1"


@dataclass
class Site:
    id: str
    name: str
    lat: float
    lon: float
    event_date: str
    date_source: str


def out_dir() -> Path:
    return data_dir() / "raw" / "openmeteo"


def sites() -> list[Site]:
    """Event sites: chip-extent centres for Sen1Floods11, box centres for candidates."""
    from terramind.evidence import store

    meta_path = data_dir() / "raw" / "sen1floods11" / "meta" / "Sen1Floods11_Metadata.geojson"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dates = {
        f["properties"]["location"]: f["properties"]["s1_date"].replace("/", "-")
        for f in meta["features"]
    }
    dates["Mekong"] = dates.get("Cambodia")  # the metadata names the Mekong event by country
    out = []
    for e in store.load_corpus():
        if e.kind == "event" and e.id.startswith("event:s1f11-") and e.bbox:
            name = e.id.removeprefix("event:s1f11-")
            if not dates.get(name):
                continue
            w, s, east, n = e.bbox
            out.append(
                Site(
                    name.lower(),
                    name,
                    (s + n) / 2,
                    (w + east) / 2,
                    dates[name],
                    "Sen1Floods11 metadata (s1_date)",
                )
            )
    for ev in registry.load_events().values():
        w, s, east, n = ev.bbox
        out.append(
            Site(
                ev.id,
                ev.name,
                (s + n) / 2,
                (w + east) / 2,
                ev.start,
                "configs/events.yaml start date (unconfirmed)",
            )
        )
    return out


def _get(url: str, params: dict, tries: int = 4) -> dict | list:
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120, headers={"User-Agent": USER_AGENT})
            if r.status_code == 429:
                time.sleep(60)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def pick_river_cell(
    site: Site, year: int = 2018, half: int = 2, step: float = GRID_STEP
) -> tuple[float, float, float]:
    """(lat, lon, mean discharge) of the largest-discharge cell near the site."""
    offsets = np.arange(-half, half + 1) * step
    lats = [round(site.lat + a, 4) for a in offsets for _ in offsets]
    lons = [round(site.lon + b, 4) for _ in offsets for b in offsets]
    res = _get(
        FLOOD_API,
        {
            "latitude": ",".join(map(str, lats)),
            "longitude": ",".join(map(str, lons)),
            "daily": "river_discharge",
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
        },
    )
    best = None
    for cell in res if isinstance(res, list) else [res]:
        q = np.array([v for v in cell["daily"]["river_discharge"] if v is not None], float)
        if len(q) and (best is None or q.mean() > best[2]):
            best = (cell["latitude"], cell["longitude"], float(q.mean()))
    if best is None:
        raise RuntimeError(f"no discharge near {site.id}")
    return best


def fetch(force: bool = False) -> Path:
    """Download one discharge and one precipitation series per site (cached)."""
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    for site in sites():
        path = out / f"{site.id}.json"
        if path.exists() and not force:
            continue
        lat, lon, mean_q = pick_river_cell(site)
        search = "5 x 5 cells, 0.05 degree"
        if mean_q < MIN_RIVER:
            # Only a trickle nearby: look for the main river within about 50 km.
            lat, lon, mean_q = pick_river_cell(site, half=5, step=0.1)
            search = "11 x 11 cells, 0.1 degree"
        q = _get(
            FLOOD_API,
            {
                "latitude": lat,
                "longitude": lon,
                "daily": "river_discharge",
                "start_date": START,
                "end_date": END,
            },
        )
        p = _get(
            ARCHIVE_API,
            {
                "latitude": round(site.lat, 4),
                "longitude": round(site.lon, 4),
                "daily": "precipitation_sum",
                "start_date": START,
                "end_date": END,
                "timezone": "UTC",
            },
        )
        record = {
            "site": asdict(site),
            "river_cell": {"lat": lat, "lon": lon, "mean_discharge_2018": mean_q, "search": search},
            "time": q["daily"]["time"],
            "discharge": q["daily"]["river_discharge"],
            "precipitation": dict(zip(p["daily"]["time"], p["daily"]["precipitation_sum"])),
            "source": {
                "discharge": "GloFAS v4 reanalysis via Open-Meteo Flood API (CC BY 4.0)",
                "precipitation": "ERA5 via Open-Meteo Historical Weather API (CC BY 4.0)",
            },
        }
        path.write_text(json.dumps(record), encoding="utf-8")
        print(f"{site.id}: river cell {lat:.3f},{lon:.3f} mean {mean_q:.1f} m3/s", flush=True)
        time.sleep(2)
    return out


def load(site_id: str) -> dict:
    return json.loads((out_dir() / f"{site_id}.json").read_text(encoding="utf-8"))
