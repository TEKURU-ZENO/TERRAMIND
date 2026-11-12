"""OpenStreetMap roads for each Sen1Floods11 chip, via the Overpass API.

One small query per chip (about 5 x 5 km), run one at a time with a pause, as the
Overpass usage policy asks. Data © OpenStreetMap contributors, ODbL 1.0: derived
road graphs stay under ODbL and must credit OpenStreetMap.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import rasterio
import requests

from terramind.config import data_dir
from terramind.ingestion import sen1floods11 as s1f11

OVERPASS = "https://overpass-api.de/api/interpreter"
VEHICLE_ROADS = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|track|road|"
    "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street"
)
USER_AGENT = "terramind-research/0.1 (flood road-graph study)"


def out_dir() -> Path:
    return data_dir() / "raw" / "osm"


def chips() -> list[tuple[str, str, tuple[float, float, float, float]]]:
    """(chip id, split, (west, south, east, north)) for every labelled chip."""
    out = []
    for split in s1f11.SPLIT_FILES:
        for s1_path, _ in s1f11.available_pairs(split):
            with rasterio.open(s1_path) as src:
                b = src.bounds
            out.append(
                (s1_path.name.replace("_S1Hand.tif", ""), split, (b.left, b.bottom, b.right, b.top))
            )
    return out


def fetch_chip(chip_id: str, bounds, tries: int = 4) -> dict:
    w, s, e, n = bounds
    query = (
        f'[out:json][timeout:90];way["highway"~"^({VEHICLE_ROADS})$"]({s},{w},{n},{e});'
        "(._;>;);out body;"
    )
    for attempt in range(tries):
        try:
            r = requests.post(
                OVERPASS, data={"data": query}, timeout=120, headers={"User-Agent": USER_AGENT}
            )
            if r.status_code in (429, 504):
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            break
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"Overpass kept refusing {chip_id}")
    nodes = {el["id"]: (el["lon"], el["lat"]) for el in data["elements"] if el["type"] == "node"}
    ways = [
        {"id": el["id"], "highway": el.get("tags", {}).get("highway"), "nodes": el["nodes"]}
        for el in data["elements"]
        if el["type"] == "way"
    ]
    return {
        "chip": chip_id,
        "bounds": bounds,
        "nodes": {str(k): v for k, v in nodes.items()},
        "ways": ways,
        "osm_base": data.get("osm3s", {}).get("timestamp_osm_base"),
        "licence": "ODbL 1.0, © OpenStreetMap contributors",
    }


def _post(query: str, tries: int = 5) -> dict:
    for attempt in range(tries):
        try:
            r = requests.post(
                OVERPASS, data={"data": query}, timeout=400, headers={"User-Agent": USER_AGENT}
            )
            if r.status_code in (429, 504):
                time.sleep(60 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("Overpass kept refusing the query")


def fetch(pause: float = 10.0, group: int = 40) -> Path:
    """One Overpass query per batch of up to `group` chips (a union of their boxes),
    then the roads are split back out per chip. Far fewer requests than per chip."""
    from shapely.geometry import LineString, box

    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in chips() if not (out / f"{c[0]}.json").exists()]
    for start in range(0, len(todo), group):
        batch = todo[start : start + group]
        parts = "".join(
            f'way["highway"~"^({VEHICLE_ROADS})$"]({s},{w},{n},{e});'
            for _, _, (w, s, e, n) in batch
        )
        data = _post(f"[out:json][timeout:300];({parts});(._;>;);out body;")
        nodes = {
            el["id"]: (el["lon"], el["lat"]) for el in data["elements"] if el["type"] == "node"
        }
        ways = [el for el in data["elements"] if el["type"] == "way"]
        for chip_id, _, bounds in batch:
            area = box(*bounds)
            mine = []
            for el in ways:
                coords = [nodes[n] for n in el["nodes"] if n in nodes]
                if len(coords) >= 2 and LineString(coords).intersects(area):
                    mine.append(el)
            used = {n for el in mine for n in el["nodes"] if n in nodes}
            rec = {
                "chip": chip_id,
                "bounds": bounds,
                "nodes": {str(n): nodes[n] for n in used},
                "ways": [
                    {
                        "id": el["id"],
                        "highway": el.get("tags", {}).get("highway"),
                        "nodes": el["nodes"],
                    }
                    for el in mine
                ],
                "osm_base": data.get("osm3s", {}).get("timestamp_osm_base"),
                "licence": "ODbL 1.0, © OpenStreetMap contributors",
            }
            (out / f"{chip_id}.json").write_text(json.dumps(rec), encoding="utf-8")
        print(f"{start + len(batch)}/{len(todo)} chips, {len(ways)} ways in this batch", flush=True)
        time.sleep(pause)
    return out


def load(chip_id: str) -> dict:
    return json.loads((out_dir() / f"{chip_id}.json").read_text(encoding="utf-8"))
