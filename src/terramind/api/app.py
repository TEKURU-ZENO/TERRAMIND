"""Read-only HTTP API for the Earth Explorer.

Serves the dataset registry, candidate events, STAC scene manifests and the
precomputed chip index and overlays. Spatial filtering is done in-process with
Shapely; PostGIS replaces this once data outgrows memory.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import lru_cache
from statistics import mean

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from shapely.geometry import box

from terramind import registry
from terramind.config import data_dir
from terramind.explorer.build import LAYERS, out_dir

app = FastAPI(title="TERRA MIND API", version="0.0.1")
CHIP_ID = re.compile(r"^[A-Za-z-]+_\d+$")


@lru_cache(maxsize=1)
def _index() -> dict:
    path = out_dir() / "index.json"
    if not path.exists():
        raise HTTPException(503, "explorer index not built; run `terramind build-explorer`")
    return json.loads(path.read_text(encoding="utf-8"))


def _feature(chip: dict) -> dict:
    w, s, e, n = chip["bounds"]
    ring = [[w, s], [e, s], [e, n], [w, n], [w, s]]
    return {
        "type": "Feature",
        "id": chip["id"],
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {k: v for k, v in chip.items() if k not in ("bounds", "layers")},
    }


@lru_cache(maxsize=1)
def _spatial_store():
    """A spatial database if TERRAMIND_DATABASE_URL or TERRAMIND_SPATIAL=duckdb is set."""
    import os

    if os.environ.get("TERRAMIND_DATABASE_URL") or os.environ.get("TERRAMIND_SPATIAL") == "duckdb":
        from terramind.storage.spatial import open_store

        return open_store()
    return None


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/datasets")
def datasets() -> list[dict]:
    return [d.model_dump() for d in registry.load_datasets().values()]


@app.get("/api/events")
def events() -> list[dict]:
    return [e.model_dump() for e in registry.load_events().values()]


@app.get("/api/events/{event_id}/scenes")
def scenes(event_id: str) -> dict:
    """Scenes found by earlier `terramind stac-search` runs for this event."""
    if event_id not in registry.load_events():
        raise HTTPException(404, "unknown event")
    found = []
    for path in sorted((data_dir() / "manifests").glob(f"{event_id}__*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for scene in manifest["scenes"]:
            found.append(
                {
                    "id": scene["id"],
                    "collection": scene["collection"],
                    "catalog": scene["catalog"],
                    "datetime": scene["datetime"],
                    "bbox": scene["bbox"],
                    "cloud_cover": scene["cloud_cover"],
                    "platform": scene["platform"],
                }
            )
    return {"event": event_id, "scenes": sorted(found, key=lambda s: s["datetime"] or "")}


@app.get("/api/chips")
def chips(
    bbox: str | None = Query(None, description="west,south,east,north in EPSG:4326"),
    event: str | None = None,
    split: str | None = None,
) -> dict:
    """Chip footprints as GeoJSON, optionally filtered by bbox, event and split."""
    selected = _index()["chips"]
    if event:
        selected = [c for c in selected if c["event"] == event]
    if split:
        selected = [c for c in selected if c["split"] == split]
    if bbox:
        try:
            values = [float(v) for v in bbox.split(",")]
            query = box(*values)
        except (ValueError, TypeError):
            raise HTTPException(422, "bbox must be west,south,east,north") from None
        db = _spatial_store()
        if db is not None:
            # Spatial database (PostGIS or DuckDB) when one is configured.
            ids = set(db.in_bbox("chips", values))
            selected = [c for c in selected if c["id"] in ids]
        else:
            selected = [c for c in selected if box(*c["bounds"]).intersects(query)]
    return {
        "type": "FeatureCollection",
        "model": _index()["model"],
        "note": _index()["note"],
        "features": [_feature(c) for c in selected],
    }


@app.get("/api/chips/{chip_id}")
def chip(chip_id: str) -> dict:
    for c in _index()["chips"]:
        if c["id"] == chip_id:
            return c
    raise HTTPException(404, "unknown chip")


@app.get("/api/chips/{chip_id}/{layer}.png")
def chip_layer(chip_id: str, layer: str) -> FileResponse:
    if not CHIP_ID.match(chip_id) or layer not in LAYERS:
        raise HTTPException(404, "unknown chip or layer")
    path = out_dir() / chip_id / f"{layer}.png"
    if not path.exists():
        raise HTTPException(404, "layer not available for this chip")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/summary")
def summary() -> dict:
    """Per-event, per-split chip counts and mean per-chip IoU of the explorer model."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in _index()["chips"]:
        groups[(c["event"], c["split"])].append(c)
    rows = []
    for (event, split), items in sorted(groups.items()):
        ious = [c["iou"] for c in items if c["iou"] is not None]
        bounds = [c["bounds"] for c in items]
        rows.append(
            {
                "event": event,
                "split": split,
                "chips": len(items),
                "mean_chip_iou": mean(ious) if ious else None,
                "bounds": [
                    min(b[0] for b in bounds),
                    min(b[1] for b in bounds),
                    max(b[2] for b in bounds),
                    max(b[3] for b in bounds),
                ],
            }
        )
    return {"model": _index()["model"], "rows": rows}


@lru_cache(maxsize=1)
def _core():
    """TerraCore is built on first use: loading the embedding model takes a few seconds."""
    from terramind.agents.workflow import TerraCore
    from terramind.evidence import store
    from terramind.rag.retriever import Retriever

    try:
        corpus = store.load_corpus()
    except FileNotFoundError as err:
        raise HTTPException(503, str(err)) from None
    return TerraCore(Retriever(corpus))


@app.get("/api/ask")
def ask(q: str = Query(..., min_length=3, max_length=400)) -> dict:
    """Answer a question from evidence only; abstains when nothing supports an answer."""
    return _core().ask(q)


@lru_cache(maxsize=2)
def _scenarios(modality: str):
    from terramind.sim import scenarios

    try:
        return scenarios.load(modality)
    except FileNotFoundError:
        raise HTTPException(503, "scenarios not built; run `terramind build-sim`") from None


@lru_cache(maxsize=8)
def _policy(name: str):
    from terramind.sim import evaluate

    try:
        return evaluate.make_policy(name)
    except LookupError as err:
        raise HTTPException(404, str(err)) from None


@app.get("/api/sim/results")
def sim_results() -> dict:
    """The E7 policy comparison (simulated), if it has been run."""
    from terramind.sim.ppo import runs_dir

    path = runs_dir() / "e7-s1.json"
    if not path.exists():
        raise HTTPException(503, "no simulation results; run `terramind sim-eval`")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/sim/episode")
def sim_episode(
    chip: str = Query(..., max_length=60),
    policy: str = Query("lookahead", max_length=40),
    depot: int = Query(0, ge=0, le=1000),
) -> dict:
    """Play one simulated flood-response episode on a chip and return every step."""
    from terramind.sim import evaluate

    if not CHIP_ID.match(chip):
        raise HTTPException(404, "unknown chip")
    try:
        return evaluate.replay(_scenarios("s1"), chip, _policy(policy), depot)
    except LookupError as err:
        raise HTTPException(404, str(err)) from None


@app.get("/api/report/{chip_id}")
def situation_report(chip_id: str, narrate: bool = False) -> dict:
    """Perception, reliability and a simulated response for one chip, with evidence ids."""
    from terramind.integration import situation

    if not CHIP_ID.match(chip_id):
        raise HTTPException(404, "unknown chip")
    try:
        report = situation.build(chip_id)
        if narrate:
            # A local LLM's summary, kept only if it adds no numbers.
            report["summary"] = situation.narrate(report)
        return report
    except LookupError as err:
        raise HTTPException(404, str(err)) from None
    except FileNotFoundError as err:
        raise HTTPException(503, f"missing input: {err}") from None
