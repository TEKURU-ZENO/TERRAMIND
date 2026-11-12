"""Road graphs on the Sen1Floods11 chips, with per-point features and flood labels.

Nodes are points along OSM vehicle roads, about every two pixels (20 m), merged
when they fall in the same pixel (so junctions join). Edges follow the roads.

Per-node features (all available without the hand labels):
- water probability from the radar U-Net trained with this chip's flood event
  held out (leave-one-event-out), so the feature is out-of-sample for every chip
- radar backscatter VV and VH
- Copernicus DEM GLO-30 elevation: relative to the chip median, relative to
  the lowest point within ~300 m, and local slope
- distance to the nearest pixel the radar model calls water
- road class (7 groups) and node degree

Label: whether the hand label marks the node's pixel as water (3 x 3 majority of
labelled pixels; nodes with no labelled pixel are dropped).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from terramind.config import data_dir
from terramind.ingestion import sen1floods11 as s1f11
from terramind.processing.raster import IGNORE_INDEX, read_label_chip, read_s1_chip
from terramind.roads import osm

ROAD_GROUPS = {
    "motorway": 0, "trunk": 0, "motorway_link": 0, "trunk_link": 0,
    "primary": 1, "primary_link": 1,
    "secondary": 2, "secondary_link": 2,
    "tertiary": 3, "tertiary_link": 3,
    "unclassified": 4, "road": 4,
    "residential": 5, "living_street": 5, "service": 5,
    "track": 6,
}  # fmt: skip
N_GROUPS = 7
FEATURES = (
    "water_prob", "vv", "vh", "elev_rel_median", "elev_rel_local_min", "slope",
    "dist_to_pred_water", *[f"road_group_{i}" for i in range(N_GROUPS)], "degree",
)  # fmt: skip
DEM_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM/"
    "Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM.tif"
)


def out_dir() -> Path:
    return data_dir() / "derived" / "roads"


def _tile(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return "/vsicurl/" + DEM_URL.format(lat=ns, lon=ew)


def dem_for_chip(s1_path: Path) -> np.ndarray:
    """Copernicus DEM GLO-30 resampled (bilinear) onto the chip's 512 x 512 grid."""
    cache = out_dir() / "dem" / s1_path.name.replace("_S1Hand.tif", ".npy")
    if cache.exists():
        return np.load(cache)
    with rasterio.open(s1_path) as src:
        transform, crs, shape, b = src.transform, src.crs, (src.height, src.width), src.bounds
    out = np.full(shape, np.nan, np.float32)
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"}
    with rasterio.Env(**env):
        for lat in range(int(np.floor(b.bottom)), int(np.floor(b.top)) + 1):
            for lon in range(int(np.floor(b.left)), int(np.floor(b.right)) + 1):
                try:
                    with rasterio.open(_tile(lat, lon)) as dem:
                        part = np.full(shape, np.nan, np.float32)
                        reproject(
                            rasterio.band(dem, 1), part, dst_transform=transform, dst_crs=crs,
                            resampling=Resampling.bilinear, dst_nodata=np.nan,
                        )  # fmt: skip
                        out = np.where(np.isnan(out), part, out)
                except rasterio.errors.RasterioIOError:
                    continue  # ocean tiles do not exist
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def loeo_probabilities(device: str | None = None) -> dict[str, np.ndarray]:
    """Out-of-event water probability per chip, from the radar leave-one-event-out models."""
    import torch

    from terramind.models.unet import UNet
    from terramind.training.runs import runs_dir, select_loeo_runs
    from terramind.training.train_baseline import load_chip

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache = out_dir() / "prob"
    cache.mkdir(parents=True, exist_ok=True)
    runs = {event: rep for (event, _seed), rep in select_loeo_runs("s1").items()}
    dirs = {}
    for path in runs_dir().glob("loeo-*-s1-*/report.json"):
        import json

        rep = json.loads(path.read_text(encoding="utf-8"))
        if rep["modality"] == "s1" and rep["holdout_event"] in runs:
            dirs[rep["holdout_event"]] = path.parent  # latest wins (sorted glob is not needed)
    out = {}
    models: dict[str, torch.nn.Module] = {}
    with torch.no_grad():
        for split in s1f11.SPLIT_FILES:
            for s1_path, label_path in s1f11.available_pairs(split):
                chip = s1_path.name.replace("_S1Hand.tif", "")
                f = cache / f"{chip}.npy"
                if f.exists():
                    out[chip] = np.load(f).astype(np.float32)
                    continue
                event = s1f11.event_of(s1_path.name)
                if event not in models:
                    m = UNet(in_channels=2).to(device).eval()
                    m.load_state_dict(torch.load(dirs[event] / "model.pt", map_location=device))
                    models[event] = m
                img, _ = load_chip(s1_path, label_path, "s1")
                prob = models[event](torch.from_numpy(img)[None].to(device)).softmax(1)[0, 1]
                out[chip] = prob.float().cpu().numpy()
                np.save(f, out[chip].astype(np.float16))
    return out


def build_chip(
    s1_path: Path, label_path: Path, prob: np.ndarray, step_px: float = 2.0
) -> dict | None:
    chip = s1_path.name.replace("_S1Hand.tif", "")
    try:
        roads = osm.load(chip)
    except FileNotFoundError:
        return None
    with rasterio.open(s1_path) as src:
        inv = ~src.transform
        h, w = src.height, src.width
    label = read_label_chip(label_path)
    s1, s1_valid = read_s1_chip(s1_path)
    dem = dem_for_chip(s1_path)

    # Points along roads, merged per pixel; edges between consecutive points.
    node_of: dict[tuple[int, int], int] = {}
    group: dict[int, int] = {}
    edges = set()
    for way in roads["ways"]:
        coords = [roads["nodes"].get(str(n)) for n in way["nodes"]]
        coords = [c for c in coords if c]
        g = ROAD_GROUPS.get(way["highway"], 4)
        prev = None
        for (x0, y0), (x1, y1) in pairwise(coords):
            c0, r0 = inv * (x0, y0)
            c1, r1 = inv * (x1, y1)
            n_pts = max(1, int(np.hypot(c1 - c0, r1 - r0) / step_px))
            for t in np.linspace(0, 1, n_pts + 1):
                r, c = int(r0 + t * (r1 - r0)), int(c0 + t * (c1 - c0))
                if not (0 <= r < h and 0 <= c < w):
                    prev = None
                    continue
                key = (r, c)
                if key not in node_of:
                    node_of[key] = len(node_of)
                i = node_of[key]
                group[i] = min(group.get(i, g), g)
                if prev is not None and prev != i:
                    edges.add((min(prev, i), max(prev, i)))
                prev = i
    if len(node_of) < 5:
        return None
    rc = np.array(list(node_of.keys()))
    rows, cols = rc[:, 0], rc[:, 1]

    def win_mean(a):
        return ndimage.uniform_filter(np.nan_to_num(a.astype(np.float32)), 3)[rows, cols]

    known = (label != IGNORE_INDEX).astype(np.float32)
    water = (label == 1).astype(np.float32)
    known_frac = ndimage.uniform_filter(known, 3)[rows, cols]
    water_frac = ndimage.uniform_filter(water, 3)[rows, cols]
    keep = known_frac > 0
    y = (water_frac / np.maximum(known_frac, 1e-6) >= 0.5).astype(np.int64)

    dem_f = np.where(np.isnan(dem), np.nanmedian(dem) if np.isfinite(dem).any() else 0.0, dem)
    rel_med = (dem_f - np.median(dem_f)) / 10.0
    rel_min = (dem_f - ndimage.minimum_filter(dem_f, size=31)) / 10.0
    gy, gx = np.gradient(dem_f, 10.0)
    slope = np.hypot(gx, gy)
    dist = ndimage.distance_transform_edt(prob < 0.5) / 100.0

    deg = np.zeros(len(node_of))
    for a, b_ in edges:
        deg[a] += 1
        deg[b_] += 1
    onehot = np.zeros((len(node_of), N_GROUPS), np.float32)
    onehot[np.arange(len(node_of)), [group[i] for i in range(len(node_of))]] = 1.0
    x = np.column_stack(
        [
            win_mean(prob), win_mean(s1[0] * s1_valid), win_mean(s1[1] * s1_valid),
            rel_med[rows, cols], rel_min[rows, cols], slope[rows, cols], dist[rows, cols],
            onehot, deg / 4.0,
        ]
    ).astype(np.float32)  # fmt: skip

    # Keep only labelled nodes, re-index edges.
    new = -np.ones(len(node_of), np.int64)
    new[keep] = np.arange(int(keep.sum()))
    e = np.array([(new[a], new[b_]) for a, b_ in edges if keep[a] and keep[b_]], np.int64).reshape(
        -1, 2
    )
    return {
        "chip": chip,
        "split": None,
        "event": s1f11.event_of(s1_path.name),
        "x": x[keep],
        "y": y[keep],
        "edges": e,
        "rows": rows[keep],
        "cols": cols[keep],
    }


def build_all() -> Path:
    import pickle

    probs = loeo_probabilities()
    graphs = []
    for split in s1f11.SPLIT_FILES:
        for s1_path, label_path in s1f11.available_pairs(split):
            chip = s1_path.name.replace("_S1Hand.tif", "")
            g = build_chip(s1_path, label_path, probs[chip])
            if g is not None and len(g["y"]):
                g["split"] = split
                graphs.append(g)
    path = out_dir() / "graphs.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"features": FEATURES, "graphs": graphs}, f)
    counts = defaultdict(lambda: [0, 0, 0])
    for g in graphs:
        counts[g["split"]][0] += 1
        counts[g["split"]][1] += len(g["y"])
        counts[g["split"]][2] += int(g["y"].sum())
    print({k: {"chips": v[0], "nodes": v[1], "flooded": v[2]} for k, v in counts.items()})
    return path
