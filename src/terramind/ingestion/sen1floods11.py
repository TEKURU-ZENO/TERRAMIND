"""Sen1Floods11 hand-labelled subset: download and split handling.

Sentinel-1 chips are 2-band (VV, VH) backscatter in dB, 512x512 at 10 m.
Labels: 1 = water, 0 = not water, -1 = no data.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from terramind.config import data_dir

BASE = "https://storage.googleapis.com/sen1floods11/v1.1"
SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}


def root() -> Path:
    return data_dir() / "raw" / "sen1floods11"


def event_of(filename: str) -> str:
    """'Sri-Lanka_101973_S1Hand.tif' -> 'Sri-Lanka'."""
    return filename.split("_", 1)[0]


def _fetch(url: str, dest: Path, retries: int = 4) -> bool:
    """Download to dest unless it already exists. Returns True if downloaded."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    f.writelines(r.iter_content(chunk_size=1 << 20))
            break
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries:
                raise
    tmp.replace(dest)
    return True


def read_split(split: str) -> list[tuple[str, str]]:
    """Return (s1_filename, label_filename) pairs for a split."""
    path = root() / "splits" / SPLIT_FILES[split]
    with open(path, newline="", encoding="utf-8") as f:
        return [(row[0].strip(), row[1].strip()) for row in csv.reader(f) if row]


def s2_name(s1_name: str) -> str:
    return s1_name.replace("_S1Hand", "_S2Hand")


def s2_path_for(s1_path: Path) -> Path:
    """Sentinel-2 chip co-registered with a Sentinel-1 chip (13-band L1C)."""
    return root() / "S2Hand" / s2_name(s1_path.name)


def download(
    limit_per_split: int | None = None, workers: int = 16, s2: bool = False
) -> dict[str, int]:
    """Download split files and chips. `limit_per_split` is for smoke tests.

    Downloads run in parallel because the bucket can be slow per connection.
    """
    counts, jobs = {}, []
    for split, name in SPLIT_FILES.items():
        _fetch(f"{BASE}/splits/flood_handlabeled/{name}", root() / "splits" / name)
        pairs = read_split(split)
        if limit_per_split is not None:
            pairs = pairs[:limit_per_split]
        for s1, label in pairs:
            chips = f"{BASE}/data/flood_events/HandLabeled"
            jobs.append((f"{chips}/S1Hand/{s1}", root() / "S1Hand" / s1))
            jobs.append((f"{chips}/LabelHand/{label}", root() / "LabelHand" / label))
            if s2:
                jobs.append((f"{chips}/S2Hand/{s2_name(s1)}", root() / "S2Hand" / s2_name(s1)))
        counts[split] = len(pairs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch, url, dest) for url, dest in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="sen1floods11"):
            future.result()
    return counts


def available_pairs(split: str) -> list[tuple[Path, Path]]:
    """Pairs from a split whose files are both on disk."""
    out = []
    for s1, label in read_split(split):
        s1_path, label_path = root() / "S1Hand" / s1, root() / "LabelHand" / label
        if s1_path.exists() and label_path.exists():
            out.append((s1_path, label_path))
    return out
