"""Dataset/licence registry and flood event catalogue."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

from terramind.config import load_yaml

Redistribution = Literal[
    "allowed",
    "allowed-with-attribution",
    "share-alike",
    "non-commercial-share-alike",
    "unknown",
]


class Dataset(BaseModel):
    id: str
    name: str
    provider: str
    access_url: str
    modality: str
    resolution: str
    licence: str
    licence_verified: bool
    licence_checked_on: str | None = None
    licence_source: str | None = None  # where the licence was read
    attribution: str | None = None  # credit line the licence asks for
    redistribution: Redistribution
    used_for: list[str]
    stac_collection: str | None = None
    notes: str | None = None


class FloodEvent(BaseModel):
    id: str
    name: str
    region: str
    bbox: tuple[float, float, float, float]
    start: str
    end: str
    confirmed: bool = False

    @field_validator("bbox")
    @classmethod
    def _valid_bbox(cls, v):
        west, south, east, north = v
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError(f"bbox must be (west, south, east, north) in EPSG:4326, got {v}")
        return v


def load_datasets() -> dict[str, Dataset]:
    items = [Dataset(**d) for d in load_yaml("datasets.yaml")["datasets"]]
    ids = [d.id for d in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate dataset ids in configs/datasets.yaml")
    return {d.id: d for d in items}


def load_events() -> dict[str, FloodEvent]:
    return {e["id"]: FloodEvent(**e) for e in load_yaml("events.yaml")["events"]}


def require_dataset(dataset_id: str) -> Dataset:
    """Fetch a registry entry, failing loudly if the dataset was never registered."""
    datasets = load_datasets()
    if dataset_id not in datasets:
        raise KeyError(
            f"'{dataset_id}' is not in configs/datasets.yaml. Register its source and "
            "licence before using it."
        )
    return datasets[dataset_id]
