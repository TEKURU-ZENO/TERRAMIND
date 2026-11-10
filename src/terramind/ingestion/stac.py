"""STAC scene discovery.

Discovery only: this finds scenes and records where their assets live, with
enough provenance to reproduce the search. It does not download rasters.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

CATALOGS = {
    # Anonymous search. Sentinel-2 COGs are public; Sentinel-1 GRD assets sit in
    # a requester-pays bucket, so use CDSE for S1 downloads.
    "earth-search": "https://earth-search.aws.element84.com/v1",
    # Anonymous search; asset download needs a free CDSE account.
    "cdse": "https://stac.dataspace.copernicus.eu/v1",
    # Anonymous search; asset hrefs must be signed with `planetary-computer`.
    "planetary-computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
}


@dataclass
class SceneRecord:
    id: str
    collection: str
    catalog: str
    datetime: str | None
    bbox: list[float] | None
    platform: str | None
    cloud_cover: float | None
    assets: dict[str, str] = field(default_factory=dict)


def item_to_record(item, catalog: str) -> SceneRecord:
    props = item.properties
    dt = item.datetime.isoformat() if item.datetime else props.get("start_datetime")
    return SceneRecord(
        id=item.id,
        collection=item.collection_id,
        catalog=catalog,
        datetime=dt,
        bbox=list(item.bbox) if item.bbox else None,
        platform=props.get("platform"),
        cloud_cover=props.get("eo:cloud_cover"),
        assets={key: asset.href for key, asset in item.assets.items()},
    )


def search(
    collection: str,
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    catalog: str = "earth-search",
    max_cloud: float | None = None,
    limit: int = 100,
) -> list[SceneRecord]:
    from pystac_client import Client

    if catalog not in CATALOGS:
        raise ValueError(f"unknown catalog '{catalog}', expected one of {sorted(CATALOGS)}")
    client = Client.open(CATALOGS[catalog])
    query = {"eo:cloud_cover": {"lt": max_cloud}} if max_cloud is not None else None
    result = client.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        query=query,
        max_items=limit,
    )
    records = [item_to_record(item, catalog) for item in result.items()]
    return sorted(records, key=lambda r: r.datetime or "")


def write_manifest(records: list[SceneRecord], path: Path, search_params: dict) -> Path:
    """Persist search results with the parameters that produced them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "searched_at": datetime.now(UTC).isoformat(),
        "search": search_params,
        "count": len(records),
        "scenes": [asdict(r) for r in records],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
