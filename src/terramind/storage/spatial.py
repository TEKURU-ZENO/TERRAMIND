"""Spatial database for chips, scenes and evidence: PostGIS, or embedded DuckDB.

Both backends hold the same three tables with a geometry column and answer the
same bounding-box queries the API needs. PostGIS (docker-compose service
`postgis`) is the deployment target; DuckDB with its spatial extension needs no
server and is the default when no database URL is configured.

    TERRAMIND_DATABASE_URL=postgresql://terramind:<password>@127.0.0.1:5432/terramind
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from terramind.config import data_dir

TABLES = {
    "chips": "id TEXT PRIMARY KEY, event TEXT, split TEXT, iou DOUBLE PRECISION, "
    "water_fraction DOUBLE PRECISION, props TEXT",
    "scenes": "id TEXT PRIMARY KEY, event TEXT, collection TEXT, datetime TEXT, "
    "cloud_cover DOUBLE PRECISION, props TEXT",
    "evidence": "id TEXT PRIMARY KEY, kind TEXT, title TEXT, start_date TEXT, end_date TEXT, "
    "provenance DOUBLE PRECISION, props TEXT",
}


def _envelope(b) -> str:
    w, s, e, n = b
    return f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"


def _rows():
    """(table, id, columns..., wkt or None) from the explorer index, manifests and evidence."""
    from terramind.evidence import store
    from terramind.explorer.build import out_dir as explorer_dir

    index = json.loads((explorer_dir() / "index.json").read_text(encoding="utf-8"))
    for c in index["chips"]:
        yield "chips", (c["id"], c["event"], c["split"], c["iou"], c["water_fraction"],
                        json.dumps(c)), _envelope(c["bounds"])  # fmt: skip
    for path in sorted((data_dir() / "manifests").glob("*.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        for sc in m["scenes"]:
            if sc.get("bbox"):
                yield "scenes", (sc["id"], m["search"]["event"], sc["collection"], sc["datetime"],
                                 sc["cloud_cover"], json.dumps(sc)), _envelope(sc["bbox"])  # fmt: skip
    for e in store.load_corpus():
        yield "evidence", (e.id, e.kind, e.title, e.start, e.end, e.provenance,
                           json.dumps(e.data)), _envelope(e.bbox) if e.bbox else None  # fmt: skip


class SpatialStore:
    """Common interface; subclasses supply the connection and SQL dialect."""

    placeholder = "?"

    def _exec(self, sql: str, params=()):
        raise NotImplementedError

    def load(self) -> dict[str, int]:
        counts = dict.fromkeys(TABLES, 0)
        for table, cols in TABLES.items():
            self._exec(f"DROP TABLE IF EXISTS {table}")
            self._exec(f"CREATE TABLE {table} ({cols}, geom {self.geometry_type})")
        p = self.placeholder
        for table, values, wkt in _rows():
            marks = ", ".join([p] * len(values))
            geom = f"ST_GeomFromText({p})" if wkt else "NULL"
            self._exec(
                f"INSERT INTO {table} VALUES ({marks}, {geom}) ON CONFLICT DO NOTHING",
                (*values, wkt) if wkt else values,
            )
            counts[table] += 1
        self._index()
        return counts

    def _index(self) -> None:
        pass

    def in_bbox(self, table: str, bbox, **filters) -> list[str]:
        """Ids of rows whose geometry intersects bbox (west, south, east, north)."""
        if table not in TABLES:
            raise ValueError(f"unknown table {table}")
        p = self.placeholder
        where = [f"ST_Intersects(geom, ST_GeomFromText({p}))"]
        params = [_envelope(bbox)]
        for col, val in filters.items():
            if val is not None:
                if col not in ("event", "split", "kind", "collection"):
                    raise ValueError(f"cannot filter on {col}")
                where.append(f"{col} = {p}")
                params.append(val)
        rows = self._exec(f"SELECT id FROM {table} WHERE {' AND '.join(where)} ORDER BY id", params)
        return [r[0] for r in rows]


class DuckDBStore(SpatialStore):
    geometry_type = "GEOMETRY"

    def __init__(self, path: Path | None = None):
        import duckdb

        self.path = path or data_dir() / "derived" / "terramind.duckdb"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path))
        self.con.execute("INSTALL spatial; LOAD spatial;")

    def _exec(self, sql, params=()):
        cur = self.con.execute(sql, list(params))
        try:
            return cur.fetchall()
        except Exception:  # noqa: BLE001 - statements without a result set
            return []

    def _index(self):
        for table in TABLES:
            self.con.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_geom ON {table} USING RTREE (geom)"
            )


class PostGISStore(SpatialStore):
    geometry_type = "geometry(Polygon, 4326)"
    placeholder = "%s"

    def __init__(self, url: str):
        import psycopg

        self.con = psycopg.connect(url, autocommit=True)
        self.con.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    def _exec(self, sql, params=()):
        sql = sql.replace("ST_GeomFromText(%s)", "ST_GeomFromText(%s, 4326)")
        cur = self.con.execute(sql, params)
        return cur.fetchall() if cur.description else []

    def _index(self):
        for table in TABLES:
            self.con.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_geom ON {table} USING GIST (geom)"
            )


def open_store() -> SpatialStore:
    url = os.environ.get("TERRAMIND_DATABASE_URL")
    return PostGISStore(url) if url else DuckDBStore()
