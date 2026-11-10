"""Spatial-temporal hybrid retrieval over the evidence corpus.

Score per evidence record d for query q:

    S(d, q) = (w_s S_semantic + w_g S_spatial + w_t S_temporal + w_p S_provenance) / sum of active w

A component is active only when the query constrains it (a query with no place
has no spatial term). Candidates come from vector search, a spatial index and
one hop of graph expansion. Everything runs in-process: Qdrant in local mode,
Shapely for spatial predicates, NetworkX for the graph. PostGIS and Neo4j
replace the last two when the corpus outgrows memory.

The default weights are hand-set, not calibrated. `terramind rag-benchmark`
reports how they do.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import date

import networkx as nx
import numpy as np
from shapely.geometry import box
from shapely.strtree import STRtree

from terramind.evidence.store import Evidence
from terramind.rag.query import ParsedQuery

DEFAULT_WEIGHTS = {"semantic": 0.45, "spatial": 0.30, "temporal": 0.15, "provenance": 0.10}
KM_PER_DEGREE = 111.0


class HashingEmbedder:
    """Dependency-free bag-of-words embedder. For tests and offline use only."""

    name = "hashing-256"
    dim = 256

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, text in enumerate(texts):
            for token in "".join(c.lower() if c.isalnum() else " " for c in text).split():
                h = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")
                out[i, h % self.dim] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class SentenceEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.name = model
        # CPU on purpose: the GPU is for training, and the corpus is small.
        self._model = SentenceTransformer(model, device="cpu")
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
        ).astype(np.float32)


@dataclass
class Hit:
    evidence: Evidence
    score: float
    components: dict[str, float]
    via: list[str] = field(default_factory=list)  # which retrievers proposed it


def _days(iso: str) -> int:
    return date.fromisoformat(iso[:10]).toordinal()


def spatial_score(bbox: list[float] | None, query_boxes: list[list[float]]) -> float:
    """1 if the record intersects any query box, decaying with distance (500 km scale)."""
    if not bbox:
        return 0.0
    geom = box(*bbox)
    best = 0.0
    for q in query_boxes:
        qbox = box(*q)
        if geom.intersects(qbox):
            return 1.0
        best = max(best, math.exp(-geom.distance(qbox) * KM_PER_DEGREE / 500.0))
    return best


def temporal_score(start: str | None, end: str | None, q_start: str, q_end: str) -> float:
    """1 if the record's dates overlap the query window, decaying with the gap (90 day scale)."""
    if not start:
        return 0.0
    a0, a1 = _days(start), _days(end or start)
    b0, b1 = _days(q_start), _days(q_end)
    if a0 <= b1 and b0 <= a1:
        return 1.0
    return math.exp(-min(abs(a0 - b1), abs(b0 - a1)) / 90.0)


class Retriever:
    def __init__(self, corpus: list[Evidence], embedder=None, weights: dict | None = None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.corpus = corpus
        self.by_id = {e.id: e for e in corpus}
        self.weights = weights or DEFAULT_WEIGHTS
        self.embedder = embedder or SentenceEmbedder()

        vectors = self.embedder.encode([f"{e.title}. {e.text}" for e in corpus])
        self._client = QdrantClient(":memory:")
        self._client.create_collection(
            "evidence", vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE)
        )
        self._client.upload_points(
            "evidence",
            [
                PointStruct(id=i, vector=v.tolist(), payload={"eid": e.id, "kind": e.kind})
                for i, (e, v) in enumerate(zip(corpus, vectors))
            ],
        )

        self._spatial_ids = [e.id for e in corpus if e.bbox]
        self._tree = STRtree([box(*self.by_id[i].bbox) for i in self._spatial_ids])

        self.graph = nx.Graph()
        for e in corpus:
            self.graph.add_node(e.id, kind=e.kind)
        for e in corpus:
            for target in e.links:
                if target in self.by_id:
                    self.graph.add_edge(e.id, target)

    # -- candidate generators -------------------------------------------------

    def _vector(self, text: str, k: int, kinds: list[str] | None) -> dict[str, float]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        flt = (
            Filter(must=[FieldCondition(key="kind", match=MatchAny(any=kinds))]) if kinds else None
        )
        vector = self.embedder.encode([text])[0].tolist()
        result = self._client.query_points("evidence", query=vector, limit=k, query_filter=flt)
        return {p.payload["eid"]: max(0.0, float(p.score)) for p in result.points}

    def _spatial(self, query_boxes: list[list[float]]) -> set[str]:
        found = set()
        for q in query_boxes:
            for i in self._tree.query(box(*q), predicate="intersects"):
                found.add(self._spatial_ids[int(i)])
        return found

    def _graph(self, seeds: list[str], limit: int) -> set[str]:
        """One hop from the seeds, skipping the bulky chip and scene neighbours."""
        found = set()
        for seed in seeds:
            if seed not in self.graph:
                continue
            for neighbour in self.graph.neighbors(seed):
                if self.graph.nodes[neighbour]["kind"] not in ("chip", "scene"):
                    found.add(neighbour)
                if len(found) >= limit:
                    return found
        return found

    # -- public ---------------------------------------------------------------

    def retrieve(
        self,
        query: ParsedQuery,
        k: int = 10,
        mode: str = "hybrid",
        kinds: list[str] | None = None,
        use_time: bool = True,
    ) -> list[Hit]:
        """mode: 'vector' (semantic only), 'spatial' (spatial/temporal filter only),
        'hybrid-nograph', or 'hybrid'."""
        boxes = query.bboxes
        timed = use_time and bool(query.start)
        via: dict[str, list[str]] = {}
        semantic: dict[str, float] = {}

        if mode != "spatial":
            semantic = self._vector(query.text, max(4 * k, 40), kinds)
            for eid in semantic:
                via.setdefault(eid, []).append("vector")
        if mode != "vector" and boxes:
            for eid in self._spatial(boxes):
                if not kinds or self.by_id[eid].kind in kinds:
                    via.setdefault(eid, []).append("spatial")
        if mode == "hybrid":
            seeds = [r.evidence_id for r in query.regions] + sorted(
                semantic, key=semantic.get, reverse=True
            )[:5]
            for eid in self._graph(seeds, limit=40):
                if not kinds or self.by_id[eid].kind in kinds:
                    via.setdefault(eid, []).append("graph")

        # Semantic similarity for candidates that vector search did not return.
        missing = [eid for eid in via if eid not in semantic]
        if missing and mode != "spatial":
            q = self.embedder.encode([query.text])[0]
            docs = self.embedder.encode(
                [f"{self.by_id[e].title}. {self.by_id[e].text}" for e in missing]
            )
            for eid, sim in zip(missing, docs @ q):
                semantic[eid] = max(0.0, float(sim))

        hits = []
        for eid, sources in via.items():
            e = self.by_id[eid]
            comp = {"provenance": e.provenance}
            if mode != "spatial":
                comp["semantic"] = semantic.get(eid, 0.0)
            if mode != "vector" and boxes:
                comp["spatial"] = spatial_score(e.bbox, boxes)
            if mode != "vector" and timed:
                comp["temporal"] = temporal_score(e.start, e.end, query.start, query.end)
            # A pure structured filter must satisfy every stated constraint.
            if mode == "spatial" and (
                comp.get("spatial", 1.0) < 1.0 or comp.get("temporal", 1.0) < 1.0
            ):
                continue
            total = sum(self.weights[c] for c in comp)
            score = sum(self.weights[c] * v for c, v in comp.items()) / total
            hits.append(Hit(e, score, comp, sources))
        hits.sort(key=lambda h: (-h.score, h.evidence.id))
        return hits[:k]
