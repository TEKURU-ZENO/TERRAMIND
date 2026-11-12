"""Evidence graph in Neo4j (docker-compose service `neo4j`).

Every evidence record becomes an (:Evidence {id, kind, title, provenance}) node;
each link becomes a [:LINKS] relationship. `neighbours` is the one-hop
expansion the retriever does in memory with NetworkX, as a Cypher query.

    TERRAMIND_NEO4J_URL=bolt://127.0.0.1:7687
    TERRAMIND_NEO4J_PASSWORD=<password from .env>
"""

from __future__ import annotations

import os

BATCH = 500


class EvidenceGraph:
    def __init__(self, url: str | None = None, user: str = "neo4j", password: str | None = None):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(
            url or os.environ["TERRAMIND_NEO4J_URL"],
            auth=(user, password or os.environ["TERRAMIND_NEO4J_PASSWORD"]),
        )

    def close(self) -> None:
        self.driver.close()

    def load(self) -> dict[str, int]:
        from terramind.evidence import store

        records = store.load_corpus()
        nodes = [
            {"id": e.id, "kind": e.kind, "title": e.title, "provenance": e.provenance}
            for e in records
        ]
        links = [{"a": e.id, "b": t} for e in records for t in e.links]
        with self.driver.session() as s:
            s.run("MATCH (n:Evidence) DETACH DELETE n")
            s.run(
                "CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (n:Evidence) REQUIRE n.id IS UNIQUE"
            )
            for i in range(0, len(nodes), BATCH):
                s.run(
                    "UNWIND $rows AS r MERGE (n:Evidence {id: r.id}) SET n += r",
                    rows=nodes[i : i + BATCH],
                )
            for i in range(0, len(links), BATCH):
                s.run(
                    "UNWIND $rows AS r MATCH (a:Evidence {id: r.a}), (b:Evidence {id: r.b}) "
                    "MERGE (a)-[:LINKS]->(b)",
                    rows=links[i : i + BATCH],
                )
        return {"nodes": len(nodes), "links": len(links)}

    def neighbours(self, ids: list[str], limit: int = 50) -> list[str]:
        with self.driver.session() as s:
            res = s.run(
                "MATCH (a:Evidence)-[:LINKS]-(b:Evidence) WHERE a.id IN $ids AND NOT b.id IN $ids "
                "RETURN DISTINCT b.id AS id LIMIT $limit",
                ids=ids,
                limit=limit,
            )
            return [r["id"] for r in res]
