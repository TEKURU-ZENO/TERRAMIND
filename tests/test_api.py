import pytest
from fastapi.testclient import TestClient

from terramind.api.app import app
from terramind.explorer.build import out_dir

client = TestClient(app)
needs_index = pytest.mark.skipif(
    not (out_dir() / "index.json").exists(), reason="explorer index not built"
)


def test_events_and_datasets():
    assert client.get("/api/health").json() == {"status": "ok"}
    assert any(e["id"] == "chennai-2015" for e in client.get("/api/events").json())
    assert any(d["id"] == "sen1floods11" for d in client.get("/api/datasets").json())
    assert client.get("/api/events/nope/scenes").status_code == 404


def test_layer_route_rejects_unknown_names():
    assert client.get("/api/chips/Bolivia_1/secrets.png").status_code == 404
    assert client.get("/api/chips/..%2F..%2Fx/sar.png").status_code == 404


@needs_index
def test_bbox_filter_returns_only_intersecting_chips():
    everything = client.get("/api/chips").json()["features"]
    bolivia = client.get("/api/chips", params={"bbox": "-70,-20,-60,-10"}).json()["features"]
    assert 0 < len(bolivia) < len(everything)
    assert {f["properties"]["event"] for f in bolivia} == {"Bolivia"}
    assert client.get("/api/chips", params={"bbox": "bad"}).status_code == 422


@needs_index
def test_chip_layer_is_served_as_png():
    chip_id = client.get("/api/chips", params={"split": "bolivia"}).json()["features"][0]["id"]
    res = client.get(f"/api/chips/{chip_id}/sar.png")
    assert res.status_code == 200 and res.headers["content-type"] == "image/png"


def test_sim_episode_route():
    from terramind.sim import scenarios

    if not scenarios.path_for("s1").exists():
        pytest.skip("scenarios not built")
    chip = scenarios.load("s1").ids[0]
    r = client.get("/api/sim/episode", params={"chip": chip, "policy": "greedy", "depot": 3})
    body = r.json()
    assert r.status_code == 200 and body["simulated"] is True
    assert len(body["positions"]) == body["horizon"] + 1 == len(body["served"])
    assert body["served"] == sorted(body["served"]) and body["served"][-1] <= 1 + 1e-5
    assert client.get("/api/sim/episode", params={"chip": chip, "policy": "nope"}).status_code == 404
    assert client.get("/api/sim/episode", params={"chip": "../x"}).status_code == 404


def test_situation_report_route():
    from terramind.evidence.store import corpus_path
    from terramind.sim import scenarios

    if not (scenarios.path_for("s1").exists() and corpus_path().exists()):
        pytest.skip("scenarios or evidence not built")
    data = scenarios.load("s1")
    chip = str(data.ids[data.select("bolivia")[0]])
    body = client.get(f"/api/report/{chip}").json()
    assert body["event"] == "Bolivia" and len(body["sections"]) == 3
    statements = [s for sec in body["sections"] for s in sec["statements"]]
    assert all(s["evidence"] for s in statements)
    assert all(s["simulated"] for s in body["sections"][2]["statements"])
    assert client.get("/api/report/Nowhere_1").status_code == 404
