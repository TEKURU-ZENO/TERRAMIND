import numpy as np
import pytest
import torch


def test_prithvi_seg_shapes_and_band_selection():
    from terramind.models.prithvi import S2_TO_HLS, BandSelect, PrithviSeg

    model = PrithviSeg(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 13, 100, 70))  # not a multiple of 16: padded, then cropped
    assert out.shape == (1, 2, 100, 70)
    picked = []
    BandSelect(lambda x: picked.append(x) or x)(torch.arange(13.0).view(1, 13, 1, 1))
    assert picked[0].flatten().tolist() == [float(i) for i in S2_TO_HLS]


def test_frozen_prithvi_trains_only_the_decoder():
    from terramind.models.prithvi import PrithviSeg

    model = PrithviSeg(pretrained=False, freeze_encoder=True)
    groups = model.param_groups(1e-3)
    assert len(groups) == 1 and all(not p.requires_grad for p in model.encoder.parameters())


def test_modality_dropout_hides_one_sensor_at_a_time(monkeypatch):
    from terramind.training import train_baseline as tb

    img = np.ones((15, 8, 8), np.float32)
    monkeypatch.setattr(tb, "cached_chip", lambda *a: (img.copy(), np.zeros((8, 8), np.int64)))
    monkeypatch.setattr(tb.s1f11, "event_of", lambda name: "X")

    class P:
        name = "X_1_S1Hand.tif"

    ds = tb.FloodChips([(P(), P())], crop=None, augment=False, modality="s1s2", dropout=0.3)
    seen = set()
    for seed in range(40):
        tb.random.seed(seed)
        x, _, _ = ds[0]
        s1, s2 = bool(x[:2].sum()), bool(x[2:].sum())
        assert s1 or s2  # never both hidden
        seen.add((s1, s2))
    assert seen == {(True, False), (False, True), (True, True)}
    x, _, _ = tb.FloodChips([(P(), P())], None, False, "s1s2", zero="s2")[0]
    assert x[2:].sum() == 0 and x[:2].sum() > 0


def test_dispatch_global_state_for_mappo():
    from terramind.sim.dispatch import DispatchEnv

    truth = np.zeros((2, 6, 6), np.float32)
    truth[:, 2, 2] = 1.0
    env = DispatchEnv(truth, truth, np.zeros_like(truth))
    env.reset([0, 1], [0, 3])
    g = env.global_state()
    assert g.shape == (2, 3, env.global_channels, 6, 6)
    assert (g[:, :, 5].sum((2, 3)) == 1).all()  # each team sees its own position once
    assert g[0, 0, 9, 2, 2] == 1.0  # truth plane for the critic


def test_forecast_features_and_scores():
    from terramind.forecast import discharge as d

    n = 400
    dates = np.arange(np.datetime64("2010-01-01"), np.datetime64("2010-01-01") + n)
    q = 10 + 5 * np.sin(np.arange(n) / 20.0)
    z = (np.log1p(q) - np.log1p(q).mean()) / np.log1p(q).std()
    s = d.SiteSeries("x", "x", "2010-06-01", dates, q, np.ones(n), z, np.log1p(q).mean(),
                     np.log1p(q).std(), float(np.percentile(q, 95)))  # fmt: skip
    X, seq, y, idx = d.features(s, lead=3)
    assert X.shape[0] == seq.shape[0] == len(y) == len(idx) == n - d.WINDOW - 3
    assert np.allclose(y, z[idx + 3] - z[idx])
    assert np.allclose(d.to_q(z, s), q)
    assert d._nse(q, q) == pytest.approx(1.0)
    assert d._scores({"tp": 2, "fp": 1, "fn": 1})["csi"] == pytest.approx(0.5)


def test_gnn_batching_and_models_run():
    from terramind.roads.gnn import MLP, SAGE, _batch

    g = {
        "x": np.ones((4, 3), np.float32),
        "y": np.array([0, 1, 1, 0]),
        "edges": np.array([[0, 1], [1, 2]]),
    }
    x, _, s, dst = _batch([g, g], "cpu")
    assert x.shape == (8, 3) and len(s) == 8  # 2 edges x 2 directions x 2 graphs
    assert int(s.max()) == 6  # second graph's edges are offset by 4
    for model in (SAGE(3), MLP(3)):
        assert model(x, s, dst).shape == (8,)


def test_spatial_store_bbox_query(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    from terramind.storage import spatial

    def rows():
        yield "chips", ("a", "E", "test", 0.5, 0.1, "{}"), spatial._envelope((0, 0, 1, 1))
        yield "chips", ("b", "E", "train", 0.5, 0.1, "{}"), spatial._envelope((5, 5, 6, 6))
        yield "evidence", ("x", "doc", "t", None, None, 1.0, "{}"), None

    monkeypatch.setattr(spatial, "_rows", rows)
    try:
        store = spatial.DuckDBStore(tmp_path / "t.duckdb")
    except Exception as err:  # noqa: BLE001 - the spatial extension needs a one-off download
        pytest.skip(f"duckdb spatial unavailable: {err}")
    assert store.load() == {"chips": 2, "scenes": 0, "evidence": 1}
    assert store.in_bbox("chips", (0.5, 0.5, 2, 2)) == ["a"]
    assert store.in_bbox("chips", (-10, -10, 10, 10), split="train") == ["b"]
    with pytest.raises(ValueError):
        store.in_bbox("chips", (0, 0, 1, 1), iou=1)


def test_llm_summary_is_dropped_when_it_changes_a_number():
    from terramind.integration.situation import narrate

    report = {
        "recommendation": "Usable in simulation.",
        "flags": [],
        "sections": [
            {
                "title": "t",
                "statements": [{"text": "Teams reach 70.2% of the area.", "simulated": True}],
            }
        ],
    }

    class Rounding:
        def generate(self, prompt):
            return "Teams reach about 70% of the area in simulation."

    class Faithful:
        def generate(self, prompt):
            return "In simulation, teams reach 70.2% of the flooded area."

    assert narrate(report, Rounding())["accepted"] is False
    assert narrate(report, Faithful())["accepted"] is True
