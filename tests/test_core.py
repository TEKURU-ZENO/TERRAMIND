from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from terramind import registry
from terramind.ingestion import sen1floods11, stac
from terramind.training.metrics import Confusion


def test_registry_loads_and_every_stac_collection_is_registered():
    datasets = registry.load_datasets()
    assert "sen1floods11" in datasets
    assert {"sentinel-1-grd", "sentinel-2-l2a"} <= {d.stac_collection for d in datasets.values()}


def test_unregistered_dataset_is_rejected():
    with pytest.raises(KeyError):
        registry.require_dataset("not-a-dataset")


def test_events_have_valid_bboxes():
    events = registry.load_events()
    assert "chennai-2015" in events
    with pytest.raises(ValueError):
        registry.FloodEvent(
            id="x", name="x", region="x", bbox=(10, 5, 0, 6), start="2020-01-01", end="2020-01-02"
        )


def test_event_of_handles_hyphenated_names():
    assert sen1floods11.event_of("Sri-Lanka_101973_S1Hand.tif") == "Sri-Lanka"


def test_confusion_ignores_nodata_and_computes_iou():
    pred = np.array([[1, 1, 0, 0, 1]])
    target = np.array([[1, 0, 1, 0, 255]])
    c = Confusion()
    c.update(pred, target)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    assert c.summary()["iou"] == pytest.approx(1 / 3)
    assert Confusion().summary()["iou"] is None


def test_item_to_record():
    item = SimpleNamespace(
        id="S2A_x",
        collection_id="sentinel-2-l2a",
        datetime=datetime(2015, 12, 1, tzinfo=UTC),
        bbox=[80.0, 12.9, 80.3, 13.2],
        properties={"platform": "sentinel-2a", "eo:cloud_cover": 12.5},
        assets={"red": SimpleNamespace(href="https://example.com/red.tif")},
    )
    rec = stac.item_to_record(item, "earth-search")
    assert rec.cloud_cover == 12.5
    assert rec.assets == {"red": "https://example.com/red.tif"}
    assert rec.datetime.startswith("2015-12-01")


def test_unet_preserves_spatial_size():
    torch = pytest.importorskip("torch")
    from terramind.models.unet import UNet

    out = UNet(base=8)(torch.zeros(1, 2, 64, 64))
    assert out.shape == (1, 2, 64, 64)


@pytest.mark.network
def test_live_stac_search_returns_sentinel2_scenes():
    event = registry.load_events()["kerala-2018"]
    records = stac.search("sentinel-2-l2a", event.bbox, event.start, event.end, limit=5)
    assert records and all(r.collection == "sentinel-2-l2a" for r in records)
