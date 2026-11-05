import importlib
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pam.inference import SegResult, resample_to_canonical


class FakeSegmenter:
    """Deterministic stand-in so API tests do not depend on the trained model."""
    version = "test"
    card = {"version": "test"}

    def predict(self, img, gsd):
        img_c, scale = resample_to_canonical(img, gsd)
        h, w = img_c.shape[:2]
        cls = np.full((h, w), 3, np.uint8)
        cls[: h // 2, : w // 2] = 1
        cls[h // 2:, w // 2:] = 2
        p = np.zeros((5, h, w), np.float32)
        for c in range(5):
            p[c][cls == c] = 0.9
        return SegResult(cls, p, img_c, scale)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PAM_API_KEYS", "k1,k2")
    monkeypatch.setenv("PAM_DATA_DIR", str(tmp_path))
    import pam.api as api
    api = importlib.reload(api)
    api._state.update({"segmenter": FakeSegmenter(), "store": None, "error": None})
    return TestClient(api.app)


def png(w=200, h=200):
    buf = io.BytesIO()
    Image.fromarray(np.random.RandomState(0).randint(0, 255, (h, w, 3), np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def test_health_open(client):
    assert client.get("/v1/health").json()["status"] == "ok"
    assert client.get("/v1/ready").json()["ready"] is True


def test_auth_required(client):
    r = client.post("/v1/measure", files={"file": ("a.png", png())}, data={"gsd": "0.15"})
    assert r.status_code == 401


def test_measure_full_flow(client):
    H = {"X-API-Key": "k1"}
    r = client.post("/v1/measure", headers=H, files={"file": ("a.png", png())},
                    data={"gsd": "0.15", "pitch": "6", "parcel": json.dumps([[0, 0], [100, 0], [100, 100], [0, 100]])})
    assert r.status_code == 200, r.text
    j = r.json()
    for k in ("id", "measurements", "confidence", "review", "geojson", "quote", "overlay_png_base64", "model_version"):
        assert k in j
    # parcel = top-left 100x100 px (lawn quadrant) at 0.15 m -> ~225 m2 lawn, no roof
    assert j["areas"]["lawn"]["m2"] == pytest.approx(225, rel=0.03)
    assert j["areas"]["roof"]["m2"] < 1.0   # boundary pixel row only
    rid = j["id"]
    assert client.get(f"/v1/results/{rid}", headers=H).json()["id"] == rid
    assert client.get(f"/v1/results/{rid}/overlay.png", headers=H).headers["content-type"] == "image/png"
    assert client.get(f"/v1/results/{rid}/geojson", headers=H).json()["type"] == "FeatureCollection"
    assert "lawn_sqft" in client.get(f"/v1/results/{rid}/csv", headers=H).text
    fb = client.post(f"/v1/results/{rid}/feedback", headers=H, json={"corrected": {"lawn_sqft": 3000}})
    assert fb.json()["stored"] and fb.json()["quote"]["total"] > 0


def test_bad_inputs(client):
    H = {"X-API-Key": "k2"}
    assert client.post("/v1/measure", headers=H, files={"file": ("a.png", png())}).status_code == 422      # no gsd
    assert client.post("/v1/measure", headers=H, files={"file": ("a.png", png())},
                       data={"gsd": "2.0"}).status_code == 422                                              # out of range
    assert client.post("/v1/measure", headers=H, files={"file": ("a.png", b"not an image")},
                       data={"gsd": "0.15"}).status_code == 422
    assert client.get("/v1/results/doesnotexist", headers=H).status_code == 404


def test_quote_and_batch(client):
    H = {"X-API-Key": "k1"}
    q = client.post("/v1/quote", headers=H, json={"measurements": {"lawn_sqft": 10000}}).json()
    assert q["items"][0]["price"] == 45.0
    r = client.post("/v1/measure/batch", headers=H, files=[("files", ("a.png", png())), ("files", ("b.png", png(120, 90)))],
                    data={"gsd": "0.2"})
    assert len(r.json()["results"]) == 2 and all("measurements" in x for x in r.json()["results"])


def test_address_without_provider_key(client, monkeypatch):
    monkeypatch.delenv("PAM_GOOGLE_MAPS_KEY", raising=False)
    r = client.post("/v1/measure/address", headers={"X-API-Key": "k1"}, json={"address": "1 Main St, Dallas, TX"})
    assert r.status_code == 503
