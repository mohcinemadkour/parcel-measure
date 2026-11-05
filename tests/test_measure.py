import math

import numpy as np
import pytest

from pam.config import CANONICAL_GSD, SQFT_PER_M2
from pam.inference import _windows, resample_to_canonical
from pam.measure import clean_mask, measure, pitch_factor, rasterize_parcel
from pam.quote import DEFAULT_PRICE_TABLE, PriceLine, PriceTable, build_quote


def onehot(cls):
    p = np.zeros((5, *cls.shape), np.float32)
    for c in range(5):
        p[c][cls == c] = 0.95
    return p


def test_pitch_factor():
    assert pitch_factor(0) == 1.0
    assert pitch_factor(6) == pytest.approx(math.sqrt(1.25), rel=1e-9)  # 1.1180
    assert pitch_factor(12) == pytest.approx(math.sqrt(2), rel=1e-9)


def test_area_is_pixels_times_gsd_squared():
    cls = np.zeros((200, 200), np.uint8)
    cls[0:100, 0:100] = 1          # lawn: 10,000 px
    cls[100:140, 100:200] = 2      # roof: 4,000 px
    r = measure(cls, onehot(cls), 1.0, CANONICAL_GSD, pitch=6, with_polygons=False)
    px = CANONICAL_GSD ** 2
    assert r["areas"]["lawn"]["m2"] == pytest.approx(10_000 * px, abs=0.01)
    assert r["measurements"]["lawn_sqft"] == pytest.approx(10_000 * px * SQFT_PER_M2, abs=0.1)
    assert r["measurements"]["roof_surface_sqft"] == pytest.approx(4_000 * px * SQFT_PER_M2 * math.sqrt(1.25), abs=0.1)
    assert r["measurements"]["roof_squares"] == pytest.approx(r["measurements"]["roof_surface_sqft"] / 100, abs=0.01)


def test_perimeter_of_square_lawn():
    cls = np.zeros((300, 300), np.uint8)
    cls[50:250, 50:250] = 1        # 30 m x 30 m square -> 120 m perimeter
    r = measure(cls, onehot(cls), 1.0, CANONICAL_GSD, with_polygons=False)
    assert r["measurements"]["lawn_perimeter_ft"] == pytest.approx(120 * 3.28084, rel=0.02)


def test_parcel_clip_in_original_pixels():
    cls = np.ones((100, 100), np.uint8)
    # original image was at 0.30 m -> scale 2: a 50x50 original-pixel square = 100x100 canonical px
    mask = rasterize_parcel([[0, 0], [25, 0], [25, 25], [0, 25]], (100, 100), 2.0)
    assert 2400 <= mask.sum() <= 2700   # 50x50 canonical px incl. boundary
    r = measure(cls, onehot(cls), 2.0, 0.30, parcel=[[0, 0], [25, 0], [25, 25], [0, 25]], with_polygons=False)
    assert r["processing"]["parcel_applied"]
    assert r["areas"]["lawn"]["m2"] == pytest.approx(mask.sum() * CANONICAL_GSD ** 2, abs=0.01)


def test_clean_mask_removes_specks_and_fills_holes():
    cls = np.full((100, 100), 3, np.uint8)
    cls[10:12, 10:12] = 2                # 4 px roof speck (0.09 m2) -> removed
    cls[40:80, 40:80] = 1                # lawn block
    cls[59:61, 59:61] = 4                # tiny hole -> filled
    out = clean_mask(cls)
    assert (out[10:12, 10:12] == 3).all()
    assert (out[59:61, 59:61] == 1).all()


def test_review_flags():
    cls = np.zeros((100, 100), np.uint8)
    cls[:, :30] = 1
    cls[:, 30:] = 4                      # canopy dominates pervious area
    r = measure(cls, onehot(cls), 1.0, 0.45, with_polygons=False)
    assert r["review"]["required"]
    assert any("canopy" in s for s in r["review"]["reasons"])
    assert any("coarser" in s for s in r["review"]["reasons"])


def test_polygons_geojson():
    cls = np.zeros((120, 120), np.uint8)
    cls[20:100, 20:100] = 2
    r = measure(cls, onehot(cls), 0.5, 0.30)
    feats = r["geojson"]["features"]
    assert len(feats) == 1 and feats[0]["properties"]["class"] == "roof"
    xs = [p[0] for p in feats[0]["geometry"]["coordinates"][0]]
    assert max(xs) == pytest.approx(200, abs=2)   # canonical 100 px -> original px at scale 0.5


def test_quote_minimums_and_rates():
    q = build_quote({"lawn_sqft": 5000, "lawn_perimeter_ft": 300, "roof_perimeter_ft": 200}, DEFAULT_PRICE_TABLE)
    items = {i["service"]: i for i in q["items"]}
    assert items["Mowing (per visit)"]["price"] == 35.0          # 5 x 4.50 = 22.50 -> minimum 35
    assert items["Core aeration"]["price"] == 75.0
    assert items["Fertilization (per application)"]["price"] == 45.0
    assert items["Edging"]["price"] == 45.0                      # 300 x 0.15
    big = build_quote({"lawn_sqft": 20000}, PriceTable(lines=[PriceLine(service="m", basis="lawn_sqft", rate=4.5, per=1000,
                                                                          minimum=35, surcharge_pct=10)], tax_pct=8.25))
    assert big["subtotal"] == pytest.approx(99.0)
    assert big["total"] == pytest.approx(99.0 * 1.0825, abs=0.01)


def test_resample_and_windows():
    img = np.zeros((300, 400, 3), np.uint8)
    out, s = resample_to_canonical(img, 0.30)
    assert s == 2.0 and out.shape[:2] == (600, 800)
    assert resample_to_canonical(img, 0.15)[1] == 1.0
    w = _windows(1000, 512, 448)
    assert w[0] == 0 and w[-1] == 1000 - 512 and all(b - a <= 448 for a, b in zip(w, w[1:]))
