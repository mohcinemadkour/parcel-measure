"""Turn a segmentation into measurements: clean, clip to parcel, area, perimeter,
roof pitch, polygons, confidence and review flags (spec section 7)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

from .config import (CANONICAL_GSD, CLASSES, FT_PER_M, HOLE_FILL_M2, MAX_CANOPY_SHARE,
                     MAX_RELIABLE_GSD, MIN_CONFIDENCE, MIN_REGION_M2, SIMPLIFY_M, SQFT_PER_M2)


def pitch_factor(pitch: float) -> float:
    """Roof surface / footprint for a pitch given as rise per 12 of run."""
    return math.sqrt(1.0 + (pitch / 12.0) ** 2)


def clean_mask(classes: np.ndarray, gsd: float = CANONICAL_GSD) -> np.ndarray:
    """Remove specks under MIN_REGION_M2 and fill small holes in lawn and roof."""
    out = classes.copy()
    px_area = gsd * gsd
    min_px = max(1, int(MIN_REGION_M2 / px_area))
    hole_px = max(1, int(HOLE_FILL_M2 / px_area))
    for cid in (1, 2):
        m = out == cid
        lab, n = ndimage.label(m)
        if n:
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            small = np.isin(lab, np.where(sizes < min_px)[0] + 1)
            if small.any():
                # reassign specks to the most common neighbouring class
                dil = ndimage.binary_dilation(small, iterations=1) & ~small
                if dil.any():
                    out[small] = np.bincount(out[dil], minlength=len(CLASSES)).argmax()
        holes = ndimage.binary_fill_holes(out == cid) & (out != cid)
        lab, n = ndimage.label(holes)
        if n:
            sizes = ndimage.sum(holes, lab, range(1, n + 1))
            out[np.isin(lab, np.where(sizes < hole_px)[0] + 1)] = cid
    return out


def rasterize_parcel(polygon: list[list[float]] | None, shape: tuple[int, int], scale: float) -> np.ndarray:
    """Polygon in ORIGINAL image pixel coords [[x,y],...] -> boolean mask at canonical GSD."""
    if not polygon:
        return np.ones(shape, bool)
    from PIL import Image, ImageDraw
    pts = [(float(x) * scale, float(y) * scale) for x, y in polygon]
    im = Image.new("1", (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon(pts, fill=1, outline=1)
    return np.array(im, bool)


def perimeter_m(mask: np.ndarray, gsd: float) -> float:
    """Boundary length from polygon outlines (more stable than pixel-edge counting)."""
    from rasterio import features
    from shapely.geometry import shape
    total = 0.0
    for geom, val in features.shapes(mask.astype(np.uint8), mask=mask):
        g = shape(geom).simplify(1.0)
        total += g.length
    return total * gsd


def class_polygons(classes: np.ndarray, valid: np.ndarray, gsd: float, inv_scale: float,
                   transform=None) -> dict[str, Any]:
    """GeoJSON FeatureCollection; coordinates in original-image pixels or in the image CRS
    when an affine transform is supplied."""
    from rasterio import features
    from shapely.geometry import mapping, shape
    from shapely.affinity import scale as sscale, affine_transform
    feats = []
    tol_px = SIMPLIFY_M / gsd
    for cid, name in enumerate(CLASSES):
        if name == "other":
            continue
        m = (classes == cid) & valid
        if not m.any():
            continue
        for geom, _ in features.shapes(m.astype(np.uint8), mask=m):
            g = shape(geom).simplify(tol_px, preserve_topology=True)
            if g.is_empty or g.area * gsd * gsd < MIN_REGION_M2:
                continue
            g = sscale(g, xfact=inv_scale, yfact=inv_scale, origin=(0, 0))
            if transform is not None:
                a = transform
                g = affine_transform(g, [a.a, a.b, a.d, a.e, a.c, a.f])
            feats.append({"type": "Feature", "properties": {"class": name,
                          "area_sqft": round(shape(geom).area * gsd * gsd * SQFT_PER_M2, 1)},
                          "geometry": mapping(g)})
    return {"type": "FeatureCollection", "features": feats}


def measure(classes: np.ndarray, probs: np.ndarray, scale: float, input_gsd: float,
            parcel: list[list[float]] | None = None, pitch: float | None = None,
            transform=None, with_polygons: bool = True) -> dict[str, Any]:
    gsd = CANONICAL_GSD
    cls = clean_mask(classes, gsd)
    valid = rasterize_parcel(parcel, cls.shape, scale)
    px_m2 = gsd * gsd
    pitch = 6.0 if pitch is None else float(pitch)
    pf = pitch_factor(pitch)

    areas, conf = {}, {}
    for cid, name in enumerate(CLASSES):
        m = (cls == cid) & valid
        a_m2 = float(m.sum()) * px_m2
        areas[name] = {"m2": round(a_m2, 2), "sqft": round(a_m2 * SQFT_PER_M2, 1)}
        conf[name] = round(float(probs[cid][m].mean()), 3) if m.any() else None

    roof_fp = areas["roof"]["m2"]
    lawn_mask = (cls == 1) & valid
    roof_mask = (cls == 2) & valid
    measurements = {
        "lawn_sqft": areas["lawn"]["sqft"],
        "lawn_perimeter_ft": round(perimeter_m(lawn_mask, gsd) * FT_PER_M, 1) if lawn_mask.any() else 0.0,
        "roof_footprint_sqft": areas["roof"]["sqft"],
        "roof_pitch": pitch,
        "roof_pitch_factor": round(pf, 4),
        "roof_surface_sqft": round(roof_fp * pf * SQFT_PER_M2, 1),
        "roof_squares": round(roof_fp * pf * SQFT_PER_M2 / 100.0, 2),
        "roof_perimeter_ft": round(perimeter_m(roof_mask, gsd) * FT_PER_M, 1) if roof_mask.any() else 0.0,
        "hardscape_sqft": areas["hardscape"]["sqft"],
        "tree_canopy_sqft": areas["tree"]["sqft"],
        "parcel_sqft": round(float(valid.sum()) * px_m2 * SQFT_PER_M2, 1),
    }

    reasons = []
    for k in ("lawn", "roof"):
        if conf[k] is not None and conf[k] < MIN_CONFIDENCE:
            reasons.append(f"{k} confidence {conf[k]:.2f} below {MIN_CONFIDENCE}")
    pervious = areas["lawn"]["m2"] + areas["tree"]["m2"]
    if pervious > 0 and areas["tree"]["m2"] / pervious > MAX_CANOPY_SHARE:
        reasons.append("tree canopy hides more than 35% of pervious area; lawn may be under-measured")
    if input_gsd > MAX_RELIABLE_GSD:
        reasons.append(f"image resolution {input_gsd:.2f} m/px is coarser than {MAX_RELIABLE_GSD} m/px")
    valid_conf = [c for c in conf.values() if c is not None]

    out = {
        "measurements": measurements,
        "areas": areas,
        "confidence": {**conf, "overall": round(float(np.mean(valid_conf)), 3) if valid_conf else None},
        "review": {"required": bool(reasons), "reasons": reasons},
        "processing": {"input_gsd_m": input_gsd, "canonical_gsd_m": gsd, "resample_scale": round(scale, 4),
                       "parcel_applied": bool(parcel)},
    }
    if with_polygons:
        out["geojson"] = class_polygons(cls, valid, gsd, 1.0 / scale, transform)
    out["_clean_classes"] = cls
    out["_valid"] = valid
    return out
