"""Acceptance evaluation (spec section 11).

Property proxy: the held-out tiles are cut into a grid of 36 m x 36 m windows
(1,296 m2 = 0.32 acre, about a US suburban lot). Ground truth areas are counted on
the 0.10 m master labels; predicted areas are counted on the model output at the
canonical 0.15 m after the same post-processing the API applies. Accuracy per window
and class = 1 - |pred - true| / true, for classes with true area >= 20 m2.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .config import CANONICAL_GSD, CLASSES, IMAGENET_MEAN, IMAGENET_STD, N_CLASSES, OVERLAP, TILE
from .inference import SegResult, _blend_weights, _windows, resample_to_canonical, tta_views
from .measure import clean_mask

MASTER_GSD = 0.10
WINDOW_M = 36.0
MIN_TRUE_M2 = 20.0
EVAL_CLASSES = ["lawn", "roof", "hardscape", "tree"]


class TorchPredictor:
    """Sliding-window predictor around a torch model (used during training)."""

    def __init__(self, model, tta: str = "none"):
        self.model, self.tta = model, tta

    def predict(self, img: np.ndarray, gsd: float) -> SegResult:
        import torch
        img_c, scale = resample_to_canonical(img, gsd)
        h, w = img_c.shape[:2]
        ph, pw = max(h, TILE), max(w, TILE)
        pad = np.zeros((ph, pw, 3), np.uint8)
        pad[:h, :w] = img_c
        x = ((pad.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(x, np.float32))
        acc = np.zeros((N_CLASSES, ph, pw), np.float32)
        wsum = np.zeros((ph, pw), np.float32)
        wt = _blend_weights(TILE)
        step = TILE - OVERLAP
        coords = [(y, x0) for y in _windows(ph, TILE, step) for x0 in _windows(pw, TILE, step)]
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            for i in range(0, len(coords), 4):
                chunk = coords[i:i + 4]
                tiles = [x[:, y:y + TILE, x0:x0 + TILE].numpy() for y, x0 in chunk]
                views = tta_views(self.tta)
                batch = torch.from_numpy(np.ascontiguousarray(np.stack([f(t) for t in tiles for f, _ in views])))
                pr = self.model(batch.contiguous(memory_format=torch.channels_last)).float().softmax(1).numpy()
                pr = pr.reshape(len(tiles), len(views), *pr.shape[1:])
                p = [np.mean([inv(v) for v, (_, inv) in zip(pt, views)], axis=0) for pt in pr]
                for (y, x0), pp in zip(chunk, p):
                    acc[:, y:y + TILE, x0:x0 + TILE] += pp * wt
                    wsum[y:y + TILE, x0:x0 + TILE] += wt
        probs = (acc / wsum)[:, :h, :w]
        return SegResult(probs.argmax(0).astype(np.uint8), probs, img_c, scale)


def _resize_img(img: np.ndarray, src_gsd: float, dst_gsd: float) -> np.ndarray:
    if abs(src_gsd - dst_gsd) < 1e-6:
        return img
    h, w = img.shape[:2]
    f = src_gsd / dst_gsd
    return np.array(Image.fromarray(img).resize((round(w * f), round(h * f)),
                                                Image.BOX if f < 1 else Image.BILINEAR))


def _miou(conf: np.ndarray) -> dict:
    ious = {}
    for c, name in enumerate(CLASSES):
        tp = conf[c, c]
        denom = conf[c, :].sum() + conf[:, c].sum() - tp
        ious[name] = round(float(tp / denom), 4) if denom else None
    vals = [v for k, v in ious.items() if v is not None and k != "other"]
    return {"per_class": ious, "mean": round(float(np.mean(vals)), 4)}


def evaluate_tiles(predictor, tiles, ids, input_gsd: float = CANONICAL_GSD,
                   window_m: float = WINDOW_M, return_rows: bool = False) -> dict:
    rows = []
    conf = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    for (img10, lbl10), tid in zip(tiles, ids):
        img_in = _resize_img(img10, MASTER_GSD, input_gsd)       # image as captured
        res = predictor.predict(img_in, input_gsd)                # resampled to 0.15 inside
        pred = clean_mask(res.classes, CANONICAL_GSD)
        # confusion at canonical grid
        gt_c = np.array(Image.fromarray(lbl10).resize(pred.shape[::-1], Image.NEAREST))
        valid = gt_c < N_CLASSES
        conf += np.bincount(gt_c[valid].astype(np.int64) * N_CLASSES + pred[valid],
                            minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
        # windows in metres
        extent_m = lbl10.shape[0] * MASTER_GSD
        n = int(extent_m // window_m)
        for i in range(n):
            for j in range(n):
                y0m, x0m = i * window_m, j * window_m
                g = lbl10[int(round(y0m / MASTER_GSD)):int(round((y0m + window_m) / MASTER_GSD)),
                          int(round(x0m / MASTER_GSD)):int(round((x0m + window_m) / MASTER_GSD))]
                p = pred[int(round(y0m / CANONICAL_GSD)):int(round((y0m + window_m) / CANONICAL_GSD)),
                         int(round(x0m / CANONICAL_GSD)):int(round((x0m + window_m) / CANONICAL_GSD))]
                row = {"tile": tid, "i": i, "j": j}
                for c, name in enumerate(CLASSES):
                    row[f"{name}_true_m2"] = float((g == c).sum()) * MASTER_GSD ** 2
                    row[f"{name}_pred_m2"] = float((p == c).sum()) * CANONICAL_GSD ** 2
                rows.append(row)
    acc = {}
    for name in EVAL_CLASSES:
        t = np.array([r[f"{name}_true_m2"] for r in rows])
        p = np.array([r[f"{name}_pred_m2"] for r in rows])
        keep = t >= MIN_TRUE_M2
        a = 1 - np.abs(p[keep] - t[keep]) / t[keep]
        acc[name] = {
            "n": int(keep.sum()),
            "median": round(float(np.median(a)), 4) if keep.any() else None,
            "mean": round(float(np.mean(np.clip(a, 0, 1))), 4) if keep.any() else None,
            "within_10pct": round(float(np.mean(np.abs(p[keep] - t[keep]) / t[keep] <= 0.10)), 4) if keep.any() else None,
            "within_20pct": round(float(np.mean(np.abs(p[keep] - t[keep]) / t[keep] <= 0.20)), 4) if keep.any() else None,
            "aggregate": round(float(1 - abs(p[keep].sum() - t[keep].sum()) / t[keep].sum()), 4) if keep.any() else None,
            "bias_pct": round(float((p[keep].sum() - t[keep].sum()) / t[keep].sum() * 100), 2) if keep.any() else None,
        }
    out = {"input_gsd": input_gsd, "window_m": window_m, "n_windows": len(rows),
           "miou": _miou(conf)["mean"], "iou": _miou(conf)["per_class"], "area_accuracy": acc}
    if return_rows:
        out["rows"] = rows
    return out
