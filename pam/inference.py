"""Segmentation inference: resample to canonical GSD, sliding-window ONNX inference."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import (CANONICAL_GSD, IMAGENET_MEAN, IMAGENET_STD, MODEL_CARD, MODEL_ONNX,
                     N_CLASSES, OVERLAP, TILE)


def resample_to_canonical(img: np.ndarray, gsd: float) -> tuple[np.ndarray, float]:
    """Return image resampled to CANONICAL_GSD and the scale factor applied."""
    scale = gsd / CANONICAL_GSD
    if abs(scale - 1.0) < 1e-3:
        return img, 1.0
    h, w = img.shape[:2]
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resample = Image.BOX if scale < 1 else Image.BILINEAR
    return np.array(Image.fromarray(img).resize((nw, nh), resample)), scale


def tta_views(mode: str):
    """(forward, inverse) transforms on CxHxW arrays."""
    ident = (lambda a: a, lambda a: a)
    if mode == "none":
        return [ident]
    hflip = (lambda a: a[..., ::-1], lambda a: a[..., ::-1])
    if mode == "flip":
        return [ident, hflip]
    vflip = (lambda a: a[..., ::-1, :], lambda a: a[..., ::-1, :])
    rots = [(lambda a, k=k: np.rot90(a, k, axes=(-2, -1)), lambda a, k=k: np.rot90(a, -k, axes=(-2, -1)))
            for k in (1, 2, 3)]
    return [ident, hflip, vflip, *rots]


def _windows(n: int, tile: int, step: int) -> list[int]:
    if n <= tile:
        return [0]
    starts = list(range(0, n - tile, step)) + [n - tile]
    return sorted(set(starts))


def _blend_weights(tile: int) -> np.ndarray:
    ramp = np.minimum(np.arange(tile) + 1, np.arange(tile)[::-1] + 1).astype(np.float32)
    ramp = np.minimum(ramp / OVERLAP, 1.0)
    return np.outer(ramp, ramp) + 1e-3


@dataclass
class SegResult:
    classes: np.ndarray   # HxW uint8 class ids at canonical GSD
    probs: np.ndarray     # CxHxW float32 softmax
    image: np.ndarray     # HxWx3 uint8 at canonical GSD
    scale: float


class Segmenter:
    """Thread-safe ONNX Runtime segmenter."""

    def __init__(self, onnx_path=MODEL_ONNX, threads: int | None = None, tta: str | None = None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(onnx_path), so, providers=ort.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name
        self._lock = threading.Lock()
        self.tta = (tta or os.environ.get("PAM_TTA", "flip")).lower()  # none | flip | d4
        try:
            self.card = json.loads(MODEL_CARD.read_text())
        except Exception:
            self.card = {"version": "unknown"}

    @property
    def version(self) -> str:
        return self.card.get("version", "unknown")

    def _run(self, batch: np.ndarray) -> np.ndarray:
        with self._lock:
            return self.session.run(None, {self.input_name: batch})[0]

    def _probs(self, tile: np.ndarray) -> np.ndarray:
        """Softmax for one CxHxW tile, averaged over test-time flips/rotations."""
        views = tta_views(self.tta)
        batch = np.stack([fwd(tile) for fwd, _ in views]).astype(np.float32)
        logits = self._run(np.ascontiguousarray(batch))
        e = np.exp(logits - logits.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        return np.mean([inv(pi) for pi, (_, inv) in zip(p, views)], axis=0)

    def predict(self, img: np.ndarray, gsd: float) -> SegResult:
        img = np.ascontiguousarray(img[..., :3])
        img_c, scale = resample_to_canonical(img, gsd)
        h, w = img_c.shape[:2]
        ph, pw = max(h, TILE), max(w, TILE)
        pad = np.zeros((ph, pw, 3), np.uint8)
        pad[:h, :w] = img_c
        x = (pad.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = x.transpose(2, 0, 1).astype(np.float32)
        acc = np.zeros((N_CLASSES, ph, pw), np.float32)
        wsum = np.zeros((ph, pw), np.float32)
        wt = _blend_weights(TILE)
        step = TILE - OVERLAP
        for y in _windows(ph, TILE, step):
            for x0 in _windows(pw, TILE, step):
                p = self._probs(x[:, y:y + TILE, x0:x0 + TILE])
                acc[:, y:y + TILE, x0:x0 + TILE] += p * wt
                wsum[y:y + TILE, x0:x0 + TILE] += wt
        probs = (acc / wsum)[:, :h, :w]
        return SegResult(probs.argmax(0).astype(np.uint8), probs, img_c, scale)
