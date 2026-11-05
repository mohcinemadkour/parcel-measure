"""Download ISPRS Potsdam patches from the GitHub mirror, rebuild full tiles,
map the 6 ISPRS classes to the 5 product classes and save a 0.10 m/pixel
master copy (image PNG + label PNG) per tile.

Usage: python scripts/prepare_potsdam.py --out data/tiles10 [--tiles 2_10 3_11]

Class mapping (ISPRS colour -> product class id):
  impervious (255,255,255) -> 3 hardscape
  building   (0,0,255)     -> 2 roof
  low veg    (0,255,255)   -> 1 lawn
  tree       (0,255,0)     -> 4 tree
  car        (255,255,0)   -> 3 hardscape
  clutter    (255,0,0)     -> 0 other
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = "https://github.com/cuiyabo0312/Potsdam.git"
PATCH = 256
GRID = 23  # 23 x 23 patches of 256 px = 5,888 px at 0.05 m
SRC_GSD = 0.05
OUT_GSD = 0.10
COLOR_TO_CLASS = {
    (255, 255, 255): 3,
    (0, 0, 255): 2,
    (0, 255, 255): 1,
    (0, 255, 0): 4,
    (255, 255, 0): 3,
    (255, 0, 0): 0,
}
ALL_TILES = [
    "2_10", "2_11", "2_12", "2_13", "2_14", "3_10", "3_11", "3_12", "3_13", "3_14",
    "4_10", "4_11", "4_12", "4_13", "4_14", "4_15", "5_10", "5_11", "5_12", "5_13",
    "5_14", "5_15", "6_10", "6_11", "6_12", "6_13", "6_14", "6_15", "6_7", "6_8",
    "6_9", "7_10", "7_11", "7_12", "7_13", "7_7", "7_8", "7_9",
]


def rgb_to_class(lbl: np.ndarray) -> np.ndarray:
    # nearest palette colour: some tiles (e.g. 4_12) carry compression noise in labels
    pal = np.array(list(COLOR_TO_CLASS.keys()), np.int32)
    ids = np.array(list(COLOR_TO_CLASS.values()), np.uint8)
    d = ((lbl[..., None, :].astype(np.int32) - pal[None, None]) ** 2).sum(-1)
    return ids[d.argmin(-1)]


def majority_downsample(cls: np.ndarray, f: int, n_classes: int = 5) -> np.ndarray:
    h, w = cls.shape
    h2, w2 = h // f, w // f
    blocks = cls[: h2 * f, : w2 * f].reshape(h2, f, w2, f).transpose(0, 2, 1, 3).reshape(h2, w2, f * f)
    counts = np.stack([(blocks == c).sum(-1) for c in range(n_classes)], -1)
    return counts.argmax(-1).astype(np.uint8)


def fetch_tile(tile: str, workdir: Path) -> Path:
    repo = workdir / f"repo_{tile}"
    if repo.exists():
        shutil.rmtree(repo)
    run = lambda *a: subprocess.run(a, cwd=repo if repo.exists() else workdir, check=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    run("git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout", REPO, str(repo))
    pats = [f"{d}/top_potsdam_{tile}__*" for d in ("train", "train_labels", "test", "test_labels")]
    run("git", "sparse-checkout", "set", "--no-cone", *pats)
    run("git", "checkout", "-q", "HEAD")
    return repo


def build_tile(repo: Path, tile: str):
    img = np.zeros((GRID * PATCH, GRID * PATCH, 3), np.uint8)
    lbl = np.full((GRID * PATCH, GRID * PATCH), 255, np.uint8)
    n = 0
    for split in ("train", "test"):
        for f in (repo / split).glob(f"top_potsdam_{tile}__*.tif"):
            r, c = map(int, re.search(r"__(\d+)_(\d+)\.tif$", f.name).groups())
            y, x = r * PATCH, c * PATCH
            img[y:y + PATCH, x:x + PATCH] = np.array(Image.open(f).convert("RGB"))
            lab = np.array(Image.open(repo / f"{split}_labels" / f.name).convert("RGB"))
            lbl[y:y + PATCH, x:x + PATCH] = rgb_to_class(lab)
            n += 1
    if n != GRID * GRID:
        raise RuntimeError(f"tile {tile}: expected {GRID*GRID} patches, got {n}")
    return img, lbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tiles10")
    ap.add_argument("--tiles", nargs="*", default=ALL_TILES)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="potsdam_"))
    f = int(round(OUT_GSD / SRC_GSD))
    for t in a.tiles:
        if (out / f"{t}_lbl.png").exists():
            print(t, "exists", flush=True)
            continue
        try:
            repo = fetch_tile(t, work)
            img, lbl = build_tile(repo, t)
            h, w = img.shape[0] // f, img.shape[1] // f
            Image.fromarray(img).resize((w, h), Image.BOX).save(out / f"{t}_img.png", optimize=False)
            Image.fromarray(majority_downsample(lbl, f)).save(out / f"{t}_lbl.png")
            counts = np.bincount(lbl.ravel(), minlength=5)[:5] / lbl.size
            print(t, "ok", " ".join(f"{x:.2f}" for x in counts), flush=True)
        except Exception as e:  # keep going; report at the end
            print(t, "FAILED", e, flush=True)
        finally:
            shutil.rmtree(work / f"repo_{t}", ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
