"""Cut sample images (with ground-truth label PNGs) from the held-out TEST tiles for
the client interface. Picks residential-looking crops: lawn and roof both present.

Usage: python scripts/make_samples.py --out web/samples --n 12
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pam.config import CANONICAL_GSD, SQFT_PER_M2  # noqa: E402
from pam.splits import SPLITS  # noqa: E402

MASTER = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tiles10")
    ap.add_argument("--out", default="web/samples")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--size_m", type=float, default=48.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(3)
    src = int(a.size_m / MASTER)
    dst = int(round(a.size_m / CANONICAL_GSD))
    cands = []
    for t in SPLITS["test"]:
        img = np.array(Image.open(Path(a.data) / f"{t}_img.png").convert("RGB"))
        lbl = np.array(Image.open(Path(a.data) / f"{t}_lbl.png"))
        for _ in range(60):
            y, x = rng.randint(0, lbl.shape[0] - src), rng.randint(0, lbl.shape[1] - src)
            l = lbl[y:y + src, x:x + src]
            f = np.bincount(l.ravel(), minlength=5) / l.size
            # residential mix: real lawn and roof, not mostly street
            score = min(f[1], 0.35) + min(f[2], 0.35) - max(0, f[3] - 0.35) - f[0]
            if f[1] > 0.12 and f[2] > 0.10:
                cands.append((score, t, y, x))
    cands.sort(reverse=True)
    chosen, used = [], set()
    for s, t, y, x in cands:
        key = (t, y // src, x // src)
        if key in used or sum(1 for c in chosen if c[1] == t) >= 2:
            continue
        used.add(key)
        chosen.append((s, t, y, x))
        if len(chosen) == a.n:
            break
    index = []
    for i, (_, t, y, x) in enumerate(chosen, 1):
        img = np.array(Image.open(Path(a.data) / f"{t}_img.png").convert("RGB"))[y:y + src, x:x + src]
        lbl = np.array(Image.open(Path(a.data) / f"{t}_lbl.png"))[y:y + src, x:x + src]
        im = Image.fromarray(img).resize((dst, dst), Image.BOX)
        lb = Image.fromarray(lbl).resize((dst, dst), Image.NEAREST)
        name = f"sample_{i:02d}"
        im.save(out / f"{name}.png", optimize=True)
        lb.save(out / f"{name}_label.png", optimize=True)
        truth = {k: round(float((lbl == c).sum()) * MASTER ** 2 * SQFT_PER_M2, 1)
                 for c, k in [(1, "lawn_sqft"), (2, "roof_footprint_sqft"), (3, "hardscape_sqft"), (4, "tree_canopy_sqft")]}
        index.append({"id": name, "file": f"{name}.png", "label": f"{name}_label.png", "gsd": CANONICAL_GSD,
                      "size_m": a.size_m, "title": f"Held-out tile {t}, {a.size_m:.0f} m square", "tile": t,
                      "truth": truth})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} samples to {out}")


if __name__ == "__main__":
    main()
