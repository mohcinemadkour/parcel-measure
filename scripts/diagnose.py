"""Validation diagnostics for a checkpoint: TTA variants, lawn/roof bias, error by
feature size, confusion matrix. VALIDATION tiles only (test tiles stay untouched).

Usage: python scripts/diagnose.py --ckpt models/run/best.pt [--tta none flip d4]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pam.evaluation import TorchPredictor, evaluate_tiles  # noqa: E402
from pam.model import build_model  # noqa: E402
from pam.splits import SPLITS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/run/best.pt")
    ap.add_argument("--data", default="data/tiles10")
    ap.add_argument("--tta", nargs="*", default=["none", "flip", "d4"])
    a = ap.parse_args()
    torch.set_num_threads(2)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = build_model(ck["encoder"])
    m.load_state_dict(ck["state_dict"])
    m.eval().to(memory_format=torch.channels_last)
    ids = SPLITS["val"]
    tiles = [(np.array(Image.open(Path(a.data) / f"{t}_img.png").convert("RGB")),
              np.array(Image.open(Path(a.data) / f"{t}_lbl.png"))) for t in ids]
    for tta in a.tta:
        r = evaluate_tiles(TorchPredictor(m, tta), tiles, ids, return_rows=True)
        rows = r.pop("rows")
        print(f"== epoch {ck['epoch']} tta={tta} mIoU={r['miou']}")
        for k, v in r["area_accuracy"].items():
            print(f"  {k:9s} median {v['median']:.3f}  within10 {v['within_10pct']:.2f}  total {v['aggregate']:.3f}  bias {v['bias_pct']:+.1f}%  n={v['n']}")
        t = np.array([x["lawn_true_m2"] for x in rows])
        p = np.array([x["lawn_pred_m2"] for x in rows])
        for lo, hi in [(20, 100), (100, 300), (300, 600), (600, 2000)]:
            k = (t >= lo) & (t < hi)
            if k.any():
                acc = 1 - np.abs(p[k] - t[k]) / t[k]
                print(f"    lawn {lo:4d}-{hi:4d} m2: n={k.sum():3d} median acc {np.median(acc):.3f} bias {(p[k].sum()-t[k].sum())/t[k].sum()*100:+.1f}%")
        print("  IoU", json.dumps(r["iou"]))


if __name__ == "__main__":
    main()
