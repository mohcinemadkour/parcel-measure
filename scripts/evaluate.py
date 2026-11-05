"""Acceptance evaluation on the held-out TEST tiles with the exported ONNX model
(exactly what the API serves). Writes models/evaluation.json and per-window CSV.

Usage: python scripts/evaluate.py [--model models/pam_unet.onnx] [--gsds 0.15 0.10 0.20 0.30]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pam.evaluation import evaluate_tiles  # noqa: E402
from pam.inference import Segmenter  # noqa: E402
from pam.splits import SPLITS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tiles10")
    ap.add_argument("--model", default="models/pam_unet.onnx")
    ap.add_argument("--out", default="models")
    ap.add_argument("--gsds", nargs="*", type=float, default=[0.15, 0.10, 0.20, 0.30])
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    ids = SPLITS[a.split]
    tiles = [(np.array(Image.open(Path(a.data) / f"{t}_img.png").convert("RGB")),
              np.array(Image.open(Path(a.data) / f"{t}_lbl.png"))) for t in ids]
    seg = Segmenter(a.model)
    results = {}
    for g in a.gsds:
        t0 = time.time()
        r = evaluate_tiles(seg, tiles, ids, input_gsd=g, return_rows=(g == a.gsds[0]))
        r["seconds"] = round(time.time() - t0, 1)
        if "rows" in r:
            rows = r.pop("rows")
            with open(Path(a.out) / f"evaluation_windows_{a.split}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        results[f"{g:.2f}"] = r
        print(f"GSD {g:.2f}: " + json.dumps({k: v["median"] for k, v in r["area_accuracy"].items()}),
              "mIoU", r["miou"], flush=True)
    main_r = results[f"{a.gsds[0]:.2f}"]
    acc = main_r["area_accuracy"]
    summary = {
        "split": a.split, "tiles": ids, "windows": main_r["n_windows"], "window_m": main_r["window_m"],
        "lawn_median_accuracy": acc["lawn"]["median"], "roof_median_accuracy": acc["roof"]["median"],
        "lawn_within_10pct": acc["lawn"]["within_10pct"], "roof_within_10pct": acc["roof"]["within_10pct"],
        "lawn_total_accuracy": acc["lawn"]["aggregate"], "roof_total_accuracy": acc["roof"]["aggregate"],
        "miou": main_r["miou"],
        "AC-01_pass": acc["lawn"]["median"] >= 0.90, "AC-02_pass": acc["roof"]["median"] >= 0.90,
    }
    out = {"summary": summary, "by_gsd": results, "model": a.model}
    Path(a.out, "evaluation.json" if a.split == "test" else f"evaluation_{a.split}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
