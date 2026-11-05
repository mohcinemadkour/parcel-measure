"""Assemble the static in-browser demo (no server): interface + samples + ONNX Runtime
Web + the weight-compressed model split into <=15 MB parts.

Usage: python scripts/build_demo.py --ort /path/to/onnxruntime-web/dist --out demo
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PART = 15 * 1024 * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ort", required=True)
    ap.add_argument("--out", default="demo")
    a = ap.parse_args()
    out = ROOT / a.out
    if out.exists():
        shutil.rmtree(out)
    (out / "model").mkdir(parents=True)
    (out / "ort").mkdir()
    shutil.copy(ROOT / "web/index.html", out / "index.html")
    shutil.copytree(ROOT / "web/samples", out / "samples")
    for f in ("ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"):
        shutil.copy(Path(a.ort) / f, out / "ort" / f)
    data = (ROOT / "models/pam_unet_int8w.onnx").read_bytes()
    parts = []
    for i in range(0, len(data), PART):
        name = f"pam_unet_int8w.part{len(parts):02d}.wasm"
        (out / "model" / name).write_bytes(data[i:i + PART])
        parts.append(name)
    card = json.loads((ROOT / "models/model_card.json").read_text())
    (out / "model/manifest.json").write_text(json.dumps({"version": card["version"], "parts": parts,
                                                         "bytes": len(data)}, indent=1))
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    print(f"{len(files)} files, {sum((out / f).stat().st_size for f in files) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
