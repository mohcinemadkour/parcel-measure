"""Export the trained checkpoint to ONNX (server) and a weight-compressed ONNX
(browser demo: int8 per-channel weights + DequantizeLinear, fp32 compute).

Usage: python scripts/export_onnx.py --ckpt models/run/best.pt --out models
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper, TensorProto

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pam.config import CANONICAL_GSD, CLASSES, TILE  # noqa: E402
from pam.model import build_model  # noqa: E402


def export(model, path: Path):
    model.eval()
    x = torch.randn(1, 3, TILE, TILE)
    torch.onnx.export(model, x, str(path), input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    m = onnx.load(str(path), load_external_data=True)
    onnx.save(m, str(path), save_as_external_data=False)
    onnx.checker.check_model(str(path))


def weight_only_int8(src: Path, dst: Path, min_size: int = 2048):
    """Replace large Conv weights with int8 + per-output-channel scale + DequantizeLinear."""
    m = onnx.load(str(src))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    conv_w = {n.input[1] for n in g.node if n.op_type == "Conv" and len(n.input) > 1}
    new_inits, new_nodes, n_q = [], [], 0
    for name in list(inits):
        t = inits[name]
        w = numpy_helper.to_array(t)
        if name not in conv_w or w.dtype != np.float32 or w.size < min_size:
            continue
        oc = w.shape[0]
        flat = w.reshape(oc, -1)
        scale = np.maximum(np.abs(flat).max(1), 1e-12) / 127.0
        q = np.clip(np.round(flat / scale[:, None]), -127, 127).astype(np.int8).reshape(w.shape)
        g.initializer.remove(t)
        new_inits += [numpy_helper.from_array(q, name + "_q"),
                      numpy_helper.from_array(scale.astype(np.float32), name + "_scale"),
                      numpy_helper.from_array(np.zeros(oc, np.int8), name + "_zp")]
        new_nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_scale", name + "_zp"],
                                          [name], axis=0, name=name + "_dq"))
        n_q += 1
    g.initializer.extend(new_inits)
    for i, n in enumerate(new_nodes):
        g.node.insert(i, n)
    onnx.save(m, str(dst))
    onnx.checker.check_model(str(dst))
    return n_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/run/best.pt")
    ap.add_argument("--out", default="models")
    ap.add_argument("--version", default="0.1.0")
    a = ap.parse_args()
    out = Path(a.out)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck["encoder"])
    model.load_state_dict(ck["state_dict"])
    fp32 = out / "pam_unet.onnx"
    export(model, fp32)
    small = out / "pam_unet_int8w.onnx"
    nq = weight_only_int8(fp32, small)

    # numerical check: fp32 torch vs onnx fp32 vs int8-weight onnx
    import onnxruntime as ort
    x = np.random.RandomState(0).randn(1, 3, TILE, TILE).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    o1 = ort.InferenceSession(str(fp32)).run(None, {"image": x})[0]
    o2 = ort.InferenceSession(str(small)).run(None, {"image": x})[0]
    agree = float((o1.argmax(1) == o2.argmax(1)).mean())
    card = {
        "name": "pam-unet",
        "version": a.version,
        "architecture": f"U-Net, {ck['encoder']} encoder (ImageNet init)",
        "classes": CLASSES,
        "input": {"shape": [1, 3, TILE, TILE], "gsd_m": CANONICAL_GSD, "normalization": "ImageNet mean/std, RGB 0-1"},
        "training_data": "ISPRS Potsdam (26 train tiles, 4 val tiles), resampled to 0.15 m +/- 30%",
        "checkpoint_epoch": ck.get("epoch"),
        "validation": ck.get("val"),
        "exported": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {
            "server": {"file": fp32.name, "sha256": hashlib.sha256(fp32.read_bytes()).hexdigest(),
                       "mb": round(fp32.stat().st_size / 1e6, 1)},
            "browser": {"file": small.name, "sha256": hashlib.sha256(small.read_bytes()).hexdigest(),
                        "mb": round(small.stat().st_size / 1e6, 1), "int8_weight_tensors": nq,
                        "pixel_agreement_vs_fp32": round(agree, 5)},
        },
        "max_abs_diff_torch_vs_onnx": float(np.abs(ref - o1).max()),
        "license_note": "Prototype trained on a research benchmark; retrain on licensed US data before commercial use.",
    }
    eval_path = out / "evaluation.json"
    if eval_path.exists():
        card["test_evaluation"] = json.loads(eval_path.read_text()).get("summary")
    (out / "model_card.json").write_text(json.dumps(card, indent=1))
    print(json.dumps({k: card[k] for k in ("files", "max_abs_diff_torch_vs_onnx")}, indent=1))


if __name__ == "__main__":
    main()
