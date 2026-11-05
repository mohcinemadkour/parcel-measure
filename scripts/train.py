"""Train the 5-class U-Net on prepared Potsdam tiles (0.10 m master copies).

Each sample: pick a training tile, draw a target GSD in [0.105, 0.195] m (canonical
0.15 +/- 30%), cut the matching ground footprint, resize it to CROP px at canonical
scale (simulating images arriving at different resolutions and being resampled), then
flips / rot90 / colour jitter / blur. Loss = CE + Dice. Model selection on validation
lawn+roof area accuracy over property-sized windows (see pam/evaluation.py).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pam.config import CANONICAL_GSD, IMAGENET_MEAN, IMAGENET_STD, N_CLASSES  # noqa: E402
from pam.model import build_model  # noqa: E402
from pam.splits import SPLITS  # noqa: E402

MASTER_GSD = 0.10


def load_tiles(root: Path, ids):
    out = []
    for t in ids:
        img = np.array(Image.open(root / f"{t}_img.png").convert("RGB"))
        lbl = np.array(Image.open(root / f"{t}_lbl.png"))
        out.append((img, lbl))
    return out


class Sampler:
    def __init__(self, tiles, crop: int, rng: random.Random):
        self.tiles, self.crop, self.rng = tiles, crop, rng
        # class-balanced-ish: sample tiles proportional to size, crops uniformly

    def sample(self):
        img, lbl = self.rng.choice(self.tiles)
        gsd = self.rng.uniform(0.105, 0.195)
        # footprint at canonical scale = crop px * 0.15 m; image arrives at `gsd`,
        # then gets resampled to 0.15 m, so extract footprint and resize via `gsd` first
        foot_m = self.crop * CANONICAL_GSD
        src = int(round(foot_m / MASTER_GSD))
        h, w = lbl.shape
        y = self.rng.randrange(0, h - src)
        x = self.rng.randrange(0, w - src)
        im = Image.fromarray(img[y:y + src, x:x + src])
        lb = Image.fromarray(lbl[y:y + src, x:x + src])
        mid = max(8, int(round(foot_m / gsd)))           # as captured at `gsd`
        im = im.resize((mid, mid), Image.BOX if mid < src else Image.BILINEAR)
        im = im.resize((self.crop, self.crop), Image.BOX if self.crop < mid else Image.BILINEAR)
        lb = lb.resize((self.crop, self.crop), Image.NEAREST)
        if self.rng.random() < 0.3:
            im = im.filter(ImageFilter.GaussianBlur(self.rng.uniform(0.3, 1.2)))
        a = np.asarray(im, np.float32) / 255.0
        b = np.asarray(lb, np.int64)
        # colour jitter: brightness, contrast, per-channel gain (seasonal/sensor shift)
        a = a * self.rng.uniform(0.75, 1.25) + self.rng.uniform(-0.08, 0.08)
        m = a.mean()
        a = (a - m) * self.rng.uniform(0.75, 1.3) + m
        a = a * np.array([self.rng.uniform(0.9, 1.1) for _ in range(3)], np.float32)
        a = np.clip(a, 0, 1)
        k = self.rng.randrange(4)
        a, b = np.rot90(a, k), np.rot90(b, k)
        if self.rng.random() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]
        a = (a - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(a.transpose(2, 0, 1), np.float32), np.ascontiguousarray(b)


def dice_loss(logits, target, eps=1.0):
    p = logits.softmax(1)
    oh = F.one_hot(target, N_CLASSES).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    union = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tiles10")
    ap.add_argument("--out", default="models/run")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--weights", default="models/resnet34-43635321.pth")
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--samples", type=int, default=1200, help="samples per epoch")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--init", default=None, help="continue from a checkpoint (fine-tuning stage)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.data)
    train_tiles = load_tiles(root, SPLITS["train"])
    print(f"train tiles: {len(train_tiles)}  val tiles: {len(SPLITS['val'])}", flush=True)

    from pam.evaluation import evaluate_tiles, TorchPredictor
    val_tiles = load_tiles(root, SPLITS["val"])

    model = build_model(a.encoder, None if a.init else a.weights)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["state_dict"])
        print("initialised from", a.init, flush=True)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = a.samples // a.batch
    total = steps_per_epoch * a.epochs
    warm = steps_per_epoch // 2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    sampler = Sampler(train_tiles, a.crop, rng)
    # class weights: mild inverse-frequency so small roofs/hardscape aren't ignored
    freq = np.bincount(np.concatenate([t[1].ravel()[::97] for t in train_tiles]), minlength=N_CLASSES) + 1
    cw = torch.tensor((freq.sum() / freq) ** 0.3, dtype=torch.float32)
    cw = cw / cw.mean()
    print("class weights", [round(float(x), 2) for x in cw], flush=True)

    history, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for s in range(steps_per_epoch):
            xs, ys = zip(*(sampler.sample() for _ in range(a.batch)))
            x = torch.from_numpy(np.stack(xs)).to(memory_format=torch.channels_last)
            y = torch.from_numpy(np.stack(ys))
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = model(x)
            logits = logits.float()
            loss = F.cross_entropy(logits, y, weight=cw) + dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item()
        train_loss = run / steps_per_epoch
        model.eval()
        metrics = evaluate_tiles(TorchPredictor(model), val_tiles, SPLITS["val"])
        score = (metrics["area_accuracy"]["lawn"]["median"] + metrics["area_accuracy"]["roof"]["median"]) / 2
        rec = {"epoch": ep, "train_loss": round(train_loss, 4), "val_score": round(score, 4),
               "val_miou": metrics["miou"], "val_lawn_median": metrics["area_accuracy"]["lawn"]["median"],
               "val_roof_median": metrics["area_accuracy"]["roof"]["median"],
               "minutes": round((time.time() - t0) / 60, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "encoder": a.encoder, "epoch": ep,
                        "val": metrics}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1))
    torch.save({"state_dict": model.state_dict(), "encoder": a.encoder, "epoch": a.epochs}, out / "last.pt")
    print("best val score", best, flush=True)


if __name__ == "__main__":
    main()
