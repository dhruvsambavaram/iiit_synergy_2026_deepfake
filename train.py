"""
train.py -- the ONLY file the research agent edits.

    python train.py --run_dir runs/r0001 [--epochs 5] [--seed 0] [--full]

Contract with evaluate.py / inference_script.py:
  * build_model(**build_kwargs) -> nn.Module taking ImageNet-normalised (B,3,224,224) floats,
    returning fake-logits (B,) or (B,1). Any test-time trick (TTA, resizing) must live inside the model.
  * saves run_dir/model.pt = {"state_dict", "build_kwargs", "config"} and run_dir/train_log.json
  * reads ONLY data_splits/train.csv (--full adds val.csv; for the final retrain only, never in the loop)
"""
import argparse
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from common import MEAN, STD, SPLITS, IMAGE_SIZE

# ============================================================== EXPERIMENT CONFIG
CONFIG = dict(
    arch="efficientnet_b0.ra_in1k",   # any timm ImageNet-pretrained model
    drop_rate=0.2,
    time_budget_min=6.0,              # fixed wall-clock training budget (Karpathy-style); --epochs overrides
    epochs=None,
    batch_size=64,
    lr=8e-3,
    weight_decay=0.05,
    warmup_epochs=1.0,
    label_smoothing=0.0,
    consistency_weight=3.0,           # EDD-style: BCE(clean)+BCE(degraded) + w*MSE(sigmoid(clean), sigmoid(degraded))
    num_workers=6,
)


def build_model(arch, drop_rate=0.0, pretrained=False):
    return timm.create_model(arch, pretrained=pretrained, num_classes=1, drop_rate=drop_rate)


def degrade(img):
    """ONE random SEEN corruption (common.py 'seen' types only; no WebP/motion/gamma/posterize/median/crop-zoom/social)."""
    k = random.randrange(8); w, h = img.size
    if k == 0: return img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.0)))
    if k == 1:
        a = np.asarray(img).astype(np.float32)
        a = a + np.random.normal(0, random.uniform(0.01, 0.05) * 255, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if k in (2, 3):
        s = random.uniform(0.4, 0.9) if k == 2 else random.uniform(0.2, 0.5)
        rs = Image.BILINEAR if k == 2 else Image.NEAREST
        return img.resize((max(8, int(w * s)), max(8, int(h * s))), rs).resize((w, h), rs)
    if k == 4:
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=random.randint(30, 95)); buf.seek(0)
        return Image.open(buf).convert("RGB")
    if k == 5: return img.convert("L").convert("RGB")
    if k == 6:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
        return ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.3))
    a = np.asarray(img).copy(); s = int(random.uniform(0.10, 0.30) * w)
    x, y = random.randint(0, w - s), random.randint(0, h - s)
    a[y:y + s, x:x + s] = random.choice([0, 128, 255]); return Image.fromarray(a)


class PairTF:
    """Same geometry for both views; returns stacked (2,3,H,W): [clean, degraded]."""
    def __init__(self):
        self.geo = T.Compose([T.RandomHorizontalFlip(),
                              T.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05), antialias=True)])
        self.tt = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])

    def __call__(self, img):
        img = self.geo(img)
        return torch.stack([self.tt(img), self.tt(degrade(img))])


def train_transforms():
    return PairTF()


def loss_fn(logits, y):
    ls = CONFIG["label_smoothing"]
    y = y * (1 - ls) + 0.5 * ls
    return nn.functional.binary_cross_entropy_with_logits(logits, y)
# ================================================================================


class TrainDS(Dataset):
    def __init__(self, df, tf):
        self.blobs = [Path(p).read_bytes() for p in df.path]   # raw JPEG bytes in RAM
        self.labels = df.label.values.astype(np.float32)
        self.tf = tf

    def __len__(self): return len(self.blobs)

    def __getitem__(self, i):
        img = Image.open(io.BytesIO(self.blobs[i])).convert("RGB")
        return self.tf(img), self.labels[i]


def device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="train on train+val (final retrain only)")
    args = ap.parse_args()
    cfg = dict(CONFIG, **({"epochs": args.epochs} if args.epochs else {}))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    dev = device()

    df = pd.read_csv(SPLITS / "train.csv")
    if args.full:
        df = pd.concat([df, pd.read_csv(SPLITS / "val.csv")], ignore_index=True)
    ds = TrainDS(df, train_transforms())
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                    num_workers=cfg["num_workers"], persistent_workers=True)

    build_kwargs = dict(arch=cfg["arch"], drop_rate=cfg["drop_rate"])
    model = build_model(**build_kwargs, pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # Schedule: progress p in [0,1] is wall-clock based (time budget) or step based (--epochs, final runs).
    budget_s = None if cfg["epochs"] else cfg["time_budget_min"] * 60
    total_steps = cfg["epochs"] * len(dl) if cfg["epochs"] else None
    warm_steps = int(cfg["warmup_epochs"] * len(dl))

    def lr_at(step, elapsed):
        if step < warm_steps:
            return cfg["lr"] * (step + 1) / max(1, warm_steps)
        p = elapsed / budget_s if budget_s else step / total_steps
        return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    log = {"config": cfg, "seed": args.seed, "full": args.full, "n_train": len(ds), "epochs": []}
    t0, step, ep, done = time.time(), 0, 0, False
    while not done:
        model.train(); tl, n, te = 0.0, 0, time.time()
        for x, y in dl:
            for g in opt.param_groups: g["lr"] = lr_at(step, time.time() - t0)
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            B = len(y)
            logits = model(x.reshape(2 * B, *x.shape[2:])).reshape(B, 2)
            lc, ld = logits[:, 0], logits[:, 1]
            loss = loss_fn(lc, y) + loss_fn(ld, y) + cfg["consistency_weight"] * \
                nn.functional.mse_loss(torch.sigmoid(lc), torch.sigmoid(ld))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            tl += loss.item() * len(y); n += len(y)
            if budget_s and time.time() - t0 >= budget_s:
                done = True; break
        ep += 1
        if cfg["epochs"] and ep >= cfg["epochs"]: done = True
        log["epochs"].append({"epoch": ep, "loss": tl / max(1, n), "sec": time.time() - te, "steps": step})
        print(f"[train] epoch {ep} loss={tl / max(1, n):.4f} ({time.time() - te:.0f}s, step {step})", flush=True)
    log["train_seconds"] = time.time() - t0

    torch.save({"state_dict": model.state_dict(), "build_kwargs": build_kwargs, "config": cfg}, run_dir / "model.pt")
    (run_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"[train] saved {run_dir / 'model.pt'} ({log['train_seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
