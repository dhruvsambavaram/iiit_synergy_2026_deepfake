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
from PIL import Image
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
    lr=3e-4,
    weight_decay=0.05,
    warmup_epochs=0.5,
    label_smoothing=0.0,
    num_workers=6,
)


def build_model(arch, drop_rate=0.0, pretrained=False):
    return timm.create_model(arch, pretrained=pretrained, num_classes=1, drop_rate=drop_rate)


class RandomBoxes:
    """Organiser-style occlusion: 1-2 boxes (side 10-35% of width, aspect 0.5-2) filled black/gray/white."""
    def __init__(self, p=0.5): self.p = p

    def __call__(self, img):
        if random.random() >= self.p: return img
        a = np.asarray(img).copy(); h, w = a.shape[:2]
        for _ in range(random.choice([1, 1, 2])):
            s = random.uniform(0.10, 0.35) * w; r = random.uniform(0.5, 2.0)
            bw, bh = max(4, int(s * math.sqrt(r))), max(4, int(s / math.sqrt(r)))
            bw, bh = min(bw, w - 1), min(bh, h - 1)
            x, y = random.randint(0, w - bw), random.randint(0, h - bh)
            a[y:y + bh, x:x + bw] = random.choice([0, 128, 255])
        return Image.fromarray(a)


def train_transforms():
    return T.Compose([
        T.RandomHorizontalFlip(),
        RandomBoxes(p=0.5),
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05), antialias=True),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
    ])


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
            loss = loss_fn(model(x).reshape(-1), y)
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
