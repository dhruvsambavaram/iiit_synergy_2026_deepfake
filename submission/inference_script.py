"""
inference_script.py -- Deepfake ML Hackathon (IIIT Bangalore) submission.

    python inference_script.py --test_dir ./test/images/ --weights ./model_weights.pt --output predictions.csv

Weights file (built by tools/pack_submission.py) is a dict:
    {"format": "dfk-ensemble-v1",
     "models": [{"arch": str, "drop_rate": float, "state_dict": {...}}, ...],
     "tta_hflip": bool, "threshold": float, "image_size": 224}
A plain single-model checkpoint {"state_dict", "build_kwargs"} is also accepted.

Output: image_id,label,confidence  (confidence = P(fake), label = confidence >= threshold).
Device: CUDA (fp16 autocast) > Apple MPS > CPU, chosen automatically.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EXTS = {".jpg", ".jpeg", ".png"}


class TestDataset(Dataset):
    def __init__(self, test_dir, image_size):
        self.paths = sorted(p for p in Path(test_dir).iterdir() if p.suffix.lower() in EXTS)
        self.size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            if im.size != (self.size, self.size):
                im = im.resize((self.size, self.size), Image.BILINEAR)
            x = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1)  # uint8, normalised on GPU
        return x, i


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(weights_path, device):
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    if ck.get("format") == "dfk-ensemble-v1":
        specs, tta, thr, size = ck["models"], ck.get("tta_hflip", False), ck.get("threshold", 0.5), ck.get("image_size", 224)
    else:  # single training checkpoint
        specs = [{**ck["build_kwargs"], "state_dict": ck["state_dict"]}]
        tta, thr, size = False, 0.5, 224
    models = []
    for s in specs:
        m = timm.create_model(s["arch"], pretrained=False, num_classes=1, drop_rate=s.get("drop_rate", 0.0))
        m.load_state_dict(s["state_dict"])
        models.append(m.eval().to(device))
    print(f"[INFO] {len(models)} model(s): {[s['arch'] for s in specs]} | tta_hflip={tta} | threshold={thr:.4f} | {device}")
    return models, tta, thr, size


@torch.no_grad()
def run_inference(models, tta, test_dir, size, device, batch_size, workers):
    ds = TestDataset(test_dir, size)
    if len(ds) == 0:
        raise SystemExit(f"No images found in {test_dir}")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    pin_memory=(device == "cuda"), persistent_workers=False)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1) * 255
    probs = np.zeros(len(ds), dtype=np.float32)
    use_amp = device == "cuda"
    for x, idx in dl:
        x = (x.to(device, non_blocking=True).float() - mean) / std
        views = [x, torch.flip(x, dims=[3])] if tta else [x]
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            p = torch.stack([torch.sigmoid(m(v).float().reshape(-1)) for m in models for v in views]).mean(0)
        probs[idx.numpy()] = p.cpu().numpy()
    return [p.stem for p in ds.paths], probs


def main():
    ap = argparse.ArgumentParser(description="Deepfake ML Hackathon - inference")
    ap.add_argument("--test_dir", required=True, help="Path to test images folder")
    ap.add_argument("--weights", required=True, help="Path to model weights file")
    ap.add_argument("--output", default="predictions.csv", help="Output CSV path")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--threshold", type=float, default=None, help="override threshold stored in weights")
    args = ap.parse_args()

    t0 = time.time()
    torch.backends.cudnn.benchmark = True
    device = pick_device()
    models, tta, thr, size = load_models(args.weights, device)
    if args.threshold is not None:
        thr = args.threshold
    ids, conf = run_inference(models, tta, args.test_dir, size, device, args.batch_size, args.num_workers)
    conf = np.clip(conf, 0.0, 1.0)
    df = pd.DataFrame({"image_id": ids, "label": (conf >= thr).astype(int), "confidence": conf})

    assert df["image_id"].is_unique, "duplicate image ids"
    assert df["confidence"].between(0, 1).all() and df["label"].isin([0, 1]).all()
    df.to_csv(args.output, index=False, float_format="%.6f")
    print(f"[DONE] {len(df)} predictions -> {args.output} in {time.time() - t0:.1f}s "
          f"| predicted fake {df.label.mean() * 100:.1f}% | mean conf {df.confidence.mean():.4f}")


if __name__ == "__main__":
    main()
