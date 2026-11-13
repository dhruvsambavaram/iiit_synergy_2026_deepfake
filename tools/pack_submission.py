"""
Pack one or more trained runs into a single submission weights file.

    python tools/pack_submission.py --runs runs/r0001 [runs/r0012 ...] --tta --out submission/model_weights.pt \
        [--threshold_from runs/r0001 runs/r0012]   # choose threshold (Youden J) from the runs' VAL predictions

Threshold note: --full retrained runs have no clean val predictions, so pass the matching screening runs
(same configs, trained on train only) via --threshold_from.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def youden_threshold(y, p):
    """Threshold maximising TPR - FPR (balanced accuracy); classes are ~balanced so this beats max-F1,
    which drifts toward predicting everything fake when the model is weak."""
    best_t, best_j = 0.5, -1
    for t in np.unique(np.round(p, 3)):
        pred = p >= t
        tpr = (pred & (y == 1)).sum() / max(1, (y == 1).sum()); fpr = (pred & (y == 0)).sum() / max(1, (y == 0).sum())
        if tpr - fpr > best_j:
            best_t, best_j = float(t), tpr - fpr
    return best_t, best_j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--threshold_from", nargs="*", default=None)
    ap.add_argument("--out", default=str(REPO / "submission" / "model_weights.pt"))
    args = ap.parse_args()

    models = []
    for r in args.runs:
        ck = torch.load(Path(r) / "model.pt", map_location="cpu", weights_only=False)
        models.append({**ck["build_kwargs"], "state_dict": ck["state_dict"]})
    thr = 0.5
    if args.threshold_from:
        P = [np.load(Path(r) / "val_preds.npz", allow_pickle=True) for r in args.threshold_from]
        y = P[0]["y"]; p = np.mean([q["clean"] for q in P], 0)
        thr, j = youden_threshold(y, p)
        print(f"[pack] threshold {thr:.3f} (val balanced-acc {(1 + j) / 2:.4f}; non-TTA val preds)")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "dfk-ensemble-v1", "models": models, "tta_hflip": args.tta,
                "threshold": thr, "image_size": 224, "source_runs": args.runs}, out)
    print(f"[pack] {len(models)} model(s) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
