"""
evaluate.py -- FROZEN. Scores a trained run on the validation split (or, rarely, the holdout).

    python evaluate.py --run_dir runs/r0001 --desc "baseline effnet-b0"
    python evaluate.py --run_dir runs/r0007 --final      # holdout; max 3 uses total

The checkpoint (run_dir/model.pt) must be a dict {"state_dict", "build_kwargs"};
the model is rebuilt via train.build_model(**build_kwargs). The model receives
ImageNet-normalised 224x224 float tensors and must return fake-logits of shape (B,) or (B,1).

Writes: run_dir/val_preds.npz, one row in results.tsv, updates best.json on KEEP.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from common import CORRUPTIONS, MEAN, STD, SPLITS, VARIANTS, apply_variant, load_rgb, sha256_file  # noqa: E402

RESULTS = REPO / "results.tsv"
BEST = REPO / "best.json"
HOLDOUT_LOG = REPO / "holdout_log.tsv"
MAX_HOLDOUT_USES = 3
N_BOOT = 500
GUARDS = {"max_params_M": 150, "max_ckpt_MB": 600, "min_img_per_s": 100}
SEEN = [k for k, (_, g) in CORRUPTIONS.items() if g == "seen"]
UNSEEN = [k for k, (_, g) in CORRUPTIONS.items() if g == "unseen"]
COLS = ["time", "run_id", "verdict", "proxy", "delta", "p_better", "auc_main", "auc_iid", "auc_ood",
        "auc_seen", "auc_unseen", "auc_worst", "worst_variant", "params_M", "ckpt_MB", "img_s",
        "train_min", "desc"]


def device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


class VariantDS(Dataset):
    def __init__(self, df, variants):
        self.items = [(v, i) for v in variants for i in range(len(df))]
        self.paths, self.ids = list(df.path), list(df.image_id)

    def __len__(self): return len(self.items)

    def __getitem__(self, k):
        v, i = self.items[k]
        img = apply_variant(load_rgb(self.paths[i]), v, self.ids[i])
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)
        x = (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
        return x, k


def auc(y, s):
    """Fast exact ROC-AUC (ties -> average rank)."""
    order = np.argsort(s, kind="mergesort"); s_sorted = s[order]
    ranks = np.empty(len(s)); r = np.arange(1, len(s) + 1, dtype=float)
    _, inv, cnt = np.unique(s_sorted, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, r); ranks[order] = (sums / cnt)[inv]
    npos = y.sum(); nneg = len(y) - npos
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def summarize(y, P, subset, idx=None):
    """P: dict variant -> scores. Returns metric dict. idx = bootstrap index sample."""
    if idx is not None:
        y, subset, P = y[idx], subset[idx], {k: v[idx] for k, v in P.items()}
    a = {v: auc(y, P[v]) for v in VARIANTS}
    iid, ood = subset == "iid", subset == "ood"
    m = dict(auc_iid=auc(y[iid], P["clean"][iid]), auc_ood=auc(y[ood], P["clean"][ood]),
             auc_seen=float(np.mean([a[v] for v in SEEN])), auc_unseen=float(np.mean([a[v] for v in UNSEEN])))
    m["auc_main"] = 0.5 * m["auc_iid"] + 0.5 * m["auc_ood"]
    rob = 0.4 * m["auc_seen"] + 0.6 * m["auc_unseen"]
    m["proxy"] = 50 * (2 * m["auc_main"] - 1) + 20 * (2 * rob - 1)
    w = min(a, key=a.get); m["auc_worst"], m["worst_variant"] = a[w], w
    m["per_variant"] = a
    return m


def verify_frozen():
    man = json.loads((SPLITS / "manifest.json").read_text())
    for f, h in man.items():
        p = SPLITS / f if f.endswith(".csv") else REPO / f
        if sha256_file(p) != h:
            sys.exit(f"[evaluate] FROZEN FILE MODIFIED: {f}. Restore it (git checkout {f}). Aborting.")


def load_model(run_dir, dev):
    import importlib
    train = importlib.import_module("train")
    ck = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    model = train.build_model(**ck["build_kwargs"])
    model.load_state_dict(ck["state_dict"])
    return model.eval().to(dev), ck


@torch.no_grad()
def predict(model, df, variants, dev, bs=128, workers=6):
    ds = VariantDS(df, variants)
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False)
    out = np.zeros(len(ds), dtype=np.float32)
    for x, k in dl:
        out[k.numpy()] = torch.sigmoid(model(x.to(dev)).float().reshape(-1)).cpu().numpy()
    n = len(df)
    return {v: out[j * n:(j + 1) * n] for j, v in enumerate(variants)}


@torch.no_grad()
def throughput(model, dev, n=1024, bs=128):
    x = torch.randn(bs, 3, 224, 224, device=dev)
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else \
        ((lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None))
    model(x); sync(); t = time.time()
    for _ in range(n // bs): model(x)
    sync(); return n / (time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--final", action="store_true", help="score on HOLDOUT (max 3 uses, ever)")
    args = ap.parse_args()
    verify_frozen()
    run_dir = (REPO / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    dev = device()
    model, ck = load_model(run_dir, dev)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    ckpt_mb = (run_dir / "model.pt").stat().st_size / 1e6
    img_s = throughput(model, dev)
    train_log = json.loads((run_dir / "train_log.json").read_text()) if (run_dir / "train_log.json").exists() else {}
    train_min = train_log.get("train_seconds", 0) / 60

    split = "holdout" if args.final else "val"
    if args.final:
        uses = len(pd.read_csv(HOLDOUT_LOG, sep="\t")) if HOLDOUT_LOG.exists() else 0
        if uses >= MAX_HOLDOUT_USES:
            sys.exit(f"[evaluate] holdout already used {uses} times. Refusing.")
    df = pd.read_csv(SPLITS / f"{split}.csv")
    t0 = time.time()
    P = predict(model, df, VARIANTS, dev)
    y, subset = df.label.values, df.subset.values
    m = summarize(y, P, subset)
    eval_s = time.time() - t0

    if args.final:
        row = {"time": time.strftime("%Y-%m-%d %H:%M"), "run_id": run_dir.name, **{k: m[k] for k in
               ["proxy", "auc_main", "auc_iid", "auc_ood", "auc_seen", "auc_unseen", "auc_worst", "worst_variant"]},
               "desc": args.desc}
        pd.DataFrame([row]).to_csv(HOLDOUT_LOG, sep="\t", index=False, mode="a", header=not HOLDOUT_LOG.exists())
        np.savez(run_dir / "holdout_preds.npz", y=y, subset=subset, **P)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
        return

    np.savez(run_dir / "val_preds.npz", y=y, subset=subset, **P)
    guard_fail = [g for g, ok in [("params", params <= GUARDS["max_params_M"]),
                                  ("ckpt_size", ckpt_mb <= GUARDS["max_ckpt_MB"]),
                                  ("throughput", img_s >= GUARDS["min_img_per_s"])] if not ok]
    delta, p_better = 0.0, 1.0
    best = json.loads(BEST.read_text()) if BEST.exists() else None
    if best:
        bp = np.load(REPO / best["preds"], allow_pickle=True)
        BP = {v: bp[v] for v in VARIANTS}
        delta = m["proxy"] - best["proxy"]
        rng = np.random.default_rng(0); wins = 0
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(y), len(y))
            wins += summarize(y, P, subset, idx)["proxy"] > summarize(y, BP, subset, idx)["proxy"]
        p_better = wins / N_BOOT
    if guard_fail: verdict = "REJECT_GUARD:" + ",".join(guard_fail)
    elif best is None: verdict = "BASELINE"
    elif delta > 0 and p_better >= 0.90: verdict = "KEEP"
    elif delta > 0: verdict = "INCONCLUSIVE"
    else: verdict = "DISCARD"
    if verdict in ("BASELINE", "KEEP"):
        BEST.write_text(json.dumps({"run_id": run_dir.name, "proxy": m["proxy"],
                                    "preds": str((run_dir / "val_preds.npz").relative_to(REPO))}, indent=1))

    row = {"time": time.strftime("%Y-%m-%d %H:%M"), "run_id": run_dir.name, "verdict": verdict,
           "proxy": m["proxy"], "delta": delta, "p_better": p_better,
           **{k: m[k] for k in ["auc_main", "auc_iid", "auc_ood", "auc_seen", "auc_unseen", "auc_worst", "worst_variant"]},
           "params_M": params, "ckpt_MB": ckpt_mb, "img_s": img_s, "train_min": train_min,
           "desc": args.desc.replace("\t", " ")}
    pd.DataFrame([row], columns=COLS).to_csv(RESULTS, sep="\t", index=False, mode="a",
                                             header=not RESULTS.exists(), float_format="%.4f")
    (run_dir / "metrics.json").write_text(json.dumps({**row, "per_variant": m["per_variant"]}, indent=1, default=float))
    pv = "  ".join(f"{k}={v:.3f}" for k, v in m["per_variant"].items())
    print(f"[evaluate] per-variant AUC: {pv}")
    print(f"[evaluate] eval {eval_s:.0f}s on {dev}")
    print("RESULT " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))


if __name__ == "__main__":
    main()
