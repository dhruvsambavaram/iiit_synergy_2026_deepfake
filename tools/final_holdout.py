"""
HUMAN-LED FINAL STAGE. Scores an ENSEMBLE (mean of probabilities, optional hflip TTA, exactly as the submission
inference_script.py does) on the untouched HOLDOUT split, clean + all corruption variants.

One invocation = one holdout use (shared budget with evaluate.py --final: max 3 rows in holdout_log.tsv, ever).

    python tools/final_holdout.py --runs runs/r0019 runs/r0034 runs/r0036 runs/r0014 --tta --desc "ens4 train-only"
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import evaluate as E  # noqa: E402
from common import SPLITS, VARIANTS  # noqa: E402

HOLDOUT_LOG = REPO / "holdout_log.tsv"
MAX_USES = 3


class ProbModel(nn.Module):
    """Returns logit(mean prob over flip views) so evaluate.predict's sigmoid recovers the TTA probability."""
    def __init__(self, m, tta):
        super().__init__(); self.m, self.tta = m, tta

    def forward(self, x):
        p = torch.sigmoid(self.m(x).float().reshape(-1))
        if self.tta:
            p = 0.5 * (p + torch.sigmoid(self.m(torch.flip(x, dims=[3])).float().reshape(-1)))
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))


def load(run, dev, tta):
    ck = torch.load(Path(run) / "model.pt", map_location="cpu", weights_only=False)
    bk = ck["build_kwargs"]
    m = timm.create_model(bk["arch"], pretrained=False, num_classes=1, drop_rate=bk.get("drop_rate", 0.0))
    m.load_state_dict(ck["state_dict"])
    return ProbModel(m.eval().to(dev), tta).eval(), bk["arch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--desc", required=True)
    ap.add_argument("--threshold", type=float, default=None, help="report accuracy at this threshold too")
    args = ap.parse_args()
    E.verify_frozen()
    uses = len(pd.read_csv(HOLDOUT_LOG, sep="\t")) if HOLDOUT_LOG.exists() else 0
    if uses >= MAX_USES:
        sys.exit(f"[final] holdout already used {uses} times. Refusing.")
    print(f"[final] holdout use {uses + 1}/{MAX_USES}: {args.desc}")

    dev = E.device()
    df = pd.read_csv(SPLITS / "holdout.csv")
    y, sub = df.label.values, df.subset.values
    per = []
    t0 = time.time()
    for r in args.runs:
        model, arch = load(r, dev, args.tta)
        P = E.predict(model, df, VARIANTS, dev)
        m = E.summarize(y, P, sub)
        per.append(P)
        print(f"  {r:22s} {arch:26s} proxy {m['proxy']:.2f} iid {m['auc_iid']:.4f} ood {m['auc_ood']:.4f} "
              f"unseen {m['auc_unseen']:.4f} worst {m['auc_worst']:.3f} ({m['worst_variant']})")
        del model
    EP = {v: np.mean([P[v] for P in per], 0) for v in VARIANTS}
    m = E.summarize(y, EP, sub)
    c = EP["clean"]
    acc05 = float(((c >= 0.5) == y).mean())
    acc_thr = float(((c >= args.threshold) == y).mean()) if args.threshold is not None else None
    print(f"  {'ENSEMBLE':22s} {'':26s} proxy {m['proxy']:.2f} iid {m['auc_iid']:.4f} ood {m['auc_ood']:.4f} "
          f"unseen {m['auc_unseen']:.4f} worst {m['auc_worst']:.3f} ({m['worst_variant']})")
    print(f"  clean AUC (all holdout) {E.auc(y, c):.4f} | acc@0.5 {acc05:.4f}"
          + (f" | acc@{args.threshold:.3f} {acc_thr:.4f}" if acc_thr is not None else "") + f" | {time.time() - t0:.0f}s")

    out = REPO / "final"; out.mkdir(exist_ok=True)
    tag = f"holdout_use{uses + 1}"
    np.savez(out / f"{tag}_preds.npz", y=y, subset=sub, **EP)
    (out / f"{tag}_metrics.json").write_text(json.dumps(
        {"desc": args.desc, "runs": args.runs, "tta": args.tta, "per_variant": m["per_variant"],
         **{k: m[k] for k in ["proxy", "auc_main", "auc_iid", "auc_ood", "auc_seen", "auc_unseen", "auc_worst", "worst_variant"]},
         "acc@0.5": acc05, "acc@thr": acc_thr, "threshold": args.threshold}, indent=1, default=float))
    row = {"time": time.strftime("%Y-%m-%d %H:%M"), "run_id": "+".join(Path(r).name for r in args.runs),
           **{k: m[k] for k in ["proxy", "auc_main", "auc_iid", "auc_ood", "auc_seen", "auc_unseen", "auc_worst", "worst_variant"]},
           "desc": args.desc}
    pd.DataFrame([row]).to_csv(HOLDOUT_LOG, sep="\t", index=False, mode="a", header=not HOLDOUT_LOG.exists(), float_format="%.4f")


if __name__ == "__main__":
    main()
