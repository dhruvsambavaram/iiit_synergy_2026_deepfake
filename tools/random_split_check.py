"""
Reference measurement: what a conventional random 80/20 (StratifiedKFold) protocol would report.

Trains the CURRENT train.py config on a plain StratifiedKFold(5, shuffle=True, random_state=42) split of all
7,000 labelled images (fold 0 = val; no near-duplicate grouping, no held-out clusters) and reports AUC /
accuracy on that fold, i.e. what a conventional random 80/20 protocol would report, for comparison with our
grouped / held-out-cluster validation.

NOT part of the research loop: the model also trains on our val/holdout images, so its run dir must never be
used for selection or submission.

    python tools/random_split_check.py --run_dir runs_aux/rs_xception [--time_budget_min 6]
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import common  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--time_budget_min", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(common.SPLITS / f"{s}.csv") for s in ("train", "val", "holdout")], ignore_index=True)
    # Deterministic fold assignment: StratifiedKFold over train_metadata.csv in file order.
    meta = pd.read_csv(Path(df.path.iloc[0]).parents[2] / "train_metadata.csv")
    df = meta[["image_id"]].merge(df, on="image_id", how="left", validate="1:1")
    assert df.path.notna().all() and len(df) == 7000
    df["fold"] = -1
    for f, (_, vi) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, df.label)):
        df.loc[vi, "fold"] = f
    tmp = Path(tempfile.mkdtemp(prefix="rs_split_"))
    df[df.fold != 0].to_csv(tmp / "train.csv", index=False)
    val = df[df.fold == 0].reset_index(drop=True)
    leak = val.group.isin(set(df[df.fold != 0].group)).mean()
    print(f"[rs] train {int((df.fold != 0).sum())} / val {len(val)} | val images with a near-dup in train: {leak:.1%}")

    import train  # train.py reads SPLITS / "train.csv" via its module-global SPLITS
    train.SPLITS = tmp
    if args.time_budget_min is not None:
        train.CONFIG["time_budget_min"] = args.time_budget_min
    sys.argv = ["train.py", "--run_dir", args.run_dir] + (["--epochs", str(args.epochs)] if args.epochs else [])
    train.main()

    import evaluate as E
    dev = E.device()
    model, _ = E.load_model(Path(args.run_dir), dev)
    P = E.predict(model, val, ["clean"], dev)["clean"]
    y = val.label.values
    ts = np.unique(np.round(P, 3))
    print(f"[rs] RANDOM-SPLIT fold0: AUC {E.auc(y, P):.4f} | acc@0.5 {((P >= .5) == y).mean():.4f} | "
          f"acc@best-thr {max(((P >= t) == y).mean() for t in ts):.4f}")
