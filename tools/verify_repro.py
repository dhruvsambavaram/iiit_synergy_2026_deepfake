"""
Check that two predictions.csv files agree (guards the -5 pt "script doesn't reproduce predictions" penalty)
and that a file is a valid submission for a test folder.

    python tools/verify_repro.py predictions_mac.csv predictions_t4.csv [--test_dir /path/to/test/images]
"""
import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b", nargs="?")
ap.add_argument("--test_dir")
args = ap.parse_args()

A = pd.read_csv(args.a)
assert list(A.columns) == ["image_id", "label", "confidence"], A.columns
assert A.image_id.is_unique and A.confidence.between(0, 1).all() and A.label.isin([0, 1]).all()
print(f"[ok] {args.a}: {len(A)} rows, valid format")
if args.test_dir:
    ids = {p.stem for p in Path(args.test_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    missing, extra = ids - set(A.image_id), set(A.image_id) - ids
    print(f"[{'ok' if not missing and not extra else 'FAIL'}] coverage: missing={len(missing)} extra={len(extra)}")
if args.b:
    B = pd.read_csv(args.b)
    m = A.merge(B, on="image_id", suffixes=("_a", "_b"))
    d = (m.confidence_a - m.confidence_b).abs()
    flips = (m.label_a != m.label_b).sum()
    print(f"[{'ok' if d.max() < 1e-2 and flips <= 3 else 'WARN'}] rows matched {len(m)}/{len(A)} | "
          f"max |dconf| {d.max():.2e} | mean {d.mean():.2e} | label flips {flips}")
