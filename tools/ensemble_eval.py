"""Score an average-of-probabilities ensemble of existing runs on val, using saved val_preds.npz (no retraining).
    python tools/ensemble_eval.py runs/r0009 runs/r0008 [...]
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import summarize, VARIANTS

def load(r):
    z = np.load(Path(r) / "val_preds.npz", allow_pickle=True)
    return z["y"], z["subset"], {v: z[v] for v in VARIANTS}

if __name__ == "__main__":
    runs = sys.argv[1:]
    data = [load(r) for r in runs]
    y, sub = data[0][0], data[0][1]
    for r, (_, _, P) in zip(runs, data):
        m = summarize(y, P, sub); print(f"{r:12s} proxy {m['proxy']:.2f} iid {m['auc_iid']:.4f} ood {m['auc_ood']:.4f} unseen {m['auc_unseen']:.4f} worst {m['auc_worst']:.3f}")
    E = {v: np.mean([P[v] for _, _, P in data], 0) for v in VARIANTS}
    m = summarize(y, E, sub)
    print(f"{'ENSEMBLE':12s} proxy {m['proxy']:.2f} iid {m['auc_iid']:.4f} ood {m['auc_ood']:.4f} unseen {m['auc_unseen']:.4f} worst {m['auc_worst']:.3f} ({m['worst_variant']})")
    if len(data) == 2:
        c = np.corrcoef(data[0][2]["clean"], data[1][2]["clean"])[0, 1]; print(f"prediction correlation (clean): {c:.3f}")
