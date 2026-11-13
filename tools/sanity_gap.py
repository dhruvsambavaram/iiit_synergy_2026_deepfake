"""Diagnostic: train-subset AUC vs val AUC for a run (checks pipeline vs generalisation)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as E
if __name__ == "__main__":
    run = Path(sys.argv[1]); dev = E.device(); model, _ = E.load_model(run, dev)
    tr = pd.read_csv(E.SPLITS / "train.csv").sample(800, random_state=0)
    P = E.predict(model, tr, ["clean"], dev)
    print("train-subset AUC:", round(E.auc(tr.label.values, P["clean"]), 4))
    v = np.load(run / "val_preds.npz", allow_pickle=True)
    print("val clean AUC:", round(E.auc(v["y"], v["clean"]), 4),
          "| mean p(fake) real/fake:", v["clean"][v["y"] == 0].mean().round(3), v["clean"][v["y"] == 1].mean().round(3))
