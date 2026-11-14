"""
Metrics requested by the hackathon README ("Report your AUC-ROC, F1, and threshold selection") computed with
sklearn exactly like evaluation_metric.md's sample scoring code, on:
  * VALIDATION (1,479 imgs): the 4 screening members trained on train only (r0019, r0034, r0036, r0014), no TTA
    -> this is where the threshold was selected
  * HOLDOUT (589 imgs): the submitted members (train+val retrain) with hflip TTA, i.e. exactly the submission
Also writes an ROC figure and docs/metrics_summary.json. Re-uses holdout predictions already computed by
tools/final_holdout.py (use 2) for the ensemble; per-member holdout scores are recomputed for reporting only
(no selection decision is made from them).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, timm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
import evaluate as E
from tools.final_holdout import ProbModel
from common import SPLITS

THR = 0.66
VAL_RUNS = {"Xception41 #1": "r0019", "Xception41 #2": "r0034", "DenseNet-121": "r0036", "EfficientNet-B0": "r0014"}
FINAL = {"Xception41 #1": "xception_s0", "Xception41 #2": "xception_s1", "DenseNet-121": "densenet121", "EfficientNet-B0": "b0_degrade"}

def cls(y, p, t):
    yh = (p >= t).astype(int); tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
    return dict(auc=roc_auc_score(y, p), f1=f1_score(y, yh), acc=accuracy_score(y, yh), precision=precision_score(y, yh),
                recall=recall_score(y, yh), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def best_thr(y, p, kind):
    ts = np.unique(np.round(p, 3)); best, bt = -1, 0.5
    for t in ts:
        yh = p >= t
        v = f1_score(y, yh) if kind == "f1" else (yh[y == 1].mean() - yh[y == 0].mean())
        if v > best: best, bt = v, float(t)
    return bt

def main():
    out = {"threshold_submitted": THR}
    # ---------------- validation
    vp = {k: np.load(REPO / "runs" / r / "val_preds.npz", allow_pickle=True) for k, r in VAL_RUNS.items()}
    y, sub = vp["Xception41 #1"]["y"], vp["Xception41 #1"]["subset"]
    ens = np.mean([v["clean"] for v in vp.values()], 0)
    out["validation"] = {"n": int(len(y)), "ensemble": {"all": cls(y, ens, THR), "iid": cls(y[sub == "iid"], ens[sub == "iid"], THR),
                         "ood": cls(y[sub == "ood"], ens[sub == "ood"], THR)},
                         "members": {k: cls(y, v["clean"], THR) for k, v in vp.items()}}
    out["threshold_selection"] = {"youden_val": best_thr(y, ens, "youden"), "f1max_val": best_thr(y, ens, "f1"),
                                  "val_at": {str(t): cls(y, ens, t) for t in (0.5, THR, best_thr(y, ens, "f1"))}}
    # ---------------- holdout (submitted ensemble, TTA)
    hz = np.load(REPO / "final" / "holdout_use2_preds.npz", allow_pickle=True)
    hy, hs, hens = hz["y"], hz["subset"], hz["clean"]
    out["holdout"] = {"n": int(len(hy)), "ensemble": {"all": cls(hy, hens, THR), "iid": cls(hy[hs == "iid"], hens[hs == "iid"], THR),
                      "ood": cls(hy[hs == "ood"], hens[hs == "ood"], THR)}, "ensemble_at_0.5": cls(hy, hens, 0.5), "members": {}}
    dev = E.device(); df = pd.read_csv(SPLITS / "holdout.csv"); assert (df.label.values == hy).all()
    mem_probs = {}
    for k, name in FINAL.items():
        ck = torch.load(REPO / "runs_final" / name / "model.pt", map_location="cpu", weights_only=False); bk = ck["build_kwargs"]
        m = timm.create_model(bk["arch"], pretrained=False, num_classes=1, drop_rate=bk.get("drop_rate", 0.0)); m.load_state_dict(ck["state_dict"])
        p = E.predict(ProbModel(m.eval().to(dev), True).eval(), df, ["clean"], dev)["clean"]
        mem_probs[k] = p; out["holdout"]["members"][k] = cls(hy, p, THR)
    assert abs(roc_auc_score(hy, np.mean(list(mem_probs.values()), 0)) - roc_auc_score(hy, hens)) < 1e-3  # consistency
    (REPO / "docs" / "metrics_summary.json").write_text(json.dumps(out, indent=1, default=float))

    # ---------------- ROC figure (holdout)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    COL = {"Ensemble (submitted)": "#2a78d6", "Xception41 #1": "#eb6834", "Xception41 #2": "#1baf7a", "DenseNet-121": "#eda100", "EfficientNet-B0": "#e87ba4"}
    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor=SURF); ax.set_facecolor(SURF)
    for k, p in mem_probs.items():
        fpr, tpr, _ = roc_curve(hy, p); ax.plot(fpr, tpr, color=COL[k], lw=1.6, label=f"{k}  AUC {roc_auc_score(hy, p):.3f}")
    fpr, tpr, _ = roc_curve(hy, hens); ax.plot(fpr, tpr, color=COL["Ensemble (submitted)"], lw=2.6, label=f"Ensemble (submitted)  AUC {roc_auc_score(hy, hens):.3f}", zorder=5)
    c = out["holdout"]["ensemble"]["all"]; op = (c["fp"] / (c["fp"] + c["tn"]), c["tp"] / (c["tp"] + c["fn"]))
    ax.scatter(*op, s=70, color=INK, zorder=6, edgecolor=SURF, linewidth=2)
    ax.annotate(f"threshold {THR:.2f}\nTPR {op[1]:.2f}, FPR {op[0]:.2f}", op, xytext=(op[0] + 0.12, op[1] - 0.2), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    ax.set_xlabel("False-positive rate (real called fake)", color=INK2, fontsize=9); ax.set_ylabel("True-positive rate (fakes caught)", color=INK2, fontsize=9)
    ax.set_title("ROC on the untouched holdout (589 images)", color=INK, fontsize=10.5, loc="left")
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"): ax.spines[s_].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8); ax.legend(loc="lower right", fontsize=7.5, frameon=False, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(REPO / "docs" / "figures" / "roc_holdout.png", dpi=170, facecolor=SURF)
    print(json.dumps({"val_ens": out["validation"]["ensemble"]["all"], "hold_ens": out["holdout"]["ensemble"]["all"],
                      "thr": out["threshold_selection"]["youden_val"], "f1max": out["threshold_selection"]["f1max_val"],
                      "hold_members": {k: round(v["auc"], 4) for k, v in out["holdout"]["members"].items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
