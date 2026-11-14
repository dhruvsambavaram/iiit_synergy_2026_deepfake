"""
Grad-CAM for the submitted ensemble (bonus +2: "Grad-CAM / attention visualizations").

Uses HOLDOUT images only: never test images, and never images the final (train+val) models were trained on.
For each model, the target is the final 7x7 feature map (timm forward_features), and the score is the fake-logit.
CAM = ReLU(sum_c alpha_c * A_c), alpha_c = spatial mean of d(logit)/dA_c (Selvaraju et al., 2017).

    python tools/gradcam.py --weights submission/model_weights.pt --out docs/gradcam
Outputs:
  examples_grid.png   : confident-correct fakes / reals + the most confident errors, per-model + ensemble CAM
  class_mean_cams.png : mean ensemble CAM over all holdout real vs fake images (faces are pre-aligned)
  gradcam_summary.json: which images, probabilities, and per-region attention shares
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common import MEAN, SPLITS, STD, load_rgb  # noqa: E402

SHORT = {"xception41.tf_in1k": "Xception41", "densenet121.ra_in1k": "DenseNet-121", "efficientnet_b0.ra_in1k": "EffNet-B0"}


def device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load_models(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = []
    for s in ck["models"]:
        m = timm.create_model(s["arch"], pretrained=False, num_classes=1, drop_rate=s.get("drop_rate", 0.0))
        m.load_state_dict(s["state_dict"]); out.append((s["arch"], m.eval().to(dev)))
    return out, ck.get("threshold", 0.5)


def to_tensor(img, dev):
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)
    x = (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
    return x.unsqueeze(0).to(dev)


def gradcam(model, x, sign=1.0):
    """Returns (cam 224x224 in [0,1], prob). sign=+1 explains the FAKE score, -1 explains the REAL score.
    Includes hflip TTA: CAM of the flipped view is un-flipped and averaged."""
    cams, probs = [], []
    for flip in (False, True):
        xi = torch.flip(x, dims=[3]) if flip else x
        feats = model.forward_features(xi)
        feats.retain_grad()
        logit = model.forward_head(feats).reshape(-1)[0]
        model.zero_grad(set_to_none=True); (sign * logit).backward()
        w = feats.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * feats).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        if flip: cam = torch.flip(cam, dims=[1])
        cam = cam / (cam.max() + 1e-8)
        cams.append(cam.detach().cpu().numpy()); probs.append(torch.sigmoid(logit).item())
    return np.mean(cams, 0), float(np.mean(probs))


def region_shares(cam):
    """Share of CAM mass in coarse face regions (images are face-aligned, 224x224)."""
    h = cam.shape[0]; tot = cam.sum() + 1e-8
    bands = {"upper (forehead/hair)": (0, int(.30 * h)), "eyes": (int(.30 * h), int(.50 * h)),
             "nose/cheeks": (int(.50 * h), int(.68 * h)), "mouth/chin": (int(.68 * h), int(.88 * h)), "bottom/neck": (int(.88 * h), h)}
    out = {k: float(cam[a:b].sum() / tot) for k, (a, b) in bands.items()}
    cx = cam[:, int(.2 * h):int(.8 * h)].sum() / tot
    out["central 60% width"] = float(cx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(REPO / "submission" / "model_weights.pt"))
    ap.add_argument("--out", default=str(REPO / "docs" / "gradcam"))
    ap.add_argument("--n_mean", type=int, default=150, help="images per class for the class-mean CAM")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = device()
    models, thr = load_models(args.weights, dev)
    df = pd.read_csv(SPLITS / "holdout.csv")

    # 1) ensemble probability for every holdout image (TTA, same as submission)
    rec = []
    for _, r in df.iterrows():
        img = load_rgb(r.path); x = to_tensor(img, dev)
        cams, ps = zip(*[gradcam(m, x, +1.0) for _, m in models])
        if np.mean(ps) < thr:  # predicted REAL -> explain the REAL score instead
            cams, _ = zip(*[gradcam(m, x, -1.0) for _, m in models])
        rec.append({"image_id": r.image_id, "label": int(r.label), "subset": r.subset, "path": r.path,
                    "p": float(np.mean(ps)), "cam": np.mean(cams, 0), "cams": cams, "ps": ps})
    R = pd.DataFrame(rec)
    R["correct"] = (R.p >= thr).astype(int) == R.label

    # 2) example grid: 3 confident fakes, 3 confident reals, 1 worst false-positive, 1 worst false-negative
    pick = (list(R[(R.label == 1) & R.correct].nlargest(3, "p").index) +
            list(R[(R.label == 0) & R.correct].nsmallest(3, "p").index) +
            list(R[(R.label == 0) & ~R.correct].nlargest(1, "p").index) +
            list(R[(R.label == 1) & ~R.correct].nsmallest(1, "p").index))
    names = [SHORT.get(a, a) for a, _ in models]
    names = [n + (f" #{names[:j].count(n) + 1}" if names.count(n) > 1 else "") for j, n in enumerate(names)]
    ncol = 2 + len(models)
    fig, axes = plt.subplots(len(pick), ncol, figsize=(2.2 * ncol, 2.3 * len(pick)))
    for i, k in enumerate(pick):
        r = R.loc[k]; img = np.asarray(load_rgb(r.path))
        tag = ("FAKE" if r.label else "REAL") + (" ✓" if r.correct else " ✗ (error)")
        expl = "explains FAKE score" if r.p >= thr else "explains REAL score"
        axes[i, 0].imshow(img); axes[i, 0].set_title(f"{tag}\nensemble p(fake)={r.p:.2f}", fontsize=8)
        for j, (cam, p) in enumerate(zip(r.cams, r.ps)):
            axes[i, 1 + j].imshow(img); axes[i, 1 + j].imshow(cam, cmap="jet", alpha=0.45)
            axes[i, 1 + j].set_title(f"{names[j]}\np={p:.2f}", fontsize=8)
        axes[i, -1].imshow(img); axes[i, -1].imshow(r.cam, cmap="jet", alpha=0.45)
        axes[i, -1].set_title(f"Ensemble CAM\n({expl})", fontsize=8)
        for a in axes[i]: a.axis("off")
    fig.suptitle("Grad-CAM on HOLDOUT images (never trained on): evidence for the PREDICTED class", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(out / "examples_grid.png", dpi=130); plt.close(fig)

    # 3) class-mean ensemble CAM (aligned faces): where does each class's evidence sit on average?
    rng = np.random.default_rng(0)
    means, shares = {}, {}
    for lab, name in ((0, "real"), (1, "fake")):
        idx = R[R.label == lab].index.values
        idx = rng.choice(idx, min(args.n_mean, len(idx)), replace=False)
        means[name] = np.mean(np.stack(R.loc[idx, "cam"].values), 0)
        means[name + "_img"] = np.mean(np.stack([np.asarray(load_rgb(p), dtype=np.float32) for p in R.loc[idx, "path"]]), 0) / 255
        shares[name] = region_shares(means[name])
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    for a, name in zip(axes, ("real", "fake")):
        a.imshow(means[name + "_img"]); a.imshow(means[name] / means[name].max(), cmap="jet", alpha=0.5)
        a.set_title(f"Mean Grad-CAM, true {name.upper()} images (n={min(args.n_mean, (R.label == (name == 'fake')).sum())})\nevidence for the predicted class", fontsize=9)
        a.axis("off")
    fig.tight_layout(); fig.savefig(out / "class_mean_cams.png", dpi=130); plt.close(fig)

    summary = {"threshold": thr, "n_holdout": len(R), "holdout_acc_at_thr": float(R.correct.mean()),
               "examples": [{k: (R.loc[i, k] if k != "p" else round(R.loc[i, k], 4)) for k in ("image_id", "label", "subset", "p", "correct")} for i in pick],
               "region_shares_mean_cam": shares}
    (out / "gradcam_summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(summary["region_shares_mean_cam"], indent=1))
    print(f"[gradcam] holdout acc@thr {summary['holdout_acc_at_thr']:.4f} | wrote {out}/examples_grid.png, class_mean_cams.png")


if __name__ == "__main__":
    main()
