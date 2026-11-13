"""
prepare.py -- FROZEN. Run ONCE. Builds the train / val / holdout splits.

    python prepare.py --data_dir /path/to/hackathon_dataset

Split design (official training images only; test images are never read):
  1. Near-duplicate grouping via 64-bit dHash (hamming <= 6) -> groups never straddle splits.
  2. ImageNet-ResNet50 features, KMeans(16) per class. Whole clusters are held out as
     "ood" (out-of-distribution) data to simulate unseen generators / capture conditions
     (idea from DeepFakeDefenders 2024 cluster-based hard splits).
  3. Remaining pool split randomly ("iid").
  Result: train ~70% | val = val_iid ~8% + val_ood 2 clusters/class | holdout = 4% iid + 1 cluster/class
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import timm
from sklearn.cluster import KMeans

from common import SPLITS, MEAN, STD, load_rgb, sha256_file, REPO

SEED = 42
K = 16


def dhash(img, size=8):
    g = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return np.packbits(bits)  # 8 bytes


def group_near_dups(hashes, thr=6):
    n = len(hashes)
    bits = np.unpackbits(hashes, axis=1).astype(np.uint8)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    pairs = 0
    for i in range(n):
        d = (bits[i + 1:] != bits[i]).sum(1)
        for j in np.nonzero(d <= thr)[0] + i + 1:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri; pairs += 1
    return np.array([find(i) for i in range(n)]), pairs


@torch.no_grad()
def features(paths):
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    m = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=0).eval().to(dev)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1).to(dev); std = torch.tensor(STD).view(1, 3, 1, 1).to(dev)
    out = []
    for i in range(0, len(paths), 128):
        x = np.stack([np.asarray(load_rgb(p), dtype=np.float32) / 255 for p in paths[i:i + 128]])
        x = (torch.from_numpy(x).permute(0, 3, 1, 2).to(dev) - mean) / std
        f = m(x); out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()
    root = __import__("pathlib").Path(args.data_dir).resolve()
    meta = pd.read_csv(root / "train_metadata.csv")
    meta["path"] = [str(root / "train" / ("fake" if l == 1 else "real") / f"{i}.jpg")
                    for i, l in zip(meta.image_id, meta.label)]
    missing = [p for p in meta.path if not __import__("os").path.exists(p)]
    assert not missing, f"{len(missing)} missing files, e.g. {missing[:3]}"
    print(f"[prepare] {len(meta)} images, label counts {meta.label.value_counts().to_dict()}")

    imgs_hash = np.stack([dhash(load_rgb(p)) for p in meta.path])
    meta["group"], pairs = group_near_dups(imgs_hash)
    print(f"[prepare] near-dup merges: {pairs}, groups: {meta.group.nunique()}")

    feats = features(list(meta.path))
    meta["cluster"] = -1
    for lab in (0, 1):
        idx = np.nonzero(meta.label.values == lab)[0]
        meta.loc[idx, "cluster"] = KMeans(K, random_state=SEED, n_init=10).fit_predict(feats[idx]) + lab * 100
    # a near-dup group takes the cluster of its first member
    meta["cluster"] = meta.groupby("group").cluster.transform("first")

    rng = np.random.default_rng(SEED)
    meta["split"], meta["subset"] = "train", "iid"
    for lab in (0, 1):
        sizes = meta[meta.label == lab].cluster.value_counts()
        n_lab = (meta.label == lab).sum()
        cand = [c for c, s in sizes.items() if 0.03 * n_lab <= s <= 0.09 * n_lab]
        if len(cand) < 3:  # fallback: the clusters closest to median size
            cand = list((sizes - sizes.median()).abs().sort_values().index[:5])
        chosen = rng.choice(cand, 3, replace=False)
        for c, sp in zip(chosen, ["val", "val", "holdout"]):
            meta.loc[meta.cluster == c, ["split", "subset"]] = [sp, "ood"]

    pool = meta[meta.split == "train"]
    groups = pool.groupby("group").label.first()
    for lab in (0, 1):
        g = groups[groups == lab].index.values.copy(); rng.shuffle(g)
        n = (meta.label == lab).sum()
        acc, val_g, hold_g = 0, [], []
        sizes = pool.group.value_counts()
        for gid in g:
            if acc < 0.08 * n: val_g.append(gid)
            elif acc < 0.12 * n: hold_g.append(gid)
            else: break
            acc += sizes[gid]
        meta.loc[meta.group.isin(val_g), "split"] = "val"
        meta.loc[meta.group.isin(hold_g), "split"] = "holdout"

    SPLITS.mkdir(exist_ok=True)
    cols = ["image_id", "label", "path", "subset", "cluster", "group"]
    manifest = {}
    for sp in ("train", "val", "holdout"):
        df = meta[meta.split == sp][cols].reset_index(drop=True)
        f = SPLITS / f"{sp}.csv"; df.to_csv(f, index=False)
        manifest[f.name] = sha256_file(f)
        print(f"[prepare] {sp:8s} n={len(df):5d}  " +
              "  ".join(f"{s}: real={((df.subset==s)&(df.label==0)).sum()} fake={((df.subset==s)&(df.label==1)).sum()}"
                        for s in ("iid", "ood")))
    for f in ("common.py", "evaluate.py", "prepare.py"):
        manifest[f] = sha256_file(REPO / f)
    (SPLITS / "manifest.json").write_text(json.dumps(manifest, indent=1))
    assert not set(meta[meta.split == "train"].group) & set(meta[meta.split != "train"].group)
    print("[prepare] done ->", SPLITS)


if __name__ == "__main__":
    main()
