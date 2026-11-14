"""Figure for the report: proxy score per AutoResearch experiment, with running best. Output: docs/figures/progress.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GRAY = "#2a78d6", "#a3a29c"

df = pd.read_csv(REPO / "results.tsv", sep="\t")
df = df[df.verdict != "CRASH"].reset_index(drop=True)
df["n"] = df.run_id.str[1:].astype(int)
keep = df.verdict.isin(["KEEP", "BASELINE"])
best = df.proxy.where(keep).cummax().ffill()

fig, ax = plt.subplots(figsize=(8.6, 4.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.scatter(df.n[~keep], df.proxy[~keep], s=40, facecolor=SURFACE, edgecolor=GRAY, linewidth=1.5, zorder=3,
           label="Not adopted (discarded, inconclusive, or comparison run)")
ax.scatter(df.n[keep], df.proxy[keep], s=56, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4, label="Kept (new best)")
ax.step(df.n, best, where="post", color=BLUE, linewidth=2, zorder=2, label="Running best")

notes = [(5, "lr 3e-4 → 1e-3\n(+6.7)", (12.5, 40.0)), (9, "lr 8e-3\nbest B0", (9.5, 52.5)), (19, "Xception41\nlr 3e-3", (16, 54.5)),
         (28, "25–30 min runs:\nlonger = worse", (27.5, 41.5)), (31, "teammate recipe\n(Swin, 10 ep)", (34, 43.0)),
         (47, "lr 6e-3 + consistency\n(collapse)", (44, 31.0))]
for n, txt, xy in notes:
    y = float(df.loc[df.n == n, "proxy"].iloc[0])
    ax.annotate(txt, xy=(n, y), xytext=xy, fontsize=7.5, color=INK2, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xlabel("Experiment number (run id)", color=INK2, fontsize=9)
ax.set_ylabel("Validation proxy score (max 70)", color=INK2, fontsize=9)
ax.set_title("AutoResearch: 55 experiments on the frozen validation harness", color=INK, fontsize=11, loc="left")
ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=8)
ax.set_ylim(25, 57)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 0.02), fontsize=7.5, frameon=False, labelcolor=INK2)
fig.tight_layout()
out = REPO / "docs" / "figures"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "progress.png", dpi=160, facecolor=SURFACE)
print("wrote", out / "progress.png")
