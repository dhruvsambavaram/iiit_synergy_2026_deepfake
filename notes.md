- r0001 baseline B0: proxy 34.5, val AUC iid 0.81 / ood 0.70, but train AUC 0.99 → heavy overfit (pipeline verified by tools/sanity_gap.py). Collapses on pixelate (0.52) and noise05 (0.58). Epoch-1 loss 1.96 is suspiciously high → try lower lr / longer warmup first.
- 2026-11-13: switched train.py to a fixed 6-min wall-clock budget (Karpathy-style). r0001 used 5 fixed epochs (3.4 min), so FIRST loop run = unchanged config under the time budget (re-baseline), THEN the lr/warmup hypothesis.
- r0002 rebaseline (6-min budget, 9 epochs vs 5): KEEP proxy 36.76 (+2.29). iid 0.826 / ood 0.717. Still worst on pixelate (0.55) and noise05. Next: lr 1e-4 + warmup 1.0.
- r0003 lr 1e-4 + warmup 1.0: DISCARD 31.16 (-5.60). Underfits (final loss 0.31 vs 0.07); high ep-1 loss was just warmup, not instability. Higher lr (1e-3) is the more promising direction; queue after cutout.
- r0004 cutout boxes p=0.5 (1-2 boxes, 10-35% side): DISCARD 35.01 (-1.75). iid flat, ood -0.025; even the cutout variant didn't improve much. Model is underfitting-in-budget rather than lacking occlusion invariance? Next: lr 1e-3 (ladder 1), then JPEG aug.
- r0005 lr 1e-3: KEEP 43.43 (+6.68!). iid 0.877 / ood 0.762, all variants up. lr was the main bottleneck in a 6-min budget. Next: lr 2e-3 to find the edge; retest cutout later (r0004 may have failed because the model was underfit at lr 3e-4).
- r0006 lr 2e-3 seed0: INCONCLUSIVE 43.83 (+0.40, p=0.68), pixelate 0.585→0.697. r0007 seed1 rerun: KEEP 44.80 (+1.36, p=0.91). Seed spread ≈1 pt → best may be slightly seed-lucky (winner's curse); treat small deltas skeptically. Next: lr 4e-3 once to bracket, then move on to JPEG aug / retest cutout.
- r0008 lr 4e-3: KEEP 46.36 (+1.56, p=0.92). ood 0.803 (+0.025), pixelate 0.74. High lr seems to regularise (flatter minima) in this overfit regime. Next: lr 8e-3.
- r0009 lr 8e-3: KEEP 48.35 (+2.00, p=0.97). ood 0.825, pixelate 0.80. Final train loss 0.39 (vs 0.03 at 1e-3), so the overfit is gone. Stable curve. Next: lr 1.2e-2 (smaller step near the edge); if it plateaus, retest cutout/JPEG aug at the new lr.
- r0010 lr 1.2e-2: DISCARD 44.66 (-3.69). Past the edge; lr optimum ≈ 8e-3 for B0/AdamW/bs64 in 6 min.

### Summary after 10 runs (2026-11-13 15:15)
- Best: r0009 (lr 8e-3, effnet-b0, flip+light RRC, 6-min budget), proxy 48.35 (iid 0.876, ood 0.825, seen 0.831, unseen 0.836).
- Biggest wins: lr sweep 3e-4 → 8e-3 (+11.6 total over the rebaseline); the 6-min budget itself gave +2.3 over 5 epochs. Higher lr removed the overfit (train loss 0.03 → 0.39).
- Failed: lr 1e-4 (underfit), cutout at lr 3e-4 (-1.75, may not hold now that the regime has changed), lr 1.2e-2.
- Weakest variants now: pixelate 0.80, gblur2 0.82, then webp30/color/gamma/noise05/gray ≈ 0.83. The variants are now fairly uniform; the ood gap (0.876 vs 0.825) is the bigger target.
- Seed spread ≈ 1 pt.
- Next: retest cutout at lr 8e-3, then JPEG aug, then blur/pixelate aug (aimed at pixelate and gblur2).
