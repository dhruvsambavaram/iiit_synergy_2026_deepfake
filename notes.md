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
- r0011 cutout p=0.5 @ lr 8e-3: DISCARD 47.95 (-0.40, p=0.26). iid +0.008 but ood -0.012. Cutout doesn't help at either lr, so drop it. Next: JPEG q U(30,95) p=0.5.
- r0012 JPEG q U(30,95) p=0.5: DISCARD 47.35 (-1.00, p=0.07). jpeg30 is unchanged and ood -0.016. CPU aug also cost ~4% of steps. Generic degradation augs aren't paying off so far; high lr is already regularising. Next: OneOf(blur/noise/downup/pixelate) p=0.4, aimed at pixelate and gblur2.
- r0013 OneOf(blur/noise/downup/pixelate) p=0.4: INCONCLUSIVE 49.31 (+0.96, p=0.88). r0014 seed1: INCONCLUSIVE 48.70 (+0.35, p=0.69). Reverted per the rules, but both seeds were positive (mean +0.66, iid +0.01). Probably a small real gain; worth a combined/longer test in the final stage.

## Ensemble candidates
(ensemble = mean of probabilities with r0009 via tools/ensemble_eval.py; corr = clean-pred correlation)
| run | config | single proxy | ens w/ r0009 | corr |
|---|---|---|---|---|
| r0008 | B0 lr 4e-3 | 46.36 | 49.62 | 0.815 |
| r0013 | B0 lr 8e-3 + OneOf degrade p=0.4 | 49.31 | 50.01 | 0.904 |
| r0015 | effnet_b2 lr 4e-3 (5 ep) | 48.00 | 49.96 | 0.841 |
| r0016 | effnet_b2 lr 8e-3 (4 ep; machine slower, 83 s/ep) | 48.45 | 49.44 | 0.892 |
| r0017 | convnext_tiny in22k-ft lr 5e-4 (only ~3.3 ep / 244 steps → DROPPED per the <4-epoch rule; best iid 0.891, weak ood 0.790) | 46.84 | 49.51 | 0.834 |
| r0018 | xception41 tf_in1k lr 1e-3 (~3.4 ep / 259 steps, 100-120 s/ep, img/s 168) | 48.99 | **50.50** | 0.852 |
- Epoch-cutoff note: xception41 (3.4 ep) is borderline like convnext_tiny (3.3 ep), but its single-model score beats B0 and it gives the best ensemble so far → continuing its lr sweep; convnext gets its lr 1e-3 run too, for consistency.
