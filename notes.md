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
| r0019 | xception41 tf_in1k lr 3e-3 (~3.9 ep / 282 steps) | **51.49** (best single model) | 51.44 (ens doesn't beat it alone) | 0.879 |
| r0009+r0018+r0019 | 3-way | — | 51.58 | — |
- r0019 got verdict KEEP vs r0009 (+3.14, p=1.0), so the harness set best.json → r0019. train.py is restored to B0 per instructions, but later B0 verdicts are now relative to xception; judge B0 runs by delta vs r0009 via ensemble_eval. Xception lr is still climbing → trying 6e-3 (3rd sweep point).
| r0020 | xception41 lr 6e-3 (~3.5 ep) | 50.64 | 50.75 | 0.889 |
| r0019+r0020 | two xception lrs | — | **51.88** (best ensemble so far) | 0.934 |
- Xception lr optimum ≈ 3e-3 (1e-3: 48.99, 3e-3: 51.49, 6e-3: 50.64). Next: seresnext50 lr 1e-3 / 3e-3.
| r0021 | seresnext50 racm lr 1e-3 (~3.2 ep; pixelate 0.67) | 46.85 | 49.37 (w/ r0019: 51.20; r0019+r0020+r0021: 51.72) | 0.838 |
| r0022 | seresnext50 lr 3e-3 (~3.4 ep) | 48.18 | 49.59 (w/ r0019: 51.16; r0019+r0020+r0022: 51.53) | 0.879 |
- seresnext50: best iid of any model (0.895) but weak ood (0.80); it doesn't add to the xception ensembles → dropped. Machine is slowing (eval 244 s, img/s 150 for seresnext — the ≥100 guard is not far off for big models).
| r0023 | convnext_tiny lr 1e-3 (~3.5 ep) | 46.07 | 48.85 (w/ r0019: 50.54; r0019+r0020+r0023: 51.15) | 0.836 |
- convnext_tiny: both lrs weak on ood (0.79/0.77) and it hurts every xception ensemble → dropped for good.
- r0024 EDD consistency on B0 (clean+degraded views, BCE both + 1.0·MSE): 49.31 = +0.96 vs r0009 (harness verdict DISCARD is vs xception r0019, so ignore it). Only 318 steps (half of B0's), so the gain comes despite half the updates. iid 0.881 / ood 0.832. Rerunning seed 1.
| r0024 | B0 consistency (5 ep, 318 steps) | 49.31 | 49.72 (w/ r0019: 51.85; **r0019+r0020+r0024: 51.97**, best) | 0.916 |
- r0025 consistency seed1: 48.81 (+0.46 vs r0009). Ensembles: r0024+r0025 49.94; r0019+r0020+r0025 51.55.
- Verdict fix: since best.json → r0019 (xception), harness verdicts for B0 runs are vs xception and meaningless. I re-ran the same paired bootstrap vs r0009 (scratch script /tmp/ar_scratch/p_vs.py, reads only val_preds, matches harness p-values exactly). r0024 vs r0009: +0.96, p=0.908 → KEEP by the program's rule. **Consistency stays in train.py; B0 base = r0024.** (r0013 degrade aug was p=0.884, just short.) B0 runs from here are judged vs r0024 this way.
- Next (ladder 3, label noise): label smoothing 0.1 on top of consistency.
- r0026/r0027 label smoothing 0.1 on consistency B0: +0.31 (p=0.67) / +0.59 (p=0.83) vs r0024; vs seed-matched r0025 the seed-1 run is +1.09 (p=0.96). Positive on both seeds but not KEEP vs base → reverted. Likely a small real gain; a candidate to fold in for the final stage. Ens r0019+r0020+r0027: 51.79.
- 2026-11-13 ~18:00: HUMAN OVERRIDE — ladder paused; architecture pool fixed (xception41 + B0). Running 3 long-budget runs (not loop experiments, not committed; config recorded in runs/<id>/train.diff).

## Long-budget runs
(not loop experiments; not committed; exact config in runs/<id>/train.diff. acc = clean val, threshold 0.5; Youden acc in brackets)
| run | config | budget | epochs (steps) | proxy | iid | ood | unseen | acc all/iid/ood [Youden] | 6-min counterpart |
|---|---|---|---|---|---|---|---|---|---|
| r0028 | xception41 lr 3e-3 (r0019 cfg) | 30 min | 16 (1193) | 47.17 | 0.862 | 0.818 | 0.833 | 0.809/0.832/0.795 [0.813] | r0019: 4 ep (282), 51.49, 0.893/0.848/0.863, acc 0.826/0.838/0.818 [0.830] |
- r0028: 4× the compute is WORSE on everything (-4.3 proxy; iid -0.031, ood -0.031, unseen -0.030). Final train loss 0.057 → overfits. The short cosine schedule is acting as early stopping, so we are NOT compute-limited at this data size (4.9k images); more epochs need more regularisation. Ens r0019+r0028 51.01 (corr 0.861) < r0019 alone.
| r0029 | B0 consistency lr 8e-3 (r0024 cfg) | 25 min | 18 (1322) | 46.63 | 0.870 | 0.806 | 0.824 | 0.780/0.813/0.759 [0.801] | r0024: 5 ep (318), 49.31, 0.881/0.832/0.843, acc 0.782/0.822/0.757 [0.801] |
- r0029: also worse (-2.68; ood -0.027). Loss drops steadily to 0.13 (the two-view sum, so ≈0.06 per view) → overfits after ~5-8 epochs even with consistency. Ens r0024+r0029 50.02 (corr 0.817).
| r0030 | B0 consistency + OneOf degrade p=0.4 (on the shared base view) | 25 min | 19 (1431) | 48.02 | 0.867 | 0.826 | 0.836 | 0.801/0.816/0.792 [0.802] | r0024: 49.31 (above); r0013 (degrade, no consistency): 8 ep (603), 49.31, 0.886/0.827/0.846, acc 0.799/0.824/0.783 [0.805] |
- r0030: worse than both 6-min counterparts (-1.29), but the extra aug halves the long-budget loss (-1.3 vs -2.7 for r0029); best B0 worst-variant (0.818) and ood acc (0.792). Ens r0024+r0030 50.89 (corr 0.800, the most diverse B0 pair); r0019+r0020+r0030 51.92.
- **Long-budget conclusion: NOT compute-limited.** 4-5× compute lost 1.3-4.3 proxy on all 3 configs; iid AUC drops the most (-0.01 to -0.03). The 6-min budget acts as early stopping. For the final retrain: keep ~6-min-equivalent epochs (xception ≈4 ep, B0 ≈5-8 ep) rather than "full epochs", or add much stronger regularisation before training longer.
- **FINAL-RETRAIN LESSON (human-confirmed):** when retraining on train+val, keep the validated EPOCH counts (xception41 ≈4 ep, B0 ≈5-8 ep; read the exact steps from runs/<id>/train_log.json), not the 6-min wall clock (machine speed varies) and not more epochs (r0028-r0030 show longer = worse).
- Human confirmed: their aug order = Resize → HFlip → ShiftScaleRotate → JPEG → GaussianBlur → ColorJitter → CoarseDropout → Normalize, matching my reimplementation. acc@best-thr is the matching comparison for their same-fold-threshold F1.
- Queue: head-to-head A/B/C → densenet121.ra_in1k lr 1e-3/3e-3 (ens vs r0019) → xception r0019 cfg seeds 2,3,4 → greedy forward ensemble selection (3/4/5 models, ≤1 per run).

## Head-to-head vs their recipe
Their recipe reimplemented from published hyperparameters only (patch: /tmp/ar_scratch/apply_theirs.py; their head swap allowed for A/B only). "Their split" = StratifiedKFold(5, rs=42) fold 0, 5600/1400, **8.4% of val images have a near-dup in train**. Their published Swin-T (best epoch): 0.927 AUC.
| run | model / recipe | split | epochs | train min | AUC | acc@0.5 | acc@best-thr | iid | ood | unseen | worst | proxy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A rs_swin_theirs | swin_tiny, their recipe | theirs (8.4% leak) | 10 | 16.1 | **0.9248** | 0.8886 | 0.8914 | — | — | — | — | — |
| B r0031 | swin_tiny, their recipe | ours | 10 | 14.4 | — | 0.797 (iid 0.838 / ood 0.772) | Youden 0.803 | 0.882 | 0.799 | 0.823 | 0.737 (pixelate) | 46.91 |
| C rs_xception | xception41 lr 3e-3, our recipe (r0019 cfg) | theirs (8.4% leak) | 3 (252 steps) | 6.0 | 0.9154 | 0.8821 | 0.8843 | — | — | — | — | — |
| r0019 (ref) | xception41 lr 3e-3, our recipe | ours | 4 | 6.0 | — | 0.826 (iid 0.838 / ood 0.818) | Youden 0.830 | 0.893 | 0.848 | 0.863 | 0.812 (pixelate) | 51.49 |
- A matches their 0.927 → our reimplementation and data pipeline agree with theirs. B: on our harness their recipe is -4.58 vs r0019, mainly ood (-0.049) and corruption robustness (worst 0.737). Same iid acc (0.838) as r0019 → the gap is generalisation to unseen domains. Ens r0019+r0031 51.31 (corr 0.836) < r0019; r0019+r0020+r0031 51.86; r0019+r0024+r0031 51.96. The Swin ensemble candidate is covered by B → skip a separate Swin run.
- C: on THEIR split our 6-min xception (3 ep) gets 0.915 vs their Swin 0.925 (10 ep, 16 min, 2.7× compute). On OUR harness the ranking flips (r0019 51.49 vs r0031 46.91). Their random split rewards iid fit (8.4% near-dup leak); our ood/corruption harness rewards robustness. Their 0.92s are not comparable to our val numbers.
| r0032 | xception41 r0019 cfg seed 2 (3 ep, 196 steps) | 50.19 | 51.80 | 0.921 |
| r0033 | xception41 r0019 cfg seed 3 (4 ep, 250 steps) | 49.92 | 51.56 | 0.926 |
| r0034 | xception41 r0019 cfg seed 4 (4 ep, 247 steps) | 50.84 | 51.80 | 0.934 |
- Seeds 2-4: mean 50.32 vs r0019 51.49 → r0019 was ~1.2 pt seed-lucky (winner's curse, as suspected). The step count also varies with machine speed (196-282 steps in the same 6 min), another reason the final retrain should use fixed EPOCHS.
| r0035 | densenet121.ra_in1k lr 1e-3 (6 ep, 409 steps) | 48.26 | 51.29 | 0.885 |
| r0036 | densenet121.ra_in1k lr 3e-3 (5 ep, 358 steps) | 50.05 | **51.95** (best pair) | 0.901 |
| r0031 | swin_tiny, their recipe + custom head (NOT submittable: non-timm head) | 46.91 | 51.31 | 0.836 |
- DenseNet-121 ra: lr 3e-3 > 1e-3 (like xception). Best worst-variant of any single model (0.829) and fast (361 img/s, 28 MB).

### Greedy forward selection (2026-11-13 21:00, tools: /tmp/ar_scratch/greedy.py, mean of probs, ≤1 model per run, all runs/ minus r0031)
| k | models | proxy | iid | ood | unseen | worst |
|---|---|---|---|---|---|---|
| 1 | r0019 | 51.49 | 0.893 | 0.848 | 0.863 | 0.812 |
| 2 | + r0036 densenet121 lr3e-3 | 51.95 | 0.894 | 0.854 | 0.866 | 0.838 |
| 3 | + r0014 B0 degrade-aug (seed 1) | **52.08** | 0.898 | 0.852 | 0.866 | 0.834 |
| 4 | + r0034 xception seed 4 | **52.21** | 0.898 | 0.853 | 0.867 | 0.836 |
| 5 | + r0030 B0 long-budget (or r0032 if long runs excluded: 52.20) | 52.22 | 0.897 | 0.854 | 0.868 | 0.843 |
- Caveats: (1) the 5th model adds +0.01, which is noise → stop at 3-4. (2) Selecting from ~35 runs on the same val overfits it: expect the real gain over a single model to be smaller than +0.6-0.7. (3) Without the lucky r0019 seed the best 4-ensemble is r0034+r0036+r0018+r0032 = 51.86, so ~0.35 of the headline is seed luck. Since configs are retrained, pick CONFIGS: xception41 lr3e-3 (≈4 ep) ×2 seeds + densenet121 ra lr3e-3 (≈5 ep) + B0 lr8e-3 with degrade aug (≈8 ep) is the robust reading of this table.
- r0037 mixup α=0.2 (same pair/λ for both views) on consistency B0: 49.44, +0.13 vs r0024 (p=0.63) → reverted. Ens r0019+r0036+r0014+r0037 51.99 (< 52.08 without it). Loop resumed after the human-requested queue. Next: consistency_weight 3.0.
- r0038 consistency_weight 3.0: 49.94, +0.63 vs r0024 (p=0.892) → seed-1 rerun r0039: 50.24, **+0.93 vs r0024 (p=0.950) → KEEP**; vs seed-matched r0025 +1.43 (p=0.998). Both seeds positive; iid 0.895/0.888, ood 0.826/0.835, worst 0.810/0.827. **B0 base = r0039 (consistency_weight 3.0).** B0 best 50.24, still below xception r0019 51.49. Ens r0019+r0036+r0014+r0039 52.01. Next: bracket with weight 6.0.
- r0040 consistency_weight 6.0: 47.99, -2.24 vs r0039 (p=0.00) → reverted. Optimum ≈3.0 (1.0: 49.31/48.81, 3.0: 49.94/50.24, 6.0: 47.99). Only 277 steps (machine slow, 443 img/s), but -2.2 is too large to be that alone.

### Summary after 40 runs (2026-11-13 21:35)
- Best single model: r0019 xception41 lr 3e-3, 51.49 (seed-lucky: seeds 2-4 average 50.32). Best B0: r0039 (lr 8e-3, consistency w=3.0), 50.24. Best extra backbone: densenet121.ra lr 3e-3 (r0036), 50.05.
- Best ensemble (greedy, see above): r0019+r0036+r0014+r0034 = 52.21; 3 models 52.08.
- Biggest wins: lr (B0 3e-4→8e-3: +11.6; xception 1e-3→3e-3: +2.5), 6-min budget over 5 fixed epochs (+2.3), consistency training (+1 on B0), backbone diversity in the ensemble (+0.5-0.7).
- What didn't help: cutout (×2), JPEG aug, mixup, longer training (all 3 configs), convnext_tiny, seresnext50, effnet_b2 (fine, but redundant with B0), Swin with their recipe (46.9).
- Near misses (positive on both seeds, never KEEP): OneOf degrade aug (+0.96/+0.35), label smoothing 0.1 (+0.31/+0.59).
- Weakest variants (r0019): pixelate 0.81, then noise05/webp30 ≈0.84; ood gap (iid 0.893 vs ood 0.848) is still the biggest lever.
- Next: OneOf degrade aug p=0.4 on the r0039 base (single change); then label smoothing 0.1 on the new base.
