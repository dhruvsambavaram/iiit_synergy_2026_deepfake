# program.md — AutoResearch instructions (deepfake detector)

You are an autonomous ML researcher. Goal: find the deepfake detector (real=0 / fake=1, 224×224 face JPEGs)
that maximises **proxy score** on our frozen validation harness, which mirrors the hackathon scoring:

```
proxy   = 50·(2·auc_main − 1) + 20·(2·auc_rob − 1)
auc_main = ½·AUC(val iid) + ½·AUC(val ood clusters)       # ood = whole feature-clusters held out (unseen-domain proxy)
auc_rob  = 0.4·mean AUC(seen corruptions) + 0.6·mean AUC(unseen corruptions)
```

## Hard rules (violating any = the run is invalid)
1. Edit **only `train.py`**. Never edit `common.py`, `evaluate.py`, `prepare.py`, `run_experiment.sh`, `data_splits/`, `best.json`, `results.tsv`
   (evaluate.py hash-checks the frozen files and aborts).
2. Never read `val.csv`, `holdout.csv`, or anything under the official `test/` folder from train.py. Never use `--full` or `evaluate.py --final`; those are for the human-led final stage.
3. Only ImageNet-pretrained weights (timm `*_in1k`, `*_in22k`, `*_in22k_ft_in1k`). No deepfake-pretrained weights, no external data, no generative synthesis of new training images.
4. Augmentation may use any **seen** corruption type in `common.py` (jpeg, gaussian blur, gaussian noise, down/up-scale,
   brightness/contrast, cutout boxes, pixelation, grayscale) and generic augs (flip, crop, color jitter, mixup/cutmix, erasing).
   It must **not** use or closely imitate **unseen** types: WebP, motion blur, gamma curves, posterize/quantise,
   center-crop-zoom, median filter, the downscale+JPEG "social" chain. Those measure true generalisation.
5. Model contract: `build_model(**build_kwargs)` gets ImageNet-normalised 224×224 input, returns fake-logits. TTA must live inside the model.
   Guards (auto-checked): ≤150M params, ≤600 MB checkpoint, ≥100 img/s inference on this machine.
6. Screening budget (Karpathy-style fixed compute): **6 minutes of training wall-clock** per experiment, so backbones are compared at
   equal compute. Set `CONFIG["time_budget_min"]` (train.py stops at the end of the epoch that crosses it and the cosine schedule is
   sized from a measured epoch time). Never raise it in the loop.
7. Models must stay expressible as `timm.create_model(arch, num_classes=1, drop_rate=…)` + optional TTA flag, because the
   submission `inference_script.py` rebuilds them that way. If you need a custom head, write it in notes.md as a proposal instead.

## The loop
0. Once per session: `git checkout -b autoresearch/<date>-<n>` from the current best commit (keeps runs per session separable).
1. Read `results.tsv` (all rows) and `git log --oneline -20`. Look at `runs/<best>/metrics.json` per-variant AUCs to find the weakest variants / ood gap.
2. Write ONE hypothesis, grounded in the evidence (e.g. "worst variant is gblur2 at 0.81 → add blur aug p=0.3").
   Change one thing at a time unless combining two individually-validated KEEPs.
3. Edit `train.py`, then `git commit -am "<hypothesis>"`.
4. Run `./run_experiment.sh "<hypothesis>" [epochs] [seed]` in the **foreground** (do not poll; it blocks ~5–10 min) and read the `RESULT` line.
5. Verdicts (computed by evaluate.py with a paired bootstrap vs the current best):
   * `KEEP` → leave the commit; it is the new best.
   * `INCONCLUSIVE` (better but p<0.90) → if delta ≥ 0.3, rerun once with seed 1; if still not KEEP, revert.
   * `DISCARD` / `REJECT_GUARD` / `CRASH` → `git revert --no-edit HEAD` (or fix a trivial crash bug once).
   Rejected-but-interesting runs (e.g. a different backbone with good ood AUC) are ensemble candidates: note them in `notes.md`.
6. Append 1–3 lines to `notes.md`: what you tried, what happened, what you'd try next. Then go to 1.
Never stop to ask the human; keep looping until interrupted.

## What we already know (data forensics, 2026-11-13)
* 4,932 train / 1,479 val (561 iid + 918 ood) / 589 holdout. Balanced classes.
* All images re-encoded identically (same JPEG tables, same metadata) → no file-level shortcut.
* The organisers applied degradations to **both** classes: occlusion boxes (gray/black/white), speckle "snow" noise, pixelation,
  grayscale, color casts, low contrast, blur. Degradation ≠ fake. Expect hidden test variants to look like these.
* Some real/fake pairs come from the same source photo (near-duplicate groups are kept in the same split).
* README warns of label noise → label smoothing / robust losses are legitimate hypotheses.
* M4 Pro MPS: do NOT use channels_last (2.4× slower). fp32 is as fast as bf16. EfficientNet-B0 ≈ 41 s/epoch.

## Research ladder (rough priority; evidence overrides it)
1. **Baseline calibration**: lr (1e-4 / 3e-4 / 1e-3), epochs (5 vs 8 is a later full-run question), drop_rate.
2. **Degradation-aware augmentation** (DFDC 1st/3rd/5th place; see github.com/SlongLiu/collection_dfdc). Try in this order, one per run:
   a. Random erasing / cutout boxes filled gray/black/white/noise (p≈0.5) — mirrors the organisers' occlusion boxes.
   b. Random JPEG re-compression, quality U(30, 95), p≈0.5 — JPEG ONLY (WebP is a held-out eval corruption).
   c. Gaussian blur σ∈[0.5,2] / gaussian noise σ∈[0.01,0.05] / down-up-scale 0.4–0.9 / pixelate, p≈0.2–0.3 each.
   d. RandAugment or TrivialAugmentWide with **posterize and solarize removed** (posterize is held-out); small rotate/shear only (faces are aligned).
   e. Grayscale p≈0.1, color jitter.
   Then combine the KEEPs. Watch the train-vs-val gap (tools/sanity_gap.py): baseline is 0.99 train vs 0.74 val AUC.
3. **Regularisation for label noise**: label smoothing 0.05–0.1, mixup/cutmix (soft labels; NTechLab used mixup), stochastic depth.
   (Real/fake *pair* mixup is not worth it: only ~74 same-source pairs exist.)
4. **Consistency training** (EDD-style): loss on clean + degraded views of the same image, plus a KL/MSE term between the two
   predictions; optionally an EMA "mean teacher".
5. **Backbones** (diversity for the final ensemble): convnext_tiny, efficientnet_b2, resnet50, tf_efficientnetv2_s, a small ViT/Swin.
   Note per-variant strengths—an ensemble wants *different* failure modes.
6. **Resolution / head tricks**: higher internal resolution (upsample inside the model), GeM pooling, layer-wise lr decay, freezing the stem for 1 epoch.

## Final stage (human-led, not part of the loop)
Top 2–3 configs → full epochs × 2 seeds → pick by val → retrain with `--full` → **one** holdout check per candidate
(`evaluate.py --final`, max 3 ever) → ensemble → `inference_script.py` → verify on a Colab T4.
