# deepfake-autoresearch

Deepfake face detection for the **Deepfake ML Hackathon (Synergy '26), IIIT Bangalore**. It's built with an
automated, leakage-free research loop in the style of Karpathy's
[AutoResearch](https://github.com/karpathy/autoresearch): a frozen evaluation harness, a single agent-editable
training file, and 55 logged experiments.

**Final submission:** an ensemble of 4 ImageNet-pretrained CNNs (Xception41 ×2, DenseNet-121, EfficientNet-B0) with
horizontal-flip TTA.

| On an untouched holdout (589 images) | |
|---|---|
| AUC-ROC | **0.908** (held-out image clusters 0.893, random images 0.919) |
| AUC under 7 never-trained corruptions | 0.899 |
| Accuracy / F1 at threshold 0.660 | 85.9% / 0.853 |
| NVIDIA T4, 3,000 test images | **65.6 s**, 5.1 GB peak GPU memory |

The full write-up is in [`docs/TECHNICAL_REPORT.pdf`](docs/TECHNICAL_REPORT.pdf) (Markdown source alongside it).

**Team members:** Dhruv Reddy Sambavaram, Sowmya Gupta

## Submission deliverables (per the hackathon README)

| # | Required deliverable | Where |
|---|---|---|
| 1 | `predictions.csv` | `submission/predictions.csv` (also at the top of the Drive folder) |
| 2 | Model weights | `model_weights.pt`, at the top of the Drive folder (245 MB, too large for this zip). Rebuild with `tools/final_retrain.sh` + `tools/pack_submission.py` |
| 3 | `inference_script.py` | `submission/inference_script.py` |
| 4 | `requirements.txt` | `submission/requirements.txt` |
| 5 | Source code with `README.md` | this repository (`README.md` = this file) |
| 6 | Technical report | `docs/TECHNICAL_REPORT.pdf` (Markdown source: `docs/TECHNICAL_REPORT.md`) |
| 7 | Training logs / loss curves (optional) | `results.tsv`, `notes.md`, `holdout_log.tsv`, `runs/*/train_log.json`, `runs_final/`, `docs/figures/progress.png`, `final/t4_report.txt` |
| 8 | Grad-CAM visualizations (optional, bonus) | `docs/gradcam/` |

---

## 1. Run the submission (what the judges run)

```bash
pip install -r submission/requirements.txt        # Python 3.10, PyTorch 2.x
python submission/inference_script.py \
    --test_dir ./test/images/ \
    --weights  ./submission/model_weights.pt \
    --output   predictions.csv
```
- The script picks its device automatically: CUDA (fp16 autocast) → Apple MPS → CPU.
- No hardcoded paths. It writes `image_id,label,confidence`, where confidence = P(fake).
- To check a re-run against the submitted file:
  `python tools/verify_repro.py predictions.csv submission/predictions.csv --test_dir ./test/images/`
  (T4 vs our machine: 0 label flips).

`model_weights.pt` is a single dict: `{"format": "dfk-ensemble-v1", "models": [{arch, drop_rate, state_dict}, ...],
"tta_hflip": True, "threshold": 0.66}`. Every member is a plain `timm.create_model(arch, num_classes=1)`.

---

## 2. Repository layout

```
prepare.py            FROZEN  builds the leakage-free train/val/holdout splits (run once)
common.py             FROZEN  image loading + the 18-corruption robustness suite
evaluate.py           FROZEN  scores a run: AUC on iid/ood val + every corruption, guards, paired-bootstrap keep/discard
run_experiment.sh     FROZEN  one experiment = train.py + evaluate.py; appends a row to results.tsv
train.py              AGENT   the only file the research agent edits (model, augmentation, optimizer, loss)
program.md                    the agent's research instructions and rules
data_splits/                  frozen split CSVs + SHA-256 manifest (evaluate.py refuses to run if anything changed)
results.tsv                   one row per experiment (55 runs): verdict, proxy, all AUCs, size, speed, description
notes.md                      the agent's and the humans' research log: every hypothesis, result and decision
holdout_log.tsv               the final holdout checks (2 of the 3 allowed uses spent)
runs/rXXXX/                   per-experiment train.py snapshot, diff, train log, metrics, val predictions
runs_final/                   the 4 final members retrained on train+val (logs + snapshots)
submission/                   inference_script.py, requirements.txt, model_weights.pt, predictions.csv
tools/                        final-stage and analysis tools (see below)
docs/                         technical report, Grad-CAM figures, progress figure, team Q&A
```

| Tool | Purpose |
|---|---|
| `tools/final_retrain.sh` | Retrain the 4 ensemble members on train+val at fixed epochs, from the exact saved snapshots |
| `tools/pack_submission.py` | Pack runs into one weights file (TTA flag + Youden threshold from val) |
| `tools/final_holdout.py` | Score an ensemble on the holdout exactly as submitted (counts as a holdout use; max 3) |
| `tools/ensemble_eval.py` | Score any combination of runs from saved val predictions (no retraining) |
| `tools/random_split_check.py` | Train the current config on a plain StratifiedKFold fold, for comparison with random-split numbers |
| `tools/gradcam.py` | Grad-CAM on holdout images for every member and for the ensemble |
| `tools/verify_repro.py` | Format/coverage check and run-to-run comparison of prediction files |
| `tools/plot_progress.py`, `tools/build_report.py` | Report figure and Markdown → PDF |
| `tools/sanity_gap.py` | Train-vs-val AUC gap diagnostic |

---

## 3. Reproduce from scratch

```bash
# 0. environment (Python 3.10)
pip install -r submission/requirements.txt scikit-learn matplotlib markdown pymupdf   # extras: splits, figures, report

# 1. frozen splits (deterministic, seed 42): near-duplicate grouping + per-class K-means held-out clusters
python prepare.py --data_dir /path/to/hackathon_dataset

# 2. (optional) the research loop: an agent follows program.md, running e.g.
./run_experiment.sh "hypothesis text" [epochs] [seed]

# 3. final members on train+val, from the archived snapshots of runs r0019, r0034, r0036, r0014
./tools/final_retrain.sh

# 4. pack + check on holdout + predict
python tools/pack_submission.py --runs runs_final/xception_s0 runs_final/xception_s1 runs_final/densenet121 runs_final/b0_degrade \
       --tta --threshold_from runs/r0019 runs/r0034 runs/r0036 runs/r0014 --out submission/model_weights.pt
python tools/final_holdout.py --runs runs_final/* --tta --threshold 0.66 --desc "my check"
python submission/inference_script.py --test_dir /path/to/test/images --weights submission/model_weights.pt --output predictions.csv
```
Set `PY=/path/to/python` for the shell scripts if `python` isn't the right interpreter. The split CSVs store absolute
image paths from our machine; re-running `prepare.py` with your `--data_dir` regenerates them.

---

## 4. Method in brief

- **Validation that can't flatter us:**
  - near-duplicates (including real/fake pairs from the same source photo) never cross splits;
  - half of val consists of **whole held-out look-alike clusters**, a stand-in for unseen generators;
  - every candidate is also scored under **18 corruptions**, 7 of which are forbidden as augmentation.
- **AutoResearch loop:**
  - a fixed 6-minute training budget per experiment (equal compute for every model);
  - the frozen harness, hash-checked;
  - KEEP only if the proxy score improves with paired-bootstrap p ≥ 0.90.
- **Key findings:**
  - learning rate was the main lever (B0: 3e-4 → 8e-3, +11.6 proxy);
  - short training acts as early stopping (4–5× longer training was worse for every config);
  - high random-split AUC doesn't imply generalization (ConvNeXt, SE-ResNeXt and Swin were strong on iid but weak on
    held-out clusters);
  - one degradation augmentation helped; most others didn't.
- **Selection:** greedy ensemble selection on val, retraining of *configurations* on train+val, and confirmation on
  the untouched holdout.

Rules followed: only the provided data; ImageNet-pretrained weights only (no JFT/DINO/deepfake-pretrained weights);
test images used only for producing predictions (never trained on, selected on, or viewed).

**Credits:** AutoResearch pattern by A. Karpathy. Ideas from the DFDC 2020 solutions (Seferbekov, NTechLab),
DeepFakeDefenders 2024 (cluster-based splits) and EDD 2026 (consistency training).
