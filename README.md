# Attention Effects in Chest X-Ray Classification: A Multi-Seed Benchmark of SE and CBAM Under Class Imbalance

Code, analysis outputs and figures for the manuscript of the same title (under review at *IEEE Access*).

**Authors:** Fariha Binta Sadeque, Farzana Islam Etee, Maliha Mostafa, Mohammad Mehedi Hasan Munna, Reefka Fabliha Tulona, Acramul Haque Kabir, Ashraful Islam.
**Corresponding author:** Reefka Fabliha Tulona (reefka@iub.edu.bd).

---

## What the study does

We test whether Squeeze-and-Excitation (SE) and the Convolutional Block Attention Module (CBAM) improve four-class chest X-ray classification (COVID-19, Normal, Pneumonia, Tuberculosis).

- **Design.** Four backbones (ResNet50, DenseNet121, MobileNetV2 and a custom CNN trained from scratch) × three attention settings (none, SE, CBAM) × two data conditions (imbalanced with weighted cross-entropy, and balanced). Each of the 24 configurations is trained with five seeds.
- **Additional runs.**
  - Loss-versus-data ablations.
  - Fine-tuning sensitivity runs.
  - Five robustness variants: lung-field cropping, patient-grouped validation, multi-depth attention, chest-radiograph pretraining and downsample-only training.
  - **215 runs in total.**
- **Statistics.** Effects are estimated on per-image test predictions with a paired hierarchical bootstrap over test images and seeds (B = 2,000), with Holm correction within families of contrasts. Pooled effects and equivalence tests (TOST) are also reported.
- **Data audit.** The dataset is audited for duplicates, leakage and acquisition-source shortcuts.

### Main findings (macro F1)

| Quantity | Result |
|---|---|
| 16 main attention effects | none with a 95% CI excluding zero; range −0.011 to +0.016 |
| Pooled main effect | +0.0001 (95% CI −0.0072 to +0.0066); equivalent to zero within ±0.01 (TOST p = 0.005) |
| Individual upper 95% limits | all below +0.05 (largest +0.047) |
| SE vs CBAM | no difference on any backbone |
| Pooled robustness effect (8 effects) | +0.0070 (95% CI +0.0011 to +0.0137); equivalent within ±0.02, not ±0.01 |
| Shortcut probes, never seeing the lung fields | file metadata 0.72; image border only 0.66 |

Full numbers are in [`results/`](results/). The section [Where each paper result comes from](#where-each-paper-result-comes-from) maps every table and figure to its source file.

---

## Data

| Dataset | Use | Source |
|---|---|---|
| Chest X-Ray (Pneumonia, Covid-19, Tuberculosis) | all experiments; predefined train / val / test split kept | Kaggle: [`jtiptj/chest-xray-pneumoniacovid19tuberculosis`](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis) |
| Balanced training set (2,000 images per class) | balanced condition | Kaggle: [`malihamostafa/balanceddatasetctbp`](https://www.kaggle.com/datasets/malihamostafa/balanceddatasetctbp) |

No images are redistributed here. File-level outputs refer to images by their relative path in the Kaggle dataset.

---

## Repository layout

```
notebooks/            the eight Kaggle notebooks, in run order (self-contained; model code shared verbatim)
results/
  main_analysis/      notebook 03 – main tables, labelled effects, per-class results, checkpoints, ablations
  calibration/        notebook 06 – ECE / Brier / NLL tables and attention effects
  lung_crop/          notebook 07 – lung bounding boxes, crop QC, shortcut probes on crops
  round3/             notebook 08 – pooled estimates, equivalence tests, replicate counts, probe CIs,
                      collapsed-run and MobileNetV2 sensitivity, per-seed table, revised figures
  supplementary/      CSV versions of Supplementary Tables S1, S4, S5 and S6
scripts/              seed_level_crosscheck.py – paired t and sign-flip permutation tests (Table S6)
figures/              figures as they appear in the manuscript
```

---

## Reproducing the results

All notebooks were run on Kaggle; training used one NVIDIA Tesla T4. Each notebook finds its inputs under `/kaggle/input/` and writes to `/kaggle/working/`.

| # | Notebook | Hardware | Inputs | Produces |
|---|---|---|---|---|
| 01 | `01_data_audit` | CPU, ~30 min | original + balanced datasets | duplicate / leakage report, test-set exclusion lists, source statistics, shortcut probes |
| 07 | `07_lungcrop_prep` | GPU ~5 min (CPU ~40 min) | original dataset | `lung_boxes.csv` (needed by the lung-crop runs), crop QC, probes on crops |
| 02 | `02_train_unified` | GPU (T4), several 12-h sessions | both datasets (+ 07 output for lung-crop runs) | `results/<run_id>/{metrics.json, preds.csv, history.csv}` for every run |
| 03 | `03_stats_analysis` | CPU, ~5 min | 02 outputs | `results/main_analysis/` |
| 04 | `04_compute_cost` | CPU (GPU optional) | 02 outputs (training time) | parameters, GMACs, latency, training time |
| 05 | `05_gradcam_quant` | GPU, ~1 h | original dataset + 02 outputs | Grad-CAM localisation tables and figures |
| 06 | `06_calibration` | CPU, ~5 min | 02 outputs | `results/calibration/` |
| 08 | `08_revision_round3` | CPU, ~15 min | 02, 05 and 07 outputs + original dataset | `results/round3/` |

How the training notebook behaves:
- **Resumable queue.** Notebook 02 skips finished runs when the previous version's output is attached, so it can be continued across sessions.
- **Fixed settings.** Training uses five seeds (42, 0, 1, 2, 3), deterministic cuDNN kernels and FP16 automatic mixed precision. The learning rates, scheduler, batch size, epoch budget and augmentation were fixed in advance; no hyperparameter search was done.
- **Internet access.** The chest-radiograph-pretraining runs and notebook 07 need internet access on Kaggle, because TorchXRayVision downloads its weights.

Python packages: see [`requirements.txt`](requirements.txt).

---

## Where each paper result comes from

| Manuscript item | Source file(s) |
|---|---|
| Table 6 (all 24 configurations) | `results/main_analysis/table4_mean_sd_ci.csv` |
| Table 7, Fig. 4 (attention effects) | `results/main_analysis/effects_labeled.csv` (family `ATTENTION`) |
| Table 8 (SE − CBAM) | `effects_labeled.csv` (`SE-vs-CBAM`), `table6_se_vs_cbam.csv` |
| Table 9 (balancing, interaction) | `effects_labeled.csv` (`BALANCING`, `INTERACTION`) |
| Table 10 (loss-versus-data ablation) | `ablation_loss_vs_data.csv`; `effects_labeled.csv` (`ABLATION`) |
| Tables 11–12, Fig. 5, Fig. 6 (per-class, triage, confusion, ROC) | `per_class.csv`, `tb_covid_recall.csv`, `triage_operating_point.csv`, `confusion_matrices_*.png`, `roc_abnormal_vs_normal.png` |
| Checkpoint selection (§IV-E) | `checkpoint_best_vs_final.csv`, `checkpoint_by_validation_scheme.csv`, `selection_bias.csv`, `rank1_probability.csv` |
| Fine-tuning sensitivity (§IV-C) | `sensitivity_variants.csv` |
| Table 13 (robustness) | `effects_labeled.csv` (`ATTENTION-REVISION`), `sensitivity_variants.csv`, `results/lung_crop/` |
| Table 14, Fig. 9 (calibration) | `results/calibration/` |
| Pooled estimates, equivalence tests (§IV-B, §IV-G; Suppl. Tables S4–S5) | `results/round3/pooled_attention.csv`, `equivalence_per_contrast.csv` |
| Replicate counts on each side of zero | `results/round3/crossing_counts.csv` |
| Shortcut probes with 95% CIs (§III-A, §IV-G) | `results/round3/shortcut_probes_ci.csv` |
| Collapsed run, MobileNetV2 matched baseline, medians | `collapsed_run_sensitivity.csv`, `mobilenet_matched_baseline.csv`, `seed_medians.csv` |
| Fig. 8 (Grad-CAM energy per class) | `results/round3/fig_gradcam_energy_v2.png`, `gradcam_class_chance.csv`, `gradcam_box_n.csv` |
| Seed-level t / permutation cross-check (Suppl. Table S6) | `python scripts/seed_level_crosscheck.py` → `results/supplementary/table_s6_seed_level_crosscheck.csv` |
| Suppl. Table S1 (every run, both checkpoints) | `results/round3/per_seed_all_runs.csv`, `results/supplementary/table_s1_per_seed_all_runs.csv` |

### Notes for reviewers

- **Checkpoint used.** Headline results use the lowest-validation-loss checkpoint ("best"). Final-epoch results are reported alongside in `checkpoint_best_vs_final.csv` and Table S1.
- **Collapsed run.** `customcnn_se_imbalanced_s1` collapsed (macro F1 0.764). The paper reports five-seed values and gives the analysis without this run as a sensitivity check.
- **Fourth seed.** The attention-learning-rate sensitivity variant has four seeds (0, 1, 2, 42); all four are used.
- **Original single-seed runs.** The original single-seed runs of the first submission (FP32) are not part of these results. All reported runs, including seed 42, were retrained under the protocol above.

---

## Still to be added

The following outputs were produced on Kaggle and will be added to this repository before publication:

- per-image softmax outputs (`preds.csv`) and training histories of all 215 runs;
- data-audit outputs of notebook 01 (duplicate groups, leakage flags, test-set exclusion lists, shortcut-probe tables);
- Grad-CAM per-image and per-seed tables of notebook 05;
- the compute table of notebook 04.

---

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The datasets remain under their original Kaggle licences.

## Citation

If you use this code or these results, please cite the article (citation will be added on publication).
