# OSIC Pulmonary Fibrosis Progression — Literature vs. Internal Model Comparison

> **Mandatory Cohort & Metric Disclaimer:** *Progression results are exploratory, separate cohort, no IPF labels.*  
> **NOT Leaderboard Scores:** All PulmoSight internal metrics below are computed via **5-Repeat x 5-Fold Nested GroupKFold by Patient on OSIC train.csv future visits (NOT Kaggle leaderboard)** on `backend/data/osic/train.csv` (`n = 175` CT patients, `1359` future visits). They are **internal cross-validation scores** and are **NOT** directly comparable to Kaggle OSIC public/private leaderboard scores evaluated on the hidden competition test set.

---

## 1. Competition Metric Definition (Modified Laplace Log-Likelihood)

All internal models are evaluated with the exact Kaggle OSIC Pulmonary Fibrosis Progression metric:

$$\sigma_{\text{clipped}} = \max(\sigma,\, 70), \qquad \Delta = \min\left(|\text{FVC}_{\text{true}} - \text{FVC}_{\text{pred}}|,\, 1000\right)$$

$$\text{Laplace LL} = -\frac{\sqrt{2}\,\Delta}{\sigma_{\text{clipped}}} - \ln\left(\sqrt{2}\,\sigma_{\text{clipped}}\right)$$

- **Higher (less negative) is better** (e.g., `-6.68` is better than `-6.83`).
- To estimate prediction uncertainty $\sigma$ without leakage: the **Quantile-Regression Neural Network (Tabular)** predicts `(q20, q50, q80)` jointly with Pinball Loss and calibrates spread strictly on an internal training-fold patient validation split; all other tabular and multimodal models fit an in-fold horizon uncertainty model $\sigma(\Delta t) = \max(70, a + b\,\Delta t)$ strictly on inner out-of-fold (OOF) training predictions.
- **Prediction Bounds & CT Branch Design:** All predicted FVC values are winsorized to the physiologically plausible range `[500, 6500]` mL. The HRCT-only, Early Fusion, and Image + Tabular branches predict change from baseline FVC (`Delta_FVC = Target_FVC - Baseline_FVC`) using in-fold 1st–99th percentile winsorization on HU/volume features, in-fold 15D ResNet-18 PCA, and strong L2 regularization, then reconstruct `Baseline_FVC + Delta_FVC`.

---

## 2. Published Approaches vs. PulmoSight Internal CV Results

### 2A. Published OSIC Competition & Literature Benchmarks

| Approach / System | Evaluation Split / Protocol | Documented Approach Details | Reported Modified Laplace LL | Source |
| :--- | :--- | :--- | :---: | :--- |
| **Kaggle 1st Place** | Kaggle OSIC Private Leaderboard | EfficientNet-B5 (CT slices plus tabular) blended with a quantile-regression neural network on tabular data | **`-6.8305`** | [1st-Place GitHub Repo](https://github.com/artkulak/osic-pulmonary-fibrosis-progression) · [Kaggle Competition](https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression) |
| **Kaggle 2nd Place** | Kaggle OSIC Private Leaderboard | — | **`-6.8311`** | [Kaggle Competition Leaderboard](https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression) |
| **Kaggle 3rd Place** | Kaggle OSIC Private Leaderboard | — | **`-6.8336`** | [Kaggle Competition Leaderboard](https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression) |
| **Fibrosis-Net** | Published paper benchmark (`arXiv:2103.04008`) | Deep convolutional neural network + clinical metadata | **`-6.8188`** | [arXiv:2103.04008 (PDF)](https://arxiv.org/pdf/2103.04008) |
| **Fibro-CoSANet** | Reported score, evaluation protocol unclear | Reported score, evaluation protocol unclear | **`-6.68`** | Reported literature score (protocol unclear) |

### 2B. PulmoSight Internal Models (`5×5` Nested `GroupKFold` CV on `train.csv` — NOT Leaderboard)

Every model below was trained and evaluated on the **exact same 25 outer patient-grouped folds** (`5` repeats × `5` folds, `n = 175` patients, `1359` future visits). All scalers, feature winsorizers, PCA projections (`15D`), hyperparameters, and uncertainty $\sigma$ parameters were fitted **strictly inside training folds**, and all FVC predictions were winsorized to `[500, 6500]` mL.

| PulmoSight Internal Model | Modalities | Modified Laplace LL (95% CI) | FVC RMSE mL (95% CI) | FVC MAE mL (95% CI) |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline (Constant Baseline FVC)** | Baseline FVC only | `-6.8666 ± 0.0504 (95% CI: -6.8864–-6.8469)` | `262.646 ± 10.434 (95% CI: 258.556–266.736)` | `183.361 ± 8.324 (95% CI: 180.098–186.624)` |
| **Ridge** **(Best Internal CV)** | Clinical tabular | `-6.8063 ± 0.0932 (95% CI: -6.8428–-6.7698)` | `232.578 ± 9.776 (95% CI: 228.746–236.410)` | `168.507 ± 13.433 (95% CI: 163.241–173.773)` |
| **ElasticNet** | Clinical tabular | `-6.8152 ± 0.0841 (95% CI: -6.8482–-6.7823)` | `233.671 ± 8.641 (95% CI: 230.284–237.059)` | `169.829 ± 12.215 (95% CI: 165.041–174.617)` |
| **Quantile-Regression NN (Tabular)** | Clinical tabular (Pinball MLP) | `-6.9341 ± 0.1285 (95% CI: -6.9845–-6.8837)` | `240.048 ± 12.060 (95% CI: 235.320–244.775)` | `177.034 ± 16.443 (95% CI: 170.588–183.479)` |
| **Random Forest** | Clinical tabular | `-6.9510 ± 0.0408 (95% CI: -6.9670–-6.9350)` | `272.419 ± 32.252 (95% CI: 259.777–285.062)` | `195.788 ± 11.030 (95% CI: 191.464–200.112)` |
| **SVM (RBF SVR)** | Clinical tabular | `-6.9701 ± 0.0971 (95% CI: -7.0081–-6.9320)` | `305.502 ± 79.151 (95% CI: 274.475–336.529)` | `204.991 ± 21.519 (95% CI: 196.556–213.426)` |
| **XGBoost** | Clinical tabular | `-6.9051 ± 0.0563 (95% CI: -6.9272–-6.8830)` | `262.871 ± 29.642 (95% CI: 251.252–274.491)` | `186.749 ± 8.551 (95% CI: 183.397–190.101)` |
| **Image + Tabular (ResNet-18 PCA 15D)** | Clinical + HU/Vol + ResNet-18 In-Fold PCA 15D (Delta-FVC) | `-6.8100 ± 0.0840 (95% CI: -6.8429–-6.7771)` | `230.786 ± 12.291 (95% CI: 225.968–235.604)` | `168.717 ± 12.802 (95% CI: 163.699–173.735)` |
| **EfficientNet-B5 (End-to-End Fine-Tuning)** | Full 3D CT Slices + Tabular | *Not run* | *Not run* | *Not run* |

> **EfficientNet-B5 Note:** EfficientNet-B5 end-to-end 3D fine-tuning was NOT run locally because the 10.4 GB raw OSIC DICOM archive is excluded from the handover package and Kaggle GPU extraction used pretrained ResNet-18 (512-D) embeddings.

---

## 3. 6-Method Multimodal Fusion Comparison & Paired Nadeau-Bengio Test (Identical 25 Folds)

- **HRCT & Fusion Design:** HRCT-only and Early Fusion predict change from Baseline_FVC (Delta_FVC = Target_FVC - Baseline_FVC) using in-fold 1st-99th percentile winsorization on HU/volume features, in-fold 15D ResNet-18 PCA, and strong L2 regularization (Ridge alpha=500). All reconstructed FVC predictions across all 6 methods are winsorized to the physiologically plausible range [500, 6500] mL.
- **Paired Statistical Test:** Nadeau-Bengio corrected resampled paired t-test on per-fold differences vs. Clinical-only (25 identical folds, df=24, mean n_test/n_train=0.2500 for FVC regression, 0.2500 for patient-level rapid-decliner AUC)
- **`selected_fusion_method`:** `late_fusion_mean` (**Late fusion (mean)**) — selected by best mean CV Laplace LL (`fusion_vs_clinical_status: nominal_only`).
- **`statistically_confirmed_method`:** `stacking` (**Stacking (OOF meta-learner)**) — only method whose paired Nadeau-Bengio 95% CI on Laplace LL strictly excludes zero (survives Bonferroni across `20` comparisons, `α = 0.0025`; clinically negligible at `< 0.1%` of mean FVC `2676.5` mL).
- **Summary Verdict:** Late fusion (mean) has the best mean FVC Laplace LL (-6.7928 vs. clinical-only -6.8076; Δ = +0.0148, p = 0.1778, nominal only), while its rapid-decliner AUC is lower (0.5636 vs. 0.6343). Stacking (OOF meta-learner) is the only method with a statistically confirmed gain over clinical-only (+0.0104 Laplace LL, p = 0.0006; -1.7 mL MAE, p = 0.0003); across 20 paired comparisons, both p-values survive Bonferroni correction (α = 0.0025), but the gain is clinically negligible (0.07%, under 0.1% of mean FVC 2676.5 mL). Exploratory.

| Fusion Method | FVC Laplace LL (95% CI) | Paired ΔLL vs. Clinical-only (Nadeau-Bengio 95% CI, p) | FVC RMSE mL (95% CI) | FVC MAE mL (95% CI) | Rapid-Decliner AUC (95% CI) | Paired Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clinical-only** | `-6.8076 ± 0.0877 (95% CI: -6.8420–-6.7732)` | Reference | `232.596 ± 8.996 (95% CI: 229.070–236.123)` | `168.656 ± 12.728 (95% CI: 163.667–173.646)` | `0.6343 ± 0.0518 (95% CI: 0.6141–0.6546)` | Reference |
| **HRCT-only (HU/Volume + ResNet-18 PCA 15D)** | `-6.8245 ± 0.0528 (95% CI: -6.8452–-6.8038)` | `-0.0169 (95% CI: -0.0726 to +0.0388, p = 0.5364)` | `235.792 ± 11.963 (95% CI: 231.102–240.481)` | `171.448 ± 8.299 (95% CI: 168.195–174.701)` | `0.4807 ± 0.0332 (95% CI: 0.4677–0.4938)` | `nominal_only` |
| **Early fusion** | `-6.8115 ± 0.0834 (95% CI: -6.8442–-6.7788)` | `-0.0039 (95% CI: -0.0337 to +0.0260, p = 0.7915)` | `231.173 ± 11.792 (95% CI: 226.550–235.795)` | `168.968 ± 12.675 (95% CI: 164.000–173.937)` | `0.5539 ± 0.0679 (95% CI: 0.5273–0.5806)` | `nominal_only` |
| **Late fusion (mean)** | `-6.7928 ± 0.0716 (95% CI: -6.8209–-6.7647)` | `+0.0148 (95% CI: -0.0072 to +0.0367, p = 0.1778)` | `230.735 ± 9.523 (95% CI: 227.002–234.468)` | `166.333 ± 10.247 (95% CI: 162.316–170.350)` | `0.5636 ± 0.0666 (95% CI: 0.5375–0.5897)` | `nominal_only` |
| **Weighted late fusion (in-fold tuned)** | `-6.7988 ± 0.0808 (95% CI: -6.8305–-6.7671)` | `+0.0088 (95% CI: -0.0023 to +0.0198, p = 0.1153)` | `232.122 ± 9.457 (95% CI: 228.415–235.829)` | `167.265 ± 11.530 (95% CI: 162.745–171.785)` | `0.6088 ± 0.0544 (95% CI: 0.5875–0.6302)` | `nominal_only` |
| **Stacking (OOF meta-learner)** | `-6.7971 ± 0.0924 (95% CI: -6.8334–-6.7609)` | `+0.0104 (95% CI: +0.0049 to +0.0159, p = 0.0006)` | `231.116 ± 10.470 (95% CI: 227.012–235.221)` | `166.911 ± 13.460 (95% CI: 161.635–172.187)` | `0.5727 ± 0.0562 (95% CI: 0.5507–0.5947)` | `statistically_confirmed` |

### Paired Nadeau-Bengio Comparison on FVC MAE, FVC RMSE & Rapid-Decliner AUC vs. Clinical-Only (`20` Total Comparisons, Bonferroni `α = 0.0025`)

| Fusion Method | Paired Δ FVC MAE mL (95% CI, p) | Paired Δ FVC RMSE mL (95% CI, p) | Paired Δ Rapid-Decliner AUC (95% CI, p) |
| :--- | :---: | :---: | :---: |
| **HRCT-only (HU/Volume + ResNet-18 PCA 15D)** | `+2.7917 (95% CI: -5.2557 to +10.8390, p = 0.4809)` | `+3.1953 (95% CI: -5.3867 to +11.7772, p = 0.4497)` | `-0.1536 (95% CI: -0.2040 to -0.1032, p = 0.0000)` |
| **Early fusion** | `+0.3116 (95% CI: -4.4831 to +5.1064, p = 0.8944)` | `-1.4237 (95% CI: -7.6135 to +4.7661, p = 0.6393)` | `-0.0804 (95% CI: -0.1127 to -0.0481, p = 0.0000)` |
| **Late fusion (mean)** | `-2.3235 (95% CI: -5.7036 to +1.0565, p = 0.1688)` | `-1.8616 (95% CI: -6.1849 to +2.4618, p = 0.3830)` | `-0.0708 (95% CI: -0.1191 to -0.0225, p = 0.0058)` |
| **Weighted late fusion (in-fold tuned)** | `-1.3917 (95% CI: -3.2449 to +0.4615, p = 0.1342)` | `-0.4746 (95% CI: -3.8246 to +2.8754, p = 0.7725)` | `-0.0255 (95% CI: -0.0815 to +0.0305, p = 0.3563)` |
| **Stacking (OOF meta-learner)** | `-1.7454 (95% CI: -2.5898 to -0.9010, p = 0.0003)` | `-1.4799 (95% CI: -3.4216 to +0.4619, p = 0.1288)` | `-0.0616 (95% CI: -0.1577 to +0.0345, p = 0.1982)` |

---

## 4. Key Takeaways & Caveats

1. **Best Internal Model (Tabular Benchmark):** `Ridge` achieved a cross-validated Modified Laplace Log-Likelihood of **`-6.8063 ± 0.0932 (95% CI: -6.8428–-6.7698)`** (FVC RMSE `232.578 ± 9.776 (95% CI: 228.746–236.410)` mL, MAE `168.507 ± 13.433 (95% CI: 163.241–173.773)` mL) on `train.csv`.
2. **Why Internal CV Differs from Kaggle Private Leaderboard:** The Kaggle private leaderboard evaluates strictly on the final three visits per patient of a separate hidden cohort with different follow-up intervals and baseline severity distribution. Internal CV on `train.csv` evaluates all future visits (`Weeks > Baseline_Weeks`), so numerical scores cannot be equated directly to leaderboard ranks.
3. **Modality / Fusion Verdict (`fusion_vs_clinical_status: nominal_only`):** Late fusion (mean) has the best mean FVC Laplace LL (-6.7928 vs. clinical-only -6.8076; Δ = +0.0148, p = 0.1778, nominal only), while its rapid-decliner AUC is lower (0.5636 vs. 0.6343). Stacking (OOF meta-learner) is the only method with a statistically confirmed gain over clinical-only (+0.0104 Laplace LL, p = 0.0006; -1.7 mL MAE, p = 0.0003); across 20 paired comparisons, both p-values survive Bonferroni correction (α = 0.0025), but the gain is clinically negligible (0.07%, under 0.1% of mean FVC 2676.5 mL). Exploratory.
4. **Exploratory Boundary:** OSIC progression models predict two separate exploratory targets — future FVC at a follow-up week (regression) and binary rapid-decliner risk (separate classifier) — in an independent fibrosis cohort without IPF diagnostic labels; they are **never** used to diagnose IPF.

---

## 5. Primary Source Links

- **Kaggle OSIC Pulmonary Fibrosis Progression Competition:** https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression
- **Kaggle 1st-Place Solution Repository (EfficientNet-B5 + Quantile-Regression Neural Network):** https://github.com/artkulak/osic-pulmonary-fibrosis-progression
- **Fibrosis-Net Paper (`arXiv:2103.04008`):** https://arxiv.org/pdf/2103.04008
