# PulmoSight CDSS — IPF Diagnostic & Exploratory Progression Support System (`v2.2.0`)

> **Clinical Disclaimer:** *Decision support only, not a clinical diagnosis.* Every API response, UI screen, and PDF report enforces this disclaimer.

---

## 1. Overview

**PulmoSight CDSS** is a full-stack clinical decision support system for **Idiopathic Pulmonary Fibrosis (IPF)** built with **FastAPI**, **XGBoost**, **scikit-learn**, **SHAP**, **PyTorch**, and **React + Tailwind CSS + Recharts**.

It consists of two strictly separated studies plus a prototype agentic orchestration layer:
1. **Part 1 — Clinical IPF vs. Non-IPF Classifier (Production / Torch-Free):** Trained on the real Zenodo ILD clinical registry (`n = 6,808` unique patients). Using the 1-SE parsimony rule on 5-fold stratified training CV, calibrated **`XGBoost`** is deployed (`requires_torch_at_serving: false`) with dual OOF-tuned clinical operating modes (**Screening** & **Confirmatory**), per-patient Top-5 TreeSHAP charts (with concise hover tooltip), a five-part evidence-grounded **"Interpretation of this result"** engine (`backend/src/interpretation.py` + `backend/config/interpretation_rules.json` + held-out test reliability bin table, included in both Page 2 UI and 2-page PDF export), and optional env-gated fact-validated LLM rephrasing (`ENABLE_LLM_INTERPRETATION_REPHRASE`, off by default).
2. **Part 2 — Exploratory Pulmonary Fibrosis Progression Study:** Evaluates longitudinal FVC decline on the separate Kaggle OSIC cohort (`n = 176` patients, `175` with cached 3D CT features) via 5-Repeat × 5-Fold Nested `GroupKFold`, reporting the official **Modified Laplace Log-Likelihood**, **8-model internal benchmark**, **6-method multimodal fusion comparison** with **Nadeau-Bengio paired resampled t-tests** (`selected_fusion_method: late_fusion_mean`, `fusion_vs_clinical_status: nominal_only`), and **3-way modality ablation**. Exposed via `POST /predict-progression-multimodal` and rendered as three clear score cards on Page 2.
3. **Part 3 — Prototype Agentic Framework (`backend/agent/`):** Optional Anthropic tool-calling orchestrator (`POST /agent/assess`, disabled unless `ANTHROPIC_API_KEY` is set) wrapping 5 existing-code tools under strict de-identification, tool-call audit logging, and automatic `human_review_required` escalation whenever Screening and Confirmatory modes disagree.

### Companion Documentation
- Quick-Start User Guide: [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)
- Clinical Result Interpretation Rules & Threshold Citations: [`docs/INTERPRETATION_RULES.md`](docs/INTERPRETATION_RULES.md)
- OSIC Literature vs. Internal Model Comparison (Modified Laplace LL): [`docs/OSIC_MODEL_COMPARISON.md`](docs/OSIC_MODEL_COMPARISON.md)
- OSIC Pretrained CT Feature Extractor Comparison (ResNet-18 vs. Med3D, RadImageNet, CT-CLIP, ResNet-50 Control): [`docs/EXTRACTOR_COMPARISON.md`](docs/EXTRACTOR_COMPARISON.md)
- Agentic Framework Architecture & Guardrails: [`docs/AGENTIC_DESIGN.md`](docs/AGENTIC_DESIGN.md)

---

## 2. Data Sources

| Study | Source | Files Included in Handover | Notes |
| :--- | :--- | :--- | :--- |
| **Part 1: Clinical IPF Classifier** | [Zenodo ILD Clinical Registry (Record 15295274)](https://zenodo.org/records/15295274) | `backend/data/database_final_anonymized_for_zenodo.csv` | Real dataset (`data_source: "real"`, verified MD5 `07f13b8826c4e67fb4064e3b2340ed5e`, `6,808` rows, `0` missing, `6,808` unique `Case_number` values; `757` IPF = `11.12%`). |
| **Part 2: Exploratory Progression** | [Kaggle OSIC Pulmonary Fibrosis Progression Competition](https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression) | `backend/data/osic/train.csv`, `backend/data/osic/cached_dicom_features.csv` | Real longitudinal table (`176` patients, `1,549` visits) + cached 3D CT features for `175/176` patients (`523` features/patient; `ID00011637202177653955184` skipped due to unreadable DICOM). Raw `.dcm` scans (~10.4 GB) are excluded; see [`backend/data/osic/README.md`](backend/data/osic/README.md). |

*Cohort Independence:* The Zenodo ILD cohort and the Kaggle OSIC cohort share **no patients**, and OSIC has **no IPF vs. Non-IPF diagnostic labels**. The system is not patient-level multi-modal for IPF diagnosis.

**Data license:** Zenodo record [15295274](https://zenodo.org/records/15295274) (*"lung function and survival in ILD patients"*, Piotr Boros & Magdalena Martusewicz-Boros, DOI `10.5281/zenodo.15295274`) is distributed under **Creative Commons Attribution 4.0 International (CC-BY-4.0)**. OSIC data (`train.csv` and derived CT features) are governed by the [Kaggle OSIC Pulmonary Fibrosis Progression competition rules](https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression/rules).

---

## 3. Setup

### Verified Runtime Versions
- **Python:** `3.12.10` (tested; requires Python `3.10+`)
- **Node.js / npm:** `v24.18.0` / `npm 11.16.0` (tested; requires Node.js `18+`)

### Install Dependencies (from Project Root)
```bash
# 1. Create and activate a Python virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 2. Install backend runtime dependencies (and optional dev/test tools)
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Install frontend dependencies
npm install
```

---

## 4. How to Run & Key API Endpoints

Open two terminals in the project root:

**Terminal 1 — FastAPI Backend**
```bash
python -m uvicorn backend.api.main:app --port 8000
```
- `GET /health` — Readiness, `scikit-learn` version check, `imaging_model_available`, `agent_enabled`.
- `POST /predict-ipf` — **Torch-free** calibrated IPF probability, Screening & Confirmatory thresholds, Top-5 SHAP, and five-part `interpretation` (structured) + `interpretation_text` (Physiology, Model result with test-set reliability bin rate, What pushed it + real cohort age medians when age drives the score, Caution with `i-NSIP`/`u-ILD`/`CTD` error rates, and ATS/ERS/JRS/ALAT clinician considerations; optional LLM rephrasing behind `ENABLE_LLM_INTERPRETATION_REPHRASE=true`, off by default, with strict number-conservation and no-diagnosis/treatment validation).
- `POST /predict-progression-multimodal` — Exploratory OSIC progression scores (`clinical_only`, `hrct_only`, `fusion` using `selected_fusion_method`, `fusion_vs_clinical_status`, `modality_used` flags, and CV 95% CIs). Accepts an optional `.zip` archive of DICOM slices (max 50 MB); returns `hrct_only: null` and `fusion: null` if omitted, or HTTP `503` if `torch`/ResNet-18 weights are missing.
- `POST /agent/assess` — Prototype Anthropic tool-calling orchestrator (disabled / HTTP `503` unless `ANTHROPIC_API_KEY` is set; model name read from `ANTHROPIC_MODEL`). Enforces strict de-identification, logs all tool calls, and sets `human_review_required: true` when Screening and Confirmatory modes disagree.
- `GET /model-info` & `GET /eda` — Live evaluation artifacts and cohort EDA.

**Terminal 2 — React Frontend**
```bash
npm run dev
```
- Web UI URL: `http://localhost:5173`

**Run Automated Test Suite**
```bash
pytest -v
```
- `backend/tests/test_api_core.py`, `backend/tests/test_interpretation.py`, & `backend/tests/test_agent.py` run **completely torch-free** and require no external API keys.
- `backend/tests/test_torch_features.py` auto-skips cleanly via `pytest.importorskip("torch")` if PyTorch is omitted.

---

## 5. How to Retrain

Pre-trained `v2.2.0` artifacts are included in `backend/models/`. To retrain all clinical, OSIC Laplace LL benchmark, and 6-method fusion models from scratch on the included real CSVs (no local DICOM scans required because `cached_dicom_features.csv` is read directly):

```bash
python -m backend.src.train
```

To re-run only the read-only OSIC Pretrained CT Feature Extractor Comparison after placing Kaggle-extracted feature CSVs/JSONs into `backend/data/osic/` (never changes deployed models):

```bash
python -m backend.src.extractor_comparison
```

---

## 6. Results Table *(Generated Directly from `backend/models/model_artifacts.json`)*

### Part 1 — Zenodo ILD Clinical IPF Classifier (Untouched Test Set, `n = 1,362`, `151` IPF)
*Selection criterion:* 5-Fold Train CV ROC-AUC with 1-SE parsimony rule: PyTorch MLP (0.9117 ± 0.0086, SE=0.0038) vs. XGBoost (0.9100 ± 0.0092) differed by 0.0017 (< 1 SE). Selected XGBoost as deployed model for torch-free API serving and exact TreeSHAP; PyTorch MLP is retained as a deep-learning comparison baseline.
*Threshold rule:* Unified rule across all models: thresholds are tuned strictly on Out-of-Fold (OOF) calibrated probability predictions on the 80% training split (cross_val_predict with CalibratedClassifierCV(method='sigmoid')). Primary 'screening' threshold targets 80% OOF training sensitivity; secondary 'confirmatory' threshold targets >=92% OOF training specificity.

| Model | 5-Fold Train CV ROC-AUC | Untouched Test ROC-AUC (95% CI) | Test PR-AUC | Test F1 | Test Sensitivity (95% CI) | Test Specificity (95% CI) | Test Brier | Screening Thresh |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Deployed, 1-SE Rule)** | `0.9100 ± 0.0092` | `0.9179` (`0.8990–0.9346`) | `0.5391` | `0.5489` | `76.16%` (`69.59%–81.99%`) | `87.37%` (`85.44%–89.11%`) | `0.0668` | `0.1868` |
| **PyTorch MLP** | `0.9117 ± 0.0086` | `0.9156` (`0.8959–0.9321`) | `0.5404` | `0.5652` | `77.48%` (`71.05%–83.22%`) | `87.94%` (`86.14%–89.73%`) | `0.0671` | `0.2054` |
| **SVM** | `0.9070 ± 0.0065` | `0.9093` (`0.8890–0.9284`) | `0.4973` | `0.5680` | `78.81%` (`71.71%–85.03%`) | `87.70%` (`85.75%–89.29%`) | `0.0690` | `0.1873` |
| **Random Forest** | `0.9068 ± 0.0090` | `0.9113` (`0.8912–0.9291`) | `0.5107` | `0.5505` | `79.47%` (`72.66%–85.14%`) | `86.37%` (`84.31%–88.19%`) | `0.0687` | `0.1697` |

#### Deployed `XGBoost` Operating Modes & Per-Group Screening Errors
- **Screening Mode (`threshold = 0.1868`):** Sensitivity `76.16%`, Specificity `87.37%`, F1 `0.5489` — misses **`23.8%`** of IPF cases (`36/151`).
- **Confirmatory Mode (`threshold = 0.3429`):** Sensitivity `60.93%`, Specificity `93.39%`, F1 `0.5697` — misses **`39.1%`** of IPF cases (`59/151`).

| Zenodo Group | Clinical Description | Test Cases | Predicted IPF | Error Type | Screening Error Rate |
| :--- | :--- | :---: | :---: | :--- | :---: |
| `IPF` | Idiopathic Pulmonary Fibrosis | `151` | `115` | False Negative (Missed IPF) | `23.84%` (`36/151`) |
| `i-NSIP` | Idiopathic Non-Specific Interstitial Pneumonia | `23` | `14` | False Positive (Misclassified as IPF) | `60.87%` (`14/23`) |
| `CTD` | Connective Tissue Disease-Associated ILD | `120` | `42` | False Positive (Misclassified as IPF) | `35.00%` (`42/120`) |
| `u-ILD` | Unclassifiable Interstitial Lung Disease | `45` | `14` | False Positive (Misclassified as IPF) | `31.11%` (`14/45`) |
| `o-ILD` | Other Interstitial Lung Diseases | `279` | `52` | False Positive (Misclassified as IPF) | `18.64%` (`52/279`) |
| `HP` | Hypersensitivity Pneumonitis | `97` | `18` | False Positive (Misclassified as IPF) | `18.56%` (`18/97`) |
| `1_SAR` | Sarcoidosis (Stage I–IV) | `647` | `13` | False Positive (Misclassified as IPF) | `2.01%` (`13/647`) |

#### Held-Out Test Calibration Reliability Table (`n = 1362`, `151` IPF; Observed Score Range `0.0082–0.6359`)
*Used directly by the five-part **"Interpretation of this result"** card (`backend/src/interpretation.py`) and PDF report to contextualize each patient's calibrated probability alongside empirical held-out test IPF rates. Zenodo cohort age medians: IPF `68.9` yrs vs. Non-IPF `48.9` yrs. Full rules & citations: [`docs/INTERPRETATION_RULES.md`](docs/INTERPRETATION_RULES.md).*

| Calibrated Score Bin | Held-Out Test Patients (`n`) | Observed IPF (`k`) | Empirical IPF Rate | 95% Wilson CI | Mean Predicted Prob |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `0%–5%` (`0.00–0.05`) | `931` | `10` | **`1.1%`** | `0.6%–2.0%` | `0.0141` |
| `5%–10%` (`0.05–0.10`) | `84` | `8` | **`9.5%`** | `4.9%–17.7%` | `0.0701` |
| `10%–20%` (`0.10–0.20`) | `85` | `18` | **`21.2%`** | `13.8%–31.0%` | `0.1451` |
| `20%–35%` (`0.20–0.35`) | `93` | `25` | **`26.9%`** | `18.9%–36.7%` | `0.2643` |
| `35%–50%` (`0.35–0.50`) | `81` | `40` | **`49.4%`** | `38.8%–60.0%` | `0.4300` |
| `≥50%` (`0.50–1.00`) | `88` | `50` | **`56.8%`** | `46.4%–66.7%` | `0.5547` |

### Part 2 — Exploratory OSIC Progression Study (`n = 176`, `imaging_model_available: true`, `patients_with_dicoms: 175`)
> **Mandatory Label:** *Progression results are exploratory, separate cohort, no IPF labels*

#### A. Official OSIC Competition Metric Benchmark — Modified Laplace Log-Likelihood (`5×5` Nested `GroupKFold`, `n = 175` CT patients, `1359` future visits)
*Note: Internal cross-validation on `train.csv` — NOT Kaggle leaderboard scores. Published references (1st place EfficientNet-B5 + tabular quantile-regression NN blend `-6.8305`, 2nd `-6.8311`, 3rd `-6.8336`, Fibrosis-Net `-6.8188` [`arXiv:2103.04008`], Fibro-CoSANet `-6.68` [reported score, evaluation protocol unclear]) are detailed with source links in [`docs/OSIC_MODEL_COMPARISON.md`](docs/OSIC_MODEL_COMPARISON.md). All internal FVC predictions are winsorized to `[500, 6500]` mL.*

| Model | Modified Laplace LL (95% CI, higher = better) | FVC RMSE mL (95% CI) | FVC MAE mL (95% CI) |
| :--- | :---: | :---: | :---: |
| **Baseline (Constant Baseline FVC)** | `-6.8666 ± 0.0504 (95% CI: -6.8864–-6.8469)` | `262.646 ± 10.434 (95% CI: 258.556–266.736)` | `183.361 ± 8.324 (95% CI: 180.098–186.624)` |
| **Ridge** *(Best Internal CV)* | `-6.8063 ± 0.0932 (95% CI: -6.8428–-6.7698)` | `232.578 ± 9.776 (95% CI: 228.746–236.410)` | `168.507 ± 13.433 (95% CI: 163.241–173.773)` |
| **ElasticNet** | `-6.8152 ± 0.0841 (95% CI: -6.8482–-6.7823)` | `233.671 ± 8.641 (95% CI: 230.284–237.059)` | `169.829 ± 12.215 (95% CI: 165.041–174.617)` |
| **Quantile-Regression NN (Tabular)** | `-6.9341 ± 0.1285 (95% CI: -6.9845–-6.8837)` | `240.048 ± 12.060 (95% CI: 235.320–244.775)` | `177.034 ± 16.443 (95% CI: 170.588–183.479)` |
| **Random Forest** | `-6.9510 ± 0.0408 (95% CI: -6.9670–-6.9350)` | `272.419 ± 32.252 (95% CI: 259.777–285.062)` | `195.788 ± 11.030 (95% CI: 191.464–200.112)` |
| **SVM (RBF SVR)** | `-6.9701 ± 0.0971 (95% CI: -7.0081–-6.9320)` | `305.502 ± 79.151 (95% CI: 274.475–336.529)` | `204.991 ± 21.519 (95% CI: 196.556–213.426)` |
| **XGBoost** | `-6.9051 ± 0.0563 (95% CI: -6.9272–-6.8830)` | `262.871 ± 29.642 (95% CI: 251.252–274.491)` | `186.749 ± 8.551 (95% CI: 183.397–190.101)` |
| **Image + Tabular (ResNet-18 PCA 15D)** | `-6.8100 ± 0.0840 (95% CI: -6.8429–-6.7771)` | `230.786 ± 12.291 (95% CI: 225.968–235.604)` | `168.717 ± 12.802 (95% CI: 163.699–173.735)` |
| **EfficientNet-B5 (End-to-End 3D CT)** | *Not run (raw 10.4 GB DICOMs excluded; ResNet-18 cached features used)* | *Not run* | *Not run* |

#### B. 6-Method Multimodal Fusion Comparison (`5×5` Nested `GroupKFold`, Identical 25 Folds — `selected_fusion_method: late_fusion_mean` [`nominal_only`], `statistically_confirmed_method: stacking`)
*HRCT-only and Early Fusion predict change from Baseline_FVC (Delta_FVC = Target_FVC - Baseline_FVC) using in-fold 1st-99th percentile winsorization on HU/volume features, in-fold 15D ResNet-18 PCA, and strong L2 regularization (Ridge alpha=500). All reconstructed FVC predictions across all 6 methods are winsorized to the physiologically plausible range [500, 6500] mL.*

**Verdict:** Late fusion (mean) has the best mean FVC Laplace LL (-6.7928 vs. clinical-only -6.8076; Δ = +0.0148, p = 0.1778, nominal only), while its rapid-decliner AUC is lower (0.5636 vs. 0.6343). Stacking (OOF meta-learner) is the only method with a statistically confirmed gain over clinical-only (+0.0104 Laplace LL, p = 0.0006; -1.7 mL MAE, p = 0.0003); across 20 paired comparisons, both p-values survive Bonferroni correction (α = 0.0025), but the gain is clinically negligible (0.07%, under 0.1% of mean FVC 2676.5 mL). Exploratory.

*Paired test:* Nadeau-Bengio corrected resampled paired t-test on per-fold differences vs. Clinical-only (25 identical folds, df=24, mean n_test/n_train=0.2500 for FVC regression, 0.2500 for patient-level rapid-decliner AUC) (20 comparisons total; Bonferroni `α = 0.0025`).

| Fusion Method | FVC Laplace LL (95% CI) | Paired ΔLL vs. Clinical-only (95% CI, p) | Paired ΔMAE mL (95% CI, p) | Rapid-Decliner AUC (95% CI) | Paired Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clinical-only** | `-6.8076 ± 0.0877 (95% CI: -6.8420–-6.7732)` | Reference | Reference | `0.6343 ± 0.0518 (95% CI: 0.6141–0.6546)` | Reference |
| **HRCT-only (HU/Volume + ResNet-18 PCA 15D)** | `-6.8245 ± 0.0528 (95% CI: -6.8452–-6.8038)` | `-0.0169 (95% CI: -0.0726 to +0.0388, p = 0.5364)` | `+2.7917 (95% CI: -5.2557 to +10.8390, p = 0.4809)` | `0.4807 ± 0.0332 (95% CI: 0.4677–0.4938)` | `nominal_only` |
| **Early fusion** | `-6.8115 ± 0.0834 (95% CI: -6.8442–-6.7788)` | `-0.0039 (95% CI: -0.0337 to +0.0260, p = 0.7915)` | `+0.3116 (95% CI: -4.4831 to +5.1064, p = 0.8944)` | `0.5539 ± 0.0679 (95% CI: 0.5273–0.5806)` | `nominal_only` |
| **Late fusion (mean)** | `-6.7928 ± 0.0716 (95% CI: -6.8209–-6.7647)` | `+0.0148 (95% CI: -0.0072 to +0.0367, p = 0.1778)` | `-2.3235 (95% CI: -5.7036 to +1.0565, p = 0.1688)` | `0.5636 ± 0.0666 (95% CI: 0.5375–0.5897)` | `nominal_only` |
| **Weighted late fusion (in-fold tuned)** | `-6.7988 ± 0.0808 (95% CI: -6.8305–-6.7671)` | `+0.0088 (95% CI: -0.0023 to +0.0198, p = 0.1153)` | `-1.3917 (95% CI: -3.2449 to +0.4615, p = 0.1342)` | `0.6088 ± 0.0544 (95% CI: 0.5875–0.6302)` | `nominal_only` |
| **Stacking (OOF meta-learner)** | `-6.7971 ± 0.0924 (95% CI: -6.8334–-6.7609)` | `+0.0104 (95% CI: +0.0049 to +0.0159, p = 0.0006)` | `-1.7454 (95% CI: -2.5898 to -0.9010, p = 0.0003)` | `0.5727 ± 0.0562 (95% CI: 0.5507–0.5947)` | `statistically_confirmed` |

#### C. 3-Way Modality Ablation — Observed ~52-Week FVC Change (mL, CT cohort) (`n = 175` CT patients, In-Fold PCA `15D`)
| Modality / Baseline | RMSE (95% CI) | MAE (95% CI) | Rapid-Decliner AUC (95% CI) | Beats Baseline (95% CI)? | Beats Clinical-Only (95% CI)? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Train Mean / Majority Prior)** | `269.557 ± 19.577 (95% CI: 261.883–277.231)` | `221.411 ± 21.203 (95% CI: 213.100–229.722)` | `0.500 ± 0.000 (95% CI: 0.500–0.500)` | Reference | — |
| **Clinical-only** | `273.017 ± 23.205 (95% CI: 263.921–282.113)` | `220.332 ± 23.581 (95% CI: 211.088–229.576)` | `0.634 ± 0.052 (95% CI: 0.614–0.655)` | AUC above 0.5 only; RMSE and MAE no better than baseline | Reference |
| **Clinical + HU/Volume** | `276.071 ± 25.932 (95% CI: 265.906–286.237)` | `225.121 ± 25.781 (95% CI: 215.015–235.227)` | `0.591 ± 0.075 (95% CI: 0.562–0.620)` | AUC above 0.5 only; RMSE and MAE no better than baseline | No |
| **Clinical + ResNet-18 (In-Fold PCA 15D)** | `276.388 ± 24.955 (95% CI: 266.605–286.170)` | `231.340 ± 29.747 (95% CI: 219.679–243.001)` | `0.559 ± 0.133 (95% CI: 0.507–0.611)` | AUC above 0.5 only; RMSE and MAE no better than baseline | No |

#### C. 3-Way Modality Ablation — Robust Theil-Sen FVC Slope (mL/week, CT cohort) (`n = 175` CT patients, In-Fold PCA `15D`)
| Modality / Baseline | RMSE (95% CI) | MAE (95% CI) | Rapid-Decliner AUC (95% CI) | Beats Baseline (95% CI)? | Beats Clinical-Only (95% CI)? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Train Mean / Majority Prior)** | `4.987 ± 0.419 (95% CI: 4.823–5.151)` | `4.031 ± 0.468 (95% CI: 3.847–4.214)` | `0.500 ± 0.000 (95% CI: 0.500–0.500)` | Reference | — |
| **Clinical-only** | `4.948 ± 0.513 (95% CI: 4.747–5.149)` | `3.903 ± 0.401 (95% CI: 3.746–4.060)` | `0.626 ± 0.048 (95% CI: 0.607–0.645)` | AUC above 0.5 only; RMSE and MAE no better than baseline | Reference |
| **Clinical + HU/Volume** | `4.951 ± 0.505 (95% CI: 4.753–5.148)` | `3.976 ± 0.495 (95% CI: 3.782–4.170)` | `0.562 ± 0.095 (95% CI: 0.525–0.599)` | AUC above 0.5 only; RMSE and MAE no better than baseline | No |
| **Clinical + ResNet-18 (In-Fold PCA 15D)** | `4.962 ± 0.408 (95% CI: 4.802–5.122)` | `3.969 ± 0.450 (95% CI: 3.793–4.145)` | `0.588 ± 0.047 (95% CI: 0.570–0.607)` | AUC above 0.5 only; RMSE and MAE no better than baseline | No |

#### D. Pretrained CT Feature Extractor Comparison (`5×5` Patient-Grouped CV, `folds_sha256: cd42bc04090b2952…` — Read-Only Benchmark)
*Protocol:* Only `emb_prefix` varies; identical `n = 175` patients, `1359` future visits, in-fold `15D` PCA, delta-from-baseline-FVC target winsorized to `[500, 6500]` mL, Modified Laplace LL, and Nadeau-Bengio paired tests (`168` executed tests, `Holm-Bonferroni (step-down) with Bonferroni reference`). Alternative-extractor Holm flags: `any_new_extractor_beats_resnet18_holm = False` (`evaluated across 3 completed alternative extractors`), `any_new_extractor_beats_clinical_only_holm = False` (`evaluated across 3 completed alternative extractors`). Full per-method tables, paired p-values, licenses, and citations are in [`docs/EXTRACTOR_COMPARISON.md`](docs/EXTRACTOR_COMPARISON.md). Read-only benchmark; never changes the deployed clinical IPF or OSIC progression model automatically.

**Verdict:** Across completed alternative extractors (med3d, radimagenet, ctclip; 168 paired Nadeau-Bengio tests, Holm-Bonferroni α = 0.05), none achieved a statistically confirmed improvement over current ResNet-18. None achieved a Holm-corrected improvement over Clinical-only. Not run: resnet50_imagenet_control (Disabled by user toggle RUN_IMAGENET_RESNET50_CONTROL = False). The deployed model is not changed automatically. Exploratory.

| Extractor | Status | Dim / Time / Peak GPU | HRCT-Only Laplace LL (AUC) | Image+Tab Laplace LL (RMSE) | Best Fusion Laplace LL | License & Redistribution |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **ResNet-18 (Current)** | `completed` | `512D` / `N/A` / `N/A` | `-6.8245` (`AUC 0.4807`) | `-6.8100` (`RMSE 230.8`) | `-6.7928` (Late fusion (mean)) | BSD-3-Clause (torchvision) |
| **Med3D (3D ResNet-18)** | `completed` | `512D` / `45.30m` / `860 MB` | `-6.9036` (`AUC 0.5354`) | `-6.8748` (`RMSE 244.7`) | `-6.8076` (Clinical-only) | MIT License |
| **RadImageNet (ResNet-50)** | `completed` | `2048D` / `39.32m` / `764 MB` | `-6.8578` (`AUC 0.4748`) | `-6.8660` (`RMSE 240.3`) | `-6.8076` (Clinical-only) | MIT License |
| **CT-CLIP (3D CT-ViT)** | `completed` | `512D` / `8.24m` / `2712 MB` | `-6.8383` (`AUC 0.5630`) | `-6.8344` (`RMSE 236.8`) | `-6.7998` (Weighted late fusion (in-fold tuned)) | CC BY-NC-SA 4.0 (Gated, Non-Commercial — do NOT redistribute weights) |
| **ResNet-50 Control (ImageNet)** | `not run` (Disabled by user toggle RUN_IMAGENET_RESNET50_CONTROL = False) | `2048D` / — / — | `not run` | `not run` | `not run` | BSD-3-Clause (torchvision) |

*Extractor Licenses & Citations:* MedicalNet (`TencentMedicalNet/MedicalNet-Resnet18`, MIT; Chen et al., `arXiv:1904.00625`), RadImageNet (`Lab-Rasool/RadImageNet`, MIT; Mei et al., *Radiology: Artificial Intelligence*, 2022, `doi:10.1148/ryai.210315`), CT-CLIP / CT-RATE (`ibrahimhamamci/CT-RATE`, **gated, CC BY-NC-SA 4.0 non-commercial — do not redistribute weights**; Hamamci et al., `arXiv:2403.17834`), paired testing per Nadeau & Bengio (*Machine Learning*, 2003). Limits: embedding dimensions equalized to 15D via in-fold PCA; 3D volumetric preprocessing differs from 2D slice-wise preprocessing; internal CV on 175 OSIC patients (not leaderboard scores).

---

## 7. Limitations

- **Missed IPF Cases (~24% in Screening Mode):** Deployed model (XGBoost): Screening mode misses 23.8% of IPF cases (36 of 151 missed on the untouched test set), and confirmatory mode misses 39.1% of IPF cases (59 of 151). In screening mode, i-NSIP (60.9% flagged), u-ILD (31.1% flagged), and CTD (35.0% flagged) are most often misclassified as IPF, which is clinically expected.
- **Non-IPF Clinical Overlap:** Fibrotic `i-NSIP`, `CTD`-ILD, and `u-ILD` share restrictive physiology and low `TLCO` with IPF, causing expected false positives when evaluated on demographics + PFTs alone.
- **Exploratory Imaging / Progression & Fusion Study:** Although `imaging_model_available` is `true` using cached 3D CT features for `175` of `176` OSIC patients (1 patient, `ID00011637202177653955184`, lacked readable DICOM slices), Late fusion (mean) has the best mean FVC Laplace LL (-6.7928 vs. clinical-only -6.8076; Δ = +0.0148, p = 0.1778, nominal only), while its rapid-decliner AUC is lower (0.5636 vs. 0.6343). Stacking (OOF meta-learner) is the only method with a statistically confirmed gain over clinical-only (+0.0104 Laplace LL, p = 0.0006; -1.7 mL MAE, p = 0.0003); across 20 paired comparisons, both p-values survive Bonferroni correction (α = 0.0025), but the gain is clinically negligible (0.07%, under 0.1% of mean FVC 2676.5 mL). Exploratory. (`fusion_vs_clinical_status: nominal_only`). In the 3-way modality ablation, neither `Clinical + HU/Volume` nor `Clinical + ResNet-18 (In-Fold PCA 15D)` beats `Clinical-only` on RMSE, MAE, or AUC with non-overlapping 95% CIs. Progression cards on Page 2 therefore carry an **"Exploratory, not validated"** badge, explicitly distinguish future-FVC regression from binary rapid-decliner classification, require a full `.zip` DICOM series (displaying `"Requires CT series"` otherwise), and never diagnose IPF.
- **Automated Result Interpretation Limits:** The five-part "Interpretation of this result" card (`backend/config/interpretation_rules.json`) uses fixed percent-predicted thresholds (`80%` volumes/TLCO, `0.70` FEV1/FVC) as approximations when GLI LLN/z-scores are not available; rules must be reviewed by a clinician before any real clinical use. Optional LLM rephrasing (`ENABLE_LLM_INTERPRETATION_REPHRASE`) is off by default and enforces post-generation numeric-conservation and no-diagnosis/treatment validation. *Automated interpretation, decision support only. Verify clinically.* See [`docs/INTERPRETATION_RULES.md`](docs/INTERPRETATION_RULES.md).
- **Agentic Prototype Status:** `POST /agent/assess` is a prototype disabled by default unless `ANTHROPIC_API_KEY` is provided and has been verified via mocked unit tests rather than a prospective clinical reader study. See [`docs/AGENTIC_DESIGN.md`](docs/AGENTIC_DESIGN.md).
- **Separate Datasets & No External Validation:** Part 1 (Zenodo ILD) and Part 2 (OSIC) come from completely separate patient cohorts with different labels, and metrics have not yet been externally validated across independent hospitals.
- **Decision Support Only:** PulmoSight CDSS is a research and decision-support prototype, **not** a cleared medical device and **not** a substitute for clinical diagnosis or multidisciplinary ILD board evaluation.

---

## 8. Security Note (`.joblib` Artifacts, CT `.zip` Uploads & Credentials)

> **Important Security Warning:** Trained model files in `backend/models/*.joblib` use Python `pickle` serialization internally, which can execute arbitrary code when deserialized. **Only load `.joblib` files generated locally or received from strictly trusted sources.**
>
> To prevent untrusted deserialization attacks at runtime, the FastAPI backend **never accepts model or pickle files** (`.joblib`, `.pkl`, `.pickle`, `.pt`, `.pth`, `.onnx`, `.h5`) via any API upload endpoint (including inside `.zip` CT archives, which also enforce path-traversal, file-count, and decompression-size limits) and explicitly blocks them with **`HTTP 415 Unsupported Media Type`**.
>
> API credentials (`ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`) are read exclusively from environment variables, never stored in the repository, and `POST /agent/assess` blocks PHI identifiers (`name`, `patient_id`, `mrn`, `dob`, etc.) before any external API request.
