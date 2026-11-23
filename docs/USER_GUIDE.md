# PulmoSight CDSS — User Guide (Quick Start)

**PulmoSight CDSS** is a clinical decision support web application that estimates the probability of **Idiopathic Pulmonary Fibrosis (IPF)** from routine patient demographics and pulmonary function test (PFT) results, and provides an optional exploratory FVC-decline progression view when a full thoracic CT DICOM series is supplied.

> **Important Medical Notice:** This tool supports doctors and **does not replace clinical diagnosis**. Always combine results with HRCT imaging, clinical history, and multidisciplinary review.

---

## Before You Begin (One-Time Setup)
Ask your IT administrator (or run once inside the unzipped `PulmoSight` project folder):
1. Install **Python 3.10+** and **Node.js 18+**.
2. Open a terminal (PowerShell on Windows, Terminal on Mac/Linux) in the project folder and run:
   ```bash
   pip install -r requirements.txt
   npm install
   ```
   > **Note on PyTorch:** Core clinical IPF prediction (`POST /predict-ipf`) runs **without PyTorch**. If you want to upload live thoracic CT `.zip` DICOM series for exploratory HRCT/Fusion progression scoring (`POST /predict-progression-multimodal`), install `torch` and `torchvision` (with pretrained ResNet-18 weights available). If missing, the API returns a clear HTTP `503` message for CT scoring while IPF prediction remains fully operational.

---

## 5 Steps to Start the App and Make a Prediction

### Step 1 — Start the Analysis Engine (Backend)
Open a terminal in the project folder and run:
```bash
python -m uvicorn backend.api.main:app --port 8000
```
Wait a few seconds until you see `Application startup complete.` Keep this window open.

### Step 2 — Start the Web Interface (Frontend)
Open a **second** terminal in the same project folder and run:
```bash
npm run dev
```
Keep this window open as well.

### Step 3 — Open PulmoSight CDSS in Your Browser
Open Chrome, Edge, Firefox, or Safari and go to:
- **`http://localhost:5173`**

You will land on **Page 1: Patient Diagnostic Input Form**.

### Step 4 — Enter Patient Details (or Load a Sample Patient)
Fill in all required clinical fields:
- **Demographics:** Biological Sex (`M` or `F`), Age (`18–110` years), Height in **meters** (e.g., `1.72`), Weight in **kg** (e.g., `78`).
- **Lung Function (PFTs):** `FEV1/FVC` as a **decimal ratio** (e.g., `0.83`, not `83`), plus `% predicted` values for `FEV1`, `FVC`, `TLC`, and `TLCO`.
- *(Optional)* Click **Load sample patient** at the top right to try a built-in example case immediately.
- *(Optional CT Series Upload)* To activate the two exploratory progression cards on Page 2, upload a **`.zip` archive of axial DICOM (`.dcm`) slices** (max 50 MB). Uploading a single preview image or leaving this blank will keep the HRCT and Fusion cards honestly labeled **"Requires CT series"** (never dummy values).

### Step 5 — Run the Prediction & Review the Report
Click the blue **Predict** button. On **Page 2: Diagnostic Result**, you can:
1. Review the **Three-Score Overview**:
   - **Card 1 — Clinical IPF probability (Validated XGBoost):** Estimates IPF vs. Non-IPF diagnostic likelihood from PFTs and demographics.
   - **Card 2 — HRCT progression-risk score (`Exploratory, not validated`):** Predicts FVC decline on the separate OSIC cohort (not an IPF diagnosis), displaying its cross-validated 95% confidence interval or `"Requires CT series"`.
   - **Card 3 — Fusion progression-risk score (`Exploratory, not validated`):** Uses the cross-validated `selected_fusion_method` (1-SE rule), displays its 95% confidence interval, and shows `"Requires CT series"` unless a full `.zip` DICOM series is uploaded.
2. Check the **Top 5 Feature Drivers (SHAP)** bar chart and read the deterministic **"What drove this prediction"** plain-language summary directly underneath (reported on the model's log-odds scale, with the mandatory caveat that SHAP shows what the model used, not causes).
3. Switch between **Screening Mode** (default, higher sensitivity to catch more IPF cases) and **Confirmatory Mode** (higher specificity, fewer false alarms).
4. Click **Download Result as PDF** to save a shareable report (`CASE-XXXXXX-Diagnostic-Report.pdf`), which includes the exact **"What drove this prediction"** summary, sensitivity disclosure, and clinical disclaimer.

---

## Additional Pages & Optional Prototype Endpoint
- **Page 3 (`Model Info`):** Shows held-out Zenodo classifier performance, per-diagnosis error rates, the **OSIC Modified Laplace Log-Likelihood 8-Model Benchmark**, the **6-Method Multimodal Fusion Comparison** (`selected_fusion_method`), and the 3-way imaging modality ablation.
- **Page 4 (`Data Insights`):** Cohort distributions and survival overview from the real Zenodo ILD registry.
- **Optional Agentic Assessment (`POST /agent/assess`):** Disabled by default (returns HTTP `503`) unless `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_MODEL`) is set as an environment variable. Accepts only de-identified numeric clinical values (rejects names/IDs), logs every tool call, and flags `human_review_required` whenever Screening and Confirmatory modes disagree. See [AGENTIC_DESIGN.md](../docs/AGENTIC_DESIGN.md).
