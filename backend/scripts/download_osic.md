# Downloading the Full Kaggle OSIC Pulmonary Fibrosis DICOM Dataset

The repository includes the real Kaggle OSIC longitudinal clinical table at `backend/data/osic/train.csv` (`176` unique patients, `1,549` longitudinal FVC visits) and sample high-resolution DICOM slices for 2 patients (`5` `.dcm` files).

Because the full Kaggle OSIC DICOM archive (`train/` directory covering all `176` patients) is approximately **10.4 GB**, it is not committed to Git. When fewer than **100 patients** have local DICOM scans under `backend/data/osic/dicom/<PatientID>/`, the training pipeline automatically sets:

```json
"imaging_model_available": false
```

in `backend/models/model_artifacts.json` and `backend/models/osic_progression_pipeline.joblib`, and both the FastAPI backend (`POST /predict-progression` and `GET /model-info`) and the React UI explicitly report that **only the clinical-only progression model is active** (`"imaging_model_available": false`).

---

## Step-by-Step Instructions to Download All 176 Patient DICOM Scans

### 1. Install and Authenticate the Kaggle CLI

1. Install the official Kaggle CLI:
   ```powershell
   pip install kaggle
   ```
2. Sign in to [Kaggle](https://www.kaggle.com/), navigate to **Settings → API → Create New Token**, and save the downloaded `kaggle.json` file to:
   - **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux / macOS:** `~/.kaggle/kaggle.json`
3. Accept the competition rules on the [OSIC Pulmonary Fibrosis Progression Competition Page](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/rules).

### 2. Download and Extract the OSIC Dataset

Run the following PowerShell commands from the project root (`D:\PROJECT 1`):

```powershell
# Download the OSIC competition archive into backend/data/osic/
kaggle competitions download -c osic-pulmonary-fibrosis-progression -p backend/data/osic

# Extract train.csv and the train/ DICOM folders
Expand-Archive -Path "backend/data/osic/osic-pulmonary-fibrosis-progression.zip" -DestinationPath "backend/data/osic/raw_extracted" -Force

# Move all 176 patient DICOM directories (<PatientID>/*.dcm) into backend/data/osic/dicom/
New-Item -ItemType Directory -Force -Path "backend/data/osic/dicom" | Out-Null
Move-Item -Path "backend/data/osic/raw_extracted/train/ID*" -Destination "backend/data/osic/dicom/" -Force
Copy-Item -Path "backend/data/osic/raw_extracted/train.csv" -Destination "backend/data/osic/train.csv" -Force

# Optional: Remove temporary zip and extraction folder
Remove-Item -Path "backend/data/osic/osic-pulmonary-fibrosis-progression.zip" -Force
Remove-Item -Path "backend/data/osic/raw_extracted" -Recurse -Force
```

### 3. Verify Directory Layout & Retrain

After extraction, verify that `backend/data/osic/dicom/` contains all 176 patient folders:

```powershell
(Get-ChildItem -Path "backend/data/osic/dicom" -Directory).Count
# Expected output: 176
```

Then re-run training from the project root:

```powershell
python -m backend.src.train
```

- The 3D DICOM feature extractor processes each patient's full 3D scan (all slices sorted by Z-position), extracts Hounsfield Unit (HU) histogram features, physical lung volume (mL), and 512-D pretrained **ResNet-18 (`ResNet18_Weights.DEFAULT`)** slice-averaged embeddings, and **caches** the extracted features per patient to `backend/data/osic/cached_dicom_features.csv` so subsequent training runs do not re-read thousands of `.dcm` files.
- Once `patients_with_dicoms >= 100`, `imaging_model_available` automatically switches to `true` and PCA-reduced ResNet-18 embeddings (fitted strictly on training fold patients) are evaluated alongside the clinical features.
