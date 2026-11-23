# OSIC Pulmonary Fibrosis Data Directory (`backend/data/osic/`)

## Included Files
- `train.csv` — Real OSIC Pulmonary Fibrosis Progression longitudinal spirometry table (`176` patients, `1,549` visits).
- `cached_dicom_features.csv` — Cached per-patient 3D CT Hounsfield Unit (HU), lung volume, and 512-D pretrained ResNet-18 features.

## Raw DICOM Scans (Not Included)
Raw thoracic CT DICOM files (`*.dcm`) are **intentionally excluded** from this handover package due to file size (~10.4 GB) and Kaggle competition data redistribution rules.

If you want to regenerate `cached_dicom_features.csv` across all 176 patients from scratch:
1. Create a Kaggle account and accept the rules of the [OSIC Pulmonary Fibrosis Progression](https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression) competition.
2. Either:
   - **Recommended (GPU in the cloud, no 10 GB local download):** Run `backend/scripts/kaggle_extraction_notebook.ipynb` on Kaggle with GPU + Internet enabled, then copy the generated `cached_dicom_features.csv` into this folder (`backend/data/osic/cached_dicom_features.csv`).
   - **Local download via Kaggle CLI:** Follow `backend/scripts/download_osic.md` to download the scans into `backend/data/osic/dicom/<PatientID>/*.dcm` and re-run `python -m backend.src.train`.

> **Note:** Regenerating CT features is **optional**. The clinical IPF diagnostic classifier (`POST /predict-ipf`) operates on clinical demographics and pulmonary function tests alone and works out of the box without DICOM scans.
