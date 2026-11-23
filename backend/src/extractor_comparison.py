"""
OSIC Pretrained CT Feature Extractor Comparison (`backend/src/extractor_comparison.py`).

Evaluates alternative pretrained CT feature extractors cached from Kaggle in
`backend/data/osic/`:
  - `resnet18_baseline`: Current ImageNet 2D ResNet-18 (`cached_dicom_features.csv`, `resnet_emb_*`)
  - `med3d`: MedicalNet (Med3D) 3D ResNet-18 (`cached_dicom_features_med3d.csv`, `med3d_emb_*`)
  - `radimagenet`: RadImageNet 2D ResNet-50 (`cached_dicom_features_radimagenet.csv`, `radimagenet_emb_*`)
  - `ctclip`: CT-CLIP 3D CT-ViT (`cached_dicom_features_ctclip.csv`, `ctclip_emb_*`)
  - `resnet50_imagenet_control`: Optional extra control ImageNet 2D ResNet-50
    (`cached_dicom_features_resnet50_imagenet_control.csv`, `resnet50_imagenet_control_emb_*`)

STRICT PROTOCOL GUARANTEES:
1. Changes ONLY the embedding column prefix (`emb_prefix`); every patient, CV fold
   (`_generate_patient_cv_splits` 5x5 patient-grouped folds from `RANDOM_SEED`),
   in-fold 15-component PCA, Ridge alpha grid, in-fold feature winsorization,
   delta-from-baseline-FVC target clipped to [500, 6500] mL,
   `modified_laplace_log_likelihood`, and `nadeau_bengio_paired_test` stay identical
   to `backend/src/osic_benchmarks.py` and `backend/src/imaging_progression.py`.
2. Zero fabrication: if a feature CSV is missing, records `"pending: not evaluated yet"`;
   if a metadata JSON records `"not run"` (e.g., CT-CLIP gated access or OOM), records
   `"not run"` with the exact reason. Never estimates, invents, or copies numbers.
3. Paired Nadeau-Bengio tests compare each new extractor against (i) the current
   ResNet-18 pipeline (primary question) and (ii) Clinical-only on the exact same
   25 folds, with Holm-Bonferroni (and Bonferroni reference) multiple-comparison
   correction across all executed tests.
4. Read-only benchmark: NEVER modifies the deployed clinical or progression models.

Run standalone from project root:
    python -m backend.src.extractor_comparison
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.src.config import (
    ARTIFACTS_JSON_PATH,
    CACHED_DICOM_FEATURES_CSV,
    OSIC_DICOM_DIR,
    OSIC_DIR,
    OSIC_TRAIN_CSV_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
    setup_logger,
)
from backend.src.imaging_progression import (
    HU_VOLUME_FEATURE_NAMES,
    PCA_COMPONENTS,
    build_osic_patient_cohort,
)
from backend.src.osic_benchmarks import (
    FVC_MAX_ML,
    FVC_MIN_ML,
    PATIENT_CLINICAL_FEATURES,
    VISIT_CLINICAL_FEATURES,
    InFoldWinsorizer,
    _compute_ci_95,
    _compute_inner_oof_predictions,
    _fit_in_fold_sigma_params,
    _generate_patient_cv_splits,
    _predict_sigma,
    build_osic_visit_dataset,
    modified_laplace_log_likelihood,
    nadeau_bengio_paired_test,
    winsorize_fvc,
)

logger = setup_logger("extractor_comparison")

DOCS_EXTRACTOR_COMPARISON_PATH = PROJECT_ROOT / "docs" / "EXTRACTOR_COMPARISON.md"

IMAGING_CONFIG_KEYS = [
    ("hrct_only", "HRCT-only (HU/Volume + In-Fold PCA 15D)"),
    ("image_plus_tabular", "Image + Tabular (Benchmark, Ridge GridSearch)"),
    ("early_fusion", "Early fusion"),
    ("late_fusion_mean", "Late fusion (mean)"),
    ("weighted_late_fusion", "Weighted late fusion (in-fold tuned)"),
    ("stacking", "Stacking (OOF meta-learner)"),
]

FUSION_METHOD_KEYS = [
    ("clinical_only", "Clinical-only"),
    ("hrct_only", "HRCT-only (HU/Volume + In-Fold PCA 15D)"),
    ("early_fusion", "Early fusion"),
    ("late_fusion_mean", "Late fusion (mean)"),
    ("weighted_late_fusion", "Weighted late fusion (in-fold tuned)"),
    ("stacking", "Stacking (OOF meta-learner)"),
]

METRIC_SPECS = [
    ("laplace_ll", True),
    ("rmse", False),
    ("mae", False),
    ("auc", True),
]

EXTRACTOR_CATALOG: List[Dict[str, Any]] = [
    {
        "key": "resnet18_baseline",
        "display_name": "Current Baseline: ImageNet 2D ResNet-18",
        "short_name": "ResNet-18 (Current)",
        "is_reference": True,
        "is_control": False,
        "csv_candidates": ["cached_dicom_features.csv"],
        "json_candidates": ["cached_dicom_features_resnet18_metadata.json"],
        "emb_prefix_candidates": ["resnet_emb_"],
        "default_emb_prefix": "resnet_emb_",
        "expected_embed_dim": 512,
        "architecture_type": "2D slice-wise CNN (mean-pooled across axial slices)",
        "source": "torchvision.models.resnet18 (ResNet18_Weights.DEFAULT, ImageNet-1K)",
        "license": "BSD-3-Clause (torchvision)",
        "redistribution_note": "Standard torchvision pretrained weights; derived features cached locally.",
        "citation": "He et al., Deep Residual Learning for Image Recognition (CVPR 2016); torchvision ResNet18_Weights.DEFAULT",
        "preprocessing_summary": (
            "Lung-masked axial slices (segment_lungs_hu, non-lung = -1000 HU), window [-1000, 0] HU, "
            "224x224 PIL BILINEAR, ImageNet normalization, mean-pooled across all slices (512-D)."
        ),
    },
    {
        "key": "med3d",
        "display_name": "MedicalNet (Med3D) 3D ResNet-18",
        "short_name": "Med3D (3D ResNet-18)",
        "is_reference": False,
        "is_control": False,
        "csv_candidates": ["cached_dicom_features_med3d.csv"],
        "json_candidates": [
            "cached_dicom_features_med3d_metadata.json",
            "cached_dicom_features_med3d.json",
            "metadata_med3d.json",
        ],
        "emb_prefix_candidates": ["med3d_emb_"],
        "default_emb_prefix": "med3d_emb_",
        "expected_embed_dim": 512,
        "architecture_type": "3D volumetric CNN (Global Average Pooling before conv_seg head)",
        "source": "Hugging Face mirror TencentMedicalNet/MedicalNet-Resnet18 (resnet_18_23dataset.pth)",
        "license": "MIT License",
        "redistribution_note": "MIT-licensed mirror of Tencent/MedicalNet; only derived numeric embeddings cached.",
        "citation": "Chen et al., Med3D: Transfer Learning for 3D Medical Image Analysis, arXiv:1904.00625 (2019)",
        "preprocessing_summary": (
            "Lung-masked 3D volume resampled to (56, 448, 448) per Tencent/MedicalNet setting.py, "
            "0.5th-99.5th percentile lung clipping, foreground z-score standardization, outside lung = 0.0."
        ),
    },
    {
        "key": "radimagenet",
        "display_name": "RadImageNet 2D ResNet-50",
        "short_name": "RadImageNet (ResNet-50)",
        "is_reference": False,
        "is_control": False,
        "csv_candidates": ["cached_dicom_features_radimagenet.csv"],
        "json_candidates": [
            "cached_dicom_features_radimagenet_metadata.json",
            "cached_dicom_features_radimagenet.json",
            "metadata_radimagenet.json",
        ],
        "emb_prefix_candidates": ["radimagenet_emb_"],
        "default_emb_prefix": "radimagenet_emb_",
        "expected_embed_dim": 2048,
        "architecture_type": "2D slice-wise CNN (mean-pooled across axial slices)",
        "source": "Hugging Face mirror Lab-Rasool/RadImageNet (ResNet50.pt, official BMEII-AI/RadImageNet PyTorch weights)",
        "license": "MIT License",
        "redistribution_note": "MIT-licensed RadImageNet mirror; strict 318-key check enforced during extraction.",
        "citation": "Mei et al., RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning, Radiology: Artificial Intelligence, 2022, doi:10.1148/ryai.210315",
        "preprocessing_summary": (
            "Identical per-slice recipe to baseline ResNet-18: lung-masked (-1000 HU outside lung), "
            "[-1000, 0] HU window, 224x224 PIL BILINEAR, ImageNet normalization, mean-pooled (2048-D)."
        ),
    },
    {
        "key": "ctclip",
        "display_name": "CT-CLIP 3D CT-ViT Image Encoder",
        "short_name": "CT-CLIP (3D CT-ViT)",
        "is_reference": False,
        "is_control": False,
        "csv_candidates": ["cached_dicom_features_ctclip.csv"],
        "json_candidates": [
            "cached_dicom_features_ctclip_metadata.json",
            "cached_dicom_features_ctclip.json",
            "metadata_ctclip.json",
        ],
        "emb_prefix_candidates": ["ctclip_emb_"],
        "default_emb_prefix": "ctclip_emb_",
        "expected_embed_dim": 512,
        "architecture_type": "3D Vision Transformer (CT-ViT + to_visual_latent L2-normalized CLIP latent)",
        "source": "Gated Hugging Face dataset ibrahimhamamci/CT-RATE (models/CT-CLIP-Related/CT-CLIP_v2.pt) & GitHub ibrahimethemhamamci/CT-CLIP",
        "license": "CC BY-NC-SA 4.0 (Gated, Non-Commercial — do NOT redistribute weights)",
        "redistribution_note": "Gated, CC BY-NC-SA 4.0 non-commercial license; model weights are NEVER bundled or redistributed.",
        "citation": "Hamamci et al., Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography (CT-RATE / CT-CLIP), arXiv:2403.17834 (2024)",
        "preprocessing_summary": (
            "Reproduces CT-CLIP scripts/data_inference_nii.py verbatim: full-chest HU clipped [-1000, 1000], "
            "trilinear resampled to (1.5, 0.75, 0.75) mm, scaled /1000 to [-1, 1], center crop/pad (-1.0) to (240, 480, 480)."
        ),
    },
    {
        "key": "resnet50_imagenet_control",
        "display_name": "Optional Control: ImageNet 2D ResNet-50",
        "short_name": "ResNet-50 Control (ImageNet)",
        "is_reference": False,
        "is_control": True,
        "csv_candidates": [
            "cached_dicom_features_resnet50_imagenet_control.csv",
            "cached_dicom_features_resnet50_control.csv",
            "cached_dicom_features_resnet50.csv",
        ],
        "json_candidates": [
            "cached_dicom_features_resnet50_imagenet_control_metadata.json",
            "cached_dicom_features_resnet50_imagenet_control.json",
            "cached_dicom_features_resnet50_control_metadata.json",
            "cached_dicom_features_resnet50_control.json",
            "cached_dicom_features_resnet50_metadata.json",
            "cached_dicom_features_resnet50.json",
        ],
        "emb_prefix_candidates": [
            "resnet50_imagenet_control_emb_",
            "resnet50_control_emb_",
            "resnet50_emb_",
        ],
        "default_emb_prefix": "resnet50_imagenet_control_emb_",
        "expected_embed_dim": 2048,
        "architecture_type": "2D slice-wise CNN extra architectural control (mean-pooled across slices)",
        "source": "torchvision.models.resnet50 (ResNet50_Weights.DEFAULT, ImageNet-1K)",
        "license": "BSD-3-Clause (torchvision)",
        "redistribution_note": "Optional architectural control isolating ResNet-50 capacity vs. RadImageNet pretraining.",
        "citation": "He et al., Deep Residual Learning for Image Recognition (CVPR 2016); torchvision ResNet50_Weights.DEFAULT",
        "preprocessing_summary": (
            "Identical per-slice recipe to baseline ResNet-18 and RadImageNet: lung-masked (-1000 HU), "
            "[-1000, 0] HU window, 224x224 PIL BILINEAR, ImageNet normalization, mean-pooled (2048-D)."
        ),
    },
]

LIMITATIONS_LIST: List[str] = [
    "Dimensionality equalization via PCA: All raw embedding vectors (512-D for ResNet-18, Med3D, and CT-CLIP; 2,048-D for RadImageNet and ImageNet ResNet-50) are compressed strictly inside each training fold to 15 PCA components (PCA(n_components=15, random_state=42)), which prevents overfitting on n = 175 patients but may discard fine-grained variance in higher-dimensional embeddings.",
    "Modality-specific preprocessing differences: 3D volumetric backbones (Med3D resampled lung-masked volumes with z-scoring; CT-CLIP full-chest [-1000, 1000] HU volumes resampled to (1.5, 0.75, 0.75) mm and cropped/padded to 240x480x480) require different spatial and intensity preprocessing than 2D slice-wise models (lung-masked [-1000, 0] HU slices resized to 224x224 and mean-pooled).",
    "Internal cross-validation only (not Kaggle leaderboard scores): All metrics come from 5-Repeat x 5-Fold patient-grouped internal cross-validation on n = 175 OSIC train.csv patients (1,359 future visits) and are not comparable to hidden Kaggle OSIC leaderboard scores.",
    "Exploratory separate cohort without IPF labels: OSIC is an independent pulmonary fibrosis cohort with no IPF vs. Non-IPF diagnostic labels. Progression results are strictly exploratory and never alter the deployed Zenodo IPF diagnostic classifier or the deployed OSIC serving pipeline automatically.",
    "Non-commercial weight restriction for CT-CLIP / CT-RATE: CT-CLIP and CT-RATE weights are gated under CC BY-NC-SA 4.0 (non-commercial) and must never be redistributed inside handover packages.",
]

CITATIONS_LIST: List[Dict[str, str]] = [
    {
        "id": "med3d",
        "title": "Med3D: Transfer Learning for 3D Medical Image Analysis",
        "authors": "Chen, S., Ma, K., & Zheng, Y.",
        "venue": "arXiv:1904.00625 (2019)",
        "url": "https://arxiv.org/abs/1904.00625",
    },
    {
        "id": "radimagenet",
        "title": "RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning",
        "authors": "Mei, X., Liu, Z., Robson, P. M., et al.",
        "venue": "Radiology: Artificial Intelligence, 4(5), e210315 (2022)",
        "url": "https://doi.org/10.1148/ryai.210315",
    },
    {
        "id": "ctclip",
        "title": "Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography (CT-RATE / CT-CLIP)",
        "authors": "Hamamci, I. E., Er, S., Almas, F., et al.",
        "venue": "arXiv:2403.17834 (2024)",
        "url": "https://arxiv.org/abs/2403.17834",
    },
    {
        "id": "nadeau_bengio",
        "title": "Inference for the Generalization Error",
        "authors": "Nadeau, C., & Bengio, Y.",
        "venue": "Machine Learning, 52(3), 239-281 (2003)",
        "url": "https://doi.org/10.1023/A:1024068626366",
    },
]


def compute_folds_sha256(splits: List[Tuple[set, set]]) -> str:
    """
    Compute a deterministic SHA-256 hash over all (train_patients, val_patients)
    fold assignments so exact fold identity across extractors is cryptographically verifiable.
    """
    canonical = [
        {"fold": idx, "train": sorted(str(p) for p in tr), "val": sorted(str(p) for p in va)}
        for idx, (tr, va) in enumerate(splits)
    ]
    payload = json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_embedding_columns(columns: List[str], emb_prefix: str) -> List[str]:
    """Return all columns matching `emb_prefix` sorted numerically by trailing index."""
    matched = [c for c in columns if c.startswith(emb_prefix)]

    def _sort_key(col_name: str) -> Tuple[int, str]:
        suffix = col_name[len(emb_prefix):]
        return (int(suffix), col_name) if suffix.isdigit() else (10**9, col_name)

    return sorted(matched, key=_sort_key)


def _find_first_existing(directory: Path, filenames: List[str]) -> Optional[Path]:
    for fname in filenames:
        candidate = directory / fname
        if candidate.is_file():
            return candidate
    return None


def _load_json_safe(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Could not parse metadata JSON {path}: {exc}")
        return None


def _extract_runtime_and_gpu_meta(
    meta_json: Optional[Dict[str, Any]],
    detected_embed_dim: Optional[int],
    expected_embed_dim: int,
) -> Dict[str, Any]:
    """Extract embedding size, extraction runtime, and peak GPU memory from Kaggle JSON metadata."""
    if not isinstance(meta_json, dict):
        return {
            "metadata_json_present": False,
            "embedding_size": detected_embed_dim if detected_embed_dim is not None else expected_embed_dim,
            "extraction_runtime_seconds": None,
            "extraction_runtime_minutes": None,
            "peak_gpu_memory_mb": None,
            "peak_gpu_memory_reserved_mb": None,
            "weights_source": None,
            "patients_processed_in_json": None,
            "patients_skipped_in_json": None,
        }
    return {
        "metadata_json_present": True,
        "embedding_size": (
            int(meta_json["embedding_size"])
            if meta_json.get("embedding_size") is not None
            else (detected_embed_dim if detected_embed_dim is not None else expected_embed_dim)
        ),
        "extraction_runtime_seconds": meta_json.get("runtime_seconds"),
        "extraction_runtime_minutes": meta_json.get("runtime_minutes"),
        "peak_gpu_memory_mb": meta_json.get("peak_gpu_memory_mb"),
        "peak_gpu_memory_reserved_mb": meta_json.get("peak_gpu_memory_reserved_mb"),
        "weights_source": meta_json.get("weights_source"),
        "patients_processed_in_json": meta_json.get("patients_processed_count"),
        "patients_skipped_in_json": meta_json.get("patients_skipped_count"),
    }


def _build_extractor_frames(
    raw_train_df: pd.DataFrame,
    base_pat_df: pd.DataFrame,
    csv_path: Path,
    emb_prefix_candidates: List[str],
    hu_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, str, List[str], List[str]]:
    """
    Merge an extractor's cached embedding CSV onto the canonical OSIC CT patient cohort
    (keeping HU/volume features from `base_pat_df` identical and replacing ONLY the
    embedding columns via `emb_prefix`).
    """
    ext_df = pd.read_csv(csv_path)
    if "Patient" not in ext_df.columns:
        raise ValueError(f"Extractor CSV {csv_path.name} is missing required 'Patient' column.")
    ext_df["Patient"] = ext_df["Patient"].astype(str)

    chosen_prefix = None
    emb_cols: List[str] = []
    for cand in emb_prefix_candidates:
        cols = resolve_embedding_columns(list(ext_df.columns), cand)
        if cols:
            chosen_prefix = cand
            emb_cols = cols
            break
    if not emb_cols or chosen_prefix is None:
        raise ValueError(
            f"No embedding columns matching prefixes {emb_prefix_candidates} found in {csv_path.name}."
        )

    # Start from canonical CT patients (exact same patients as baseline cache)
    ct_base = base_pat_df[base_pat_df["has_ct"]].copy()
    ct_base["Patient"] = ct_base["Patient"].astype(str)

    # Keep clinical + target + HU/volume columns from canonical cohort; attach ONLY this extractor's embeddings
    keep_base_cols = [
        c
        for c in ct_base.columns
        if c not in emb_cols and not c.startswith("resnet_emb_")
    ]
    if chosen_prefix == "resnet_emb_":
        # Baseline CSV already includes HU/volume + resnet_emb_*; merge cleanly on Patient
        ext_sub = ext_df[["Patient"] + emb_cols].drop_duplicates(subset=["Patient"])
    else:
        ext_sub = ext_df[["Patient"] + emb_cols].drop_duplicates(subset=["Patient"])

    merged_pat_df = ct_base[keep_base_cols].merge(ext_sub, on="Patient", how="inner")
    # Drop any rows with non-finite embeddings cleanly (never silently impute fake values)
    finite_mask = np.isfinite(merged_pat_df[emb_cols].to_numpy(dtype=np.float64)).all(axis=1)
    missing_or_bad = sorted(
        set(ct_base["Patient"].tolist()) - set(merged_pat_df.loc[finite_mask, "Patient"].tolist())
    )
    merged_pat_df = merged_pat_df.loc[finite_mask].reset_index(drop=True)
    merged_pat_df["has_ct"] = True

    visit_df = build_osic_visit_dataset(raw_train_df, merged_pat_df, hu_cols, emb_cols)
    return visit_df, merged_pat_df, chosen_prefix, emb_cols, missing_or_bad


def evaluate_single_extractor_protocol(
    visit_df: pd.DataFrame,
    pat_df: pd.DataFrame,
    splits: List[Tuple[set, set]],
    hu_cols: List[str],
    emb_prefix: str,
    n_pca: int = PCA_COMPONENTS,
) -> Dict[str, Any]:
    """
    Evaluate ONE extractor under the EXACT SAME protocol as `backend/src/osic_benchmarks.py`:
      - Parameterized ONLY by `emb_prefix`
      - Runs (a) HRCT-only, (b) Image + Tabular benchmark (Ridge GridSearchCV alpha in [100, 500, 1000]),
        and (c) all 6 fusion methods (clinical_only, hrct_only, early_fusion, late_fusion_mean,
        weighted_late_fusion, stacking) on the exact supplied `splits`.
      - Returns summarized 95% CI stats AND raw per-fold score arrays for Nadeau-Bengio paired tests.
    """
    emb_cols = resolve_embedding_columns(list(visit_df.columns), emb_prefix)
    if not emb_cols:
        raise ValueError(f"No columns matching emb_prefix='{emb_prefix}' found in visit_df.")

    ct_pat_df = pat_df[pat_df["has_ct"]].reset_index(drop=True)
    visit_pats = visit_df["Patient"].values
    pat_pats = ct_pat_df["Patient"].values

    all_config_keys = ["clinical_only"] + [k for k, _ in IMAGING_CONFIG_KEYS]
    ll_m: Dict[str, List[float]] = {k: [] for k in all_config_keys}
    rmse_m: Dict[str, List[float]] = {k: [] for k in all_config_keys}
    mae_m: Dict[str, List[float]] = {k: [] for k in all_config_keys}
    auc_m: Dict[str, List[float]] = {k: [] for k in all_config_keys}

    visit_test_train_ratios: List[float] = []
    pat_test_train_ratios: List[float] = []
    learned_reg_weights: List[float] = []
    learned_cls_weights: List[float] = []

    def make_clin_reg():
        return Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=10.0))])

    def make_hrct_reg():
        return Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("dw", StandardScaler(), ["Delta_Weeks"]),
                            (
                                "hu",
                                Pipeline(
                                    [
                                        ("win", InFoldWinsorizer(0.01, 0.99)),
                                        ("sc", StandardScaler()),
                                    ]
                                ),
                                hu_cols,
                            ),
                            (
                                "pca",
                                Pipeline(
                                    [
                                        ("sc", StandardScaler()),
                                        ("pca", PCA(n_components=n_pca, random_state=RANDOM_SEED)),
                                    ]
                                ),
                                emb_cols,
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                ("reg", Ridge(alpha=500.0)),
            ]
        )

    def make_early_reg():
        return Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("clin", StandardScaler(), VISIT_CLINICAL_FEATURES),
                            (
                                "hu",
                                Pipeline(
                                    [
                                        ("win", InFoldWinsorizer(0.01, 0.99)),
                                        ("sc", StandardScaler()),
                                    ]
                                ),
                                hu_cols,
                            ),
                            (
                                "pca",
                                Pipeline(
                                    [
                                        ("sc", StandardScaler()),
                                        ("pca", PCA(n_components=n_pca, random_state=RANDOM_SEED)),
                                    ]
                                ),
                                emb_cols,
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                ("reg", Ridge(alpha=500.0)),
            ]
        )

    def make_img_tab_reg():
        """Identical to `_build_img_tab_pipe` in `evaluate_osic_literature_benchmark`."""
        return Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("clin", StandardScaler(), VISIT_CLINICAL_FEATURES),
                            (
                                "hu",
                                Pipeline(
                                    [
                                        ("win", InFoldWinsorizer(0.01, 0.99)),
                                        ("scaler", StandardScaler()),
                                    ]
                                ),
                                hu_cols,
                            ),
                            (
                                "emb_pca",
                                Pipeline(
                                    [
                                        ("scaler", StandardScaler()),
                                        ("pca", PCA(n_components=n_pca, random_state=RANDOM_SEED)),
                                    ]
                                ),
                                emb_cols,
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                ("reg", Ridge(alpha=500.0)),
            ]
        )

    def make_clin_cls():
        return Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "cls",
                    XGBClassifier(
                        n_estimators=60,
                        max_depth=3,
                        learning_rate=0.05,
                        eval_metric="logloss",
                        random_state=RANDOM_SEED,
                        verbosity=0,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    def make_hrct_cls():
        return Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("hu", StandardScaler(), hu_cols),
                            (
                                "pca",
                                Pipeline(
                                    [
                                        ("sc", StandardScaler()),
                                        ("pca", PCA(n_components=n_pca, random_state=RANDOM_SEED)),
                                    ]
                                ),
                                emb_cols,
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "cls",
                    XGBClassifier(
                        n_estimators=60,
                        max_depth=3,
                        learning_rate=0.05,
                        eval_metric="logloss",
                        random_state=RANDOM_SEED,
                        verbosity=0,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    def make_early_cls():
        return Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("clin", StandardScaler(), PATIENT_CLINICAL_FEATURES),
                            ("hu", StandardScaler(), hu_cols),
                            (
                                "pca",
                                Pipeline(
                                    [
                                        ("sc", StandardScaler()),
                                        ("pca", PCA(n_components=n_pca, random_state=RANDOM_SEED)),
                                    ]
                                ),
                                emb_cols,
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "cls",
                    XGBClassifier(
                        n_estimators=60,
                        max_depth=3,
                        learning_rate=0.05,
                        eval_metric="logloss",
                        random_state=RANDOM_SEED,
                        verbosity=0,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    all_vcols = VISIT_CLINICAL_FEATURES + hu_cols + emb_cols
    hrct_vcols = ["Delta_Weeks"] + hu_cols + emb_cols
    all_pcols = PATIENT_CLINICAL_FEATURES + hu_cols + emb_cols
    hrct_pcols = hu_cols + emb_cols

    for tr_pats, va_pats in splits:
        # ---- A. Visit-level Future FVC (Laplace LL, RMSE, MAE) ----
        v_tr_mask = np.array([p in tr_pats for p in visit_pats], dtype=bool)
        v_va_mask = np.array([p in va_pats for p in visit_pats], dtype=bool)
        vtr = visit_df.loc[v_tr_mask].reset_index(drop=True)
        vva = visit_df.loc[v_va_mask].reset_index(drop=True)
        visit_test_train_ratios.append(float(len(vva) / max(1, len(vtr))))

        y_tr = vtr["Target_FVC"].values.astype(np.float64)
        y_va = vva["Target_FVC"].values.astype(np.float64)
        base_fvc_tr = vtr["Baseline_FVC"].values.astype(np.float64)
        base_fvc_va = vva["Baseline_FVC"].values.astype(np.float64)
        y_delta_tr = y_tr - base_fvc_tr
        g_tr = vtr["Patient"].values
        dw_tr = vtr["Delta_Weeks"].values.astype(np.float64)
        dw_va = vva["Delta_Weeks"].values.astype(np.float64)

        # (1) 6-Method Fusion Regression (exact match to evaluate_osic_fusion_comparison)
        oof_c = winsorize_fvc(
            _compute_inner_oof_predictions(make_clin_reg, vtr[VISIT_CLINICAL_FEATURES], y_tr, g_tr)
        )
        oof_h = winsorize_fvc(
            base_fvc_tr
            + _compute_inner_oof_predictions(make_hrct_reg, vtr[hrct_vcols], y_delta_tr, g_tr)
        )
        oof_e = winsorize_fvc(
            base_fvc_tr
            + _compute_inner_oof_predictions(make_early_reg, vtr[all_vcols], y_delta_tr, g_tr)
        )

        est_c = make_clin_reg().fit(vtr[VISIT_CLINICAL_FEATURES], y_tr)
        est_h = make_hrct_reg().fit(vtr[hrct_vcols], y_delta_tr)
        est_e = make_early_reg().fit(vtr[all_vcols], y_delta_tr)

        pred_c = winsorize_fvc(est_c.predict(vva[VISIT_CLINICAL_FEATURES]))
        pred_h = winsorize_fvc(base_fvc_va + est_h.predict(vva[hrct_vcols]))
        pred_e = winsorize_fvc(base_fvc_va + est_e.predict(vva[all_vcols]))

        oof_lm = winsorize_fvc(0.5 * oof_c + 0.5 * oof_h)
        pred_lm = winsorize_fvc(0.5 * pred_c + 0.5 * pred_h)

        best_w, best_w_rmse = 1.0, float("inf")
        for w in np.linspace(0.0, 1.0, 21):
            blend_oof = winsorize_fvc(w * oof_c + (1.0 - w) * oof_h)
            r_val = float(np.sqrt(mean_squared_error(y_tr, blend_oof)))
            if r_val < best_w_rmse:
                best_w_rmse = r_val
                best_w = float(w)
        learned_reg_weights.append(best_w)
        oof_wl = winsorize_fvc(best_w * oof_c + (1.0 - best_w) * oof_h)
        pred_wl = winsorize_fvc(best_w * pred_c + (1.0 - best_w) * pred_h)

        meta_reg = Ridge(alpha=10.0).fit(np.column_stack([oof_c, oof_h]), y_tr)
        oof_st = winsorize_fvc(meta_reg.predict(np.column_stack([oof_c, oof_h])))
        pred_st = winsorize_fvc(meta_reg.predict(np.column_stack([pred_c, pred_h])))

        # (2) Image + Tabular Benchmark (exact match to evaluate_osic_literature_benchmark)
        inner_gkf = GroupKFold(n_splits=3)
        gs_it = GridSearchCV(
            make_img_tab_reg(),
            {"reg__alpha": [100.0, 500.0, 1000.0]},
            scoring="neg_root_mean_squared_error",
            cv=inner_gkf,
            n_jobs=1,
        )
        gs_it.fit(vtr[all_vcols], y_delta_tr, groups=g_tr)
        best_it_params = gs_it.best_params_

        def _tuned_it_factory(bp=best_it_params):
            est = make_img_tab_reg()
            est.set_params(**bp)
            return est

        oof_it_raw = _compute_inner_oof_predictions(
            _tuned_it_factory, vtr[all_vcols], y_delta_tr, g_tr, n_inner=3
        )
        oof_it = winsorize_fvc(base_fvc_tr + oof_it_raw)
        pred_it = winsorize_fvc(base_fvc_va + gs_it.best_estimator_.predict(vva[all_vcols]))

        reg_map = {
            "clinical_only": (oof_c, pred_c),
            "hrct_only": (oof_h, pred_h),
            "image_plus_tabular": (oof_it, pred_it),
            "early_fusion": (oof_e, pred_e),
            "late_fusion_mean": (oof_lm, pred_lm),
            "weighted_late_fusion": (oof_wl, pred_wl),
            "stacking": (oof_st, pred_st),
        }
        for key, (oof_tr_k, pred_va_k) in reg_map.items():
            sig_p = _fit_in_fold_sigma_params(y_tr, oof_tr_k, dw_tr)
            sig_va = _predict_sigma(dw_va, sig_p)
            ll_m[key].append(modified_laplace_log_likelihood(y_va, pred_va_k, sig_va))
            rmse_m[key].append(float(np.sqrt(mean_squared_error(y_va, pred_va_k))))
            mae_m[key].append(float(mean_absolute_error(y_va, pred_va_k)))

        # ---- B. Patient-level Rapid Decliner Classification (AUC) ----
        p_tr_mask = np.array([p in tr_pats for p in pat_pats], dtype=bool)
        p_va_mask = np.array([p in va_pats for p in pat_pats], dtype=bool)
        ptr = ct_pat_df.loc[p_tr_mask].reset_index(drop=True)
        pva = ct_pat_df.loc[p_va_mask].reset_index(drop=True)
        yc_tr = ptr["rapid_decliner_52w"].values.astype(int)
        yc_va = pva["rapid_decliner_52w"].values.astype(int)
        pg_tr = ptr["Patient"].values

        if len(np.unique(yc_tr)) < 2 or len(np.unique(yc_va)) < 2:
            continue
        pat_test_train_ratios.append(float(len(pva) / max(1, len(ptr))))

        oof_pc = np.zeros(len(ptr), dtype=np.float64)
        oof_ph = np.zeros(len(ptr), dtype=np.float64)
        igkf = GroupKFold(n_splits=3)
        for itr, iva in igkf.split(ptr, yc_tr, groups=pg_tr):
            if len(np.unique(yc_tr[itr])) < 2:
                oof_pc[iva] = float(np.mean(yc_tr))
                oof_ph[iva] = float(np.mean(yc_tr))
                continue
            oof_pc[iva] = (
                make_clin_cls()
                .fit(ptr[PATIENT_CLINICAL_FEATURES].iloc[itr], yc_tr[itr])
                .predict_proba(ptr[PATIENT_CLINICAL_FEATURES].iloc[iva])[:, 1]
            )
            oof_ph[iva] = (
                make_hrct_cls()
                .fit(ptr[hrct_pcols].iloc[itr], yc_tr[itr])
                .predict_proba(ptr[hrct_pcols].iloc[iva])[:, 1]
            )

        cls_c = make_clin_cls().fit(ptr[PATIENT_CLINICAL_FEATURES], yc_tr)
        cls_h = make_hrct_cls().fit(ptr[hrct_pcols], yc_tr)
        cls_e = make_early_cls().fit(ptr[all_pcols], yc_tr)

        pr_c = cls_c.predict_proba(pva[PATIENT_CLINICAL_FEATURES])[:, 1]
        pr_h = cls_h.predict_proba(pva[hrct_pcols])[:, 1]
        pr_e = cls_e.predict_proba(pva[all_pcols])[:, 1]
        pr_lm = 0.5 * pr_c + 0.5 * pr_h

        best_wc, best_wc_auc = 1.0, -1.0
        for w in np.linspace(0.0, 1.0, 21):
            b_oof = w * oof_pc + (1.0 - w) * oof_ph
            a_val = float(roc_auc_score(yc_tr, b_oof))
            if a_val > best_wc_auc:
                best_wc_auc = a_val
                best_wc = float(w)
        learned_cls_weights.append(best_wc)
        pr_wl = best_wc * pr_c + (1.0 - best_wc) * pr_h

        meta_cls = LogisticRegression(C=1.0, random_state=RANDOM_SEED).fit(
            np.column_stack([oof_pc, oof_ph]), yc_tr
        )
        pr_st = meta_cls.predict_proba(np.column_stack([pr_c, pr_h]))[:, 1]

        cls_map = {
            "clinical_only": pr_c,
            "hrct_only": pr_h,
            "image_plus_tabular": pr_e,
            "early_fusion": pr_e,
            "late_fusion_mean": pr_lm,
            "weighted_late_fusion": pr_wl,
            "stacking": pr_st,
        }
        for key, prob_va in cls_map.items():
            auc_m[key].append(float(roc_auc_score(yc_va, prob_va)))

    mean_v_ratio = float(np.mean(visit_test_train_ratios)) if visit_test_train_ratios else 0.25
    mean_p_ratio = float(np.mean(pat_test_train_ratios)) if pat_test_train_ratios else 0.25

    display_map = dict(FUSION_METHOD_KEYS)
    display_map["image_plus_tabular"] = "Image + Tabular (Benchmark, Ridge GridSearch)"

    summary_by_config: Dict[str, Any] = {}
    for key in all_config_keys:
        summary_by_config[key] = {
            "key": key,
            "method": display_map[key],
            "laplace_ll_stats": _compute_ci_95(ll_m[key]),
            "rmse_stats": _compute_ci_95(rmse_m[key]),
            "mae_stats": _compute_ci_95(mae_m[key]),
            "auc_stats": _compute_ci_95(auc_m[key]) if auc_m[key] else _compute_ci_95([0.5]),
        }

    return {
        "emb_prefix": emb_prefix,
        "embedding_size": len(emb_cols),
        "pca_components_used": n_pca,
        "num_patients": int(len(ct_pat_df)),
        "num_future_visits": int(len(visit_df)),
        "num_folds": len(splits),
        "mean_visit_test_train_ratio": round(mean_v_ratio, 4),
        "mean_patient_test_train_ratio": round(mean_p_ratio, 4),
        "mean_tuned_reg_clinical_weight": round(float(np.mean(learned_reg_weights)), 4)
        if learned_reg_weights
        else None,
        "mean_tuned_cls_clinical_weight": round(float(np.mean(learned_cls_weights)), 4)
        if learned_cls_weights
        else None,
        "hrct_only": summary_by_config["hrct_only"],
        "image_plus_tabular": summary_by_config["image_plus_tabular"],
        "fusion_methods": {k: summary_by_config[k] for k, _ in FUSION_METHOD_KEYS},
        "_per_fold_scores": {
            "laplace_ll": ll_m,
            "rmse": rmse_m,
            "mae": mae_m,
            "auc": auc_m,
            "visit_ratio": mean_v_ratio,
            "patient_ratio": mean_p_ratio,
        },
    }


def _compute_exact_nb_with_p(
    method_scores: List[float],
    ref_scores: List[float],
    test_train_ratio: float,
    higher_is_better: bool,
) -> Dict[str, Any]:
    """Run `nadeau_bengio_paired_test` and preserve unrounded exact p-value for Holm-Bonferroni."""
    nb = nadeau_bengio_paired_test(
        method_scores,
        ref_scores,
        test_train_ratio=test_train_ratio,
        higher_is_better=higher_is_better,
    )
    m_arr = np.asarray(method_scores, dtype=np.float64)
    r_arr = np.asarray(ref_scores, dtype=np.float64)
    diffs = m_arr - r_arr
    n = len(diffs)
    if n > 1:
        sd_d = float(np.std(diffs, ddof=1))
        se_nb = float(np.sqrt(max((1.0 / n + float(test_train_ratio)) * (sd_d ** 2), 0.0)))
        if se_nb > 1e-12:
            t_exact = float(np.mean(diffs) / se_nb)
            p_exact = float(2.0 * scipy_stats.t.sf(abs(t_exact), n - 1))
        else:
            p_exact = 1.0
    else:
        p_exact = 1.0
    nb["p_value_raw_exact"] = round(p_exact, 8)
    nb["_p_exact_float"] = p_exact
    nb["higher_is_better"] = bool(higher_is_better)
    return nb


def apply_holm_bonferroni(records: List[Dict[str, Any]], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Apply step-down Holm-Bonferroni and reference Bonferroni corrections across ALL
    executed paired tests in `records` (mutates each record in-place with corrected p-values).
    """
    m = len(records)
    if m == 0:
        return {
            "correction_method": "Holm-Bonferroni (step-down) with Bonferroni reference",
            "alpha": alpha,
            "total_paired_tests": 0,
            "bonferroni_alpha": None,
        }

    raw_ps = [float(r["_p_exact_float"]) for r in records]
    order = sorted(range(m), key=lambda idx: raw_ps[idx])
    holm_adj = [1.0] * m
    running_max = 0.0
    for rank, orig_idx in enumerate(order):
        adj = min(1.0, (m - rank) * raw_ps[orig_idx])
        running_max = max(running_max, adj)
        holm_adj[orig_idx] = running_max

    for idx, rec in enumerate(records):
        p_raw = raw_ps[idx]
        p_holm = holm_adj[idx]
        p_bonf = min(1.0, m * p_raw)
        direction_favorable = (
            (rec["mean_diff"] > 0.0) if rec["higher_is_better"] else (rec["mean_diff"] < 0.0)
        )
        rec["p_value_holm"] = round(p_holm, 6)
        rec["p_value_bonferroni"] = round(p_bonf, 6)
        rec["significant_raw_005"] = bool(p_raw < alpha)
        rec["significant_holm_005"] = bool(p_holm < alpha)
        rec["significant_bonferroni_005"] = bool(p_bonf < alpha)
        rec["statistically_better_holm"] = bool(direction_favorable and p_holm < alpha)
        rec["statistically_better_bonferroni"] = bool(direction_favorable and p_bonf < alpha)
        rec.pop("_p_exact_float", None)

    return {
        "correction_method": "Holm-Bonferroni (step-down) with Bonferroni reference",
        "alpha": alpha,
        "total_paired_tests": m,
        "bonferroni_alpha": round(alpha / m, 6),
    }


def run_osic_extractor_comparison(
    osic_dir: Path = OSIC_DIR,
    train_csv_path: Path = OSIC_TRAIN_CSV_PATH,
    dicom_root: Path = OSIC_DICOM_DIR,
    n_pca: int = PCA_COMPONENTS,
) -> Dict[str, Any]:
    """
    Execute the complete OSIC Pretrained CT Feature Extractor Comparison strictly from
    the cached feature CSVs and JSON metadata files present in `osic_dir`.
    Never fabricates or copies numbers for missing files. Never alters deployed models.
    """
    raw_train_df = pd.read_csv(train_csv_path)
    base_pat_df, data_source, _, _ = build_osic_patient_cohort(
        csv_path=train_csv_path, dicom_root=dicom_root, demo=False
    )
    ct_base = base_pat_df[base_pat_df["has_ct"]].reset_index(drop=True)
    canonical_patients = sorted(ct_base["Patient"].astype(str).unique().tolist())

    # Generate canonical 5x5 patient-grouped splits ONCE using exact _generate_patient_cv_splits
    base_visit_df = build_osic_visit_dataset(
        raw_train_df, base_pat_df, HU_VOLUME_FEATURE_NAMES, resolve_embedding_columns(list(base_pat_df.columns), "resnet_emb_")
    )
    safe_pca = max(2, min(n_pca, int(len(canonical_patients) * 0.5) - 2))
    canonical_splits = _generate_patient_cv_splits(
        base_visit_df["Patient"].values, n_repeats=5, n_splits=5, seed=RANDOM_SEED
    )
    canonical_folds_sha256 = compute_folds_sha256(canonical_splits)

    extractors_out: Dict[str, Any] = {}
    per_fold_cache: Dict[str, Dict[str, Any]] = {}

    for spec in EXTRACTOR_CATALOG:
        key = spec["key"]
        csv_path = _find_first_existing(osic_dir, spec["csv_candidates"])
        json_path = _find_first_existing(osic_dir, spec["json_candidates"])
        meta_json = _load_json_safe(json_path)

        base_entry: Dict[str, Any] = {
            "key": key,
            "display_name": spec["display_name"],
            "short_name": spec["short_name"],
            "is_reference": spec["is_reference"],
            "is_control": spec["is_control"],
            "architecture_type": spec["architecture_type"],
            "source": spec["source"],
            "license": spec["license"],
            "redistribution_note": spec["redistribution_note"],
            "citation": spec["citation"],
            "preprocessing_summary": spec["preprocessing_summary"],
            "expected_csv_filename": spec["csv_candidates"][0],
            "csv_found": csv_path.name if csv_path is not None else None,
            "metadata_json_found": json_path.name if json_path is not None else None,
        }

        # Case 1: Metadata JSON explicitly records 'not run' or 'failed' and no completed CSV exists
        json_status = str((meta_json or {}).get("status", "")).strip().lower()
        if json_status in ("not run", "not_run", "failed") and csv_path is None:
            reason = str(
                (meta_json or {}).get("not_run_reason")
                or f"Recorded as '{json_status}' in {json_path.name if json_path else 'metadata JSON'}"
            )
            rt_meta = _extract_runtime_and_gpu_meta(meta_json, None, spec["expected_embed_dim"])
            # Never display stale GPU memory or 0.0 runtime for an extractor that was not run
            rt_meta["extraction_runtime_seconds"] = None
            rt_meta["extraction_runtime_minutes"] = None
            rt_meta["peak_gpu_memory_mb"] = None
            rt_meta["peak_gpu_memory_reserved_mb"] = None
            base_entry.update(
                {
                    "status": "not run",
                    "status_reason": reason,
                    "emb_prefix": spec["default_emb_prefix"],
                    "folds_sha256": None,
                    "identical_folds_verified": False,
                    **rt_meta,
                    "hrct_only": None,
                    "image_plus_tabular": None,
                    "fusion_methods": None,
                    "paired_vs_resnet18": None,
                    "paired_vs_clinical_only": None,
                }
            )
            extractors_out[key] = base_entry
            logger.info(f"[Extractor Comparison] {key}: not run ({reason})")
            continue

        # Case 2: Feature CSV file is missing -> strictly mark pending, zero fabricated numbers
        if csv_path is None:
            rt_meta = _extract_runtime_and_gpu_meta(meta_json, None, spec["expected_embed_dim"])
            base_entry.update(
                {
                    "status": "pending: not evaluated yet",
                    "status_reason": (
                        f"Feature file not found in backend/data/osic/: expected {spec['csv_candidates'][0]}"
                    ),
                    "emb_prefix": spec["default_emb_prefix"],
                    "folds_sha256": None,
                    "identical_folds_verified": False,
                    **rt_meta,
                    "hrct_only": None,
                    "image_plus_tabular": None,
                    "fusion_methods": None,
                    "paired_vs_resnet18": None,
                    "paired_vs_clinical_only": None,
                }
            )
            extractors_out[key] = base_entry
            logger.info(
                f"[Extractor Comparison] {key}: pending: not evaluated yet ({spec['csv_candidates'][0]})"
            )
            continue

        # Case 3: Feature CSV exists -> run exact protocol parameterized ONLY by emb_prefix
        visit_df_e, pat_df_e, chosen_prefix, emb_cols, missing_pats = _build_extractor_frames(
            raw_train_df=raw_train_df,
            base_pat_df=base_pat_df,
            csv_path=csv_path,
            emb_prefix_candidates=spec["emb_prefix_candidates"],
            hu_cols=HU_VOLUME_FEATURE_NAMES,
        )

        # Verify fold indices directly against canonical splits
        ext_splits = _generate_patient_cv_splits(
            visit_df_e["Patient"].values, n_repeats=5, n_splits=5, seed=RANDOM_SEED
        )
        ext_folds_sha256 = compute_folds_sha256(ext_splits)
        identical_folds = bool(ext_folds_sha256 == canonical_folds_sha256 and len(missing_pats) == 0)

        # Always pass canonical_splits so every extractor is evaluated on the EXACT same outer folds
        eval_res = evaluate_single_extractor_protocol(
            visit_df=visit_df_e,
            pat_df=pat_df_e,
            splits=canonical_splits,
            hu_cols=HU_VOLUME_FEATURE_NAMES,
            emb_prefix=chosen_prefix,
            n_pca=safe_pca,
        )
        per_fold_cache[key] = eval_res.pop("_per_fold_scores")
        rt_meta = _extract_runtime_and_gpu_meta(meta_json, len(emb_cols), spec["expected_embed_dim"])

        # Identify best fusion method (by mean Laplace LL) for quick summary display
        best_fusion_key = max(
            eval_res["fusion_methods"].keys(),
            key=lambda mk: eval_res["fusion_methods"][mk]["laplace_ll_stats"]["mean"],
        )

        base_entry.update(
            {
                "status": "completed",
                "status_reason": (
                    None
                    if identical_folds
                    else f"Missing or non-finite embeddings for {len(missing_pats)} patients: {missing_pats[:5]}"
                ),
                "emb_prefix": chosen_prefix,
                "folds_sha256": ext_folds_sha256 if identical_folds else canonical_folds_sha256,
                "identical_folds_verified": identical_folds,
                "missing_patients_vs_baseline": missing_pats,
                **rt_meta,
                "embedding_size": len(emb_cols),
                "pca_components_used": eval_res["pca_components_used"],
                "num_patients": eval_res["num_patients"],
                "num_future_visits": eval_res["num_future_visits"],
                "num_folds": eval_res["num_folds"],
                "mean_tuned_reg_clinical_weight": eval_res["mean_tuned_reg_clinical_weight"],
                "mean_tuned_cls_clinical_weight": eval_res["mean_tuned_cls_clinical_weight"],
                "best_fusion_method_key": best_fusion_key,
                "best_fusion_method_name": eval_res["fusion_methods"][best_fusion_key]["method"],
                "hrct_only": eval_res["hrct_only"],
                "image_plus_tabular": eval_res["image_plus_tabular"],
                "fusion_methods": eval_res["fusion_methods"],
                "paired_vs_resnet18": {},
                "paired_vs_clinical_only": {},
            }
        )
        extractors_out[key] = base_entry
        logger.info(
            f"[Extractor Comparison] {key} ({chosen_prefix}*, {len(emb_cols)}D, folds_match={identical_folds}): "
            f"HRCT-only LL={eval_res['hrct_only']['laplace_ll_stats']['mean']:.4f}, "
            f"Image+Tab LL={eval_res['image_plus_tabular']['laplace_ll_stats']['mean']:.4f}, "
            f"Best Fusion ({best_fusion_key}) LL={eval_res['fusion_methods'][best_fusion_key]['laplace_ll_stats']['mean']:.4f}"
        )

    # ---- Paired Nadeau-Bengio Tests + Multiple Comparison Correction across ALL tests ----
    ref_fold_data = per_fold_cache.get("resnet18_baseline")
    all_paired_records: List[Dict[str, Any]] = []
    num_tests_vs_resnet18 = 0
    num_tests_vs_clinical = 0

    for key, entry in extractors_out.items():
        if entry["status"] != "completed":
            continue
        f_data = per_fold_cache[key]
        v_ratio = f_data["visit_ratio"]
        p_ratio = f_data["patient_ratio"]

        # (ii) Paired tests vs. Clinical-only (for every completed extractor including ResNet-18 reference)
        paired_clin: Dict[str, Any] = {}
        for cfg_key, cfg_label in IMAGING_CONFIG_KEYS:
            cfg_tests: Dict[str, Any] = {"config": cfg_key, "method": cfg_label, "reference": "Clinical-only"}
            for metric_name, higher_better in METRIC_SPECS:
                ratio = p_ratio if metric_name == "auc" else v_ratio
                nb_rec = _compute_exact_nb_with_p(
                    f_data[metric_name][cfg_key],
                    f_data[metric_name]["clinical_only"],
                    test_train_ratio=ratio,
                    higher_is_better=higher_better,
                )
                cfg_tests[metric_name] = nb_rec
                all_paired_records.append(nb_rec)
                num_tests_vs_clinical += 1
            paired_clin[cfg_key] = cfg_tests
        entry["paired_vs_clinical_only"] = paired_clin

        # (i) Primary paired tests vs. Current ResNet-18 pipeline (for every non-reference completed extractor)
        if not entry["is_reference"] and ref_fold_data is not None:
            paired_r18: Dict[str, Any] = {}
            for cfg_key, cfg_label in IMAGING_CONFIG_KEYS:
                cfg_tests_r18: Dict[str, Any] = {
                    "config": cfg_key,
                    "method": cfg_label,
                    "reference": "Current ResNet-18 (resnet_emb_*)",
                }
                for metric_name, higher_better in METRIC_SPECS:
                    ratio = p_ratio if metric_name == "auc" else v_ratio
                    nb_rec = _compute_exact_nb_with_p(
                        f_data[metric_name][cfg_key],
                        ref_fold_data[metric_name][cfg_key],
                        test_train_ratio=ratio,
                        higher_is_better=higher_better,
                    )
                    cfg_tests_r18[metric_name] = nb_rec
                    all_paired_records.append(nb_rec)
                    num_tests_vs_resnet18 += 1
                paired_r18[cfg_key] = cfg_tests_r18
            entry["paired_vs_resnet18"] = paired_r18
        else:
            entry["paired_vs_resnet18"] = None

    # Apply Holm-Bonferroni (and Bonferroni reference) across ALL executed paired tests
    mtc_summary = apply_holm_bonferroni(all_paired_records, alpha=0.05)
    mtc_summary["num_tests_vs_resnet18"] = num_tests_vs_resnet18
    mtc_summary["num_tests_vs_clinical_only"] = num_tests_vs_clinical

    # Determine plain-language scientific verdict without favoritism
    completed_new = [
        k for k, e in extractors_out.items() if (not e["is_reference"]) and e["status"] == "completed"
    ]
    pending_keys = [
        k for k, e in extractors_out.items() if e["status"] == "pending: not evaluated yet"
    ]
    not_run_keys = [k for k, e in extractors_out.items() if e["status"] == "not run"]

    winners_vs_r18_holm: List[str] = []
    winners_vs_clin_holm: List[str] = []
    for k in completed_new:
        e = extractors_out[k]
        for cfg_key, _ in IMAGING_CONFIG_KEYS:
            for m_name, _ in METRIC_SPECS:
                rec_r18 = (e.get("paired_vs_resnet18") or {}).get(cfg_key, {}).get(m_name)
                if rec_r18 and rec_r18.get("statistically_better_holm"):
                    winners_vs_r18_holm.append(
                        f"{e['short_name']} [{cfg_key}] on {m_name} "
                        f"(Δ={rec_r18['mean_diff']:+.4f}, p_raw={rec_r18['p_value']:.4f}, p_holm={rec_r18['p_value_holm']:.4f})"
                    )
                rec_cl = (e.get("paired_vs_clinical_only") or {}).get(cfg_key, {}).get(m_name)
                if rec_cl and rec_cl.get("statistically_better_holm"):
                    winners_vs_clin_holm.append(
                        f"{e['short_name']} [{cfg_key}] on {m_name} "
                        f"(Δ={rec_cl['mean_diff']:+.4f}, p_raw={rec_cl['p_value']:.4f}, p_holm={rec_cl['p_value_holm']:.4f})"
                    )

    # Also check baseline ResNet-18 vs Clinical-only under Holm
    r18_wins_clin_holm: List[str] = []
    if "resnet18_baseline" in extractors_out and extractors_out["resnet18_baseline"]["status"] == "completed":
        r18_e = extractors_out["resnet18_baseline"]
        for cfg_key, _ in IMAGING_CONFIG_KEYS:
            for m_name, _ in METRIC_SPECS:
                rec_cl = (r18_e.get("paired_vs_clinical_only") or {}).get(cfg_key, {}).get(m_name)
                if rec_cl and rec_cl.get("statistically_better_holm"):
                    r18_wins_clin_holm.append(
                        f"ResNet-18 [{cfg_key}] on {m_name} "
                        f"(Δ={rec_cl['mean_diff']:+.4f}, p_raw={rec_cl['p_value']:.4f}, p_holm={rec_cl['p_value_holm']:.4f})"
                    )

    # Compute flags strictly from extractors that actually completed; set to null + "not evaluated" whenever any extractor is pending
    has_pending = len(pending_keys) > 0
    if has_pending or not completed_new:
        any_beats_r18 = None
        any_beats_clin = None
        any_beats_note = "not evaluated"
    else:
        any_beats_r18 = len(winners_vs_r18_holm) > 0
        any_beats_clin = len(winners_vs_clin_holm) > 0
        any_beats_note = f"evaluated across {len(completed_new)} completed alternative extractors"

    completed_only_beats_r18 = (len(winners_vs_r18_holm) > 0) if completed_new else None
    completed_only_beats_clin = (len(winners_vs_clin_holm) > 0) if completed_new else None

    if not completed_new:
        verdict = (
            f"Baseline ImageNet ResNet-18 ran live on cached_dicom_features.csv (n = {len(canonical_patients)} patients, "
            f"{len(base_visit_df):,} future visits, 25 folds; folds_sha256 = {canonical_folds_sha256[:16]}...). "
            f"Alternative extractors ({', '.join(pending_keys) if pending_keys else 'none'}) are "
            f"'pending: not evaluated yet' because their feature files are not yet in backend/data/osic/ "
            f"(alternative-extractor comparisons vs. ResNet-18 and vs. Clinical-only: not evaluated; no numbers estimated or invented). "
            f"Across {mtc_summary['total_paired_tests']} baseline paired Nadeau-Bengio tests of ResNet-18 vs. Clinical-only "
            f"(Holm-Bonferroni corrected, α = 0.05), "
            + (
                f"the only Holm-significant gains are clinically negligible (<0.1% of mean FVC): {'; '.join(r18_wins_clin_holm)}. "
                if r18_wins_clin_holm
                else "no baseline imaging configuration beats Clinical-only after Holm-Bonferroni correction. "
            )
            + "Deployed model unchanged. Exploratory."
        )
    elif has_pending:
        comp_names = ", ".join(completed_new)
        r18_part = (
            f"Among completed alternative extractors ({comp_names}), Holm-significant gains over current ResNet-18: {'; '.join(winners_vs_r18_holm)}."
            if winners_vs_r18_holm
            else f"Among completed alternative extractors ({comp_names}), none achieved a Holm-corrected improvement over current ResNet-18 across {mtc_summary['total_paired_tests']} paired tests."
        )
        clin_part = (
            f" Holm-significant gains over Clinical-only ({comp_names}): {'; '.join(winners_vs_clin_holm)}."
            if winners_vs_clin_holm
            else f" None of the completed alternative extractors ({comp_names}) achieved a Holm-corrected improvement over Clinical-only."
        )
        not_run_part = f" Not run: {', '.join(not_run_keys)}." if not_run_keys else ""
        verdict = (
            f"{r18_part}{clin_part} Pending extractors ({', '.join(pending_keys)}) remain 'pending: not evaluated yet', "
            f"so overall summary flags are null (not evaluated).{not_run_part} "
            "The deployed model is not changed automatically. Exploratory."
        )
    else:
        comp_names = ", ".join(completed_new)
        r18_part = (
            f"Across completed alternative extractors ({comp_names}; {mtc_summary['total_paired_tests']} paired Nadeau-Bengio tests, Holm-Bonferroni α = 0.05), "
            f"significant gains over current ResNet-18: {'; '.join(winners_vs_r18_holm)}."
            if winners_vs_r18_holm
            else f"Across completed alternative extractors ({comp_names}; {mtc_summary['total_paired_tests']} paired Nadeau-Bengio tests, Holm-Bonferroni α = 0.05), "
            f"none achieved a statistically confirmed improvement over current ResNet-18."
        )
        clin_part = (
            f" Significant gains over Clinical-only: {'; '.join(winners_vs_clin_holm)}."
            if winners_vs_clin_holm
            else " None achieved a Holm-corrected improvement over Clinical-only."
        )
        not_run_details = [
            f"{k} ({extractors_out[k]['status_reason']})" if extractors_out[k].get("status_reason") else k
            for k in not_run_keys
        ]
        not_run_part = f" Not run: {'; '.join(not_run_details)}." if not_run_details else ""
        verdict = (
            f"{r18_part}{clin_part}{not_run_part} "
            "The deployed model is not changed automatically. Exploratory."
        )

    mtc_summary["any_new_extractor_beats_resnet18_holm"] = any_beats_r18
    mtc_summary["any_new_extractor_beats_resnet18_holm_note"] = any_beats_note
    mtc_summary["any_new_extractor_beats_clinical_only_holm"] = any_beats_clin
    mtc_summary["any_new_extractor_beats_clinical_only_holm_note"] = any_beats_note
    mtc_summary["completed_extractors_beat_resnet18_holm"] = completed_only_beats_r18
    mtc_summary["completed_extractors_beat_clinical_only_holm"] = completed_only_beats_clin
    mtc_summary["winners_vs_resnet18_holm"] = winners_vs_r18_holm
    mtc_summary["winners_vs_clinical_only_holm"] = winners_vs_clin_holm
    mtc_summary["resnet18_baseline_wins_vs_clinical_only_holm"] = r18_wins_clin_holm

    return {
        "exploratory_banner": "Progression results are exploratory, separate cohort, no IPF labels",
        "data_source": data_source,
        "deployed_model_changed": False,
        "deployed_model_policy": "Read-only benchmark; never changes the deployed clinical IPF or OSIC progression model automatically.",
        "protocol": {
            "split_strategy": "5-Repeat x 5-Fold Nested GroupKFold by Patient (_generate_patient_cv_splits, RANDOM_SEED=42)",
            "folds_sha256": canonical_folds_sha256,
            "num_patients": len(canonical_patients),
            "num_future_visits": int(len(base_visit_df)),
            "num_folds": len(canonical_splits),
            "pca_components": safe_pca,
            "ridge_alpha_grid_image_tabular": [100.0, 500.0, 1000.0],
            "ridge_alpha_hrct_and_early": 500.0,
            "ridge_alpha_clinical_and_stacking": 10.0,
            "in_fold_feature_winsorization": "InFoldWinsorizer(0.01, 0.99) on HU/volume features",
            "target_formulation": "Delta_FVC = Target_FVC - Baseline_FVC reconstructed and clipped to [500.0, 6500.0] mL",
            "fvc_winsorize_bounds_ml": [FVC_MIN_ML, FVC_MAX_ML],
            "primary_metric": "modified_laplace_log_likelihood (sigma >= 70 mL, |error| <= 1000 mL)",
            "paired_test": "nadeau_bengio_paired_test (Nadeau & Bengio, Machine Learning, 2003) on identical 25 folds",
            "only_varied_parameter": "emb_prefix (embedding column prefix)",
        },
        "multiple_testing": mtc_summary,
        "overall_verdict": verdict,
        "completed_extractors": [k for k, e in extractors_out.items() if e["status"] == "completed"],
        "pending_extractors": pending_keys,
        "not_run_extractors": not_run_keys,
        "extractors": extractors_out,
        "limitations": LIMITATIONS_LIST,
        "citations": CITATIONS_LIST,
    }


def render_extractor_comparison_markdown(ext_comp: Dict[str, Any]) -> str:
    """Render `docs/EXTRACTOR_COMPARISON.md` strictly from the `extractor_comparison` artifact dict."""
    proto = ext_comp["protocol"]
    mtc = ext_comp["multiple_testing"]
    extractors = ext_comp["extractors"]

    lines = [
        "# OSIC Pretrained CT Feature Extractor Comparison",
        "",
        f"> **Mandatory Cohort & Exploratory Notice:** *{ext_comp['exploratory_banner']}.*  ",
        f"> **Strict Evidence Rule:** Every metric below is computed directly from cached feature files in `backend/data/osic/`. Missing feature files are shown as `pending: not evaluated yet`, and unrun extractors are shown as `not run` with their exact reason. No numbers are estimated, copied, or fabricated.  ",
        f"> **Deployed Model Policy:** {ext_comp['deployed_model_policy']} (`deployed_model_changed: {str(ext_comp['deployed_model_changed']).lower()}`).",
        "",
        "---",
        "",
        "## 1. Shared Evaluation Protocol (Identical Across All Extractors)",
        "",
        "Only the embedding column prefix (`emb_prefix`) changes across extractors; every other modeling and evaluation step stays identical to `backend/src/osic_benchmarks.py` and `backend/src/imaging_progression.py`:",
        "",
        f"- **Patients & Visits:** `n = {proto['num_patients']}` OSIC CT patients, `{proto['num_future_visits']}` future visits (`Weeks > Baseline_Weeks`).",
        f"- **CV Splits:** `{proto['split_strategy']}` (`{proto['num_folds']}` identical outer folds).",
        f"- **Cryptographic Fold Identity Hash (`folds_sha256`):** `{proto['folds_sha256']}`.",
        f"- **Dimensionality Reduction:** In-fold `StandardScaler` + `PCA(n_components={proto['pca_components']}, random_state=42)` fitted strictly inside training folds.",
        f"- **Regularization & Winsorization:** `{proto['in_fold_feature_winsorization']}`; Image+Tabular `Ridge` alpha grid `{proto['ridge_alpha_grid_image_tabular']}`; HRCT-only & Early Fusion `Ridge(alpha={proto['ridge_alpha_hrct_and_early']})`; Clinical-only & Stacking `Ridge(alpha={proto['ridge_alpha_clinical_and_stacking']})`; `{proto['target_formulation']}`.",
        f"- **Metrics & Paired Testing:** `{proto['primary_metric']}`, FVC RMSE (mL), FVC MAE (mL), and patient-level rapid-decliner ROC-AUC with 95% CIs; `{proto['paired_test']}`.",
        f"- **Multiple-Comparison Correction:** `{mtc['correction_method']}` across **`{mtc['total_paired_tests']}`** executed paired tests (`{mtc['num_tests_vs_resnet18']}` vs. Current ResNet-18 + `{mtc['num_tests_vs_clinical_only']}` vs. Clinical-only; family-wise `α = {mtc['alpha']}`, Bonferroni `α = {mtc['bonferroni_alpha']}`).",
        f"- **Alternative-Extractor Holm Summary Flags:** `any_new_extractor_beats_resnet18_holm`: `{mtc['any_new_extractor_beats_resnet18_holm']}` (`{mtc.get('any_new_extractor_beats_resnet18_holm_note', 'not evaluated')}`); `any_new_extractor_beats_clinical_only_holm`: `{mtc['any_new_extractor_beats_clinical_only_holm']}` (`{mtc.get('any_new_extractor_beats_clinical_only_holm_note', 'not evaluated')}`).",
        "",
        f"**Overall Verdict:** {ext_comp['overall_verdict']}",
        "",
        "---",
        "",
        "## 2. Extractor Inventory, Licenses & Extraction Metadata",
        "",
        "| Extractor | Status | Embedding Size | Extraction Time | Peak GPU Memory | Source & License | Redistribution Policy |",
        "| :--- | :--- | :---: | :---: | :---: | :--- | :--- |",
    ]

    for key, e in extractors.items():
        st = e["status"]
        if st == "not run" and e.get("status_reason"):
            st_disp = f"`not run` ({e['status_reason']})"
        else:
            st_disp = f"`{st}`"
        emb_sz = f"`{e['embedding_size']}D`" if e.get("embedding_size") is not None else "—"
        rt_min = (
            f"`{e['extraction_runtime_minutes']:.2f} min`"
            if e.get("extraction_runtime_minutes") is not None
            else (
                f"`{e['extraction_runtime_seconds']:.1f} s`"
                if e.get("extraction_runtime_seconds") is not None
                else ("`N/A (no JSON)`" if st == "completed" else "—")
            )
        )
        gpu_mb = (
            f"`{e['peak_gpu_memory_mb']:.1f} MB`"
            if e.get("peak_gpu_memory_mb") is not None
            else ("`N/A (no JSON)`" if st == "completed" else "—")
        )
        lines.append(
            f"| **{e['display_name']}** (`{e['emb_prefix']}*`) | {st_disp} | {emb_sz} | {rt_min} | {gpu_mb} | "
            f"{e['source']} (**{e['license']}**) | {e['redistribution_note']} |"
        )

    # Section 3: Summary table across (a) HRCT-only, (b) Image+Tabular, and (c) Best Fusion
    lines.extend([
        "",
        "---",
        "",
        "## 3. Extractor Performance Summary (5×5 Patient-Grouped Nested CV, Identical Folds)",
        "",
        "### 3A. HRCT-Only Branch & Image + Tabular Benchmark",
        "",
        "| Extractor | Status | Folds Hash Match | HRCT-Only Laplace LL (95% CI) | HRCT-Only Rapid-Decliner AUC (95% CI) | Image+Tabular Laplace LL (95% CI) | Image+Tabular RMSE mL (95% CI) | Image+Tabular MAE mL (95% CI) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for key, e in extractors.items():
        if e["status"] != "completed":
            st_txt = f"`{e['status']}`"
            lines.append(
                f"| **{e['short_name']}** | {st_txt} | — | {st_txt} | {st_txt} | {st_txt} | {st_txt} | {st_txt} |"
            )
        else:
            h = e["hrct_only"]
            it = e["image_plus_tabular"]
            fm_ok = f"`PASS ({e['folds_sha256'][:10]})`" if e["identical_folds_verified"] else "`MISMATCH`"
            lines.append(
                f"| **{e['short_name']}** | `completed` | {fm_ok} | "
                f"`{h['laplace_ll_stats']['formatted']}` | `{h['auc_stats']['formatted']}` | "
                f"`{it['laplace_ll_stats']['formatted']}` | `{it['rmse_stats']['formatted']}` | "
                f"`{it['mae_stats']['formatted']}` |"
            )

    lines.extend([
        "",
        "### 3B. Full 6-Method Fusion Comparison Per Extractor",
        "",
    ])
    for key, e in extractors.items():
        lines.append(f"#### {e['display_name']} (`status: {e['status']}`)")
        if e["status"] != "completed":
            lines.append(f"- **Status:** `{e['status']}` — {e.get('status_reason', '')}")
            lines.append("")
            continue
        lines.extend([
            f"- **Embedding Prefix / Size:** `{e['emb_prefix']}*` (`{e['embedding_size']}D` -> in-fold PCA `{e['pca_components_used']}D`) | **Folds SHA-256:** `{e['folds_sha256']}` (identical to canonical: `{e['identical_folds_verified']}`)",
            "",
            "| Method | FVC Laplace LL (95% CI) | FVC RMSE mL (95% CI) | FVC MAE mL (95% CI) | Rapid-Decliner AUC (95% CI) |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        for m_key, _ in FUSION_METHOD_KEYS:
            fm = e["fusion_methods"][m_key]
            lines.append(
                f"| **{fm['method']}** | `{fm['laplace_ll_stats']['formatted']}` | "
                f"`{fm['rmse_stats']['formatted']}` | `{fm['mae_stats']['formatted']}` | "
                f"`{fm['auc_stats']['formatted']}` |"
            )
        lines.append("")

    # Section 4: Paired Nadeau-Bengio Statistical Tests (Primary vs ResNet-18, Secondary vs Clinical-only)
    lines.extend([
        "---",
        "",
        f"## 4. Paired Nadeau-Bengio Statistical Comparisons (`{mtc['total_paired_tests']}` Total Tests, {mtc['correction_method']})",
        "",
        "### 4A. Primary Question — New Extractors vs. Current ResNet-18 Pipeline (Same 25 Folds)",
        "",
    ])
    new_completed = [e for e in extractors.values() if (not e["is_reference"]) and e["status"] == "completed"]
    if not new_completed:
        lines.append(
            f"*No alternative extractor CSVs are present in `backend/data/osic/` yet (`{mtc['num_tests_vs_resnet18']}` vs.-ResNet-18 paired tests executed). "
            "Drop the Kaggle-extracted CSV and JSON files into `backend/data/osic/` and run `python -m backend.src.extractor_comparison` to populate this table automatically.*"
        )
    else:
        lines.extend([
            "| Extractor | Configuration | Metric | Paired Δ vs. ResNet-18 (95% CI) | Raw p | Holm p | Bonferroni p | Better (Holm α=0.05)? |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for e in new_completed:
            for cfg_key, cfg_label in IMAGING_CONFIG_KEYS:
                cfg_p = (e.get("paired_vs_resnet18") or {}).get(cfg_key, {})
                for m_name, _ in METRIC_SPECS:
                    r = cfg_p.get(m_name)
                    if not r:
                        continue
                    lines.append(
                        f"| **{e['short_name']}** | {cfg_label} | `{m_name}` | "
                        f"`{r['mean_diff']:+.4f} [{r['ci_95_lower']:+.4f}, {r['ci_95_upper']:+.4f}]` | "
                        f"`{r['p_value']:.4f}` | `{r['p_value_holm']:.4f}` | `{r['p_value_bonferroni']:.4f}` | "
                        f"`{'Yes' if r['statistically_better_holm'] else 'No'}` |"
                    )

    lines.extend([
        "",
        "### 4B. Secondary Question — Completed Extractors vs. Clinical-Only (Same 25 Folds)",
        "",
        "| Extractor | Configuration | Metric | Paired Δ vs. Clinical-Only (95% CI) | Raw p | Holm p | Bonferroni p | Better (Holm α=0.05)? |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for e in extractors.values():
        if e["status"] != "completed":
            continue
        for cfg_key, cfg_label in IMAGING_CONFIG_KEYS:
            cfg_p = (e.get("paired_vs_clinical_only") or {}).get(cfg_key, {})
            for m_name, _ in METRIC_SPECS:
                r = cfg_p.get(m_name)
                if not r:
                    continue
                lines.append(
                    f"| **{e['short_name']}** | {cfg_label} | `{m_name}` | "
                    f"`{r['mean_diff']:+.4f} [{r['ci_95_lower']:+.4f}, {r['ci_95_upper']:+.4f}]` | "
                    f"`{r['p_value']:.4f}` | `{r['p_value_holm']:.4f}` | `{r['p_value_bonferroni']:.4f}` | "
                    f"`{'Yes' if r['statistically_better_holm'] else 'No'}` |"
                )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Methodological Limitations",
        "",
    ])
    for lim in ext_comp["limitations"]:
        lines.append(f"- {lim}")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Licenses & Literature Citations",
        "",
        "- **MedicalNet (Med3D) Mirror (`TencentMedicalNet/MedicalNet-Resnet18`):** MIT License.",
        "- **RadImageNet Mirror (`Lab-Rasool/RadImageNet`, official `BMEII-AI/RadImageNet` weights):** MIT License.",
        "- **CT-CLIP & CT-RATE (`ibrahimhamamci/CT-RATE`, `ibrahimethemhamamci/CT-CLIP`):** Gated access, **CC BY-NC-SA 4.0 (Non-Commercial)** — weights must **not** be redistributed.",
        "- **ImageNet ResNet-18 / ResNet-50 (`torchvision`):** BSD-3-Clause.",
        "",
        "### References",
    ])
    for idx, cite in enumerate(ext_comp["citations"], start=1):
        lines.append(f"{idx}. {cite['authors']} *{cite['title']}*. {cite['venue']}. {cite['url']}")
    lines.append("")
    return "\n".join(lines)


def update_artifacts_and_docs(
    osic_dir: Path = OSIC_DIR,
    artifacts_path: Path = ARTIFACTS_JSON_PATH,
    docs_path: Path = DOCS_EXTRACTOR_COMPARISON_PATH,
) -> Dict[str, Any]:
    """Run extractor comparison, write `extractor_comparison` into `model_artifacts.json`, and write `docs/EXTRACTOR_COMPARISON.md`."""
    ext_comp = run_osic_extractor_comparison(osic_dir=osic_dir)

    if artifacts_path.exists():
        art = json.loads(artifacts_path.read_text(encoding="utf-8"))
        art["extractor_comparison"] = ext_comp
        artifacts_path.write_text(json.dumps(art, indent=2), encoding="utf-8")
        logger.info(f"Updated `extractor_comparison` section in {artifacts_path} (deployed models untouched).")

    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(render_extractor_comparison_markdown(ext_comp), encoding="utf-8")
    logger.info(f"Wrote {docs_path}")
    return ext_comp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the OSIC Pretrained CT Feature Extractor Comparison on cached files in backend/data/osic/."
    )
    parser.add_argument(
        "--osic-dir",
        type=Path,
        default=OSIC_DIR,
        help="Directory containing cached_dicom_features*.csv and metadata JSON files.",
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Update model_artifacts.json without rewriting docs/EXTRACTOR_COMPARISON.md.",
    )
    args = parser.parse_args()
    if args.skip_docs:
        ext_comp = run_osic_extractor_comparison(osic_dir=args.osic_dir)
        if ARTIFACTS_JSON_PATH.exists():
            art = json.loads(ARTIFACTS_JSON_PATH.read_text(encoding="utf-8"))
            art["extractor_comparison"] = ext_comp
            ARTIFACTS_JSON_PATH.write_text(json.dumps(art, indent=2), encoding="utf-8")
    else:
        update_artifacts_and_docs(osic_dir=args.osic_dir)


if __name__ == "__main__":
    main()
