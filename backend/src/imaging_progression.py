"""
Part 2: Exploratory OSIC Pulmonary Fibrosis Progression Study (`backend/src/imaging_progression.py`).

MANDATORY COHORT & REPORTING NOTE:
"Progression results are exploratory, separate cohort, no IPF labels"

Key Methodological Guarantees:
1. Minimum 100-Patient Gate for Imaging Models (`MIN_IMAGING_PATIENTS = 100`):
   - An "Imaging + Clinical" progression model is ONLY trained/enabled when `>= 100` OSIC patients
     have 3D DICOM scans under `backend/data/osic/dicom/<PatientID>/`.
   - When `< 100` patients have scans, `imaging_model_available` is `False` and only the
     clinical-only progression model is active.
2. Full 3D Volume Processing & Per-Patient CSV Caching (`cached_dicom_features.csv`):
   - Processes all slices sorted along cranio-caudal Z-position, computes true physical lung
     volume (`mL`), HU histogram features, and 512-D pretrained `torchvision` ResNet-18 embeddings
     (`ResNet18_Weights.DEFAULT`, loaded once at module level, never per slice, no random fallback).
3. In-Fold PCA Reduction (15 Components) & 3-Way Nested GroupKFold Modality Ablation:
   - Never feeds 512 raw ResNet features into a model trained on 176 patients.
   - `build_resnet_pca_pipeline(n_components=15)` fits `PCA(n_components=15)` strictly inside
     each training fold on training patients only.
   - `evaluate_modality_ablation_nested_gkf(...)` compares `Clinical-only` vs. `Clinical + HU/Volume`
     vs. `Clinical + ResNet-18 (In-Fold PCA)` in the same nested `GroupKFold` with 95% CIs.
4. Dual Target Variant Evaluation (`min_visits >= 3` per patient):
   - Variant 1: Observed ~52-Week FVC Change (mL) (`fvc_change_52w_observed_ml`, rapid decliner `<= -200` mL/yr).
   - Variant 2: Robust Theil-Sen FVC Slope (mL/week) (`fvc_slope`, rapid decliner `<= -4.5` mL/wk).
   - Compares Random Forest, SVM, and XGBoost against explicit Dummy Baselines (`DummyRegressor(mean)`
     and `DummyClassifier(prior)`) via 5-Repeat × 5-Fold Repeated Nested `GroupKFold` by Patient.
"""

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from scipy import ndimage, stats
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from backend.src.config import (
    CACHED_DICOM_FEATURES_CSV,
    EXPLORATORY_PROGRESSION_NOTE,
    MIN_IMAGING_PATIENTS,
    OSIC_DICOM_DIR,
    OSIC_TRAIN_CSV_PATH,
    PROGRESSION_MODEL_PATH,
    RANDOM_SEED,
    set_global_seed,
    setup_logger,
)
from backend.src.generate_sample_data import ensure_osic_dataset
from backend.src.osic_benchmarks import run_osic_benchmarks_and_fusion

logger = setup_logger("imaging_progression")

RESNET_EMBED_DIM = 512
PCA_COMPONENTS = 15  # 10–20 components fitted strictly on training patients inside each fold

CLINICAL_PROGRESSION_FEATURES = [
    "Baseline_FVC",
    "Baseline_Percent",
    "Age",
    "Sex_Male",
    "Smoking_Code",
]

HU_VOLUME_FEATURE_NAMES = [
    "hu_mean",
    "hu_std",
    "hu_skew",
    "hu_kurtosis",
    "hu_p10",
    "hu_p50",
    "hu_p90",
    "haa_pct",
    "laa_pct",
    "lung_volume_ml",
    "num_slices_processed",
]

RESNET_FEATURE_NAMES = [f"resnet_emb_{i}" for i in range(RESNET_EMBED_DIM)]
IMAGING_FEATURE_NAMES = HU_VOLUME_FEATURE_NAMES + RESNET_FEATURE_NAMES

# Module-level singleton for Pretrained ResNet-18 Backbone (loaded once, never per slice)
_PRETRAINED_RESNET18_MODEL = None


def get_pretrained_resnet18_backbone():
    """
    Load `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)` ONCE at module level.
    Fails clearly with a `RuntimeError` if `torch`, `torchvision`, or pretrained ImageNet
    weights cannot be loaded — never silently falls back to a random network.
    """
    global _PRETRAINED_RESNET18_MODEL
    if _PRETRAINED_RESNET18_MODEL is not None:
        return _PRETRAINED_RESNET18_MODEL

    try:
        import torch
        import torch.nn as nn
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and torchvision are required for pretrained ResNet-18 CT feature extraction. "
            "Install them via `pip install torch torchvision`."
        ) from exc

    try:
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
        _PRETRAINED_RESNET18_MODEL = backbone
        logger.info("Loaded pretrained torchvision ResNet-18 (ImageNet DEFAULT weights) at module level.")
        return _PRETRAINED_RESNET18_MODEL
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load pretrained ResNet-18 ImageNet weights: {exc}. "
            "Silent fallback to random weights is disabled."
        ) from exc


def extract_pretrained_resnet18_slice_embedding(hu_slice: np.ndarray) -> np.ndarray:
    """
    Extract a 512-D embedding from an axial thoracic CT slice using the module-level
    pretrained ResNet-18 backbone (`ResNet18_Weights.DEFAULT`).
    """
    import torch

    backbone = get_pretrained_resnet18_backbone()

    norm_img = np.clip((hu_slice + 1000.0) / 1000.0, 0.0, 1.0).astype(np.float32)
    pil_img = Image.fromarray((norm_img * 255.0).astype(np.uint8)).resize(
        (224, 224), Image.Resampling.BILINEAR
    )
    arr_224 = np.asarray(pil_img, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    rgb = np.repeat(arr_224[np.newaxis, :, :], 3, axis=0)
    rgb_norm = (rgb - mean) / std

    with torch.no_grad():
        tensor_in = torch.from_numpy(rgb_norm).unsqueeze(0)
        embedding = backbone(tensor_in).squeeze(0).cpu().numpy()

    return embedding.astype(np.float32)


def load_dicom_as_hu(
    dcm_path_or_bytes: Any,
) -> Tuple[np.ndarray, Tuple[float, float], float, float]:
    """Load a DICOM slice and convert raw pixel values to Hounsfield Units (HU)."""
    if isinstance(dcm_path_or_bytes, (str, Path)):
        ds = pydicom.dcmread(str(dcm_path_or_bytes))
    else:
        ds = pydicom.dcmread(io.BytesIO(dcm_path_or_bytes))

    pixel_array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", -1024.0))
    hu_image = pixel_array * slope + intercept
    hu_image = np.where(hu_image < -1024.0, -1000.0, hu_image)

    raw_spacing = getattr(ds, "PixelSpacing", [0.75, 0.75])
    row_spacing = float(raw_spacing[0]) if len(raw_spacing) >= 1 else 0.75
    col_spacing = float(raw_spacing[1]) if len(raw_spacing) >= 2 else row_spacing

    thickness = float(
        getattr(ds, "SpacingBetweenSlices", getattr(ds, "SliceThickness", 1.5))
    )

    if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
        z_pos = float(ds.ImagePositionPatient[2])
    elif hasattr(ds, "SliceLocation"):
        z_pos = float(ds.SliceLocation)
    else:
        z_pos = float(getattr(ds, "InstanceNumber", 0)) * thickness

    return hu_image, (row_spacing, col_spacing), thickness, z_pos


def segment_lungs_hu(hu_image: np.ndarray) -> np.ndarray:
    """
    Radiological lung segmentation on an axial thoracic CT slice in Hounsfield Units:
    removes exterior air and CT scanner table, keeps the two largest internal lung
    connected components, and fills internal holes.
    """
    tissue_mask = hu_image > -500.0
    labeled_tissue, num_tissue = ndimage.label(tissue_mask)
    if num_tissue > 0:
        tissue_sizes = ndimage.sum(tissue_mask, labeled_tissue, range(1, num_tissue + 1))
        largest_tissue_label = int(np.argmax(tissue_sizes)) + 1
        body_wall = labeled_tissue == largest_tissue_label
        thorax_interior = ndimage.binary_fill_holes(body_wall)
        thorax_interior = ndimage.binary_erosion(thorax_interior, iterations=2)
    else:
        thorax_interior = np.ones_like(hu_image, dtype=bool)

    candidate_lungs = (hu_image >= -1000.0) & (hu_image <= -320.0) & thorax_interior
    candidate_lungs = ndimage.binary_opening(candidate_lungs, structure=np.ones((2, 2)))

    labeled_lungs, num_components = ndimage.label(candidate_lungs)
    if num_components == 0:
        fallback_mask = (hu_image >= -1000.0) & (hu_image <= -320.0)
        return fallback_mask if fallback_mask.sum() > 0 else np.ones_like(hu_image, dtype=bool)

    comp_sizes = ndimage.sum(candidate_lungs, labeled_lungs, range(1, num_components + 1))
    sorted_comp_indices = np.argsort(comp_sizes)[::-1]
    total_candidate_area = float(np.sum(comp_sizes))
    kept_mask = np.zeros_like(candidate_lungs, dtype=bool)

    for rank, idx in enumerate(sorted_comp_indices[:2]):
        label_id = int(idx) + 1
        if rank == 0 or (comp_sizes[idx] >= 0.05 * total_candidate_area):
            kept_mask |= labeled_lungs == label_id

    kept_mask = ndimage.binary_closing(kept_mask, structure=np.ones((3, 3)))
    kept_mask = ndimage.binary_fill_holes(kept_mask)
    return kept_mask


def extract_ct_features_from_3d_slices(
    slice_tuples: List[Tuple[np.ndarray, Tuple[float, float], float, float]],
    extract_cnn: bool = True,
) -> Dict[str, float]:
    """Process all slices in a 3D volume sorted along Z-position."""
    sorted_slices = sorted(slice_tuples, key=lambda item: item[3])
    n_slices = len(sorted_slices)

    z_positions = [item[3] for item in sorted_slices]
    z_diffs = [
        abs(z_positions[i + 1] - z_positions[i])
        for i in range(n_slices - 1)
        if abs(z_positions[i + 1] - z_positions[i]) > 0.1
    ]
    median_dz = float(np.median(z_diffs)) if z_diffs else None

    all_lung_voxels = []
    total_volume_ml = 0.0
    slice_embeddings = []

    for hu_img, (row_mm, col_mm), thickness_mm, _ in sorted_slices:
        dz_mm = median_dz if (median_dz is not None and median_dz <= 15.0) else thickness_mm
        voxel_vol_ml = (row_mm * col_mm * dz_mm) / 1000.0

        mask = segment_lungs_hu(hu_img)
        voxels = hu_img[mask]
        if len(voxels) > 0:
            all_lung_voxels.append(voxels)
        total_volume_ml += float(mask.sum()) * voxel_vol_ml

        if extract_cnn:
            masked_hu = np.where(mask, hu_img, -1000.0)
            slice_embeddings.append(extract_pretrained_resnet18_slice_embedding(masked_hu))

    concat_hu = (
        np.concatenate(all_lung_voxels).astype(np.float64)
        if all_lung_voxels
        else sorted_slices[0][0].ravel().astype(np.float64)
    )

    hu_mean_val = float(np.mean(concat_hu))
    hu_std_val = float(np.std(concat_hu))
    hu_range_val = float(np.ptp(concat_hu))

    if hu_std_val > 1.0 and hu_range_val > 2.0:
        hu_skew_val = float(stats.skew(concat_hu, bias=False))
        hu_kurt_val = float(stats.kurtosis(concat_hu, bias=False))
        hu_skew_val = 0.0 if not np.isfinite(hu_skew_val) else hu_skew_val
        hu_kurt_val = 0.0 if not np.isfinite(hu_kurt_val) else hu_kurt_val
    else:
        hu_skew_val = 0.0
        hu_kurt_val = 0.0

    haa_pct = float(np.mean((concat_hu >= -600.0) & (concat_hu <= -250.0)) * 100.0)
    laa_pct = float(np.mean(concat_hu < -950.0) * 100.0)

    features = {
        "hu_mean": round(hu_mean_val, 3),
        "hu_std": round(hu_std_val, 3),
        "hu_skew": round(hu_skew_val, 4),
        "hu_kurtosis": round(hu_kurt_val, 4),
        "hu_p10": round(float(np.percentile(concat_hu, 10)), 2),
        "hu_p50": round(float(np.percentile(concat_hu, 50)), 2),
        "hu_p90": round(float(np.percentile(concat_hu, 90)), 2),
        "haa_pct": round(haa_pct, 3),
        "laa_pct": round(laa_pct, 3),
        "lung_volume_ml": round(total_volume_ml, 2),
        "num_slices_processed": float(n_slices),
    }

    if extract_cnn and slice_embeddings:
        mean_embed = np.mean(slice_embeddings, axis=0)
        for i, val in enumerate(mean_embed):
            features[f"resnet_emb_{i}"] = round(float(val), 5)

    return features


def load_or_extract_cached_dicom_features(
    dicom_root: Path = OSIC_DICOM_DIR,
    cache_csv_path: Path = CACHED_DICOM_FEATURES_CSV,
) -> Dict[str, Dict[str, float]]:
    """
    Extract 3D volumetric CT + pretrained ResNet-18 features for every patient folder
    in `dicom_root` and cache them to `backend/data/osic/cached_dicom_features.csv`.
    """
    cached_map: Dict[str, Dict[str, float]] = {}
    if cache_csv_path.exists():
        try:
            df_cache = pd.read_csv(cache_csv_path)
            for _, row in df_cache.iterrows():
                pid = str(row["Patient"])
                cached_map[pid] = {
                    col: float(row[col])
                    for col in df_cache.columns
                    if col != "Patient" and pd.notna(row[col])
                }
        except Exception as exc:
            logger.warning(f"Rebuilding DICOM feature cache due to read error: {exc}")
            cached_map = {}

    if not dicom_root.exists():
        return cached_map

    updated = False
    for pat_dir in sorted(dicom_root.iterdir()):
        if not pat_dir.is_dir():
            continue
        dcm_files = sorted(pat_dir.glob("*.dcm"))
        if not dcm_files:
            continue
        pid = pat_dir.name
        cached_entry = cached_map.get(pid)
        if (
            cached_entry is not None
            and int(cached_entry.get("num_slices_processed", -1)) == len(dcm_files)
            and "resnet_emb_0" in cached_entry
        ):
            continue

        slice_tuples = []
        for dcm_f in dcm_files:
            try:
                slice_tuples.append(load_dicom_as_hu(dcm_f))
            except Exception as exc:
                logger.warning(f"Skipping unreadable DICOM file {dcm_f}: {exc}")

        if slice_tuples:
            logger.info(
                f"Extracting 3D CT + Pretrained ResNet-18 features for {pid} ({len(slice_tuples)} slices)..."
            )
            feats = extract_ct_features_from_3d_slices(slice_tuples, extract_cnn=True)
            cached_map[pid] = feats
            updated = True

    if updated and cached_map:
        cache_rows = [{"Patient": pid, **feats} for pid, feats in sorted(cached_map.items())]
        cache_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cache_rows).to_csv(cache_csv_path, index=False)
        logger.info(f"Cached 3D DICOM features for {len(cache_rows)} patients to {cache_csv_path}")

    return cached_map


def extract_features_from_upload_bytes(
    file_bytes: bytes,
    filename: str = "",
) -> Dict[str, float]:
    """Parse an uploaded CT file (DICOM `.dcm` or PNG/JPG image). Never returns a fake scan."""
    if not file_bytes:
        raise ValueError("Empty CT scan file uploaded.")

    lower_name = (filename or "").lower()
    if lower_name.endswith(".dcm") or lower_name.endswith(".dicom"):
        try:
            slice_tuple = load_dicom_as_hu(file_bytes)
            return extract_ct_features_from_3d_slices([slice_tuple], extract_cnn=True)
        except RuntimeError:
            raise
        except Exception as exc:
            raise ValueError(f"Invalid or corrupted DICOM file: {exc}") from exc

    if lower_name.endswith((".png", ".jpg", ".jpeg")):
        try:
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
            gray = np.asarray(pil_img, dtype=np.float32)
            hu_approx = -1000.0 + (gray / 255.0) * 900.0
            slice_tuple = (hu_approx, (0.75, 0.75), 1.5, 0.0)
            return extract_ct_features_from_3d_slices([slice_tuple], extract_cnn=True)
        except RuntimeError:
            raise
        except Exception as exc:
            raise ValueError(f"Invalid or corrupted raster image file: {exc}") from exc

    raise ValueError(
        f"Unsupported CT file format for '{filename}'. Allowed extensions: .dcm, .dicom, .png, .jpg, .jpeg"
    )


FORBIDDEN_ARCHIVE_EXTENSIONS = (
    ".joblib",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".exe",
    ".dll",
    ".sh",
    ".bat",
    ".ps1",
    ".py",
)


def extract_features_from_dicom_zip_bytes(
    zip_bytes: bytes,
    filename: str = "series.zip",
    max_files: int = 600,
    max_uncompressed_bytes: int = 350 * 1024 * 1024,
) -> Dict[str, float]:
    """
    Safely inspect a `.zip` archive of axial thoracic CT DICOM slices (`.dcm`/`.dicom`),
    extract true 3D HU/volume statistics and pretrained ResNet-18 embeddings across all slices.

    Security & Serving Rules:
    - Strictly requires a valid `.zip` file.
    - Blocks path traversal (`..`, absolute paths), zip bombs (`max_files`, `max_uncompressed_bytes`),
      and forbidden serialized/executable files (`.joblib`, `.pkl`, `.pt`, etc.).
    - Fails fast with `RuntimeError` (mapped to HTTP 503 by the API) if `torch`, `torchvision`,
      or pretrained ResNet-18 ImageNet weights are missing.
    """
    if not zip_bytes:
        raise ValueError("Empty CT series archive uploaded.")

    lower_name = (filename or "").lower()
    if not lower_name.endswith(".zip"):
        raise ValueError(
            f"Full CT series upload requires a .zip archive of DICOM (.dcm) slices; received '{filename}'."
        )

    # Validate archive structure and security limits BEFORE loading torch or DICOM pixels
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid or corrupted .zip archive: {exc}") from exc

    with zf:
        infos = zf.infolist()
        if len(infos) > max_files:
            raise ValueError(
                f"CT series archive contains {len(infos)} entries, exceeding maximum limit of {max_files} files."
            )

        total_uncompressed = 0
        dcm_members: List[zipfile.ZipInfo] = []

        for info in infos:
            member_name = info.filename
            if info.is_dir() or member_name.startswith("__MACOSX/"):
                continue

            norm_parts = Path(member_name.replace("\\", "/")).parts
            if (
                member_name.startswith(("/", "\\"))
                or ".." in norm_parts
                or (norm_parts and ":" in norm_parts[0])
            ):
                raise ValueError(
                    f"Security policy violation: unsafe path '{member_name}' inside CT .zip archive."
                )

            low_member = member_name.lower()
            if any(low_member.endswith(ext) for ext in FORBIDDEN_ARCHIVE_EXTENSIONS):
                raise ValueError(
                    f"Security policy violation: forbidden file type '{member_name}' inside CT .zip archive."
                )

            total_uncompressed += int(info.file_size)
            if total_uncompressed > max_uncompressed_bytes:
                raise ValueError(
                    f"CT series archive uncompressed size exceeds {max_uncompressed_bytes // (1024 * 1024)} MB safety limit."
                )

            if low_member.endswith((".dcm", ".dicom")):
                dcm_members.append(info)

        if not dcm_members:
            raise ValueError(
                "No DICOM (.dcm/.dicom) slices found inside the uploaded .zip archive."
            )

        # Verify torch + pretrained ResNet-18 weights upfront so missing dependencies return HTTP 503 cleanly
        get_pretrained_resnet18_backbone()

        slice_tuples = []
        skipped_count = 0
        for member in dcm_members:
            try:
                raw_slice = zf.read(member)
                slice_tuples.append(load_dicom_as_hu(raw_slice))
            except Exception:
                skipped_count += 1

    if not slice_tuples:
        raise ValueError(
            f"None of the {len(dcm_members)} DICOM entries in '{filename}' could be decoded as valid CT slices."
        )

    feats = extract_ct_features_from_3d_slices(slice_tuples, extract_cnn=True)
    feats["num_slices_skipped"] = float(skipped_count)
    return feats


def build_in_fold_resnet_pca_preprocessor(
    n_components: int = PCA_COMPONENTS,
    include_hu: bool = False,
) -> ColumnTransformer:
    """
    Build a leak-free `ColumnTransformer` that fits `PCA(n_components)` on the 512-D
    ResNet-18 embeddings strictly inside each training fold (10–20 components, default 15).
    Never passes 512 raw ResNet dimensions directly to a classifier/regressor.
    """
    base_cols = (
        CLINICAL_PROGRESSION_FEATURES + HU_VOLUME_FEATURE_NAMES
        if include_hu
        else CLINICAL_PROGRESSION_FEATURES
    )
    return ColumnTransformer(
        transformers=[
            ("clin_base", StandardScaler(), base_cols),
            (
                "resnet_pca",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=n_components, random_state=RANDOM_SEED)),
                    ]
                ),
                RESNET_FEATURE_NAMES,
            ),
        ],
        remainder="drop",
    )


def build_osic_patient_cohort(
    csv_path: Path = OSIC_TRAIN_CSV_PATH,
    dicom_root: Path = OSIC_DICOM_DIR,
    demo: bool = False,
    min_visits: int = 3,
) -> Tuple[pd.DataFrame, str, int, int]:
    """
    Build the patient-level OSIC cohort (`min_visits >= 3`) with BOTH target variants:
    1. `fvc_change_52w_observed_ml`: Observed ~52-week FVC change (mL), computed from the
       visit closest to `t_0 + 52` weeks within `[26, 78]` weeks, annualized to 52 weeks
       (`(FVC_52 - FVC_0) * (52.0 / delta_weeks)`), clipped to `[-800, +250]` mL.
       Binary rapid decliner label: `rapid_decliner_52w = 1` if `<= -200` mL/yr.
    2. `fvc_slope`: Robust Theil-Sen FVC slope (`mL/week`), clipped to `[-15.0, +4.0]` mL/wk.
       Binary rapid decliner label: `rapid_decliner_slope = 1` if `<= -4.5` mL/wk.
    """
    verified_csv, data_source, _ = ensure_osic_dataset(csv_path, dicom_root, demo=demo)
    df = pd.read_csv(verified_csv)
    cached_dicom_map = load_or_extract_cached_dicom_features(dicom_root)

    patient_rows = []
    for pat_id, group in df.groupby("Patient"):
        group_sorted = group.sort_values("Weeks").reset_index(drop=True)
        if len(group_sorted) < min_visits:
            continue

        weeks = group_sorted["Weeks"].values.astype(float)
        fvcs = group_sorted["FVC"].values.astype(float)
        span_weeks = float(np.ptp(weeks))
        if span_weeks < 12.0:
            continue

        # Target Variant B: Robust Theil-Sen slope (mL/week)
        try:
            res = stats.theilslopes(fvcs, weeks)
            raw_slope = float(res[0])
        except Exception:
            raw_slope, _, _, _, _ = stats.linregress(weeks, fvcs)
        robust_slope = float(np.clip(raw_slope, -15.0, 4.0))
        rapid_decliner_slope = 1 if robust_slope <= -4.5 else 0

        # Target Variant A: Observed ~52-week FVC change (mL)
        rel_weeks = weeks - weeks[0]
        # Prefer visits within [26, 78] weeks from baseline; fallback to furthest visit >= 20w
        window_mask = (rel_weeks >= 26.0) & (rel_weeks <= 78.0)
        if np.any(window_mask):
            window_indices = np.where(window_mask)[0]
            best_idx = window_indices[int(np.argmin(np.abs(rel_weeks[window_indices] - 52.0)))]
            delta_w = float(rel_weeks[best_idx])
            raw_52w = (float(fvcs[best_idx]) - float(fvcs[0])) * (52.0 / delta_w)
            has_52w_visit = True
        elif rel_weeks[-1] >= 20.0:
            raw_52w = (float(fvcs[-1]) - float(fvcs[0])) * (52.0 / float(rel_weeks[-1]))
            has_52w_visit = True
        else:
            raw_52w = robust_slope * 52.0
            has_52w_visit = False

        fvc_change_52w = float(np.clip(raw_52w, -800.0, 250.0))
        rapid_decliner_52w = 1 if fvc_change_52w <= -200.0 else 0

        first_row = group_sorted.iloc[0]
        sex_male = 1.0 if str(first_row["Sex"]).strip().lower().startswith("m") else 0.0
        smoke_str = str(first_row["SmokingStatus"]).strip().lower()
        smoking_code = (
            2.0 if "current" in smoke_str else 1.0 if "ex" in smoke_str else 0.0
        )

        ct_feats = cached_dicom_map.get(str(pat_id))
        has_ct = ct_feats is not None

        row_data = {
            "Patient": str(pat_id),
            "num_visits": int(len(group_sorted)),
            "follow_up_weeks_span": span_weeks,
            "Baseline_FVC": float(first_row["FVC"]),
            "Baseline_Percent": float(first_row["Percent"]),
            "Age": float(first_row["Age"]),
            "Sex_Male": sex_male,
            "Smoking_Code": smoking_code,
            "fvc_slope": round(robust_slope, 4),
            "rapid_decliner_slope": int(rapid_decliner_slope),
            "fvc_change_52w_observed_ml": round(fvc_change_52w, 2),
            "rapid_decliner_52w": int(rapid_decliner_52w),
            "has_52w_visit": has_52w_visit,
            "has_ct": has_ct,
        }
        if has_ct:
            row_data.update(ct_feats)

        patient_rows.append(row_data)

    pat_df = pd.DataFrame(patient_rows)
    dicom_patients_count = int(pat_df["has_ct"].sum()) if len(pat_df) > 0 else 0
    logger.info(
        f"Constructed OSIC cohort (min_visits={min_visits}): {len(pat_df)} patients "
        f"from {len(df)} visits (patients_with_dicoms={dicom_patients_count})"
    )
    return pat_df, data_source, int(len(df)), dicom_patients_count


def _compute_ci_95(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    se = std_v / np.sqrt(max(1, len(arr)))
    return {
        "mean": round(mean_v, 4),
        "std": round(std_v, 4),
        "ci_95_lower": round(mean_v - 1.96 * se, 4),
        "ci_95_upper": round(mean_v + 1.96 * se, 4),
        "formatted": f"{mean_v:.3f} ± {std_v:.3f} (95% CI: {mean_v - 1.96 * se:.3f}–{mean_v + 1.96 * se:.3f})",
    }


def _evaluate_target_nested_gkf(
    X_df: pd.DataFrame,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    patients: np.ndarray,
    target_label: str,
    units: str,
    n_repeats: int = 5,
    n_outer_splits: int = 5,
) -> Dict[str, Any]:
    """
    Run 5-Repeat × 5-Fold Repeated Nested `GroupKFold` by Patient for Random Forest,
    SVM, XGBoost, and Dummy Baselines (`DummyRegressor(mean)`, `DummyClassifier(prior)`).
    """
    base_rmses, base_maes, base_aucs, base_f1s = [], [], [], []
    model_names = ["Random Forest", "SVM", "XGBoost"]
    reg_rmses = {m: [] for m in model_names}
    reg_maes = {m: [] for m in model_names}
    cls_aucs = {m: [] for m in model_names}
    cls_f1s = {m: [] for m in model_names}

    for rep in range(n_repeats):
        rng = np.random.default_rng(RANDOM_SEED + rep * 101)
        perm = rng.permutation(len(X_df))
        X_rep = X_df.iloc[perm].reset_index(drop=True)
        yr_rep = y_reg[perm]
        yc_rep = y_cls[perm]
        g_rep = patients[perm]

        outer_gkf = GroupKFold(n_splits=n_outer_splits)
        inner_gkf = GroupKFold(n_splits=3)

        for tr_idx, va_idx in outer_gkf.split(X_rep, yr_rep, groups=g_rep):
            X_tr, X_va = X_rep.iloc[tr_idx], X_rep.iloc[va_idx]
            yr_tr, yr_va = yr_rep[tr_idx], yr_rep[va_idx]
            yc_tr, yc_va = yc_rep[tr_idx], yc_rep[va_idx]
            g_tr = g_rep[tr_idx]

            # Dummy Baselines
            dummy_reg = DummyRegressor(strategy="mean").fit(X_tr, yr_tr)
            pred_base_r = dummy_reg.predict(X_va)
            base_rmses.append(float(np.sqrt(mean_squared_error(yr_va, pred_base_r))))
            base_maes.append(float(mean_absolute_error(yr_va, pred_base_r)))

            dummy_cls = DummyClassifier(strategy="prior").fit(X_tr, yc_tr)
            prob_base_c = dummy_cls.predict_proba(X_va)[:, 1]
            pred_base_c = dummy_cls.predict(X_va)
            base_aucs.append(
                float(roc_auc_score(yc_va, prob_base_c)) if len(np.unique(yc_va)) > 1 else 0.5
            )
            base_f1s.append(float(f1_score(yc_va, pred_base_c, zero_division=0)))

            reg_specs = {
                "Random Forest": (
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("reg", RandomForestRegressor(n_estimators=90, random_state=RANDOM_SEED)),
                        ]
                    ),
                    {"reg__max_depth": [3, 5], "reg__min_samples_leaf": [5, 10]},
                ),
                "SVM": (
                    Pipeline([("scaler", StandardScaler()), ("reg", SVR(kernel="rbf"))]),
                    {"reg__C": [0.5, 2.0]},
                ),
                "XGBoost": (
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("reg", XGBRegressor(n_estimators=70, random_state=RANDOM_SEED, verbosity=0)),
                        ]
                    ),
                    {"reg__max_depth": [2, 3], "reg__learning_rate": [0.03, 0.06]},
                ),
            }

            cls_specs = {
                "Random Forest": (
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "cls",
                                RandomForestClassifier(
                                    n_estimators=90,
                                    class_weight="balanced",
                                    random_state=RANDOM_SEED,
                                ),
                            ),
                        ]
                    ),
                    {"cls__max_depth": [3, 5], "cls__min_samples_leaf": [5, 10]},
                ),
                "SVM": (
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("cls", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED)),
                        ]
                    ),
                    {"cls__C": [0.5, 2.0]},
                ),
                "XGBoost": (
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "cls",
                                XGBClassifier(
                                    n_estimators=70,
                                    eval_metric="logloss",
                                    random_state=RANDOM_SEED,
                                    verbosity=0,
                                ),
                            ),
                        ]
                    ),
                    {"cls__max_depth": [2, 3], "cls__learning_rate": [0.03, 0.06]},
                ),
            }

            for m_name in model_names:
                r_pipe, r_grid = reg_specs[m_name]
                r_search = GridSearchCV(
                    r_pipe,
                    r_grid,
                    scoring="neg_root_mean_squared_error",
                    cv=inner_gkf,
                    n_jobs=1,
                )
                r_search.fit(X_tr, yr_tr, groups=g_tr)
                pred_r = r_search.best_estimator_.predict(X_va)
                reg_rmses[m_name].append(float(np.sqrt(mean_squared_error(yr_va, pred_r))))
                reg_maes[m_name].append(float(mean_absolute_error(yr_va, pred_r)))

                c_pipe, c_grid = cls_specs[m_name]
                c_search = GridSearchCV(
                    c_pipe, c_grid, scoring="roc_auc", cv=inner_gkf, n_jobs=1
                )
                c_search.fit(X_tr, yc_tr, groups=g_tr)
                best_c = c_search.best_estimator_
                if hasattr(best_c, "predict_proba"):
                    prob_c = best_c.predict_proba(X_va)[:, 1]
                    pred_c = (prob_c >= 0.45).astype(int)
                else:
                    scores_c = best_c.decision_function(X_va)
                    prob_c = 1.0 / (1.0 + np.exp(-np.clip(scores_c, -30, 30)))
                    pred_c = best_c.predict(X_va)

                cls_aucs[m_name].append(
                    float(roc_auc_score(yc_va, prob_c)) if len(np.unique(yc_va)) > 1 else 0.5
                )
                cls_f1s[m_name].append(float(f1_score(yc_va, pred_c, zero_division=0)))

    baseline_summary = {
        "model": "Baseline (Train Mean / Majority Prior)",
        "rmse_stats": _compute_ci_95(base_rmses),
        "mae_stats": _compute_ci_95(base_maes),
        "auc_stats": _compute_ci_95(base_aucs),
        "f1_stats": _compute_ci_95(base_f1s),
        "rmse_mean": round(float(np.mean(base_rmses)), 3),
        "rmse_std": round(float(np.std(base_rmses, ddof=1)), 3),
        "mae_mean": round(float(np.mean(base_maes)), 3),
        "auc_mean": round(float(np.mean(base_aucs)), 4),
        "f1_mean": round(float(np.mean(base_f1s)), 4),
    }

    models_out = {"Baseline (Mean / Majority)": baseline_summary}
    best_model = "XGBoost"
    best_rmse = float("inf")
    clearly_beats_baseline = False

    for m_name in model_names:
        rmse_info = _compute_ci_95(reg_rmses[m_name])
        mae_info = _compute_ci_95(reg_maes[m_name])
        auc_info = _compute_ci_95(cls_aucs[m_name])
        f1_info = _compute_ci_95(cls_f1s[m_name])

        # Strict non-overlapping CI test for "clearly beats baseline"
        beats_reg = rmse_info["mean"] < baseline_summary["rmse_stats"]["mean"]
        clearly_beats_reg = rmse_info["ci_95_upper"] < baseline_summary["rmse_stats"]["ci_95_lower"]
        beats_cls = auc_info["ci_95_lower"] > 0.50
        if clearly_beats_reg and auc_info["ci_95_lower"] >= 0.65:
            clearly_beats_baseline = True

        models_out[m_name] = {
            "model": m_name,
            "rmse_stats": rmse_info,
            "mae_stats": mae_info,
            "auc_stats": auc_info,
            "f1_stats": f1_info,
            "cv_rmse_mean": rmse_info["mean"],
            "cv_rmse_std": rmse_info["std"],
            "cv_mae_mean": mae_info["mean"],
            "cv_auc_mean": auc_info["mean"],
            "cv_auc_std": auc_info["std"],
            "cv_f1_mean": f1_info["mean"],
            "beats_regression_baseline": bool(beats_reg),
            "clearly_beats_regression_ci": bool(clearly_beats_reg),
            "beats_classification_baseline": bool(beats_cls),
        }
        if rmse_info["mean"] < best_rmse:
            best_rmse = rmse_info["mean"]
            best_model = m_name

    return {
        "target_label": target_label,
        "units": units,
        "num_patients": int(len(X_df)),
        "best_model": best_model,
        "clearly_beats_baseline": clearly_beats_baseline,
        "baseline_metrics": baseline_summary,
        "models": models_out,
    }


def evaluate_modality_ablation_nested_gkf(
    df_with_dicoms: pd.DataFrame,
    target_reg_col: str = "fvc_change_52w_observed_ml",
    target_cls_col: str = "rapid_decliner_52w",
    target_label: str = "Observed ~52-Week FVC Change (mL)",
    units: str = "mL/52w",
    n_pca_components: int = PCA_COMPONENTS,
    n_splits: int = 5,
    n_repeats: int = 5,
) -> Dict[str, Any]:
    """
    Compare Baseline (Mean / Majority) vs. Clinical-only vs. Clinical + HU/Volume vs.
    Clinical + ResNet-18 (In-Fold PCA) inside the exact same Repeated GroupKFold splits
    by patient, fitting `PCA(n_components)` strictly on training-fold patients.
    Reports RMSE, MAE, and AUC with 95% CIs and non-overlapping CI verdicts.
    """
    y_reg = df_with_dicoms[target_reg_col].values.astype(float)
    y_cls = df_with_dicoms[target_cls_col].values.astype(int)
    patients = df_with_dicoms["Patient"].values

    max_safe_pca = max(2, min(n_pca_components, int(len(df_with_dicoms) * 0.65) - 2))

    res_key = f"Clinical + ResNet-18 (In-Fold PCA {max_safe_pca}D)"
    modality_specs = {
        "Clinical-only": lambda: ColumnTransformer(
            [("clin", StandardScaler(), CLINICAL_PROGRESSION_FEATURES)], remainder="drop"
        ),
        "Clinical + HU/Volume": lambda: ColumnTransformer(
            [("clin_hu", StandardScaler(), CLINICAL_PROGRESSION_FEATURES + HU_VOLUME_FEATURE_NAMES)],
            remainder="drop",
        ),
        res_key: lambda: build_in_fold_resnet_pca_preprocessor(
            n_components=max_safe_pca, include_hu=False
        ),
    }

    base_rmses, base_maes, base_aucs = [], [], []
    mod_rmses = {k: [] for k in modality_specs}
    mod_maes = {k: [] for k in modality_specs}
    mod_aucs = {k: [] for k in modality_specs}

    for rep in range(n_repeats):
        rng = np.random.default_rng(RANDOM_SEED + rep * 37)
        perm = rng.permutation(len(df_with_dicoms))
        df_rep = df_with_dicoms.iloc[perm].reset_index(drop=True)
        yr_rep = y_reg[perm]
        yc_rep = y_cls[perm]
        g_rep = patients[perm]

        gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(g_rep))))
        for tr_idx, va_idx in gkf.split(df_rep, yr_rep, groups=g_rep):
            X_tr, X_va = df_rep.iloc[tr_idx], df_rep.iloc[va_idx]
            yr_tr, yr_va = yr_rep[tr_idx], yr_rep[va_idx]
            yc_tr, yc_va = yc_rep[tr_idx], yc_rep[va_idx]

            # Mean / Majority Prior Baseline on the exact same fold
            dummy_reg = DummyRegressor(strategy="mean").fit(X_tr, yr_tr)
            pred_base = dummy_reg.predict(X_va)
            base_rmses.append(float(np.sqrt(mean_squared_error(yr_va, pred_base))))
            base_maes.append(float(mean_absolute_error(yr_va, pred_base)))

            dummy_cls = DummyClassifier(strategy="prior").fit(X_tr, yc_tr)
            prob_base = dummy_cls.predict_proba(X_va)[:, 1]
            if len(np.unique(yc_va)) > 1:
                base_aucs.append(float(roc_auc_score(yc_va, prob_base)))
            else:
                base_aucs.append(0.5)

            for mod_name, prep_fn in modality_specs.items():
                reg_pipe = Pipeline(
                    [
                        ("prep", prep_fn()),
                        (
                            "reg",
                            XGBRegressor(
                                n_estimators=60,
                                max_depth=3,
                                learning_rate=0.05,
                                random_state=RANDOM_SEED,
                                verbosity=0,
                            ),
                        ),
                    ]
                )
                reg_pipe.fit(X_tr, yr_tr)
                pred_va = reg_pipe.predict(X_va)
                mod_rmses[mod_name].append(float(np.sqrt(mean_squared_error(yr_va, pred_va))))
                mod_maes[mod_name].append(float(mean_absolute_error(yr_va, pred_va)))

                cls_pipe = Pipeline(
                    [
                        ("prep", prep_fn()),
                        (
                            "cls",
                            XGBClassifier(
                                n_estimators=60,
                                max_depth=3,
                                learning_rate=0.05,
                                eval_metric="logloss",
                                random_state=RANDOM_SEED,
                                verbosity=0,
                            ),
                        ),
                    ]
                )
                if len(np.unique(yc_tr)) > 1 and len(np.unique(yc_va)) > 1:
                    cls_pipe.fit(X_tr, yc_tr)
                    prob_va = cls_pipe.predict_proba(X_va)[:, 1]
                    mod_aucs[mod_name].append(float(roc_auc_score(yc_va, prob_va)))

    base_stats = {
        "modality": "Baseline (Train Mean / Majority Prior)",
        "rmse_stats": _compute_ci_95(base_rmses),
        "mae_stats": _compute_ci_95(base_maes),
        "auc_stats": _compute_ci_95(base_aucs) if base_aucs else _compute_ci_95([0.5]),
    }

    modality_results: Dict[str, Any] = {"Baseline (Mean / Majority)": base_stats}
    for mod_name in modality_specs:
        modality_results[mod_name] = {
            "modality": mod_name,
            "rmse_stats": _compute_ci_95(mod_rmses[mod_name]),
            "mae_stats": _compute_ci_95(mod_maes[mod_name]),
            "auc_stats": _compute_ci_95(mod_aucs[mod_name]) if mod_aucs[mod_name] else _compute_ci_95([0.5]),
        }

    clin_info = modality_results["Clinical-only"]

    def _ci_comparison(
        mod_info: Dict[str, Any], ref_info: Dict[str, Any], is_baseline: bool = False
    ) -> Dict[str, Any]:
        beats_rmse = bool(
            mod_info["rmse_stats"]["ci_95_upper"] < ref_info["rmse_stats"]["ci_95_lower"]
        )
        beats_mae = bool(
            mod_info["mae_stats"]["ci_95_upper"] < ref_info["mae_stats"]["ci_95_lower"]
        )
        beats_auc = bool(
            mod_info["auc_stats"]["ci_95_lower"] > ref_info["auc_stats"]["ci_95_upper"]
        )
        beats_reg = bool(beats_rmse or beats_mae)
        if is_baseline and beats_auc and not beats_reg:
            summary = "AUC above 0.5 only; RMSE and MAE no better than baseline"
        else:
            wins = [
                lbl
                for lbl, ok in (("RMSE", beats_rmse), ("MAE", beats_mae), ("AUC", beats_auc))
                if ok
            ]
            summary = f"Yes ({', '.join(wins)})" if wins else "No"
        return {
            "beats_rmse_ci": beats_rmse,
            "beats_mae_ci": beats_mae,
            "beats_auc_ci": beats_auc,
            "beats_regression_ci": beats_reg,
            "beats_any_metric_ci": bool(beats_reg or beats_auc),
            "summary": summary,
        }

    for mod_name in modality_specs:
        modality_results[mod_name]["vs_baseline_ci"] = _ci_comparison(
            modality_results[mod_name], base_stats, is_baseline=True
        )
        if mod_name != "Clinical-only":
            modality_results[mod_name]["vs_clinical_only_ci"] = _ci_comparison(
                modality_results[mod_name], clin_info, is_baseline=False
            )

    hu_vs_clin = modality_results["Clinical + HU/Volume"]["vs_clinical_only_ci"]
    res_vs_clin = modality_results[res_key]["vs_clinical_only_ci"]
    hu_vs_base = modality_results["Clinical + HU/Volume"]["vs_baseline_ci"]
    res_vs_base = modality_results[res_key]["vs_baseline_ci"]

    return {
        "status": "completed",
        "target_label": target_label,
        "units": units,
        "num_patients": int(len(df_with_dicoms)),
        "pca_components_used": max_safe_pca,
        "modalities": modality_results,
        "hu_beats_baseline_ci": bool(hu_vs_base["beats_any_metric_ci"]),
        "resnet_pca_beats_baseline_ci": bool(res_vs_base["beats_any_metric_ci"]),
        "hu_beats_clinical_only": bool(hu_vs_clin["beats_any_metric_ci"]),
        "resnet_pca_beats_clinical_only": bool(res_vs_clin["beats_any_metric_ci"]),
        "any_imaging_beats_clinical_only_ci": bool(
            hu_vs_clin["beats_any_metric_ci"] or res_vs_clin["beats_any_metric_ci"]
        ),
    }


def train_osic_progression_models(
    save_path: Path = PROGRESSION_MODEL_PATH,
    demo: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate BOTH OSIC target variants (`Observed ~52-Week FVC Change (mL)` and
    `Robust Theil-Sen FVC Slope (mL/week)`) on patients with `min_visits >= 3` using
    5-Repeat × 5-Fold Repeated Nested `GroupKFold` against Dummy Baselines, and run
    the 3-way modality ablation on BOTH targets if `>= 100` patients have cached 3D CT features.
    """
    set_global_seed(RANDOM_SEED)
    pat_df, data_source, total_visits, patients_with_dicoms = build_osic_patient_cohort(
        demo=demo, min_visits=3
    )

    imaging_model_available = patients_with_dicoms >= MIN_IMAGING_PATIENTS
    if not imaging_model_available:
        logger.warning(
            f"Only {patients_with_dicoms} OSIC patients have cached 3D CT features (< {MIN_IMAGING_PATIENTS} required). "
            "Setting imaging_model_available=False. Only the clinical-only progression model is active."
        )

    X_clin = pat_df[CLINICAL_PROGRESSION_FEATURES].astype(float)
    patients = pat_df["Patient"].values

    # Target Variant 1: Observed ~52-Week FVC Change (mL) for patients with >=3 visits
    df_52w = pat_df[pat_df["has_52w_visit"]].reset_index(drop=True)
    variant_52w = _evaluate_target_nested_gkf(
        X_df=df_52w[CLINICAL_PROGRESSION_FEATURES].astype(float),
        y_reg=df_52w["fvc_change_52w_observed_ml"].values.astype(float),
        y_cls=df_52w["rapid_decliner_52w"].values.astype(int),
        patients=df_52w["Patient"].values,
        target_label="Observed ~52-Week FVC Change (mL, min_visits>=3)",
        units="mL/52w",
    )
    logger.info(
        f"[OSIC Variant 1: ~52w FVC Change (n={len(df_52w)})] Baseline RMSE={variant_52w['baseline_metrics']['rmse_mean']:.1f} mL | "
        f"Best ({variant_52w['best_model']}) RMSE={variant_52w['models'][variant_52w['best_model']]['cv_rmse_mean']:.1f} mL | "
        f"AUC={variant_52w['models'][variant_52w['best_model']]['cv_auc_mean']:.3f}"
    )

    # Target Variant 2: Robust Theil-Sen FVC Slope (mL/week, min_visits>=3)
    variant_slope = _evaluate_target_nested_gkf(
        X_df=X_clin,
        y_reg=pat_df["fvc_slope"].values.astype(float),
        y_cls=pat_df["rapid_decliner_slope"].values.astype(int),
        patients=patients,
        target_label="Robust Theil-Sen FVC Slope (mL/week, min_visits>=3)",
        units="mL/week",
    )
    logger.info(
        f"[OSIC Variant 2: Robust Slope (n={len(pat_df)})] Baseline RMSE={variant_slope['baseline_metrics']['rmse_mean']:.3f} mL/wk | "
        f"Best ({variant_slope['best_model']}) RMSE={variant_slope['models'][variant_slope['best_model']]['cv_rmse_mean']:.3f} mL/wk | "
        f"AUC={variant_slope['models'][variant_slope['best_model']]['cv_auc_mean']:.3f}"
    )

    # 3-Way Modality Ablation on BOTH targets (only runs if >= 100 patients have cached 3D CT features)
    imaging_regressor = None
    imaging_classifier = None
    osic_competition_benchmark: Dict[str, Any] = {"status": "skipped (<100 CT patients)"}
    fusion_comparison: Dict[str, Any] = {"status": "skipped (<100 CT patients)"}
    selected_fusion_method: str = "clinical_only"
    multimodal_serving_models: Optional[Dict[str, Any]] = None
    multimodal_feature_cols = CLINICAL_PROGRESSION_FEATURES + IMAGING_FEATURE_NAMES

    if imaging_model_available:
        ct_sub = pat_df[pat_df["has_ct"]].reset_index(drop=True)
        ct_sub_52w = ct_sub[ct_sub["has_52w_visit"]].reset_index(drop=True)

        ablation_52w = evaluate_modality_ablation_nested_gkf(
            ct_sub_52w,
            target_reg_col="fvc_change_52w_observed_ml",
            target_cls_col="rapid_decliner_52w",
            target_label="Observed ~52-Week FVC Change (mL, CT cohort)",
            units="mL/52w",
            n_pca_components=PCA_COMPONENTS,
            n_splits=5,
            n_repeats=5,
        )
        ablation_slope = evaluate_modality_ablation_nested_gkf(
            ct_sub,
            target_reg_col="fvc_slope",
            target_cls_col="rapid_decliner_slope",
            target_label="Robust Theil-Sen FVC Slope (mL/week, CT cohort)",
            units="mL/week",
            n_pca_components=PCA_COMPONENTS,
            n_splits=5,
            n_repeats=5,
        )
        any_img_beats_clin = bool(
            ablation_52w["any_imaging_beats_clinical_only_ci"]
            or ablation_slope["any_imaging_beats_clinical_only_ci"]
        )
        modality_ablation = {
            "status": "completed",
            "num_ct_patients": int(len(ct_sub)),
            "pca_components_used": ablation_52w["pca_components_used"],
            "split_strategy": "5-Repeat × 5-Fold GroupKFold by Patient (PCA fitted strictly inside each training fold)",
            "modalities": ablation_52w["modalities"],
            "by_target": {
                "fvc_change_52w": ablation_52w,
                "robust_theil_sen_slope": ablation_slope,
            },
            "hu_beats_clinical_only": bool(
                ablation_52w["hu_beats_clinical_only"] or ablation_slope["hu_beats_clinical_only"]
            ),
            "resnet_pca_beats_clinical_only": bool(
                ablation_52w["resnet_pca_beats_clinical_only"]
                or ablation_slope["resnet_pca_beats_clinical_only"]
            ),
            "any_imaging_beats_clinical_only_ci": any_img_beats_clin,
        }
        logger.info(
            f"[OSIC 3-Way Modality Ablation (n={len(ct_sub)} CT patients)] Completed for both targets | "
            f"Any imaging beats Clinical-only (non-overlapping 95% CI): {any_img_beats_clin}"
        )

        # Requirements A & B: Official Modified Laplace Log-Likelihood Literature Benchmark + 6-Method Fusion Comparison
        verified_osic_csv, _, _ = ensure_osic_dataset(OSIC_TRAIN_CSV_PATH, OSIC_DICOM_DIR, demo=demo)
        raw_train_df = pd.read_csv(verified_osic_csv)
        logger.info(
            "Running OSIC Competition Metric Benchmark (Modified Laplace LL) and 6-Method Fusion Comparison (5x5 Nested GroupKFold)..."
        )
        (
            osic_competition_benchmark,
            fusion_comparison,
            multimodal_serving_models,
        ) = run_osic_benchmarks_and_fusion(
            raw_train_df=raw_train_df,
            pat_df=pat_df,
            hu_cols=HU_VOLUME_FEATURE_NAMES,
            resnet_cols=RESNET_FEATURE_NAMES,
            n_pca=PCA_COMPONENTS,
        )
        selected_fusion_method = str(fusion_comparison["selected_fusion_method"])

        # Fit final multimodal pipelines (separate fresh preprocessors for regressor and classifier)
        imaging_regressor = Pipeline(
            [
                ("prep", build_in_fold_resnet_pca_preprocessor(n_components=PCA_COMPONENTS, include_hu=True)),
                (
                    "reg",
                    XGBRegressor(
                        n_estimators=80,
                        max_depth=3,
                        learning_rate=0.05,
                        random_state=RANDOM_SEED,
                        verbosity=0,
                    ),
                ),
            ]
        ).fit(ct_sub[multimodal_feature_cols], ct_sub["fvc_slope"].values)
        imaging_classifier = Pipeline(
            [
                ("prep", build_in_fold_resnet_pca_preprocessor(n_components=PCA_COMPONENTS, include_hu=True)),
                (
                    "cls",
                    XGBClassifier(
                        n_estimators=80,
                        max_depth=3,
                        learning_rate=0.05,
                        eval_metric="logloss",
                        random_state=RANDOM_SEED,
                        verbosity=0,
                    ),
                ),
            ]
        ).fit(ct_sub[multimodal_feature_cols], ct_sub["rapid_decliner_slope"].values)
    else:
        modality_ablation = {
            "status": f"skipped ({patients_with_dicoms}/{len(pat_df)} patients have cached CT features; >={MIN_IMAGING_PATIENTS} required)",
            "pca_components_configured": PCA_COMPONENTS,
            "modalities": {
                "Clinical-only": {
                    "status": "evaluated",
                    "slope_rmse": variant_slope["models"][variant_slope["best_model"]]["rmse_stats"],
                    "fvc_52w_rmse": variant_52w["models"][variant_52w["best_model"]]["rmse_stats"],
                },
                "Clinical + HU/Volume": {"status": "blocked_missing_dicoms (<100 patients)"},
                f"Clinical + ResNet-18 (In-Fold PCA {PCA_COMPONENTS}D)": {
                    "status": "blocked_missing_dicoms (<100 patients)"
                },
            },
            "hu_beats_clinical_only": None,
            "resnet_pca_beats_clinical_only": None,
            "any_imaging_beats_clinical_only_ci": False,
        }

    # Fit final clinical-only estimators on all eligible patients
    final_clin_reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "reg",
                XGBRegressor(
                    n_estimators=70,
                    max_depth=2,
                    learning_rate=0.05,
                    random_state=RANDOM_SEED,
                    verbosity=0,
                ),
            ),
        ]
    ).fit(X_clin, pat_df["fvc_slope"].values)

    final_clin_cls = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "cls",
                XGBClassifier(
                    n_estimators=70,
                    max_depth=2,
                    learning_rate=0.05,
                    eval_metric="logloss",
                    random_state=RANDOM_SEED,
                    verbosity=0,
                ),
            ),
        ]
    ).fit(X_clin, pat_df["rapid_decliner_slope"].values)

    best_slope_m = variant_slope["best_model"]
    best_52w_m = variant_52w["best_model"]
    any_clearly_beats = bool(
        variant_slope["clearly_beats_baseline"] or variant_52w["clearly_beats_baseline"]
    )
    img_verdict = (
        "At least one imaging modality improved over Clinical-only with non-overlapping 95% CIs."
        if modality_ablation.get("any_imaging_beats_clinical_only_ci")
        else "Neither Clinical + HU/Volume nor Clinical + ResNet-18 (In-Fold PCA 15D) clearly beat Clinical-only with non-overlapping 95% CIs; progression modeling remains strictly exploratory."
        if imaging_model_available
        else f"3-way imaging ablation skipped ({patients_with_dicoms}/{len(pat_df)} CT patients; >={MIN_IMAGING_PATIENTS} required)."
    )

    honest_assessment = (
        f"{EXPLORATORY_PROGRESSION_NOTE}. Evaluated via 5-Repeat × 5-Fold Nested GroupKFold by Patient "
        f"(min_visits>=3). Variant 1 (~52-week FVC change, n={variant_52w['num_patients']}): Baseline RMSE = "
        f"{variant_52w['baseline_metrics']['rmse_mean']:.1f} mL vs. {best_52w_m} RMSE = "
        f"{variant_52w['models'][best_52w_m]['cv_rmse_mean']:.1f} mL (AUC = "
        f"{variant_52w['models'][best_52w_m]['cv_auc_mean']:.3f}). Variant 2 (Robust Theil-Sen slope, "
        f"n={variant_slope['num_patients']}): Baseline RMSE = {variant_slope['baseline_metrics']['rmse_mean']:.3f} mL/wk "
        f"vs. {best_slope_m} RMSE = {variant_slope['models'][best_slope_m]['cv_rmse_mean']:.3f} mL/wk "
        f"(AUC = {variant_slope['models'][best_slope_m]['cv_auc_mean']:.3f}). "
        f"imaging_model_available={imaging_model_available} ({patients_with_dicoms}/{len(pat_df)} patients have cached 3D CT features). "
        f"{img_verdict}"
    )

    bundle = {
        "exploratory_banner": EXPLORATORY_PROGRESSION_NOTE,
        "study_description": honest_assessment,
        "data_source": data_source,
        "imaging_model_available": imaging_model_available,
        "min_imaging_patients_required": MIN_IMAGING_PATIENTS,
        "patients_with_dicoms": patients_with_dicoms,
        "usable_as_standalone_predictor": False,
        "any_model_beats_both_baselines": any_clearly_beats,
        "best_model_name": best_slope_m,
        "clinical_feature_cols": CLINICAL_PROGRESSION_FEATURES,
        "multimodal_feature_cols": multimodal_feature_cols,
        "clinical_regressor": final_clin_reg,
        "clinical_classifier": final_clin_cls,
        "imaging_regressor": imaging_regressor,
        "imaging_classifier": imaging_classifier,
        "baseline_metrics": variant_slope["baseline_metrics"],
        "metrics": variant_slope["models"],
        "target_variants": {
            "fvc_change_52w": variant_52w,
            "robust_theil_sen_slope": variant_slope,
        },
        "modality_ablation": modality_ablation,
        "osic_competition_benchmark": osic_competition_benchmark,
        "fusion_comparison": fusion_comparison,
        "selected_fusion_method": selected_fusion_method,
        "multimodal_serving_models": multimodal_serving_models,
        "split_strategy": "5-Repeat × 5-Fold Repeated Nested GroupKFold by Patient (25 outer folds, inner 3-Fold GroupKFold tuning)",
        "num_patients": int(len(pat_df)),
        "num_visits": total_visits,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, save_path)
    logger.info(
        f"Saved Exploratory OSIC Progression Bundle (imaging_model_available={imaging_model_available}) to {save_path}"
    )
    return bundle
