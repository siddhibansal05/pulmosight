"""
Dataset Verification & Optional Demo Generator (`backend/src/generate_sample_data.py`).

CRITICAL SAFETY GUARANTEES:
1. `ensure_zenodo_dataset(..., demo=False)` and `ensure_osic_dataset(..., demo=False)`
   NEVER silently generate synthetic data. If the required real dataset files are missing
   and `demo=False`, they immediately raise a `FileNotFoundError`.
2. Even when `demo=True` (`--demo` flag) is passed, this module checks whether the existing
   file on disk is real data (by MD5 `07f13b8826c4e67fb4064e3b2340ed5e` for Zenodo, or `>=170`
   patients for OSIC `train.csv`) and NEVER overwrites real data under any circumstances.
3. Running this file directly as a script requires an explicit `--demo` CLI flag.
"""

import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from backend.src.config import (
    OSIC_DICOM_DIR,
    OSIC_TRAIN_CSV_PATH,
    RANDOM_SEED,
    REAL_ZENODO_MD5,
    ZENODO_CSV_PATH,
    compute_file_md5,
    set_global_seed,
    setup_logger,
)

logger = setup_logger("data_verification")


def ensure_zenodo_dataset(
    csv_path: Path = ZENODO_CSV_PATH,
    demo: bool = False,
) -> Tuple[Path, str]:
    """
    Verify that the Zenodo ILD dataset exists at `csv_path`.
    - Never overwrites a real dataset (MD5 == REAL_ZENODO_MD5), even if `demo=True`.
    - Raises `FileNotFoundError` if `csv_path` is missing and `demo=False`.
    """
    if csv_path.exists():
        md5_hash = compute_file_md5(csv_path)
        if md5_hash == REAL_ZENODO_MD5:
            logger.info(
                f"Verified REAL Zenodo CSV at {csv_path} (md5={md5_hash}, data_source='real'). "
                "Real data is write-protected and will never be overwritten."
            )
            return csv_path, "real"
        if not demo:
            raise FileNotFoundError(
                f"File at '{csv_path}' does not match real Zenodo MD5 ({REAL_ZENODO_MD5}, got {md5_hash}) "
                "and `--demo` was not set. Replace with the real Zenodo file or pass `--demo`."
            )
        return csv_path, "synthetic"

    if not demo:
        raise FileNotFoundError(
            f"Real Zenodo ILD dataset not found at '{csv_path}'. "
            "Please place the real 'database_final_anonymized_for_zenodo.csv' "
            f"(expected MD5: {REAL_ZENODO_MD5}) in '{csv_path.parent}', "
            "or run with `--demo` to explicitly generate synthetic demo data."
        )

    logger.warning(
        f"[--demo mode] Generating synthetic demo Zenodo dataset at {csv_path} (data_source='synthetic')..."
    )
    set_global_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    n_total = 6808
    n_ipf = 757
    n_non_ipf = n_total - n_ipf

    non_ipf_groups = rng.choice(
        ["1_SAR", "o-ILD", "CTD", "HP", "u-ILD", "i-NSIP"],
        size=n_non_ipf,
        p=[0.30, 0.25, 0.16, 0.11, 0.11, 0.07],
    )
    diagnosis_group = np.concatenate([np.repeat("IPF", n_ipf), non_ipf_groups])
    perm = rng.permutation(n_total)
    diagnosis_group = diagnosis_group[perm]
    is_ipf = (diagnosis_group == "IPF").astype(int)

    sex_male = rng.binomial(1, np.where(is_ipf == 1, 0.73, 0.48))
    sex = np.where(sex_male == 1, "M", "F")
    baseline_age = np.clip(
        np.where(
            is_ipf == 1,
            rng.normal(69.5, 7.8, n_total),
            rng.normal(57.5, 12.4, n_total),
        ),
        18.0,
        92.0,
    )
    survival_days = np.round(
        np.clip(
            np.where(
                is_ipf == 1,
                rng.exponential(scale=1150, size=n_total) + 60,
                rng.exponential(scale=2100, size=n_total) + 120,
            ),
            30,
            5400,
        )
    ).astype(int)
    age_at_event = np.round(baseline_age + (survival_days / 365.25), 2)

    height = np.round(
        np.clip(
            np.where(
                sex_male == 1,
                rng.normal(1.74, 0.068, n_total),
                rng.normal(1.62, 0.062, n_total),
            ),
            1.40,
            2.05,
        ),
        2,
    )
    weight = np.round(
        np.clip(
            np.where(
                sex_male == 1,
                rng.normal(81.5, 13.5, n_total),
                rng.normal(68.5, 13.0, n_total),
            ),
            40.0,
            155.0,
        ),
        1,
    )
    fvc_pp = np.round(
        np.clip(
            np.where(
                is_ipf == 1,
                rng.normal(68.5, 15.5, n_total),
                rng.normal(85.0, 18.5, n_total),
            ),
            25.0,
            145.0,
        ),
        3,
    )
    tlc_pp = np.round(np.clip(fvc_pp * 0.96 + rng.normal(0.0, 7.0, n_total), 28.0, 145.0), 3)
    tlco_pp = np.round(
        np.clip(
            np.where(
                is_ipf == 1,
                rng.normal(48.0, 14.5, n_total),
                rng.normal(69.5, 18.0, n_total),
            ),
            14.0,
            140.0,
        ),
        3,
    )
    fev1fvc_abs = np.round(
        np.clip(
            np.where(
                is_ipf == 1,
                rng.normal(0.832, 0.058, n_total),
                rng.normal(0.775, 0.076, n_total),
            ),
            0.40,
            0.98,
        ),
        6,
    )
    fev1_pp = np.round(
        np.clip(fvc_pp * (fev1fvc_abs / 0.79) + rng.normal(0, 4.2, n_total), 22.0, 145.0),
        3,
    )

    fev1_z = np.round((fev1_pp - 100.0) / 13.5, 3)
    fvc_z = np.round((fvc_pp - 100.0) / 13.0, 3)
    fev1fvc_z = np.round((fev1fvc_abs - 0.78) / 0.06, 3)
    tlc_z = np.round((tlc_pp - 100.0) / 12.5, 3)
    tlco_z = np.round((tlco_pp - 100.0) / 14.0, 3)

    df = pd.DataFrame(
        {
            "Case_number": np.arange(1, 1 + n_total),
            "First test": ["15.01.2015"] * n_total,
            "sex": sex,
            "fev1fvc_abs": fev1fvc_abs,
            "fev1_z": fev1_z,
            "fev1_pp": fev1_pp,
            "fvc_z": fvc_z,
            "fvc_pp": fvc_pp,
            "fev1fvc_z": fev1fvc_z,
            "tlc_z": tlc_z,
            "tlc_pp": tlc_pp,
            "tlco_z": tlco_z,
            "tlco_pp": tlco_pp,
            "Height": height,
            "Weight": weight,
            "event_date": ["15.01.2018"] * n_total,
            "dead or alive": np.where(is_ipf == 1, "dead", "alive"),
            "age at event": age_at_event,
            "diagnosis group": diagnosis_group,
            "survival time": survival_days,
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, sep=";", decimal=",", index=False)
    return csv_path, "synthetic"


def generate_demo_dicom_series(
    patient_dir: Path,
    patient_id: str,
    num_slices: int = 12,
    rows: int = 512,
    cols: int = 512,
    fibrosis_severity: float = 0.5,
) -> None:
    """
    Generate a multi-slice 512x512 thoracic CT DICOM series ONLY when `--demo` is explicitly requested
    or inside isolated `tmp_path` unit tests. Refuses to overwrite existing real DICOM directories.
    """
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    if patient_dir.exists() and any(patient_dir.glob("*.dcm")):
        raise FileExistsError(
            f"Refusing to overwrite existing DICOM directory '{patient_dir}'."
        )

    patient_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(patient_id)) % (2**32))

    y, x = np.ogrid[:rows, :cols]
    cy, cx = rows // 2, cols // 2

    for s_idx in range(1, num_slices + 1):
        dcm_path = patient_dir / f"{s_idx}.dcm"
        z_frac = (s_idx - 1) / max(1, num_slices - 1)
        radius_scale = 0.75 + 0.25 * np.sin(np.pi * z_frac)

        hu_image = np.full((rows, cols), -1000.0, dtype=np.float32)
        hu_image[int(rows * 0.86) : int(rows * 0.90), int(cols * 0.20) : int(cols * 0.80)] = 120.0

        chest_mask = (
            ((x - cx) ** 2) / ((int(200 * radius_scale)) ** 2)
            + ((y - cy) ** 2) / ((int(150 * radius_scale)) ** 2)
        ) <= 1.0
        hu_image[chest_mask] = 45.0

        r_rx = max(25, int(72 * radius_scale))
        r_ry = max(35, int(105 * radius_scale))
        right_lung = (
            ((x - int(cx - 82 * radius_scale)) ** 2) / (r_rx**2) + ((y - cy) ** 2) / (r_ry**2)
        ) <= 1.0
        left_lung = (
            ((x - int(cx + 82 * radius_scale)) ** 2) / (r_rx**2) + ((y - cy) ** 2) / (r_ry**2)
        ) <= 1.0
        lung_mask = right_lung | left_lung

        basal_weight = 0.6 + 0.4 * z_frac
        mean_hu = -810.0 + (fibrosis_severity * basal_weight * 210.0)
        std_hu = 50.0 + (fibrosis_severity * 40.0)
        lung_voxels = rng.normal(mean_hu, std_hu, size=int(lung_mask.sum()))
        hu_image[lung_mask] = np.clip(lung_voxels, -995.0, -180.0)

        intercept = -1024.0
        slope = 1.0
        raw_pixels = np.clip(np.round((hu_image - intercept) / slope), 0, 4095).astype(np.uint16)

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(str(dcm_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.PatientID = patient_id
        ds.Modality = "CT"
        ds.InstanceNumber = s_idx
        ds.SliceLocation = str(round(-240.0 + s_idx * 2.5, 2))
        ds.ImagePositionPatient = ["0.0", "0.0", str(round(-240.0 + s_idx * 2.5, 2))]
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleIntercept = str(intercept)
        ds.RescaleSlope = str(slope)
        ds.PixelSpacing = ["0.703125", "0.703125"]
        ds.SliceThickness = "2.5"
        ds.SpacingBetweenSlices = "2.5"
        ds.PixelData = raw_pixels.tobytes()
        ds.save_as(str(dcm_path), enforce_file_format=True)


def ensure_osic_dataset(
    csv_path: Path = OSIC_TRAIN_CSV_PATH,
    dicom_root: Path = OSIC_DICOM_DIR,
    demo: bool = False,
    require_dicoms: bool = False,
) -> Tuple[Path, str, bool]:
    """
    Verify that the OSIC `train.csv` dataset exists.
    Never overwrites real OSIC data (`>= 170` patients).
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Real Kaggle OSIC 'train.csv' not found at '{csv_path}'. "
            "See 'backend/scripts/download_osic.md' for instructions to download from Kaggle."
        )

    df = pd.read_csv(csv_path)
    n_patients = int(df["Patient"].nunique())
    n_rows = int(len(df))

    is_real_csv = n_patients >= 170 and n_rows >= 1500
    if not is_real_csv and not demo:
        raise FileNotFoundError(
            f"OSIC 'train.csv' at '{csv_path}' appears to be synthetic ({n_patients} patients, {n_rows} rows). "
            "Real Kaggle OSIC 'train.csv' has 176 patients and 1,549 rows. "
            "Replace it with the real Kaggle file or pass `--demo`."
        )

    dcm_files = list(dicom_root.rglob("*.dcm")) if dicom_root.exists() else []
    has_dicoms = len(dcm_files) > 0

    if require_dicoms and not has_dicoms and not demo:
        raise FileNotFoundError(
            f"No OSIC DICOM (.dcm) files found in '{dicom_root}'. "
            "See 'backend/scripts/download_osic.md' for instructions."
        )

    data_source = "real" if is_real_csv else "synthetic"
    logger.info(
        f"Verified OSIC train.csv at {csv_path}: {n_patients} patients, {n_rows} rows "
        f"(data_source={data_source}, dicom_files_found={len(dcm_files)})"
    )
    return csv_path, data_source, has_dicoms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional demo data generator. Requires explicit --demo flag and never overwrites real data."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Required flag to allow generating synthetic demo data if real data is missing.",
    )
    args = parser.parse_args()
    if not args.demo:
        raise RuntimeError(
            "generate_sample_data.py requires an explicit `--demo` flag and will never overwrite real data."
        )
    ensure_zenodo_dataset(ZENODO_CSV_PATH, demo=True)


if __name__ == "__main__":
    main()
