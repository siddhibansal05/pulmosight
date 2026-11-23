"""
Configuration, constants, fixed random seed, and logging setup for the
IPF Diagnostic Support System backend (`backend/src/config.py`).
"""

import hashlib
import logging
import os
import random
from pathlib import Path
import numpy as np

MODEL_VERSION = "2.2.0"
RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds across standard libraries, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# Directory paths inside `backend/`
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
CONFIG_DIR = BACKEND_DIR / "config"
OSIC_DIR = DATA_DIR / "osic"
OSIC_DICOM_DIR = OSIC_DIR / "dicom"
MODELS_DIR = BACKEND_DIR / "models"
PLOTS_DIR = MODELS_DIR / "plots"
NOTEBOOKS_DIR = BACKEND_DIR / "notebooks"
SCRIPTS_DIR = BACKEND_DIR / "scripts"

for directory in [
    DATA_DIR,
    CONFIG_DIR,
    OSIC_DIR,
    OSIC_DICOM_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    NOTEBOOKS_DIR,
    SCRIPTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Configurable clinical interpretation rules path
INTERPRETATION_RULES_PATH = CONFIG_DIR / "interpretation_rules.json"

# Dataset file paths
ZENODO_CSV_PATH = DATA_DIR / "database_final_anonymized_for_zenodo.csv"
REAL_ZENODO_MD5 = "07f13b8826c4e67fb4064e3b2340ed5e"
REAL_ZENODO_DIAGNOSIS_GROUPS = {"1_SAR", "o-ILD", "IPF", "CTD", "HP", "u-ILD", "i-NSIP"}

OSIC_TRAIN_CSV_PATH = OSIC_DIR / "train.csv"
CACHED_DICOM_FEATURES_CSV = OSIC_DIR / "cached_dicom_features.csv"

# Minimum number of patients with 3D DICOM scans required before training an Imaging + Clinical model
MIN_IMAGING_PATIENTS = 100

# Saved model artifact paths
CLINICAL_MODEL_PATH = MODELS_DIR / "ipf_clinical_pipeline.joblib"
PROGRESSION_MODEL_PATH = MODELS_DIR / "osic_progression_pipeline.joblib"
ARTIFACTS_JSON_PATH = MODELS_DIR / "model_artifacts.json"

# Part 1: Zenodo ILD Clinical Dataset Specification
TARGET_COLUMN = "diagnosis group"
POSITIVE_CLASS_LABEL = "IPF"

# Columns that MUST NOT be used as features (data leakage or administrative identifiers)
LEAKAGE_AND_EXCLUDED_COLUMNS = [
    "Case_number",
    "First test",
    "event_date",
    "dead or alive",
    "survival time",
    "age at event",
]

# Base demographic & spirometric ratio features (Units: Height in meters, Weight in kg, fev1fvc_abs as decimal ratio)
DEMOGRAPHIC_FEATURES = ["sex", "Height", "Weight", "age", "fev1fvc_abs"]

# Percent-predicted (pp) lung function features
PP_FEATURES = ["fev1_pp", "fvc_pp", "tlc_pp", "tlco_pp"]

# Z-score lung function features
Z_FEATURES = ["fev1_z", "fvc_z", "tlc_z", "tlco_z"]

FEATURE_SETS = {
    "pp_only": DEMOGRAPHIC_FEATURES + PP_FEATURES,
    "z_only": DEMOGRAPHIC_FEATURES + Z_FEATURES,
    "both": DEMOGRAPHIC_FEATURES + PP_FEATURES + Z_FEATURES,
}

MANDATORY_DISCLAIMER = "Decision support only, not a clinical diagnosis."
AUTOMATED_INTERPRETATION_NOTICE = (
    "Automated interpretation, decision support only. Verify clinically."
)
EXPLORATORY_PROGRESSION_NOTE = (
    "Progression results are exploratory, separate cohort, no IPF labels"
)


def compute_file_md5(path: Path) -> str:
    """Compute the hexadecimal MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_sklearn_major_minor_match(trained_ver: str, runtime_ver: str) -> bool:
    """Compare major.minor version of scikit-learn (e.g. '1.9.0' vs '1.9.1' -> True)."""
    try:
        t_parts = str(trained_ver).split(".")[:2]
        r_parts = str(runtime_ver).split(".")[:2]
        return t_parts == r_parts and len(t_parts) == 2
    except Exception:
        return False


def setup_logger(name: str = "ipf_cdss") -> logging.Logger:
    """Configure and return a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
