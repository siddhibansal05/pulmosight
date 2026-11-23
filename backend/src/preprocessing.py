"""
Part 1 Preprocessing Pipeline for the Real Zenodo ILD Clinical Dataset (`backend/src/preprocessing.py`).

Responsibilities:
- Verify and load `database_final_anonymized_for_zenodo.csv` using `pd.read_csv(file, sep=';', decimal=',')`
  (raises `FileNotFoundError` if missing and `demo=False`).
- Verify whether `Case_number` is unique across all rows (6,808 unique `Case_number`s in 6,808 rows).
  If any `Case_number` appears more than once, automatically enforce patient-grouped splitting
  (`StratifiedGroupKFold` / `GroupShuffleSplit` by `Case_number`).
- Compute baseline `age = age at event - survival time / 365.25`.
- Explicitly drop leakage/administrative columns (`Case_number`, `First test`, `event_date`,
  `dead or alive`, `survival time`, `age at event`).
- Construct binary label: `IPF = 1`, all other real Zenodo diagnosis groups
  (`1_SAR`, `o-ILD`, `CTD`, `HP`, `u-ILD`, `i-NSIP`) = `0`.
- Use EXPLICIT physical units matching the real Zenodo dataset:
  * `Height`: meters (`m`), e.g. `1.72` (no heuristic threshold guessing).
  * `Weight`: kilograms (`kg`), e.g. `78.5`.
  * `fev1fvc_abs`: decimal ratio (`FEV1 / FVC`), e.g. `0.82` (no heuristic threshold guessing).
  * `fev1_pp`, `fvc_pp`, `tlc_pp`, `tlco_pp`: percent predicted (`%`).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.src.config import (
    FEATURE_SETS,
    LEAKAGE_AND_EXCLUDED_COLUMNS,
    POSITIVE_CLASS_LABEL,
    RANDOM_SEED,
    TARGET_COLUMN,
    ZENODO_CSV_PATH,
    setup_logger,
)
from backend.src.generate_sample_data import ensure_zenodo_dataset

logger = setup_logger("preprocessing")

DIAGNOSIS_GROUP_DESCRIPTIONS = {
    "IPF": "Idiopathic Pulmonary Fibrosis",
    "1_SAR": "Sarcoidosis (Stage I–IV)",
    "o-ILD": "Other Interstitial Lung Diseases",
    "CTD": "Connective Tissue Disease-Associated ILD",
    "HP": "Hypersensitivity Pneumonitis",
    "u-ILD": "Unclassifiable Interstitial Lung Disease",
    "i-NSIP": "Idiopathic Non-Specific Interstitial Pneumonia",
}


def verify_case_number_uniqueness(df: pd.DataFrame) -> Dict[str, int | bool]:
    """
    Verify whether `Case_number` is unique in the Zenodo dataset.
    Returns metadata dictionary `{total_rows, unique_case_numbers, is_unique}`.
    """
    total = int(len(df))
    if "Case_number" not in df.columns:
        return {"total_rows": total, "unique_case_numbers": total, "is_unique": True}
    unique_cases = int(df["Case_number"].nunique())
    is_unique = unique_cases == total
    logger.info(
        f"Zenodo Case_number uniqueness check: total_rows={total}, "
        f"unique_case_numbers={unique_cases}, is_unique={is_unique}"
    )
    return {
        "total_rows": total,
        "unique_case_numbers": unique_cases,
        "is_unique": is_unique,
    }


def load_raw_zenodo_dataframe(
    csv_path: Path = ZENODO_CSV_PATH,
    demo: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """
    Load the Zenodo ILD CSV with `sep=';'` and `decimal=','`.
    Raises `FileNotFoundError` if `csv_path` does not exist and `demo=False`.
    Returns `(df, data_source)` where `data_source` is `'real'` or `'synthetic'`.
    """
    verified_path, data_source = ensure_zenodo_dataset(csv_path, demo=demo)
    df = pd.read_csv(verified_path, sep=";", decimal=",")
    logger.info(
        f"Loaded Zenodo dataset from {verified_path}: shape={df.shape}, data_source={data_source}"
    )
    return df, data_source


def compute_eda_summary(df: pd.DataFrame, data_source: str = "real") -> Dict:
    """
    Extract Exploratory Data Analysis (EDA) statistics from the real Zenodo dataset
    (diagnosis group counts across `1_SAR`, `o-ILD`, `IPF`, `CTD`, `HP`, `u-ILD`, `i-NSIP`,
    computed baseline age distribution, and survival rates by diagnosis group)
    BEFORE dropping survival columns.
    """
    case_check = verify_case_number_uniqueness(df)
    work = df.copy()
    work["age"] = work["age at event"].astype(float) - (
        work["survival time"].astype(float) / 365.25
    )
    work["is_ipf"] = (
        work[TARGET_COLUMN].astype(str).str.strip() == POSITIVE_CLASS_LABEL
    ).astype(int)

    group_counts = work[TARGET_COLUMN].value_counts()
    total_rows = int(len(work))
    diagnosis_distribution = []
    for grp, count in group_counts.items():
        grp_str = str(grp).strip()
        sub = work[work[TARGET_COLUMN] == grp]
        diagnosis_distribution.append(
            {
                "group": grp_str,
                "full_name": DIAGNOSIS_GROUP_DESCRIPTIONS.get(grp_str, grp_str),
                "category": "IPF" if grp_str == POSITIVE_CLASS_LABEL else "Non-IPF",
                "count": int(count),
                "percentage": round(float(count) / total_rows * 100.0, 2),
                "mean_fvc_pp": round(float(sub["fvc_pp"].mean()), 1),
                "mean_tlco_pp": round(float(sub["tlco_pp"].mean()), 1),
                "median_age": round(float(sub["age"].median()), 1),
            }
        )

    age_bins = [-np.inf, 45, 55, 65, 75, 85, np.inf]
    age_labels = ["<45 yrs", "45–54 yrs", "55–64 yrs", "65–74 yrs", "75–84 yrs", "≥85 yrs"]
    work["age_bracket"] = pd.cut(work["age"], bins=age_bins, labels=age_labels, right=False)

    age_distribution = []
    for label in age_labels:
        bracket_df = work[work["age_bracket"] == label]
        ipf_c = int((bracket_df["is_ipf"] == 1).sum())
        non_ipf_c = int((bracket_df["is_ipf"] == 0).sum())
        age_distribution.append(
            {
                "ageRange": label,
                "ipf": ipf_c,
                "nonIpf": non_ipf_c,
                "total": ipf_c + non_ipf_c,
            }
        )

    ipf_sub = work[work["is_ipf"] == 1]
    non_ipf_sub = work[work["is_ipf"] == 0]
    ctd_nsip_sub = work[work[TARGET_COLUMN].isin(["CTD", "i-NSIP", "1_SAR"])]
    other_non_ipf_sub = work[work[TARGET_COLUMN].isin(["o-ILD", "HP", "u-ILD"])]

    months_grid = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
    survival_by_diagnosis = []

    def _surv_pct(sub_df: pd.DataFrame, days_thresh: float) -> float:
        if len(sub_df) == 0:
            return 0.0
        survived = (sub_df["survival time"].astype(float) >= days_thresh) | (
            sub_df["dead or alive"].astype(str).str.lower().str.strip() == "alive"
        )
        return round(float(survived.mean() * 100.0), 1)

    for m in months_grid:
        days_threshold = m * 30.4375
        if m == 0:
            survival_by_diagnosis.append(
                {
                    "month": 0,
                    "ipf": 100.0,
                    "nonIpf": 100.0,
                    "nonIpfFibrotic": 100.0,
                    "ctdNsip": 100.0,
                }
            )
        else:
            survival_by_diagnosis.append(
                {
                    "month": m,
                    "ipf": _surv_pct(ipf_sub, days_threshold),
                    "nonIpf": _surv_pct(non_ipf_sub, days_threshold),
                    "nonIpfFibrotic": _surv_pct(other_non_ipf_sub, days_threshold),
                    "ctdNsip": _surv_pct(ctd_nsip_sub, days_threshold),
                }
            )

    return {
        "data_source": data_source,
        "case_number_uniqueness": case_check,
        "total_rows": total_rows,
        "ipf_rows": int(work["is_ipf"].sum()),
        "non_ipf_rows": int((work["is_ipf"] == 0).sum()),
        "ipf_prevalence_pct": round(float(work["is_ipf"].mean() * 100.0), 2),
        "median_age_ipf": round(float(ipf_sub["age"].median()), 1),
        "median_age_non_ipf": round(float(non_ipf_sub["age"].median()), 1),
        "mean_fvc_ipf": round(float(ipf_sub["fvc_pp"].mean()), 1),
        "mean_tlco_ipf": round(float(ipf_sub["tlco_pp"].mean()), 1),
        "median_survival_ipf_months": round(float(ipf_sub["survival time"].median() / 30.4375), 1),
        "diagnosis_distribution": diagnosis_distribution,
        "age_distribution": age_distribution,
        "survival_by_diagnosis": survival_by_diagnosis,
    }


def encode_sex_column(series: pd.Series) -> pd.Series:
    """
    Encode biological sex: F/Female/0 -> 0.0, M/Male/1 -> 1.0.
    Preserves NaN values so downstream SimpleImputer can handle missing entries if present.
    """
    mapping = {
        "F": 0.0,
        "FEMALE": 0.0,
        "0": 0.0,
        "0.0": 0.0,
        "M": 1.0,
        "MALE": 1.0,
        "1": 1.0,
        "1.0": 1.0,
    }

    def _map_val(val):
        if pd.isna(val):
            return np.nan
        cleaned = str(val).strip().upper()
        if cleaned in ("NAN", "NONE", ""):
            return np.nan
        if cleaned not in mapping:
            raise ValueError(f"Invalid sex value '{val}'. Expected 'M' or 'F'.")
        return mapping[cleaned]

    return series.map(_map_val).astype(float)


def validate_height_meters(series: pd.Series) -> pd.Series:
    """
    Validate and convert Height in explicit meters (m).
    Does NOT use threshold-guessing heuristics (e.g. <3.0) to multiply by 100.
    """
    return pd.to_numeric(series, errors="coerce").astype(float)


def validate_fev1fvc_ratio(series: pd.Series) -> pd.Series:
    """
    Validate and convert `fev1fvc_abs` in explicit decimal ratio units (e.g. 0.82).
    Does NOT use threshold-guessing heuristics (e.g. <=1.5) to multiply by 100.
    """
    return pd.to_numeric(series, errors="coerce").astype(float)


def prepare_clinical_features_and_target(
    df: pd.DataFrame,
    feature_set_name: str = "pp_only",
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.Series]:
    """
    Clean raw Zenodo dataframe:
    1. Compute `age = age at event - survival time / 365.25`
    2. Create binary target `y` (`IPF = 1`, all other groups = `0`) and preserve `raw_diagnosis_groups`
    3. Encode `sex` (`F=0.0, M=1.0`), keep `Height` in meters (m) and `fev1fvc_abs` as decimal ratio
    4. Explicitly drop leakage columns (`Case_number`, `First test`, `event_date`,
       `dead or alive`, `survival time`, `age at event`)
    5. Select requested feature set (`pp_only`, `z_only`, or `both`)
    """
    if feature_set_name not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature_set_name '{feature_set_name}'. Choose from {list(FEATURE_SETS.keys())}."
        )

    work = df.copy()

    if "age" not in work.columns:
        work["age"] = pd.to_numeric(work["age at event"], errors="coerce") - (
            pd.to_numeric(work["survival time"], errors="coerce") / 365.25
        )

    raw_diagnosis_groups = work[TARGET_COLUMN].astype(str).str.strip()
    y = (raw_diagnosis_groups == POSITIVE_CLASS_LABEL).astype(int)

    cols_to_drop = [c for c in LEAKAGE_AND_EXCLUDED_COLUMNS if c in work.columns]
    work = work.drop(columns=cols_to_drop, errors="ignore")

    work["sex"] = encode_sex_column(work["sex"])
    work["Height"] = validate_height_meters(work["Height"])
    work["fev1fvc_abs"] = validate_fev1fvc_ratio(work["fev1fvc_abs"])

    feature_cols = FEATURE_SETS[feature_set_name]
    X = work[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float).copy()

    return X, y, feature_cols, raw_diagnosis_groups


def build_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    """
    Build an in-fold scikit-learn ColumnTransformer with median imputation and StandardScaler.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_cols)],
        remainder="drop",
    )


def get_stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    diagnosis_groups: Optional[pd.Series] = None,
    case_numbers: Optional[pd.Series] = None,
    test_size: float = 0.20,
    random_state: int = RANDOM_SEED,
):
    """
    Perform an 80/20 stratified train/test split.
    If `case_numbers` is provided and any `Case_number` appears more than once,
    automatically uses `GroupShuffleSplit` grouped by `Case_number` to prevent patient leakage.
    """
    if case_numbers is not None and not case_numbers.is_unique:
        logger.warning(
            "Duplicate Case_number detected! Splitting via GroupShuffleSplit by Case_number."
        )
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr_idx, te_idx = next(gss.split(X, y, groups=case_numbers.values))
        if diagnosis_groups is not None:
            return (
                X.iloc[tr_idx],
                X.iloc[te_idx],
                y.iloc[tr_idx],
                y.iloc[te_idx],
                diagnosis_groups.iloc[tr_idx],
                diagnosis_groups.iloc[te_idx],
            )
        return X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]

    if diagnosis_groups is not None:
        return train_test_split(
            X,
            y,
            diagnosis_groups,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
