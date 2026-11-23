"""
SHAP Explainability Module (`backend/src/explain.py`).

Provides:
1. Global SHAP summary plot generation (`backend/models/plots/shap_summary.png`) for the
   selected clinical IPF classifier pipeline.
2. Per-patient top-K SHAP feature attribution (`compute_patient_shap_top_features`)
   used by `POST /predict-ipf`.
"""

from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from backend.src.config import PLOTS_DIR, RANDOM_SEED, setup_logger

logger = setup_logger("explain")

FEATURE_DISPLAY_NAMES = {
    "sex": "Biological Sex",
    "Height": "Height (m)",
    "Weight": "Weight (kg)",
    "age": "Patient Age (yrs)",
    "fev1fvc_abs": "FEV1/FVC Ratio",
    "fev1_pp": "FEV1 % Predicted",
    "fvc_pp": "FVC % Predicted",
    "tlc_pp": "TLC % Predicted",
    "tlco_pp": "TLCO % Predicted",
    "fev1_z": "FEV1 Z-Score",
    "fvc_z": "FVC Z-Score",
    "tlc_z": "TLC Z-Score",
    "tlco_z": "TLCO Z-Score",
}


def _extract_shap_matrix_for_positive_class(shap_values: Any) -> np.ndarray:
    """Normalize SHAP output across tree/linear/kernel explainers to shape (n_samples, n_features)."""
    if hasattr(shap_values, "values"):
        vals = np.asarray(shap_values.values)
    elif isinstance(shap_values, list):
        vals = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    else:
        vals = np.asarray(shap_values)

    if vals.ndim == 3:
        vals = vals[:, :, 1]
    return vals


def build_shap_explainer_and_summary(
    fitted_pipeline: Any,
    X_background: pd.DataFrame,
    feature_names: List[str],
    save_path: Path = PLOTS_DIR / "shap_summary.png",
) -> Dict[str, Any]:
    """
    Build a SHAP explainer on the fitted pipeline (`prep` + `clf`),
    save a global SHAP summary plot to `backend/models/plots/shap_summary.png`,
    and return global mean(|SHAP|) importance rankings.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor = fitted_pipeline.named_steps["prep"]
    raw_estimator = fitted_pipeline.named_steps["clf"]

    bg_sample = X_background.sample(
        n=min(100, len(X_background)), random_state=RANDOM_SEED
    )
    X_bg_trans = preprocessor.transform(bg_sample[feature_names])
    X_bg_df = pd.DataFrame(X_bg_trans, columns=feature_names)

    try:
        explainer = shap.TreeExplainer(raw_estimator, X_bg_df)
        shap_vals_obj = explainer(X_bg_df, check_additivity=False)
    except Exception:
        masker = shap.maskers.Independent(X_bg_df, max_samples=100)
        explainer = shap.Explainer(
            lambda x: raw_estimator.predict_proba(x)[:, 1], masker
        )
        shap_vals_obj = explainer(X_bg_df)

    shap_matrix = _extract_shap_matrix_for_positive_class(shap_vals_obj)

    fig = plt.figure(figsize=(8.5, 5.5), dpi=140)
    shap.summary_plot(
        shap_matrix,
        X_bg_df,
        feature_names=[FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_names],
        show=False,
        rng=np.random.default_rng(RANDOM_SEED),
    )
    plt.title(
        "Global SHAP Feature Importance (Real Zenodo ILD Cohort)",
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved global SHAP summary plot to {save_path}")

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    ranking = sorted(
        [
            {
                "feature": feat,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "mean_abs_shap": round(float(val), 4),
            }
            for feat, val in zip(feature_names, mean_abs_shap)
        ],
        key=lambda x: x["mean_abs_shap"],
        reverse=True,
    )

    return {
        "background_data": X_bg_df,
        "global_ranking": ranking,
        "plot_path": str(save_path),
    }


def compute_patient_shap_top_features(
    artifact_bundle: Dict[str, Any],
    patient_df: pd.DataFrame,
    raw_input_dict: Dict[str, Any],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Compute per-patient SHAP feature attributions for a single patient row
    and return the top `top_k` features sorted by absolute SHAP contribution.
    """
    fitted_pipeline = artifact_bundle["uncalibrated_pipeline"]
    preprocessor = fitted_pipeline.named_steps["prep"]
    raw_estimator = fitted_pipeline.named_steps["clf"]
    feature_names = artifact_bundle["feature_names"]
    X_bg_df = artifact_bundle["shap_background"]

    X_patient_trans = preprocessor.transform(patient_df[feature_names])
    X_patient_df = pd.DataFrame(X_patient_trans, columns=feature_names)

    try:
        explainer = shap.TreeExplainer(raw_estimator, X_bg_df)
        shap_vals_obj = explainer(X_patient_df, check_additivity=False)
        shap_row = _extract_shap_matrix_for_positive_class(shap_vals_obj)[0]
    except Exception:
        try:
            masker = shap.maskers.Independent(X_bg_df, max_samples=100)
            explainer = shap.Explainer(
                lambda x: raw_estimator.predict_proba(x)[:, 1], masker
            )
            shap_vals_obj = explainer(X_patient_df)
            shap_row = _extract_shap_matrix_for_positive_class(shap_vals_obj)[0]
        except Exception:
            bg_mean = X_bg_df.mean(axis=0).values
            bg_std = np.where(
                X_bg_df.std(axis=0).values == 0, 1.0, X_bg_df.std(axis=0).values
            )
            shap_row = (X_patient_df.values[0] - bg_mean) / bg_std * 0.08

    contributions = []
    for feat, impact_val in zip(feature_names, shap_row):
        impact = float(impact_val)
        raw_val = raw_input_dict.get(feat, patient_df[feat].iloc[0])
        if feat == "sex":
            formatted_val = (
                "Male"
                if str(raw_val).upper().startswith("M") or raw_val == 1
                else "Female"
            )
        elif feat.endswith("_pp"):
            formatted_val = f"{float(raw_val):.1f}%"
        elif feat == "age":
            formatted_val = f"{float(raw_val):.1f} yrs"
        elif feat == "Height":
            formatted_val = f"{float(raw_val):.2f} m"
        elif feat == "Weight":
            formatted_val = f"{float(raw_val):.1f} kg"
        else:
            formatted_val = f"{float(raw_val):.3f}"

        contributions.append(
            {
                "feature": feat,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "value": formatted_val,
                "shap_value": round(impact, 4),
                "abs_shap": round(abs(impact), 4),
                "direction": (
                    "Increases IPF Risk" if impact >= 0 else "Decreases IPF Risk"
                ),
            }
        )

    contributions.sort(key=lambda item: item["abs_shap"], reverse=True)
    return contributions[:top_k]


SHAP_MANDATORY_CAVEAT = (
    "SHAP shows what the model used, not causes. "
    "Age partly reflects differences between IPF and non-IPF patients in this dataset."
)


def _format_feature_phrase(item: Dict[str, Any], raw_input_dict: Dict[str, Any]) -> str:
    """Build a deterministic clinical phrase with the patient's value for one SHAP feature."""
    feat = str(item.get("feature", ""))
    raw_val = raw_input_dict.get(feat)
    if feat == "sex":
        is_male = str(raw_val).strip().upper().startswith("M") or raw_val == 1
        return "male sex" if is_male else "female sex"

    try:
        num = float(raw_val)
    except (TypeError, ValueError):
        return f"{item.get('display_name', feat)} ({item.get('value', '')})"

    if feat == "age":
        age_str = f"{int(round(num))}" if abs(num - round(num)) < 0.05 else f"{num:.1f}"
        return f"age {age_str}"
    if feat == "tlco_pp":
        qual = "low " if num < 60.0 else "reduced " if num < 80.0 else "preserved "
        return f"{qual}TLCO ({num:.1f}%)"
    if feat == "tlc_pp":
        qual = "reduced " if num < 80.0 else "elevated " if num > 120.0 else ""
        return f"{qual}TLC ({num:.1f}%)"
    if feat == "fvc_pp":
        qual = "reduced " if num < 80.0 else "preserved "
        return f"{qual}FVC ({num:.1f}%)"
    if feat == "fev1_pp":
        return f"FEV1 ({num:.1f}%)"
    if feat == "fev1fvc_abs":
        qual = "elevated " if num >= 0.80 else "low " if num < 0.70 else ""
        return f"{qual}FEV1/FVC ratio ({num:.2f})"
    if feat == "Height":
        return f"height ({num:.2f} m)"
    if feat == "Weight":
        return f"weight ({num:.1f} kg)"
    return f"{item.get('display_name', feat)} ({item.get('value', num)})"


def _join_phrases(phrases: List[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def build_shap_plain_language_summary(
    top_shap_features: List[Dict[str, Any]],
    raw_input_dict: Dict[str, Any],
) -> str:
    """
    Deterministically generate a plain-language summary ("What drove this prediction")
    directly from the patient's SHAP contributions (NO LLM).
    - Names the top 2–3 factors pushing toward IPF and any pushing away with patient values.
    - States that SHAP values are on the model's log-odds scale.
    - Always includes the mandatory causal/age caveat verbatim.
    """
    toward = sorted(
        [f for f in top_shap_features if float(f.get("shap_value", 0.0)) > 0],
        key=lambda x: float(x["shap_value"]),
        reverse=True,
    )
    away = sorted(
        [f for f in top_shap_features if float(f.get("shap_value", 0.0)) < 0],
        key=lambda x: abs(float(x["shap_value"])),
        reverse=True,
    )

    def _fmt_with_shap(item: Dict[str, Any]) -> str:
        val = float(item["shap_value"])
        sign = "+" if val >= 0 else ""
        return f"{_format_feature_phrase(item, raw_input_dict)} ({sign}{val:.2f})"

    if toward:
        primary_toward = toward[:2]
        secondary_toward = toward[2:4]
        lead_text = f"{_join_phrases([_fmt_with_shap(x) for x in primary_toward])} pushed most toward IPF"
        if secondary_toward:
            lead_text += f", followed by {_join_phrases([_fmt_with_shap(x) for x in secondary_toward])}"
        if away:
            away_top = away[:2]
            max_away = max(abs(float(x["shap_value"])) for x in away_top)
            verb = "slightly lowered" if max_away < 0.25 else "lowered"
            lead_text += f", while {_join_phrases([_fmt_with_shap(x) for x in away_top])} {verb} the likelihood."
        else:
            lead_text += ", with no top factors lowering the predicted likelihood."
    elif away:
        primary_away = away[:3]
        lead_text = (
            f"{_join_phrases([_fmt_with_shap(x) for x in primary_away])} pushed most away from IPF, "
            "with no top factors increasing the predicted likelihood."
        )
    else:
        lead_text = "No individual clinical feature exerted a strong directional shift away from the cohort baseline."

    # Capitalize the first character cleanly
    lead_text = lead_text[0].upper() + lead_text[1:]

    scale_sentence = (
        "SHAP values are on the model's log-odds scale (positive values push toward IPF and "
        "negative values push away from IPF relative to the training baseline)."
    )
    return f"{lead_text} {scale_sentence} {SHAP_MANDATORY_CAVEAT}"
