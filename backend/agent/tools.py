"""
Tool Definitions & Deterministic Wrappers for the PulmoSight Agentic Prototype
(`backend/agent/tools.py`).

All five tools wrap existing validated PulmoSight code:
  1. `predict_ipf`         -> Calibrated Zenodo clinical IPF classifier (screening + confirmatory)
  2. `explain_shap`        -> Per-patient SHAP attributions + deterministic plain-language summary
  3. `extract_ct_features` -> Optional 3D DICOM `.zip` / `.dcm` HU + ResNet-18 feature extraction
  4. `predict_progression` -> Exploratory OSIC multimodal progression risk (clinical, HRCT, fusion)
  5. `generate_report`     -> Structured de-identified clinical decision support summary

Privacy Guarantee:
  `sanitize_deidentified_inputs` enforces a strict numeric/categorical whitelist and strips
  or blocks any PHI identifiers (patient names, IDs, MRNs, DOBs, contact info).
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from backend.src.config import EXPLORATORY_PROGRESSION_NOTE, MANDATORY_DISCLAIMER, MODEL_VERSION
from backend.src.explain import (
    SHAP_MANDATORY_CAVEAT,
    build_shap_plain_language_summary,
    compute_patient_shap_top_features,
)
from backend.src.imaging_progression import (
    extract_features_from_dicom_zip_bytes,
    extract_features_from_upload_bytes,
)
from backend.src.osic_benchmarks import predict_multimodal_progression
from backend.src.preprocessing import (
    encode_sex_column,
    validate_fev1fvc_ratio,
    validate_height_meters,
)

ALLOWED_CLINICAL_INPUT_KEYS = {
    "sex",
    "age",
    "height",
    "weight",
    "fev1fvc_abs",
    "fev1_pp",
    "fvc_pp",
    "tlc_pp",
    "tlco_pp",
    "baseline_fvc_ml",
    "smoking_status",
    "horizon_weeks",
    "operating_mode",
}

FORBIDDEN_PHI_KEYS = {
    "name",
    "patient_name",
    "full_name",
    "first_name",
    "last_name",
    "patient_id",
    "case_number",
    "mrn",
    "dob",
    "date_of_birth",
    "ssn",
    "email",
    "phone",
    "address",
    "clinician_notes",
    "free_text",
}


def sanitize_deidentified_inputs(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strictly enforce de-identification before any data reaches tools or the LLM.
    Raises `ValueError` if explicit PHI keys are supplied with non-empty values,
    and returns ONLY whitelisted numeric/categorical clinical fields.
    """
    flagged_phi = [
        k
        for k, v in payload.items()
        if str(k).strip().lower() in FORBIDDEN_PHI_KEYS and v not in (None, "")
    ]
    if flagged_phi:
        raise ValueError(
            f"De-identification policy violation: PHI/identifier fields {sorted(flagged_phi)} "
            "must never be sent to the agentic endpoint."
        )

    cleaned: Dict[str, Any] = {}
    for key in ALLOWED_CLINICAL_INPUT_KEYS:
        if key not in payload or payload[key] is None:
            continue
        val = payload[key]
        if key == "sex":
            s = str(val).strip().upper()
            if s in ("M", "MALE"):
                cleaned["sex"] = "M"
            elif s in ("F", "FEMALE"):
                cleaned["sex"] = "F"
            else:
                raise ValueError("sex must be 'M', 'F', 'Male', or 'Female'")
        elif key == "smoking_status":
            s = str(val).strip()
            low = s.lower()
            if "current" in low:
                cleaned["smoking_status"] = "Currently smokes"
            elif "ex" in low:
                cleaned["smoking_status"] = "Ex-smoker"
            elif "never" in low:
                cleaned["smoking_status"] = "Never smoked"
            else:
                cleaned["smoking_status"] = "Ex-smoker"
        elif key == "operating_mode":
            mode = str(val).strip().lower()
            cleaned["operating_mode"] = mode if mode in ("screening", "confirmatory") else "screening"
        else:
            cleaned[key] = float(val)
    return cleaned


AGENT_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "predict_ipf",
        "description": (
            "Run the validated Zenodo ILD calibrated XGBoost clinical classifier on de-identified "
            "demographics and PFTs. Returns calibrated IPF probability, Screening and Confirmatory "
            "thresholds/labels, whether the two modes disagree, and sensitivity disclosures."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sex": {"type": "string", "enum": ["M", "F"]},
                "age": {"type": "number"},
                "height": {"type": "number", "description": "Height in meters (1.0-2.5)"},
                "weight": {"type": "number", "description": "Weight in kg (25-260)"},
                "fev1fvc_abs": {"type": "number", "description": "Decimal ratio (0.20-1.15)"},
                "fev1_pp": {"type": "number", "description": "FEV1 % predicted"},
                "fvc_pp": {"type": "number", "description": "FVC % predicted"},
                "tlc_pp": {"type": "number", "description": "TLC % predicted"},
                "tlco_pp": {"type": "number", "description": "TLCO % predicted"},
                "operating_mode": {"type": "string", "enum": ["screening", "confirmatory"]},
            },
            "required": [
                "sex",
                "age",
                "height",
                "weight",
                "fev1fvc_abs",
                "fev1_pp",
                "fvc_pp",
                "tlc_pp",
                "tlco_pp",
            ],
        },
    },
    {
        "name": "explain_shap",
        "description": (
            "Compute per-patient TreeSHAP feature contributions (top 5) on the model's log-odds "
            "scale and return the deterministic plain-language explanation and causal caveat."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sex": {"type": "string", "enum": ["M", "F"]},
                "age": {"type": "number"},
                "height": {"type": "number"},
                "weight": {"type": "number"},
                "fev1fvc_abs": {"type": "number"},
                "fev1_pp": {"type": "number"},
                "fvc_pp": {"type": "number"},
                "tlc_pp": {"type": "number"},
                "tlco_pp": {"type": "number"},
            },
            "required": [
                "sex",
                "age",
                "height",
                "weight",
                "fev1fvc_abs",
                "fev1_pp",
                "fvc_pp",
                "tlc_pp",
                "tlco_pp",
            ],
        },
    },
    {
        "name": "extract_ct_features",
        "description": (
            "Extract quantitative 3D HU/volume statistics and pretrained ResNet-18 embeddings "
            "from an attached de-identified CT DICOM series (.zip) if provided in the session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "predict_progression",
        "description": (
            "Run the exploratory OSIC longitudinal progression models (separate cohort, predicts "
            "FVC decline, NOT an IPF diagnosis). Returns clinical-only score and, only when a CT "
            "series was extracted, HRCT-only and selected fusion scores with CV 95% CIs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "age": {"type": "number"},
                "sex": {"type": "string", "enum": ["M", "F"]},
                "baseline_fvc_ml": {"type": "number", "description": "Baseline FVC in mL (400-7000)"},
                "fvc_pp": {"type": "number", "description": "Baseline FVC % predicted"},
                "smoking_status": {
                    "type": "string",
                    "enum": ["Ex-smoker", "Never smoked", "Currently smokes"],
                },
                "horizon_weeks": {"type": "number", "description": "Forecast horizon in weeks (default 52)"},
            },
            "required": ["age", "sex", "fvc_pp"],
        },
    },
    {
        "name": "generate_report",
        "description": (
            "Compile a structured de-identified clinical decision support summary assembling all "
            "previously executed tool results, exact thresholds, human-review flag, limitations, "
            "and mandatory disclaimer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "include_exploratory_progression": {"type": "boolean"},
            },
            "required": [],
        },
    },
]


def _build_clinical_df(clin: Dict[str, Any]) -> tuple[pd.DataFrame, Dict[str, Any]]:
    raw_df = pd.DataFrame(
        [
            {
                "sex": float(np.asarray(encode_sex_column(pd.Series([clin["sex"]])))[0]),
                "Height": float(np.asarray(validate_height_meters(pd.Series([clin["height"]])))[0]),
                "Weight": float(clin["weight"]),
                "age": float(clin["age"]),
                "fev1fvc_abs": float(
                    np.asarray(validate_fev1fvc_ratio(pd.Series([clin["fev1fvc_abs"]])))[0]
                ),
                "fev1_pp": float(clin["fev1_pp"]),
                "fvc_pp": float(clin["fvc_pp"]),
                "tlc_pp": float(clin["tlc_pp"]),
                "tlco_pp": float(clin["tlco_pp"]),
            }
        ]
    )
    raw_display = {
        "sex": clin["sex"],
        "Height": float(raw_df["Height"].iloc[0]),
        "Weight": float(clin["weight"]),
        "age": float(clin["age"]),
        "fev1fvc_abs": float(raw_df["fev1fvc_abs"].iloc[0]),
        "fev1_pp": float(clin["fev1_pp"]),
        "fvc_pp": float(clin["fvc_pp"]),
        "tlc_pp": float(clin["tlc_pp"]),
        "tlco_pp": float(clin["tlco_pp"]),
    }
    return raw_df, raw_display


class PulmoSightToolExecutor:
    """
    Stateful per-assessment executor wrapping existing PulmoSight ML pipelines.
    Holds de-identified patient parameters and optional uploaded CT bytes locally
    so raw DICOM binaries never leave the backend or enter the LLM context.
    """

    def __init__(
        self,
        clinical_bundle: Dict[str, Any],
        progression_bundle: Dict[str, Any],
        artifacts_json: Dict[str, Any],
        deidentified_inputs: Dict[str, Any],
        ct_bytes: Optional[bytes] = None,
        ct_filename: Optional[str] = None,
    ) -> None:
        self.clinical_bundle = clinical_bundle
        self.progression_bundle = progression_bundle
        self.artifacts_json = artifacts_json
        self.deidentified_inputs = dict(deidentified_inputs)
        self.ct_bytes = ct_bytes
        self.ct_filename = ct_filename or ""
        self.extracted_ct_full: Optional[Dict[str, float]] = None
        self.tool_results: Dict[str, Any] = {}

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        handlers = {
            "predict_ipf": self.tool_predict_ipf,
            "explain_shap": self.tool_explain_shap,
            "extract_ct_features": self.tool_extract_ct_features,
            "predict_progression": self.tool_predict_progression,
            "generate_report": self.tool_generate_report,
        }
        if tool_name not in handlers:
            return {"error": f"Unknown tool '{tool_name}'. Allowed tools: {sorted(handlers.keys())}"}
        # Merge sanitized session defaults with sanitized tool args (never allow PHI)
        merged = dict(self.deidentified_inputs)
        if tool_input:
            safe_args = sanitize_deidentified_inputs(tool_input)
            merged.update(safe_args)
        result = handlers[tool_name](merged)
        self.tool_results[tool_name] = result
        return result

    def tool_predict_ipf(self, clin: Dict[str, Any]) -> Dict[str, Any]:
        bundle = self.clinical_bundle
        raw_df, _ = _build_clinical_df(clin)
        feature_names = bundle["feature_names"]
        prob = float(bundle["calibrated_classifier"].predict_proba(raw_df[feature_names])[0, 1])
        prob = round(float(np.clip(prob, 0.001, 0.999)), 4)

        screen_t = round(float(bundle["optimal_threshold"]), 4)
        confirm_t = round(float(bundle.get("confirmatory_threshold", screen_t)), 4)
        screen_label = "IPF" if prob >= screen_t else "Non-IPF"
        confirm_label = "IPF" if prob >= confirm_t else "Non-IPF"
        modes_disagree = screen_label != confirm_label

        active_mode = clin.get("operating_mode", "screening")
        active_t = confirm_t if active_mode == "confirmatory" else screen_t
        active_label = confirm_label if active_mode == "confirmatory" else screen_label
        risk_level = "high" if prob >= confirm_t else "medium" if prob >= screen_t else "low"

        return {
            "ipf_probability": prob,
            "ipf_probability_pct": round(prob * 100.0, 1),
            "risk_level": risk_level,
            "active_operating_mode": active_mode,
            "active_threshold": active_t,
            "active_predicted_label": active_label,
            "screening_threshold": screen_t,
            "screening_predicted_label": screen_label,
            "confirmatory_threshold": confirm_t,
            "confirmatory_predicted_label": confirm_label,
            "modes_disagree": modes_disagree,
            "model_used": f"{bundle['best_model_name']} (Calibrated, Zenodo ILD Cohort)",
            "model_version": str(bundle.get("model_version", MODEL_VERSION)),
            "sensitivity_disclosure": str(bundle.get("sensitivity_disclosure", "")),
            "disclaimer": MANDATORY_DISCLAIMER,
        }

    def tool_explain_shap(self, clin: Dict[str, Any]) -> Dict[str, Any]:
        bundle = self.clinical_bundle
        raw_df, raw_display = _build_clinical_df(clin)
        top_5 = compute_patient_shap_top_features(
            artifact_bundle=bundle,
            patient_df=raw_df,
            raw_input_dict=raw_display,
            top_k=5,
        )
        plain_summary = build_shap_plain_language_summary(top_5, raw_display)
        return {
            "top_5_shap_features": top_5,
            "shap_plain_summary": plain_summary,
            "scale": "model log-odds scale",
            "caveat": SHAP_MANDATORY_CAVEAT,
        }

    def tool_extract_ct_features(self, _clin: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ct_bytes:
            return {
                "ct_available": False,
                "message": "No CT series (.zip) or DICOM scan was uploaded; HRCT and fusion progression scores require a CT series.",
            }
        try:
            if self.ct_filename.lower().endswith(".zip"):
                feats = extract_features_from_dicom_zip_bytes(self.ct_bytes, self.ct_filename)
            else:
                feats = extract_features_from_upload_bytes(self.ct_bytes, self.ct_filename)
        except RuntimeError as exc:
            return {
                "ct_available": False,
                "error_code": 503,
                "error": str(exc),
            }
        except ValueError as exc:
            return {
                "ct_available": False,
                "error_code": 400,
                "error": str(exc),
            }

        self.extracted_ct_full = feats
        return {
            "ct_available": True,
            "hu_mean": feats["hu_mean"],
            "hu_std": feats["hu_std"],
            "haa_pct": feats["haa_pct"],
            "laa_pct": feats["laa_pct"],
            "lung_volume_ml": feats["lung_volume_ml"],
            "num_slices_processed": feats["num_slices_processed"],
            "resnet18_embedding_dim": 512,
        }

    def tool_predict_progression(self, clin: Dict[str, Any]) -> Dict[str, Any]:
        serving_models = self.progression_bundle.get("multimodal_serving_models")
        fusion_comp = (
            self.progression_bundle.get("fusion_comparison")
            or self.artifacts_json.get("osic_progression_study", {}).get("fusion_comparison", {})
        )
        if not serving_models:
            return {
                "available": False,
                "exploratory_banner": EXPLORATORY_PROGRESSION_NOTE,
                "message": "Multimodal progression serving models are unavailable; re-run training.",
            }

        sex_male = 1.0 if str(clin.get("sex", "M")).upper().startswith("M") else 0.0
        smoke_str = str(clin.get("smoking_status", "Ex-smoker")).lower()
        smoke_code = 2.0 if "current" in smoke_str else 1.0 if "ex" in smoke_str else 0.0

        # Estimate baseline FVC in mL if only FVC % predicted was supplied (clearly marked)
        fvc_pct = float(clin.get("fvc_pp", 75.0))
        if "baseline_fvc_ml" in clin and clin["baseline_fvc_ml"] is not None:
            base_fvc_ml = float(clin["baseline_fvc_ml"])
            fvc_ml_estimated = False
        else:
            ref_fvc = 4200.0 if sex_male >= 0.5 else 3100.0
            base_fvc_ml = round(float(np.clip(ref_fvc * (fvc_pct / 100.0), 800.0, 6000.0)), 1)
            fvc_ml_estimated = True

        horizon_weeks = float(clin.get("horizon_weeks", 52.0))
        res = predict_multimodal_progression(
            serving_models=serving_models,
            fusion_comparison=fusion_comp,
            clinical_inputs={
                "Baseline_FVC": base_fvc_ml,
                "Baseline_Percent": fvc_pct,
                "Age": float(clin.get("age", 65.0)),
                "Sex_Male": sex_male,
                "Smoking_Code": smoke_code,
            },
            ct_features=self.extracted_ct_full,
            horizon_weeks=horizon_weeks,
        )
        res["exploratory_banner"] = EXPLORATORY_PROGRESSION_NOTE
        res["baseline_fvc_ml_used"] = base_fvc_ml
        res["baseline_fvc_ml_estimated_from_pct"] = fvc_ml_estimated
        res["note"] = (
            "Progression scores are exploratory (OSIC cohort, n=175 CT patients, no IPF labels) "
            "and predict FVC decline, NOT an IPF diagnosis."
        )
        return res

    def tool_generate_report(self, _clin: Dict[str, Any]) -> Dict[str, Any]:
        ipf_res = self.tool_results.get("predict_ipf")
        shap_res = self.tool_results.get("explain_shap")
        ct_res = self.tool_results.get("extract_ct_features")
        prog_res = self.tool_results.get("predict_progression")

        disagree = bool(ipf_res and ipf_res.get("modes_disagree", False))
        limitations = [
            EXPLORATORY_PROGRESSION_NOTE,
            "Validated IPF classifier was trained on a single retrospective German ILD center (Zenodo, n=6,808; external multi-center validation pending).",
            "Clinical overlap between IPF and fibrotic i-NSIP, u-ILD, and CTD-ILD is expected; screening and confirmatory modes miss a non-zero fraction of IPF cases.",
            SHAP_MANDATORY_CAVEAT,
        ]
        if ipf_res and ipf_res.get("sensitivity_disclosure"):
            limitations.insert(1, ipf_res["sensitivity_disclosure"])

        return {
            "report_type": "De-Identified PulmoSight Clinical Decision Support Summary",
            "deidentified": True,
            "ipf_assessment": ipf_res,
            "shap_explanation": shap_res,
            "ct_extraction": ct_res,
            "progression_assessment": prog_res,
            "human_review_required": disagree,
            "human_review_reason": (
                "Screening mode and Confirmatory mode disagree on predicted label; multidisciplinary clinical review is required."
                if disagree
                else None
            ),
            "limitations": limitations,
            "disclaimer": MANDATORY_DISCLAIMER,
        }
