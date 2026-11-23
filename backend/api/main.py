"""
FastAPI Backend Application (`backend/api/main.py`).

Run from repository root:
    uvicorn backend.api.main:app --reload --port 8000
Or from `backend/`:
    uvicorn api.main:app --reload --port 8000

Key Features & Safeguards:
- Fails fast with a clear error message if trained model artifacts are missing.
- Security Note: `.joblib` files use Python pickle under the hood and must only be loaded
  from trusted local sources (`backend/models/`).
- Compares `scikit-learn` major.minor version at load time (`check_sklearn_major_minor_match`)
  so minor patch differences (e.g., 1.9.0 vs 1.9.1) log a warning without failing health checks.
- Supports both `"screening"` (80% OOF sensitivity) and `"confirmatory"` (≥92% OOF specificity)
  operating modes in `POST /predict-ipf`.
- In `POST /predict-progression`:
  * Always includes `"exploratory_banner": "Progression results are exploratory, separate cohort, no IPF labels"`.
  * Exposes `"imaging_model_available": False` when `< 100` OSIC patients have local DICOMs,
    clearly stating that only the clinical-only progression model is active.
  * If a CT scan is uploaded when `torch`/`torchvision` is not installed, returns a clear HTTP 503 error.
"""

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import asynccontextmanager
import json
import os
import time
from typing import Any, Deque, Dict, Optional
import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.agent import (
    AgentDisabledError,
    AgentOrchestrator,
    is_agent_enabled,
    sanitize_deidentified_inputs,
)
from backend.api.schemas import (
    AgentAssessRequest,
    AgentAssessResponse,
    HealthResponse,
    PredictIPFRequest,
    PredictIPFResponse,
    PredictProgressionMultimodalResponse,
    PredictProgressionResponse,
)
from backend.src.config import (
    ARTIFACTS_JSON_PATH,
    CLINICAL_MODEL_PATH,
    EXPLORATORY_PROGRESSION_NOTE,
    MANDATORY_DISCLAIMER,
    MODEL_VERSION,
    PROGRESSION_MODEL_PATH,
    check_sklearn_major_minor_match,
    setup_logger,
)
from backend.src.explain import (
    build_shap_plain_language_summary,
    compute_patient_shap_top_features,
)
from backend.src.interpretation import (
    build_patient_interpretation,
    load_interpretation_rules,
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

logger = setup_logger("api")

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
MAX_CT_UPLOAD_MB = float(os.getenv("MAX_CT_UPLOAD_MB", "25.0"))
MAX_CT_UPLOAD_BYTES = int(MAX_CT_UPLOAD_MB * 1024 * 1024)
MAX_CT_ZIP_UPLOAD_MB = float(os.getenv("MAX_CT_ZIP_UPLOAD_MB", "50.0"))
MAX_CT_ZIP_UPLOAD_BYTES = int(MAX_CT_ZIP_UPLOAD_MB * 1024 * 1024)
DICOM_PARSE_TIMEOUT_SECONDS = float(os.getenv("DICOM_PARSE_TIMEOUT_SECONDS", "12.0"))
DICOM_ZIP_PARSE_TIMEOUT_SECONDS = float(os.getenv("DICOM_ZIP_PARSE_TIMEOUT_SECONDS", "60.0"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))

ALLOWED_CT_EXTENSIONS = (".dcm", ".dicom", ".png", ".jpg", ".jpeg")
ALLOWED_CT_ZIP_MIMES = {
    "application/zip",
    "application/x-zip-compressed",
    "multipart/x-zip",
    "application/octet-stream",
    "",
}
ALLOWED_CT_MIMES = {
    "application/dicom",
    "application/octet-stream",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "",
}

STATE: Dict[str, Any] = {
    "clinical_bundle": None,
    "progression_bundle": None,
    "artifacts_json": None,
    "agent_client_override": None,
}

RATE_LIMIT_BUCKETS: Dict[str, Deque[float]] = defaultdict(deque)


def reset_rate_limiter() -> None:
    """Clear in-memory rate limit buckets (used by unit tests)."""
    RATE_LIMIT_BUCKETS.clear()


def load_trained_models() -> None:
    """
    Load pre-trained `.joblib` pipelines and `model_artifacts.json` from trusted local disk.
    NEVER auto-trains on startup. Compares `scikit-learn` major.minor version compatibility.
    """
    missing_paths = [
        str(p)
        for p in (CLINICAL_MODEL_PATH, PROGRESSION_MODEL_PATH, ARTIFACTS_JSON_PATH)
        if not p.exists()
    ]
    if missing_paths:
        msg = (
            f"Trained model artifacts are missing: {missing_paths}. "
            "Auto-training on API startup is disabled. Please train the models first by running: "
            "`python -m backend.src.train` (from project root) or `cd backend && python -m src.train`."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    clinical_bundle = joblib.load(CLINICAL_MODEL_PATH)
    progression_bundle = joblib.load(PROGRESSION_MODEL_PATH)
    with open(ARTIFACTS_JSON_PATH, "r", encoding="utf-8") as f:
        artifacts_json = json.load(f)

    trained_sklearn = str(clinical_bundle.get("sklearn_version", "unknown"))
    runtime_sklearn = sklearn.__version__
    if trained_sklearn != runtime_sklearn:
        logger.warning(
            f"scikit-learn patch difference detected at model load time: "
            f"trained={trained_sklearn}, runtime={runtime_sklearn} "
            f"(major.minor match={check_sklearn_major_minor_match(trained_sklearn, runtime_sklearn)})."
        )
    else:
        logger.info(f"Verified exact scikit-learn version match: {runtime_sklearn}")

    STATE["clinical_bundle"] = clinical_bundle
    STATE["progression_bundle"] = progression_bundle
    STATE["artifacts_json"] = artifacts_json
    logger.info(
        f"Loaded trained clinical ({clinical_bundle['best_model_name']}, "
        f"screening_threshold={clinical_bundle['optimal_threshold']}, "
        f"confirmatory_threshold={clinical_bundle.get('confirmatory_threshold')}, "
        f"data_source={clinical_bundle['data_source']}) and progression models into memory."
    )


def ensure_models_loaded() -> None:
    """Ensure trained models are in memory or raise HTTP 503 with training instructions."""
    if (
        STATE["clinical_bundle"] is None
        or STATE["progression_bundle"] is None
        or STATE["artifacts_json"] is None
    ):
        try:
            load_trained_models()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_trained_models()
    yield


app = FastAPI(
    title="IPF Diagnostic & Progression Support System API",
    description=(
        "Two-part Clinical Decision Support API trained on real data: "
        "(1) Zenodo ILD Clinical IPF vs. Non-IPF Classifier with SHAP explainability & dual operating modes, and "
        "(2) Exploratory OSIC Pulmonary Fibrosis Longitudinal Progression Study."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)


@app.middleware("http")
async def rate_limit_and_audit_log_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "local"
    now = time.monotonic()
    bucket = RATE_LIMIT_BUCKETS[client_ip]
    while bucket and (now - bucket[0]) > 60.0:
        bucket.popleft()

    limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", str(RATE_LIMIT_PER_MINUTE)))
    if len(bucket) >= limit:
        logger.warning(f"Rate limit exceeded for client={client_ip} on path={request.url.path}")
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Rate limit exceeded ({limit} requests/minute). Please retry shortly.",
                "disclaimer": MANDATORY_DISCLAIMER,
            },
        )
    bucket.append(now)

    start_ts = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
    logger.info(
        f"AUDIT request method={request.method} path={request.url.path} "
        f"status={response.status_code} latency_ms={elapsed_ms:.1f} (PHI_LOGGED=False)"
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health check verifying model readiness, scikit-learn major.minor version compatibility,
    `imaging_model_available`, and `data_source`.
    """
    ensure_models_loaded()
    bundle = STATE["clinical_bundle"]
    prog_bundle = STATE["progression_bundle"]
    trained_ver = str(bundle.get("sklearn_version", "unknown"))
    runtime_ver = sklearn.__version__
    return HealthResponse(
        status="ok",
        clinical_model_loaded=STATE["clinical_bundle"] is not None,
        progression_model_loaded=STATE["progression_bundle"] is not None,
        imaging_model_available=bool(prog_bundle.get("imaging_model_available", False)),
        agent_enabled=bool(is_agent_enabled() or STATE.get("agent_client_override") is not None),
        model_version=str(bundle.get("model_version", MODEL_VERSION)),
        sklearn_version_trained=trained_ver,
        sklearn_version_runtime=runtime_ver,
        sklearn_version_match=check_sklearn_major_minor_match(trained_ver, runtime_ver),
        data_source=bundle.get("data_source", "real"),
        disclaimer=MANDATORY_DISCLAIMER,
    )


@app.post("/predict-ipf", response_model=PredictIPFResponse)
@app.post("/predict", response_model=PredictIPFResponse)
def predict_ipf(payload: PredictIPFRequest) -> PredictIPFResponse:
    """
    Predict IPF vs. Non-IPF from patient demographics and pulmonary function tests (PFTs).
    Supports two OOF-tuned operating modes:
    - `operating_mode="screening"` (default): High-sensitivity threshold (80% OOF training sensitivity).
    - `operating_mode="confirmatory"`: High-specificity threshold (≥92% OOF training specificity).
    """
    ensure_models_loaded()
    bundle = STATE["clinical_bundle"]
    calibrated_clf = bundle["calibrated_classifier"]
    feature_names = bundle["feature_names"]
    screening_thresh = float(bundle["optimal_threshold"])
    confirmatory_thresh = float(bundle.get("confirmatory_threshold", screening_thresh))
    active_mode = payload.operating_mode
    active_threshold = (
        confirmatory_thresh if active_mode == "confirmatory" else screening_thresh
    )
    data_source = bundle.get("data_source", "real")

    raw_df = pd.DataFrame(
        [
            {
                "sex": float(np.asarray(encode_sex_column(pd.Series([payload.sex])))[0]),
                "Height": float(
                    np.asarray(validate_height_meters(pd.Series([payload.height])))[0]
                ),
                "Weight": float(payload.weight),
                "age": float(payload.age),
                "fev1fvc_abs": float(
                    np.asarray(validate_fev1fvc_ratio(pd.Series([payload.fev1fvc_abs])))[0]
                ),
                "fev1_pp": float(payload.fev1_pp),
                "fvc_pp": float(payload.fvc_pp),
                "tlc_pp": float(payload.tlc_pp),
                "tlco_pp": float(payload.tlco_pp),
            }
        ]
    )

    prob_ipf = float(calibrated_clf.predict_proba(raw_df[feature_names])[0, 1])
    prob_ipf = round(float(np.clip(prob_ipf, 0.001, 0.999)), 4)

    predicted_label = "IPF" if prob_ipf >= active_threshold else "Non-IPF"

    if prob_ipf >= confirmatory_thresh:
        risk_level = "high"
    elif prob_ipf >= screening_thresh:
        risk_level = "medium"
    else:
        risk_level = "low"

    raw_input_display = {
        "sex": payload.sex,
        "Height": raw_df["Height"].iloc[0],
        "Weight": payload.weight,
        "age": payload.age,
        "fev1fvc_abs": raw_df["fev1fvc_abs"].iloc[0],
        "fev1_pp": payload.fev1_pp,
        "fvc_pp": payload.fvc_pp,
        "tlc_pp": payload.tlc_pp,
        "tlco_pp": payload.tlco_pp,
    }

    top_5_shap = compute_patient_shap_top_features(
        artifact_bundle=bundle,
        patient_df=raw_df,
        raw_input_dict=raw_input_display,
        top_k=5,
    )
    shap_plain_summary = build_shap_plain_language_summary(
        top_shap_features=top_5_shap,
        raw_input_dict=raw_input_display,
    )
    interp_out = build_patient_interpretation(
        raw_inputs=raw_input_display,
        prob_ipf=prob_ipf,
        screening_threshold=screening_thresh,
        confirmatory_threshold=confirmatory_thresh,
        active_mode=active_mode,
        top_shap_features=top_5_shap,
        artifacts=STATE.get("artifacts_json") or {},
        llm_client=STATE.get("interpretation_llm_client_override"),
    )

    operating_modes_summary = {
        "screening": {
            "threshold": round(screening_thresh, 4),
            "predicted_label": "IPF" if prob_ipf >= screening_thresh else "Non-IPF",
            "rule": "80% Target Sensitivity on OOF Calibrated Training Predictions",
        },
        "confirmatory": {
            "threshold": round(confirmatory_thresh, 4),
            "predicted_label": "IPF" if prob_ipf >= confirmatory_thresh else "Non-IPF",
            "rule": "≥92% Target Specificity on OOF Calibrated Training Predictions",
        },
    }

    return PredictIPFResponse(
        ipf_probability=prob_ipf,
        predicted_label=predicted_label,
        risk_level=risk_level,
        operating_mode=active_mode,
        threshold=round(active_threshold, 4),
        screening_threshold=round(screening_thresh, 4),
        confirmatory_threshold=round(confirmatory_thresh, 4),
        operating_modes_summary=operating_modes_summary,
        sensitivity_disclosure=str(bundle.get("sensitivity_disclosure", "")),
        top_5_shap_features=top_5_shap,
        shap_plain_summary=shap_plain_summary,
        interpretation=interp_out["interpretation"],
        interpretation_text=interp_out["interpretation_text"],
        interpretation_source=interp_out["interpretation_source"],
        model_used=f"{bundle['best_model_name']} (Calibrated, {active_mode.capitalize()} Mode)",
        model_version=str(bundle.get("model_version", MODEL_VERSION)),
        data_source=data_source,
        disclaimer=MANDATORY_DISCLAIMER,
    )


@app.post("/predict-progression", response_model=PredictProgressionResponse)
async def predict_progression(
    age: float = Form(..., ge=18.0, le=110.0, description="Patient age in years (18–110)"),
    sex: str = Form(..., description="Biological sex ('M'/'F' or 'Male'/'Female')"),
    fvc: float = Form(..., ge=400.0, le=7000.0, description="Baseline FVC in mL (400–7000)"),
    percent: float = Form(
        ..., ge=10.0, le=250.0, description="Baseline FVC % predicted (10–250%)"
    ),
    smoking_status: str = Form(
        "Ex-smoker",
        description="Smoking history ('Ex-smoker', 'Never smoked', 'Currently smokes')",
    ),
    ct_scan: Optional[UploadFile] = File(
        None, description="Optional thoracic CT DICOM (.dcm) or axial image (.png/.jpg)"
    ),
) -> PredictProgressionResponse:
    """
    Part 2 Endpoint (Exploratory OSIC Fibrosis Progression Study):
    Always includes `"exploratory_banner": "Progression results are exploratory, separate cohort, no IPF labels"`
    and `"imaging_model_available": False` when `< 100` OSIC patients have local DICOM scans.
    """
    ensure_models_loaded()
    prog_bundle = STATE["progression_bundle"]
    imaging_model_available = bool(prog_bundle.get("imaging_model_available", False))

    sex_clean = str(sex).strip().upper()
    if sex_clean not in ("M", "MALE", "F", "FEMALE"):
        raise HTTPException(status_code=422, detail="sex must be 'M', 'F', 'Male', or 'Female'")
    sex_male = 1.0 if sex_clean in ("M", "MALE") else 0.0

    smoke_lower = str(smoking_status).strip().lower()
    if not any(k in smoke_lower for k in ("ex", "never", "current")):
        raise HTTPException(
            status_code=422,
            detail="smoking_status must be 'Ex-smoker', 'Never smoked', or 'Currently smokes'",
        )
    smoking_code = (
        2.0 if "current" in smoke_lower else 1.0 if "ex" in smoke_lower else 0.0
    )

    clinical_row = {
        "Baseline_FVC": float(fvc),
        "Baseline_Percent": float(percent),
        "Age": float(age),
        "Sex_Male": sex_male,
        "Smoking_Code": smoking_code,
    }

    ct_uploaded = False
    extracted_ct_summary: Optional[Dict[str, float]] = None
    ct_features: Optional[Dict[str, float]] = None

    if ct_scan is not None and (ct_scan.filename or "").strip() != "":
        filename = (ct_scan.filename or "").strip()
        lower_name = filename.lower()
        # Security policy (Item 9): Never accept model/pickle files via any API upload
        if lower_name.endswith((".joblib", ".pkl", ".pickle", ".pt", ".pth", ".onnx", ".h5")):
            raise HTTPException(
                status_code=415,
                detail="Security policy: model or serialized pickle/joblib files are never accepted through API uploads.",
            )
        if not lower_name.endswith(ALLOWED_CT_EXTENSIONS):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file extension for '{filename}'. Allowed: {ALLOWED_CT_EXTENSIONS}",
            )

        content_type = (ct_scan.content_type or "").lower()
        if content_type not in ALLOWED_CT_MIMES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported MIME type '{content_type}'. Allowed: {sorted(list(ALLOWED_CT_MIMES))}",
            )

        file_bytes = await ct_scan.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded CT scan file is empty.")
        if len(file_bytes) > MAX_CT_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded CT file exceeds maximum allowed size of {MAX_CT_UPLOAD_MB:.1f} MB.",
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(extract_features_from_upload_bytes, file_bytes, filename)
            try:
                ct_features = future.result(timeout=DICOM_PARSE_TIMEOUT_SECONDS)
            except FuturesTimeoutError as exc:
                raise HTTPException(
                    status_code=408,
                    detail=f"CT scan parsing timed out after {DICOM_PARSE_TIMEOUT_SECONDS} seconds.",
                ) from exc
            except RuntimeError as exc:
                # Raised when torch / torchvision or pretrained ResNet-18 weights are missing
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        ct_uploaded = True
        extracted_ct_summary = {
            "hu_mean": ct_features["hu_mean"],
            "hu_std": ct_features["hu_std"],
            "haa_pct": ct_features["haa_pct"],
            "laa_pct": ct_features["laa_pct"],
            "lung_volume_ml": ct_features["lung_volume_ml"],
            "num_slices_processed": ct_features["num_slices_processed"],
        }

    # Only use the Imaging + Clinical model if BOTH a CT was uploaded AND >=100 OSIC patients had DICOMs at training time
    if (
        ct_uploaded
        and imaging_model_available
        and ct_features is not None
        and prog_bundle.get("imaging_regressor") is not None
    ):
        mm_cols = prog_bundle["multimodal_feature_cols"]
        X_in = pd.DataFrame([{**clinical_row, **ct_features}])[mm_cols].astype(float)
        pred_slope = float(prog_bundle["imaging_regressor"].predict(X_in)[0])
        rapid_prob = float(prog_bundle["imaging_classifier"].predict_proba(X_in)[0, 1])
        imaging_used_in_model = True
        model_label = f"OSIC {prog_bundle['best_model_name']} (Imaging + Clinical, Exploratory)"
    else:
        clin_cols = prog_bundle["clinical_feature_cols"]
        X_in = pd.DataFrame([clinical_row])[clin_cols].astype(float)
        pred_slope = float(prog_bundle["clinical_regressor"].predict(X_in)[0])
        rapid_prob = float(prog_bundle["clinical_classifier"].predict_proba(X_in)[0, 1])
        # Even if a CT scan was uploaded for HU feature inspection, if <100 patients had training DICOMs,
        # only the clinical-only progression model is active!
        imaging_used_in_model = bool(ct_uploaded and imaging_model_available)
        model_label = (
            f"OSIC {prog_bundle['best_model_name']} "
            + (
                "(Clinical-Only Fallback — No CT Uploaded; Exploratory)"
                if imaging_model_available
                else "(Clinical-Only Active — <100 training CT scans; Exploratory)"
            )
        )

    annual_change_ml = round(pred_slope * 52.0, 1)

    if pred_slope <= -5.5 or rapid_prob >= 0.65:
        progression_label = "Rapid Decliner (Exploratory)"
        severity = "severe"
    elif pred_slope <= -3.5 or rapid_prob >= 0.40:
        progression_label = "Moderate Decliner (Exploratory)"
        severity = "moderate"
    else:
        progression_label = "Slow / Stable Decliner (Exploratory)"
        severity = "mild"

    return PredictProgressionResponse(
        exploratory_banner=EXPLORATORY_PROGRESSION_NOTE,
        imaging_model_available=imaging_model_available,
        usable_as_standalone_predictor=bool(
            prog_bundle.get("usable_as_standalone_predictor", False)
        ),
        predicted_progression=progression_label,
        severity=severity,
        predicted_fvc_slope_ml_per_week=round(pred_slope, 3),
        estimated_annual_fvc_change_ml=annual_change_ml,
        rapid_decliner_probability=round(rapid_prob, 4),
        imaging_used=ct_uploaded,
        extracted_ct_features=extracted_ct_summary,
        model_used=model_label,
        model_version=MODEL_VERSION,
        data_source=prog_bundle.get("data_source", "real"),
        study_note=prog_bundle.get("study_description", EXPLORATORY_PROGRESSION_NOTE),
        disclaimer=MANDATORY_DISCLAIMER,
    )


@app.post(
    "/predict-progression-multimodal",
    response_model=PredictProgressionMultimodalResponse,
)
async def predict_progression_multimodal(
    age: float = Form(..., ge=18.0, le=110.0, description="Patient age in years (18–110)"),
    sex: str = Form(..., description="Biological sex ('M'/'F' or 'Male'/'Female')"),
    fvc: float = Form(..., ge=400.0, le=7000.0, description="Baseline FVC in mL (400–7000)"),
    percent: float = Form(
        ..., ge=10.0, le=250.0, description="Baseline FVC % predicted (10–250%)"
    ),
    smoking_status: str = Form(
        "Ex-smoker",
        description="Smoking history ('Ex-smoker', 'Never smoked', 'Currently smokes')",
    ),
    horizon_weeks: float = Form(
        52.0,
        ge=4.0,
        le=156.0,
        description="Forecast horizon in weeks (default 52 weeks)",
    ),
    ct_series_zip: Optional[UploadFile] = File(
        None,
        description="Optional full CT series as a .zip archive of axial DICOM (.dcm) slices (max 50 MB)",
    ),
) -> PredictProgressionMultimodalResponse:
    """
    Requirement C: Multimodal OSIC progression endpoint returning `clinical_only`,
    `hrct_only`, and `fusion` (using `selected_fusion_method`) outputs plus `modality_used` flags.

    - `/predict-ipf` remains strictly torch-free.
    - When `ct_series_zip` is omitted, `hrct_only` and `fusion` are strictly `None` (never dummy values).
    - When `ct_series_zip` is uploaded, requires `torch` + pretrained ResNet-18 weights and returns
      HTTP 503 clearly if they are unavailable.
    """
    ensure_models_loaded()
    prog_bundle = STATE["progression_bundle"]
    art = STATE["artifacts_json"]
    imaging_model_available = bool(prog_bundle.get("imaging_model_available", False))
    serving_models = prog_bundle.get("multimodal_serving_models")
    fusion_comp = (
        prog_bundle.get("fusion_comparison")
        or art.get("osic_progression_study", {}).get("fusion_comparison", {})
    )

    if not serving_models:
        raise HTTPException(
            status_code=503,
            detail=(
                "Multimodal progression serving models are not yet trained in "
                "osic_progression_pipeline.joblib. Re-run `python -m backend.src.train`."
            ),
        )

    sex_clean = str(sex).strip().upper()
    if sex_clean not in ("M", "MALE", "F", "FEMALE"):
        raise HTTPException(
            status_code=422, detail="sex must be 'M', 'F', 'Male', or 'Female'"
        )
    sex_male = 1.0 if sex_clean in ("M", "MALE") else 0.0

    smoke_lower = str(smoking_status).strip().lower()
    if not any(k in smoke_lower for k in ("ex", "never", "current")):
        raise HTTPException(
            status_code=422,
            detail="smoking_status must be 'Ex-smoker', 'Never smoked', or 'Currently smokes'",
        )
    smoking_code = (
        2.0 if "current" in smoke_lower else 1.0 if "ex" in smoke_lower else 0.0
    )

    ct_features: Optional[Dict[str, float]] = None
    extracted_ct_summary: Optional[Dict[str, float]] = None

    if ct_series_zip is not None and (ct_series_zip.filename or "").strip() != "":
        filename = (ct_series_zip.filename or "").strip()
        lower_name = filename.lower()
        if lower_name.endswith((".joblib", ".pkl", ".pickle", ".pt", ".pth", ".onnx", ".h5")):
            raise HTTPException(
                status_code=415,
                detail="Security policy: model or serialized pickle/joblib files are never accepted through API uploads.",
            )
        if not lower_name.endswith(".zip"):
            raise HTTPException(
                status_code=415,
                detail=f"HRCT and Fusion scoring require a full CT series as a .zip archive of DICOM (.dcm) slices; received '{filename}'.",
            )
        content_type = (ct_series_zip.content_type or "").lower()
        if content_type not in ALLOWED_CT_ZIP_MIMES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported MIME type '{content_type}' for CT series archive. Expected application/zip.",
            )

        zip_bytes = await ct_series_zip.read()
        if len(zip_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded CT series .zip archive is empty.")
        if len(zip_bytes) > MAX_CT_ZIP_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded CT series archive exceeds maximum allowed size of {MAX_CT_ZIP_UPLOAD_MB:.1f} MB.",
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(extract_features_from_dicom_zip_bytes, zip_bytes, filename)
            try:
                ct_features = future.result(timeout=DICOM_ZIP_PARSE_TIMEOUT_SECONDS)
            except FuturesTimeoutError as exc:
                raise HTTPException(
                    status_code=408,
                    detail=f"CT series parsing timed out after {DICOM_ZIP_PARSE_TIMEOUT_SECONDS} seconds.",
                ) from exc
            except RuntimeError as exc:
                # Missing torch / torchvision / pretrained ResNet-18 weights -> clear HTTP 503
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except ValueError as exc:
                msg = str(exc)
                status_code = 415 if "Security policy" in msg else 400
                raise HTTPException(status_code=status_code, detail=msg) from exc

        extracted_ct_summary = {
            "hu_mean": ct_features["hu_mean"],
            "hu_std": ct_features["hu_std"],
            "haa_pct": ct_features["haa_pct"],
            "laa_pct": ct_features["laa_pct"],
            "lung_volume_ml": ct_features["lung_volume_ml"],
            "num_slices_processed": ct_features["num_slices_processed"],
        }

    result = predict_multimodal_progression(
        serving_models=serving_models,
        fusion_comparison=fusion_comp,
        clinical_inputs={
            "Baseline_FVC": float(fvc),
            "Baseline_Percent": float(percent),
            "Age": float(age),
            "Sex_Male": sex_male,
            "Smoking_Code": smoking_code,
        },
        ct_features=ct_features,
        horizon_weeks=float(horizon_weeks),
    )

    sel_display = result["selected_fusion_display_name"]
    fusion_status = result["fusion_vs_clinical_status"]
    stat_confirmed_method = (fusion_comp or {}).get("statistically_confirmed_method")
    sel_rationale = str(
        (fusion_comp or {}).get("selection_rationale")
        or (
            "Late fusion (mean) has the best mean FVC Laplace LL (nominal only), while its rapid-decliner AUC "
            "is lower. Stacking is the only method with a statistically confirmed gain over clinical-only "
            "(+0.0104 Laplace LL, p = 0.0006; -1.7 mL MAE, p = 0.0003); across 20 paired comparisons, both "
            "p-values survive Bonferroni correction (α = 0.0025), but the gain is clinically negligible "
            "(under 0.1% of mean FVC). Exploratory."
        )
    )
    rel_note = (
        "Card 1 (Clinical IPF probability, Zenodo cohort) predicts IPF vs. Non-IPF diagnostic "
        "likelihood and is validated on held-out test patients. Cards 2 (HRCT) and 3 (Fusion: "
        f"{sel_display}) come from the separate OSIC cohort and predict two distinct exploratory "
        "targets — future FVC at week 52 (regression, evaluated by Modified Laplace LL / RMSE / MAE) and "
        "binary rapid-decliner risk (≤−200 mL/yr, separate classifier evaluated by AUC) — NOT an IPF "
        f"diagnosis. {sel_rationale}"
    )

    methods_meta = (fusion_comp or {}).get("methods", {})
    paired_meta = (fusion_comp or {}).get("paired_vs_clinical_only", {})
    sel_method_key = result["selected_fusion_method"]
    selected_paired = paired_meta.get(sel_method_key)
    cv_ci_summary = {
        "clinical_only": {
            "laplace_ll_95ci": methods_meta.get("clinical_only", {}).get("laplace_ll_stats", {}).get("formatted", "N/A"),
            "rmse_95ci": methods_meta.get("clinical_only", {}).get("rmse_stats", {}).get("formatted", "N/A"),
            "mae_95ci": methods_meta.get("clinical_only", {}).get("mae_stats", {}).get("formatted", "N/A"),
            "auc_95ci": methods_meta.get("clinical_only", {}).get("auc_stats", {}).get("formatted", "N/A"),
            "auc_mean": methods_meta.get("clinical_only", {}).get("auc_stats", {}).get("mean"),
        },
        "hrct_only": {
            "laplace_ll_95ci": methods_meta.get("hrct_only", {}).get("laplace_ll_stats", {}).get("formatted", "N/A"),
            "rmse_95ci": methods_meta.get("hrct_only", {}).get("rmse_stats", {}).get("formatted", "N/A"),
            "mae_95ci": methods_meta.get("hrct_only", {}).get("mae_stats", {}).get("formatted", "N/A"),
            "auc_95ci": methods_meta.get("hrct_only", {}).get("auc_stats", {}).get("formatted", "N/A"),
            "auc_mean": methods_meta.get("hrct_only", {}).get("auc_stats", {}).get("mean"),
            "paired_laplace_ll": paired_meta.get("hrct_only", {}).get("laplace_ll", {}).get("formatted"),
        },
        "fusion": {
            "method": sel_method_key,
            "method_display_name": sel_display,
            "laplace_ll_95ci": methods_meta.get(sel_method_key, {}).get("laplace_ll_stats", {}).get("formatted", "N/A"),
            "rmse_95ci": methods_meta.get(sel_method_key, {}).get("rmse_stats", {}).get("formatted", "N/A"),
            "mae_95ci": methods_meta.get(sel_method_key, {}).get("mae_stats", {}).get("formatted", "N/A"),
            "auc_95ci": methods_meta.get(sel_method_key, {}).get("auc_stats", {}).get("formatted", "N/A"),
            "auc_mean": methods_meta.get(sel_method_key, {}).get("auc_stats", {}).get("mean"),
            "paired_laplace_ll": (selected_paired or {}).get("laplace_ll", {}).get("formatted"),
            "paired_auc": (selected_paired or {}).get("auc", {}).get("formatted"),
            "status": fusion_status,
        },
    }

    return PredictProgressionMultimodalResponse(
        exploratory_banner=EXPLORATORY_PROGRESSION_NOTE,
        exploratory_badge="Exploratory, not validated",
        imaging_model_available=imaging_model_available,
        usable_as_standalone_predictor=False,
        selected_fusion_method=result["selected_fusion_method"],
        selected_fusion_display_name=sel_display,
        statistically_confirmed_method=stat_confirmed_method,
        fusion_vs_clinical_status=fusion_status,
        selection_rationale=sel_rationale,
        paired_vs_clinical_summary=selected_paired,
        modality_used=result["modality_used"],
        clinical_only=result["clinical_only"],
        hrct_only=result["hrct_only"],
        fusion=result["fusion"],
        extracted_ct_features=extracted_ct_summary,
        cv_ci_summary=cv_ci_summary,
        requires_ct_series_message=(
            None
            if ct_features is not None
            else "Requires CT series (.zip archive of axial DICOM slices)"
        ),
        score_relationship_note=rel_note,
        model_version=MODEL_VERSION,
        data_source=prog_bundle.get("data_source", "real"),
        disclaimer=MANDATORY_DISCLAIMER,
    )


@app.post("/agent/assess", response_model=AgentAssessResponse)
def agent_assess(payload: AgentAssessRequest) -> AgentAssessResponse:
    """
    Requirement E: Prototype LLM tool-calling assessment endpoint (Anthropic Messages API).
    - Disabled (HTTP 503) unless `ANTHROPIC_API_KEY` is configured (or test client override active).
    - Strictly rejects PHI/identifiers and forwards ONLY de-identified numeric clinical values.
    - Returns full `tool_call_log` and `human_review_required` flag whenever Screening and
      Confirmatory modes disagree.
    """
    ensure_models_loaded()
    override_client = STATE.get("agent_client_override")
    if override_client is None and not is_agent_enabled():
        raise HTTPException(
            status_code=503,
            detail=(
                "POST /agent/assess is disabled because ANTHROPIC_API_KEY is not set in the environment. "
                "Set ANTHROPIC_API_KEY (and optionally ANTHROPIC_MODEL) to enable the agentic prototype."
            ),
        )

    raw_payload = payload.model_dump()
    try:
        deidentified = sanitize_deidentified_inputs(raw_payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        orchestrator = AgentOrchestrator(
            clinical_bundle=STATE["clinical_bundle"],
            progression_bundle=STATE["progression_bundle"],
            artifacts_json=STATE["artifacts_json"],
            client=override_client,
        )
        result = orchestrator.run_assessment(patient_payload=deidentified)
    except AgentDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AgentAssessResponse(**result)


@app.get("/model-info")
def get_model_info() -> Dict[str, Any]:
    """
    Return the real evaluation metrics table for all models (Random Forest, SVM, XGBoost, PyTorch MLP)
    evaluated under the unified 80%-OOF-sensitivity rule, plus both Screening & Confirmatory operating
    modes, 95% bootstrap confidence intervals, per-diagnosis-group error analysis, and the exploratory
    OSIC progression study comparison against Dummy Baselines.
    """
    ensure_models_loaded()
    art = STATE["artifacts_json"]
    clinical_models = art["clinical_models"]
    best_name = art["best_clinical_model"]

    metrics_table = []
    for name, m in clinical_models.items():
        metrics_table.append(
            {
                "model": name,
                "is_best": name == best_name,
                "cv_roc_auc_mean": m["cv_roc_auc_mean"],
                "cv_roc_auc_std": m["cv_roc_auc_std"],
                "cv_roc_auc_formatted": m["cv_roc_auc_formatted"],
                "threshold_rule": m.get(
                    "threshold_rule",
                    "80% Target Sensitivity on Out-of-Fold (OOF) Training Predictions",
                ),
                "threshold": m["threshold"],
                "oof_tuned_threshold": m["oof_tuned_threshold"],
                "confirmatory_threshold": m.get("confirmatory_threshold", m["threshold"]),
                "accuracy": m["accuracy"],
                "auc": m["auc"],
                "pr_auc": m["pr_auc"],
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "brier_score": m["brier_score"],
                "bootstrap_ci_95": m.get("bootstrap_ci_95", {}),
                "operating_modes": m.get("operating_modes", {}),
                "best_params": m.get("best_params", {}),
                "confusion_matrix": m["confusion_matrix"],
            }
        )

    fpr_grid = [pt["fpr"] for pt in clinical_models[best_name]["roc_curve"]]
    roc_curve_combined = []
    for idx, fpr_val in enumerate(fpr_grid):
        point = {"fpr": fpr_val, "baseline": fpr_val}
        for name, m in clinical_models.items():
            key = (
                "xgb"
                if "XGB" in name
                else "rf"
                if "Forest" in name
                else "svm"
                if "SVM" in name
                else "mlp"
            )
            point[key] = m["roc_curve"][idx]["tpr"]
        roc_curve_combined.append(point)

    return {
        "model_version": art.get("model_version", MODEL_VERSION),
        "requires_torch_at_serving": art.get("requires_torch_at_serving", False),
        "data_source": art.get("data_source", "real"),
        "zenodo_md5": art.get("zenodo_md5", ""),
        "case_number_uniqueness": art.get("case_number_uniqueness", {}),
        "imaging_model_available": art.get("imaging_model_available", False),
        "best_model": best_name,
        "highest_cv_model": art.get("highest_cv_model", best_name),
        "selection_criterion": art.get(
            "selection_criterion",
            "5-Fold Stratified Cross-Validation ROC-AUC on Training Split Only",
        ),
        "threshold_tuning_method": art.get("threshold_tuning_method", ""),
        "optimal_threshold": art["optimal_threshold"],
        "screening_threshold": art.get("screening_threshold", art["optimal_threshold"]),
        "confirmatory_threshold": art.get("confirmatory_threshold", art["optimal_threshold"]),
        "operating_modes": art.get("operating_modes", {}),
        "sensitivity_disclosure": art.get("sensitivity_disclosure", ""),
        "metrics_table": metrics_table,
        "confusion_matrix": clinical_models[best_name]["confusion_matrix"],
        "confusion_matrices_by_model": {
            name: m["confusion_matrix"] for name, m in clinical_models.items()
        },
        "per_group_error_analysis": art.get("per_group_error_analysis", []),
        "clinical_error_analysis_note": art.get("clinical_error_analysis_note", ""),
        "roc_curve": roc_curve_combined,
        "pr_curves_by_model": {
            name: m["pr_curve"] for name, m in clinical_models.items()
        },
        "calibration_curve": art["calibration_curve"],
        "calibration_reliability_table": art.get("calibration_reliability_table"),
        "cohort_age_stats": art.get("cohort_age_stats"),
        "interpretation_rules": load_interpretation_rules(),
        "feature_set_ablation": art["feature_set_ablation"],
        "imbalance_comparison": art["imbalance_comparison"],
        "global_shap_ranking": art["global_shap_ranking"],
        "osic_progression_study": art["osic_progression_study"],
        "extractor_comparison": art.get("extractor_comparison"),
        "dataset_note": (
            f"{EXPLORATORY_PROGRESSION_NOTE}. Part 1 (Zenodo ILD Registry, n=6,808, 757 IPF = 11.12%) "
            "and Part 2 (OSIC Pulmonary Fibrosis Progression Study, n=176 fibrosis patients, 1,549 visits) "
            "are separate studies on independent cohorts with no shared patients and no IPF labels in OSIC."
        ),
        "disclaimer": MANDATORY_DISCLAIMER,
    }


@app.get("/eda")
def get_eda_insights() -> Dict[str, Any]:
    """
    Return Exploratory Data Analysis (EDA) insights computed directly from the real
    Zenodo ILD dataset (`1_SAR`, `o-ILD`, `IPF`, `CTD`, `HP`, `u-ILD`, `i-NSIP`).
    """
    ensure_models_loaded()
    art = STATE["artifacts_json"]
    eda = art["eda_summary"]
    return {
        "data_source": eda.get("data_source", art.get("data_source", "real")),
        "zenodo_md5": art.get("zenodo_md5", ""),
        "case_number_uniqueness": eda.get("case_number_uniqueness", {}),
        "total_rows": eda["total_rows"],
        "ipf_rows": eda["ipf_rows"],
        "non_ipf_rows": eda["non_ipf_rows"],
        "ipf_prevalence_pct": eda["ipf_prevalence_pct"],
        "median_age_ipf": eda["median_age_ipf"],
        "median_age_non_ipf": eda["median_age_non_ipf"],
        "mean_fvc_ipf": eda["mean_fvc_ipf"],
        "mean_tlco_ipf": eda["mean_tlco_ipf"],
        "median_survival_ipf_months": eda.get("median_survival_ipf_months", 38.0),
        "diagnosis_group_counts": eda["diagnosis_distribution"],
        "age_distribution": eda["age_distribution"],
        "survival_by_diagnosis": eda["survival_by_diagnosis"],
        "disclaimer": MANDATORY_DISCLAIMER,
    }
