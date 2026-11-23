"""
Pydantic Request and Response Schemas (`backend/api/schemas.py`).

Enforces explicit physical units (Height in meters `1.0–2.5 m`, `fev1fvc_abs` as a
decimal ratio `0.20–1.15`, percent predicted `10–250%`) without heuristic unit guessing.
Supports `"screening"` (80% OOF sensitivity) and `"confirmatory"` (≥92% OOF specificity)
operating modes on `/predict-ipf`.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.src.config import EXPLORATORY_PROGRESSION_NOTE, MANDATORY_DISCLAIMER, MODEL_VERSION


class PredictIPFRequest(BaseModel):
    """
    Input payload for `POST /predict-ipf`.
    All units are explicit and match the real Zenodo ILD dataset:
    - `sex`: 'M' or 'F' (or 'Male'/'Female')
    - `age`: Baseline age in years (18.0 to 110.0)
    - `height`: Height strictly in METERS (1.0 to 2.5 m, e.g., 1.72)
    - `weight`: Weight strictly in KILOGRAMS (25.0 to 260.0 kg, e.g., 78.5)
    - `fev1fvc_abs`: Absolute FEV1/FVC strictly as a DECIMAL RATIO (0.20 to 1.15, e.g., 0.82)
    - `fev1_pp`, `fvc_pp`, `tlc_pp`, `tlco_pp`: Percent predicted values (10.0 to 250.0 %)
    - `operating_mode`: Optional operating point ('screening' for 80% OOF sensitivity, or
      'confirmatory' for >=92% OOF specificity). Defaults to 'screening'.
    """

    sex: str = Field(..., description="Biological sex ('M' or 'F')")
    age: float = Field(..., ge=18.0, le=110.0, description="Patient age in years (18–110)")
    height: float = Field(
        ...,
        ge=1.0,
        le=2.5,
        description="Patient height strictly in meters (1.00–2.50 m, e.g., 1.72)",
    )
    weight: float = Field(
        ..., ge=25.0, le=260.0, description="Patient weight in kilograms (25–260 kg)"
    )
    fev1fvc_abs: float = Field(
        ...,
        ge=0.20,
        le=1.15,
        description="Absolute FEV1/FVC strictly as a decimal ratio (0.20–1.15, e.g., 0.82)",
    )
    fev1_pp: float = Field(
        ..., ge=10.0, le=250.0, description="FEV1 percent predicted (10–250%)"
    )
    fvc_pp: float = Field(
        ..., ge=10.0, le=250.0, description="FVC percent predicted (10–250%)"
    )
    tlc_pp: float = Field(
        ..., ge=10.0, le=250.0, description="TLC percent predicted (10–250%)"
    )
    tlco_pp: float = Field(
        ..., ge=10.0, le=250.0, description="TLCO percent predicted (10–250%)"
    )
    operating_mode: Literal["screening", "confirmatory"] = Field(
        default="screening",
        description="Operating threshold mode: 'screening' (80% OOF sensitivity) or 'confirmatory' (≥92% OOF specificity)",
    )

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        cleaned = str(v).strip().upper()
        if cleaned in ("M", "MALE"):
            return "M"
        if cleaned in ("F", "FEMALE"):
            return "F"
        raise ValueError("sex must be 'M', 'F', 'Male', or 'Female'")


class ShapContributionItem(BaseModel):
    feature: str
    display_name: str
    value: str
    shap_value: float
    abs_shap: float
    direction: str


class PredictIPFResponse(BaseModel):
    ipf_probability: float
    predicted_label: Literal["IPF", "Non-IPF"]
    risk_level: Literal["low", "medium", "high"]
    operating_mode: Literal["screening", "confirmatory"]
    threshold: float
    screening_threshold: float
    confirmatory_threshold: float
    operating_modes_summary: Dict[str, Any]
    sensitivity_disclosure: str
    top_5_shap_features: List[ShapContributionItem]
    shap_plain_summary: str
    interpretation: Dict[str, Any]
    interpretation_text: str
    interpretation_source: Literal["deterministic", "llm_rephrased"] = "deterministic"
    model_used: str
    model_version: str = Field(default=MODEL_VERSION)
    data_source: Literal["real", "synthetic"]
    disclaimer: str = Field(default=MANDATORY_DISCLAIMER)


class PredictProgressionResponse(BaseModel):
    exploratory_banner: str = Field(default=EXPLORATORY_PROGRESSION_NOTE)
    imaging_model_available: bool
    usable_as_standalone_predictor: bool = Field(default=False)
    predicted_progression: str
    severity: Literal["mild", "moderate", "severe"]
    predicted_fvc_slope_ml_per_week: float
    estimated_annual_fvc_change_ml: float
    rapid_decliner_probability: float
    imaging_used: bool
    extracted_ct_features: Optional[Dict[str, float]] = None
    model_used: str
    model_version: str = Field(default=MODEL_VERSION)
    data_source: Literal["real", "synthetic"]
    study_note: str
    disclaimer: str = Field(default=MANDATORY_DISCLAIMER)


class ProgressionScoreDetail(BaseModel):
    method: str
    method_display_name: str
    progression_risk_score: float
    risk_tier: str
    predicted_fvc_at_horizon_ml: float
    predicted_fvc_change_ml: float
    uncertainty_sigma_ml: float
    horizon_weeks: float
    exploratory: bool = True
    predicts_target: str
    cv_laplace_ll_95ci: str
    cv_rmse_95ci: str
    cv_mae_95ci: str
    cv_auc_95ci: str
    cv_auc_mean: Optional[float] = None
    cv_auc_lower: Optional[float] = None
    cv_auc_upper: Optional[float] = None
    cv_laplace_ll_mean: Optional[float] = None
    cv_laplace_ll_lower: Optional[float] = None
    cv_laplace_ll_upper: Optional[float] = None


class ModalityUsedFlags(BaseModel):
    clinical: bool
    hrct: bool
    fusion: bool


class PredictProgressionMultimodalResponse(BaseModel):
    exploratory_banner: str = Field(default=EXPLORATORY_PROGRESSION_NOTE)
    exploratory_badge: str = Field(default="Exploratory, not validated")
    imaging_model_available: bool
    usable_as_standalone_predictor: bool = Field(default=False)
    selected_fusion_method: str
    selected_fusion_display_name: str
    statistically_confirmed_method: Optional[str] = None
    fusion_vs_clinical_status: Literal["statistically_confirmed", "nominal_only"]
    selection_rationale: str = ""
    paired_vs_clinical_summary: Optional[Dict[str, Any]] = None
    modality_used: ModalityUsedFlags
    clinical_only: ProgressionScoreDetail
    hrct_only: Optional[ProgressionScoreDetail] = None
    fusion: Optional[ProgressionScoreDetail] = None
    extracted_ct_features: Optional[Dict[str, float]] = None
    cv_ci_summary: Dict[str, Any] = Field(default_factory=dict)
    requires_ct_series_message: Optional[str] = None
    score_relationship_note: str
    model_version: str = Field(default=MODEL_VERSION)
    data_source: Literal["real", "synthetic"]
    disclaimer: str = Field(default=MANDATORY_DISCLAIMER)


class AgentAssessRequest(BaseModel):
    """
    Strictly de-identified numeric/categorical clinical input for `POST /agent/assess`.
    Never accepts patient names, IDs, MRNs, DOBs, or free-text identifiers.
    """

    model_config = ConfigDict(extra="allow")

    sex: str = Field(..., description="Biological sex ('M' or 'F')")
    age: float = Field(..., ge=18.0, le=110.0)
    height: float = Field(..., ge=1.0, le=2.5, description="Height in meters (1.0-2.5)")
    weight: float = Field(..., ge=25.0, le=260.0, description="Weight in kg (25-260)")
    fev1fvc_abs: float = Field(..., ge=0.20, le=1.15, description="Decimal FEV1/FVC ratio")
    fev1_pp: float = Field(..., ge=10.0, le=250.0)
    fvc_pp: float = Field(..., ge=10.0, le=250.0)
    tlc_pp: float = Field(..., ge=10.0, le=250.0)
    tlco_pp: float = Field(..., ge=10.0, le=250.0)
    baseline_fvc_ml: Optional[float] = Field(None, ge=400.0, le=7000.0)
    smoking_status: str = Field("Ex-smoker")
    horizon_weeks: float = Field(52.0, ge=4.0, le=156.0)
    operating_mode: Literal["screening", "confirmatory"] = Field(default="screening")

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        cleaned = str(v).strip().upper()
        if cleaned in ("M", "MALE"):
            return "M"
        if cleaned in ("F", "FEMALE"):
            return "F"
        raise ValueError("sex must be 'M', 'F', 'Male', or 'Female'")


class ToolCallLogEntry(BaseModel):
    step: int
    tool_use_id: str
    tool_name: str
    inputs: Dict[str, Any]
    status: str
    latency_ms: float
    output: Dict[str, Any]


class AgentAssessResponse(BaseModel):
    status: str
    model_used: str
    deidentified_inputs_sent: Dict[str, Any]
    final_summary: str
    human_review_required: bool
    human_review_reasons: List[str]
    modes_disagree: bool
    guardrail_appendix_added: bool
    tool_call_log: List[ToolCallLogEntry]
    tool_outputs: Dict[str, Any]
    disclaimer: str = Field(default=MANDATORY_DISCLAIMER)


class HealthResponse(BaseModel):
    status: str
    clinical_model_loaded: bool
    progression_model_loaded: bool
    imaging_model_available: bool
    agent_enabled: bool = False
    model_version: str
    sklearn_version_trained: str
    sklearn_version_runtime: str
    sklearn_version_match: bool
    data_source: Literal["real", "synthetic"]
    disclaimer: str = Field(default=MANDATORY_DISCLAIMER)
