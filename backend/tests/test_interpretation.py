"""
Comprehensive Unit & API Tests for the Five-Part Clinical Result Interpretation Engine
(`backend/tests/test_interpretation.py`).

ZERO `torch` dependency and ZERO live external API calls (LLM rephrasing tested via mocks).
Covers:
  - >= 15 distinct patient profiles including the 3 real Zenodo sample patients
  - All ventilatory patterns (restrictive, obstructive, mixed, normal, non-specific, indeterminate)
  - All TLCO gas-transfer severity bands + isolated gas-transfer deficit
  - Edge cases: non-IPF, borderline, conflicting Screening/Confirmatory modes, missing values,
    extreme values, and score above max observed held-out test probability
  - Configurable rules loaded from `backend/config/interpretation_rules.json`
  - Real held-out test reliability bin rates, cohort age medians (68.9 vs. 48.9 yrs), and
    per-group misclassification rates (i-NSIP, u-ILD, CTD) from `model_artifacts.json`
  - Optional env-gated LLM rephrasing + post-generation validation (number conservation,
    no diagnosis/treatment wording, de-identified payload only, safe deterministic fallback)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi.testclient import TestClient
import pytest

import backend.api.main as api_main
from backend.api.main import app, reset_rate_limiter
from backend.src.config import (
    ARTIFACTS_JSON_PATH,
    AUTOMATED_INTERPRETATION_NOTICE,
    INTERPRETATION_RULES_PATH,
)
from backend.src.interpretation import (
    build_patient_interpretation,
    load_interpretation_rules,
    validate_llm_rephrased_interpretation,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    reset_rate_limiter()
    monkeypatch.delenv("ENABLE_LLM_INTERPRETATION_REPHRASE", raising=False)
    api_main.STATE.pop("interpretation_llm_client_override", None)
    yield
    api_main.STATE.pop("interpretation_llm_client_override", None)


@pytest.fixture(scope="module")
def real_artifacts() -> Dict[str, Any]:
    with open(ARTIFACTS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def real_rules() -> Dict[str, Any]:
    return load_interpretation_rules(INTERPRETATION_RULES_PATH, use_cache=False)


# ---------------------------------------------------------------------------
# 1. Reliability Table & Rules Integrity Checks
# ---------------------------------------------------------------------------


def test_reliability_table_and_age_stats_stored_in_artifacts(real_artifacts):
    rel = real_artifacts.get("calibration_reliability_table")
    assert rel is not None
    assert rel["total_patients"] == 1362
    assert rel["total_ipf"] == 151
    assert len(rel["bins"]) == 6
    assert sum(b["n_patients"] for b in rel["bins"]) == 1362
    assert sum(b["n_ipf"] for b in rel["bins"]) == 151
    for b in rel["bins"]:
        assert b["n_patients"] > 0
        assert 0.0 <= b["empirical_ipf_pct"] <= 100.0

    age_stats = real_artifacts.get("cohort_age_stats")
    assert age_stats is not None
    assert age_stats["ipf_median_age"] == pytest.approx(68.9, abs=0.2)
    assert age_stats["non_ipf_median_age"] == pytest.approx(48.9, abs=0.2)

    res = client.get("/model-info")
    assert res.status_code == 200
    info = res.json()
    assert info["calibration_reliability_table"]["total_patients"] == 1362
    assert info["cohort_age_stats"]["ipf_median_age"] == age_stats["ipf_median_age"]
    assert info["interpretation_rules"]["mandatory_notice"] == AUTOMATED_INTERPRETATION_NOTICE


# ---------------------------------------------------------------------------
# 2. Three Sample Patients End-to-End via POST /predict-ipf
# ---------------------------------------------------------------------------


SAMPLE_PATIENTS = [
    (
        "Sample 1 (Zenodo IPF #3: 66M, Restrictive + Moderate/Borderline-Severe TLCO)",
        {
            "sex": "M",
            "age": 66.0,
            "height": 1.71,
            "weight": 90.0,
            "fev1fvc_abs": 0.83,
            "fev1_pp": 65.6,
            "fvc_pp": 60.5,
            "tlc_pp": 64.2,
            "tlco_pp": 40.1,
            "operating_mode": "screening",
        },
        "restrictive",
        "moderate",
        True,   # screening positive
        True,   # confirmatory positive
        True,   # age is a top driver
    ),
    (
        "Sample 2 (Zenodo Sarcoidosis #16: 51F, Non-Specific Preserved TLC + Normal TLCO)",
        {
            "sex": "F",
            "age": 51.0,
            "height": 1.68,
            "weight": 76.0,
            "fev1fvc_abs": 0.72,
            "fev1_pp": 71.0,
            "fvc_pp": 77.9,
            "tlc_pp": 95.5,
            "tlco_pp": 90.8,
            "operating_mode": "screening",
        },
        "non_specific",
        "normal",
        False,
        False,
        False,
    ),
    (
        "Sample 3 (Zenodo CTD-ILD #12: 55M, Normal Volumes + Preserved TLCO)",
        {
            "sex": "M",
            "age": 55.0,
            "height": 1.65,
            "weight": 66.0,
            "fev1fvc_abs": 0.72,
            "fev1_pp": 92.4,
            "fvc_pp": 101.9,
            "tlc_pp": 105.0,
            "tlco_pp": 81.6,
            "operating_mode": "screening",
        },
        "normal",
        "normal",
        False,
        False,
        False,
    ),
]


@pytest.mark.parametrize(
    "label,payload,expected_pattern,expected_tlco,exp_screen_pos,exp_conf_pos,exp_age_top",
    SAMPLE_PATIENTS,
)
def test_three_sample_patients_end_to_end_api(
    label,
    payload,
    expected_pattern,
    expected_tlco,
    exp_screen_pos,
    exp_conf_pos,
    exp_age_top,
    real_artifacts,
):
    res = client.post("/predict-ipf", json=payload)
    assert res.status_code == 200, f"Failed for {label}: {res.text}"
    data = res.json()

    assert data["interpretation_source"] == "deterministic"
    interp = data["interpretation"]
    text = data["interpretation_text"]

    # Five structured sections always present
    for key in (
        "physiology",
        "model_result",
        "what_pushed_it",
        "caution",
        "clinician_considerations",
        "notice",
    ):
        assert key in interp

    assert interp["notice"] == AUTOMATED_INTERPRETATION_NOTICE
    assert AUTOMATED_INTERPRETATION_NOTICE in text

    # No "how SHAP works" or generic "not causes" prose inside the interpretation text
    assert "log-odds scale" not in text.lower()
    assert "not causes" not in text.lower()

    # (a) Physiology checks
    assert interp["physiology"]["pattern"] == expected_pattern
    assert interp["physiology"]["gas_transfer_severity"] == expected_tlco

    # (b) Model result checks (both modes + empirical reliability bin rate)
    mr = interp["model_result"]
    assert mr["screening_positive"] is exp_screen_pos
    assert mr["confirmatory_positive"] is exp_conf_pos
    assert mr["reliability_bin"] is not None
    assert mr["reliability_bin"]["n_patients"] > 0
    assert f"{mr['reliability_bin']['empirical_ipf_pct']:.1f}%" in mr["summary"]

    # (c) Drivers + real cohort age medians when age is a top driver
    wp = interp["what_pushed_it"]
    assert wp["age_is_top_driver"] is exp_age_top
    if exp_age_top:
        assert "68.9" in wp["summary"]
        assert "48.9" in wp["summary"]
    else:
        assert wp["age_context_sentence"] is None

    # (d) Caution quotes stored i-NSIP, u-ILD, CTD misclassification rates + PFT overlap statement
    caut = interp["caution"]
    assert "i-NSIP" in caut["summary"]
    assert "u-ILD" in caut["summary"]
    assert "CTD" in caut["summary"]
    assert "lung function alone cannot separate ipf" in caut["summary"].lower()

    # (e) Generic clinician considerations cited to ATS/ERS/JRS/ALAT (no diagnosis/treatment)
    cc = interp["clinician_considerations"]
    assert len(cc["items"]) == 3
    joined_cc = " ".join(cc["items"]).lower()
    assert "hrct" in joined_cc
    assert "exposure" in joined_cc and "autoimmune" in joined_cc
    assert "multidisciplinary" in joined_cc
    assert "ats/ers/jrs/alat" in cc["guideline_reference"].lower()


# ---------------------------------------------------------------------------
# 3. Twelve Additional Diverse & Edge-Case Patient Profiles (Total = 15 Profiles)
# ---------------------------------------------------------------------------


ADDITIONAL_PROFILES = [
    # 4. Obstructive with hyperinflation, non-IPF
    {
        "name": "obstructive_hyperinflation_non_ipf",
        "inputs": {"sex": "M", "age": 68.0, "Height": 1.76, "Weight": 78.0, "fev1fvc_abs": 0.56, "fev1_pp": 62.0, "fvc_pp": 88.0, "tlc_pp": 126.0, "tlco_pp": 68.0},
        "prob": 0.0420,
        "shap": [
            {"feature": "tlc_pp", "display_name": "TLC % Predicted", "value": "126.0%", "shap_value": -0.92},
            {"feature": "fev1fvc_abs", "display_name": "FEV1/FVC Ratio", "value": "0.56", "shap_value": -0.71},
            {"feature": "age", "display_name": "Age", "value": "68 yrs", "shap_value": 0.45},
        ],
        "exp_pattern": "obstructive",
        "exp_tlco": "mild",
        "exp_disagree": False,
        "exp_borderline": False,
        "exp_age_top": True,
        "extra_check": lambda res: res["interpretation"]["physiology"]["tlc_elevated"] is True
        and "hyperinflation" in res["interpretation"]["physiology"]["summary"].lower(),
    },
    # 5. Mixed obstructive + restrictive + severe TLCO, high IPF probability
    {
        "name": "mixed_severe_tlco_high_prob",
        "inputs": {"sex": "M", "age": 72.0, "Height": 1.72, "Weight": 74.0, "fev1fvc_abs": 0.61, "fev1_pp": 46.0, "fvc_pp": 59.0, "tlc_pp": 68.0, "tlco_pp": 33.5},
        "prob": 0.5410,
        "shap": [
            {"feature": "tlco_pp", "display_name": "TLCO % Predicted", "value": "33.5%", "shap_value": 1.48},
            {"feature": "age", "display_name": "Age", "value": "72 yrs", "shap_value": 1.12},
            {"feature": "fev1fvc_abs", "display_name": "FEV1/FVC Ratio", "value": "0.61", "shap_value": -0.55},
        ],
        "exp_pattern": "mixed",
        "exp_tlco": "severe",
        "exp_disagree": False,
        "exp_borderline": False,
        "exp_age_top": True,
        "extra_check": lambda res: res["interpretation"]["model_result"]["screening_positive"] is True
        and res["interpretation"]["model_result"]["confirmatory_positive"] is True,
    },
    # 6. Isolated gas-transfer deficit with normal lung volumes
    {
        "name": "isolated_gas_transfer_deficit",
        "inputs": {"sex": "F", "age": 60.0, "Height": 1.63, "Weight": 65.0, "fev1fvc_abs": 0.81, "fev1_pp": 94.0, "fvc_pp": 96.0, "tlc_pp": 93.0, "tlco_pp": 52.0},
        "prob": 0.1120,
        "shap": [
            {"feature": "tlco_pp", "display_name": "TLCO % Predicted", "value": "52.0%", "shap_value": 0.68},
            {"feature": "sex", "display_name": "Sex", "value": "F", "shap_value": -0.45},
            {"feature": "tlc_pp", "display_name": "TLC % Predicted", "value": "93.0%", "shap_value": -0.32},
        ],
        "exp_pattern": "normal",
        "exp_tlco": "moderate",
        "exp_disagree": False,
        "exp_borderline": False,
        "exp_age_top": False,
        "extra_check": lambda res: res["interpretation"]["physiology"]["isolated_gas_transfer_deficit"] is True
        and "isolated" in res["interpretation"]["physiology"]["summary"].lower(),
    },
    # 7. Conflicting Screening (Positive) vs Confirmatory (Negative) modes
    {
        "name": "conflicting_screening_confirmatory_modes",
        "inputs": {"sex": "M", "age": 64.0, "Height": 1.74, "Weight": 82.0, "fev1fvc_abs": 0.80, "fev1_pp": 76.0, "fvc_pp": 73.0, "tlc_pp": 74.0, "tlco_pp": 58.0},
        "prob": 0.2650,
        "shap": [
            {"feature": "tlco_pp", "display_name": "TLCO % Predicted", "value": "58.0%", "shap_value": 0.62},
            {"feature": "tlc_pp", "display_name": "TLC % Predicted", "value": "74.0%", "shap_value": 0.34},
            {"feature": "fvc_pp", "display_name": "FVC % Predicted", "value": "73.0%", "shap_value": 0.21},
        ],
        "exp_pattern": "restrictive",
        "exp_tlco": "moderate",
        "exp_disagree": True,
        "exp_borderline": True,
        "exp_age_top": False,
        "extra_check": lambda res: "conflict" in res["interpretation"]["model_result"]["summary"].lower()
        and res["interpretation"]["model_result"]["screening_positive"] is True
        and res["interpretation"]["model_result"]["confirmatory_positive"] is False
        and "negative call does not exclude ipf" in res["interpretation"]["caution"]["summary"].lower(),
    },
    # 8. Borderline just below Screening threshold (0.175 vs 0.1868)
    {
        "name": "borderline_just_below_screening",
        "inputs": {"sex": "F", "age": 63.0, "Height": 1.62, "Weight": 68.0, "fev1fvc_abs": 0.78, "fev1_pp": 79.0, "fvc_pp": 78.0, "tlc_pp": 77.5, "tlco_pp": 62.0},
        "prob": 0.1750,
        "shap": [
            {"feature": "tlco_pp", "display_name": "TLCO % Predicted", "value": "62.0%", "shap_value": 0.29},
            {"feature": "sex", "display_name": "Sex", "value": "F", "shap_value": -0.35},
        ],
        "exp_pattern": "restrictive",
        "exp_tlco": "mild",
        "exp_disagree": False,
        "exp_borderline": True,
        "exp_age_top": False,
        "extra_check": lambda res: "borderline" in res["interpretation"]["model_result"]["summary"].lower()
        and len(res["interpretation"]["physiology"]["borderline_flags"]) >= 1,
    },
    # 9. Borderline just above Confirmatory threshold (0.362 vs 0.3429)
    {
        "name": "borderline_just_above_confirmatory",
        "inputs": {"sex": "M", "age": 69.0, "Height": 1.75, "Weight": 84.0, "fev1fvc_abs": 0.84, "fev1_pp": 71.0, "fvc_pp": 67.0, "tlc_pp": 71.0, "tlco_pp": 49.0},
        "prob": 0.3620,
        "shap": [
            {"feature": "age", "display_name": "Age", "value": "69 yrs", "shap_value": 0.88},
            {"feature": "tlco_pp", "display_name": "TLCO % Predicted", "value": "49.0%", "shap_value": 0.79},
        ],
        "exp_pattern": "restrictive",
        "exp_tlco": "moderate",
        "exp_disagree": False,
        "exp_borderline": True,
        "exp_age_top": True,
        "extra_check": lambda res: res["interpretation"]["model_result"]["confirmatory_positive"] is True
        and "borderline" in res["interpretation"]["model_result"]["summary"].lower(),
    },
    # 10. Severe restrictive + severe TLCO elderly patient
    {
        "name": "severe_restrictive_elderly",
        "inputs": {"sex": "M", "age": 78.0, "Height": 1.70, "Weight": 72.0, "fev1fvc_abs": 0.89, "fev1_pp": 52.0, "fvc_pp": 47.0, "tlc_pp": 51.0, "tlco_pp": 27.0},
        "prob": 0.6120,
        "shap": [
            {"feature": "tlco_pp", "display_name": "TLCO % Predicted", "value": "27.0%", "shap_value": 1.65},
            {"feature": "age", "display_name": "Age", "value": "78 yrs", "shap_value": 1.44},
            {"feature": "tlc_pp", "display_name": "TLC % Predicted", "value": "51.0%", "shap_value": 0.72},
        ],
        "exp_pattern": "restrictive",
        "exp_tlco": "severe",
        "exp_disagree": False,
        "exp_borderline": False,
        "exp_age_top": True,
        "extra_check": lambda res: "56.8%" in res["interpretation"]["model_result"]["summary"],
    },
    # 11. Missing TLC -> graceful FVC fallback for restrictive pattern
    {
        "name": "missing_tlc_fvc_fallback",
        "inputs": {"sex": "M", "age": 65.0, "Height": 1.73, "Weight": 80.0, "fev1fvc_abs": 0.83, "fev1_pp": 68.0, "fvc_pp": 64.0, "tlc_pp": None, "tlco_pp": 55.0},
        "prob": 0.2900,
        "shap": [
            {"feature": "tlco_pp", "display_name": "TLCO % Predicted", "value": "55.0%", "shap_value": 0.55},
            {"feature": "fvc_pp", "display_name": "FVC % Predicted", "value": "64.0%", "shap_value": 0.30},
        ],
        "exp_pattern": "restrictive",
        "exp_tlco": "moderate",
        "exp_disagree": True,
        "exp_borderline": True,
        "exp_age_top": False,
        "extra_check": lambda res: res["interpretation"]["physiology"]["used_fvc_proxy_for_restriction"] is True
        and "tlc_pp" in res["interpretation"]["physiology"]["missing_fields"],
    },
    # 12. Missing multiple PFTs -> indeterminate pattern, zero crash
    {
        "name": "missing_multiple_pfts_indeterminate",
        "inputs": {"sex": "F", "age": 58.0, "Height": 1.65, "Weight": 66.0, "fev1fvc_abs": None, "fev1_pp": 82.0, "fvc_pp": None, "tlc_pp": None, "tlco_pp": None},
        "prob": 0.0800,
        "shap": [],
        "exp_pattern": "indeterminate",
        "exp_tlco": "missing",
        "exp_disagree": False,
        "exp_borderline": False,
        "exp_age_top": False,
        "extra_check": lambda res: len(res["interpretation"]["physiology"]["missing_fields"]) == 4
        and "not provided" in res["interpretation"]["physiology"]["summary"].lower(),
    },
    # 13. Extreme low values + very advanced age (>90)
    {
        "name": "extreme_low_values_very_elderly",
        "inputs": {"sex": "M", "age": 94.0, "Height": 1.68, "Weight": 62.0, "fev1fvc_abs": 0.85, "fev1_pp": 22.0, "fvc_pp": 20.0, "tlc_pp": 26.0, "tlco_pp": 18.0},
        "prob": 0.5800,
        "shap": [
            {"feature": "tlco_pp", "display_name": "TLCO % Predicted", "value": "18.0%", "shap_value": 1.50},
            {"feature": "age", "display_name": "Age", "value": "94 yrs", "shap_value": 1.40},
        ],
        "exp_pattern": "restrictive",
        "exp_tlco": "severe",
        "exp_disagree": False,
        "exp_borderline": False,
        "exp_age_top": True,
        "extra_check": lambda res: len(res["interpretation"]["physiology"]["extreme_value_warnings"]) >= 4,
    },
    # 14. Extreme high values + young adult (<25)
    {
        "name": "extreme_high_values_young_adult",
        "inputs": {"sex": "F", "age": 21.0, "Height": 1.70, "Weight": 60.0, "fev1fvc_abs": 1.08, "fev1_pp": 162.0, "fvc_pp": 165.0, "tlc_pp": 158.0, "tlco_pp": 155.0},
        "prob": 0.0090,
        "shap": [
            {"feature": "age", "display_name": "Age", "value": "21 yrs", "shap_value": -1.35},
            {"feature": "tlco_pp", "display_name": "TLCO % Predicted", "value": "155.0%", "shap_value": -0.95},
        ],
        "exp_pattern": "normal",
        "exp_tlco": "normal",
        "exp_disagree": False,
        "exp_borderline": False,
        "exp_age_top": True,
        "extra_check": lambda res: len(res["interpretation"]["physiology"]["extreme_value_warnings"]) >= 5,
    },
    # 15. Score above maximum observed held-out test probability (0.85 > 0.6359)
    {
        "name": "score_above_observed_test_max",
        "inputs": {"sex": "M", "age": 75.0, "Height": 1.74, "Weight": 85.0, "fev1fvc_abs": 0.88, "fev1_pp": 60.0, "fvc_pp": 55.0, "tlc_pp": 58.0, "tlco_pp": 35.0},
        "prob": 0.8500,
        "shap": [
            {"feature": "tlco_pp", "display_name": "TLCO % Predicted", "value": "35.0%", "shap_value": 1.70},
            {"feature": "tlc_pp", "display_name": "TLC % Predicted", "value": "58.0%", "shap_value": 0.80},
            {"feature": "fvc_pp", "display_name": "FVC % Predicted", "value": "55.0%", "shap_value": 0.55},
        ],
        "exp_pattern": "restrictive",
        "exp_tlco": "severe",
        "exp_disagree": False,
        "exp_borderline": False,
        "exp_age_top": False,
        "extra_check": lambda res: res["interpretation"]["model_result"]["reliability_bin"]["above_observed_test_max"] is True
        and "exceeds the highest calibrated score observed" in res["interpretation"]["model_result"]["summary"].lower(),
    },
]


@pytest.mark.parametrize("case", ADDITIONAL_PROFILES, ids=[c["name"] for c in ADDITIONAL_PROFILES])
def test_twelve_additional_patient_profiles_deterministic(case, real_artifacts, real_rules):
    out = build_patient_interpretation(
        raw_inputs=case["inputs"],
        prob_ipf=case["prob"],
        screening_threshold=real_artifacts["screening_threshold"],
        confirmatory_threshold=real_artifacts["confirmatory_threshold"],
        active_mode="screening",
        top_shap_features=case["shap"],
        artifacts=real_artifacts,
        rules=real_rules,
    )
    interp = out["interpretation"]
    text = out["interpretation_text"]

    assert out["interpretation_source"] == "deterministic"
    assert AUTOMATED_INTERPRETATION_NOTICE in text
    assert interp["physiology"]["pattern"] == case["exp_pattern"]
    assert interp["physiology"]["gas_transfer_severity"] == case["exp_tlco"]
    assert interp["model_result"]["modes_disagree"] is case["exp_disagree"]
    assert interp["model_result"]["is_borderline"] is case["exp_borderline"]
    assert interp["what_pushed_it"]["age_is_top_driver"] is case["exp_age_top"]
    assert case["extra_check"](out) is True

    # Ensure deterministic output itself passes zero-diagnosis/treatment validation
    ok, reasons = validate_llm_rephrased_interpretation(
        candidate_text=text,
        structured_facts=interp,
        deterministic_text=text,
        rules=real_rules,
    )
    assert ok is True, f"Deterministic text unexpectedly triggered validator: {reasons}"


# ---------------------------------------------------------------------------
# 4. Configurable Rules Override Test
# ---------------------------------------------------------------------------


def test_configurable_rules_thresholds_change_pattern(real_artifacts, real_rules):
    """Changing thresholds in interpretation_rules.json changes physiology pattern dynamically."""
    sample2_inputs = {
        "sex": "F",
        "age": 51.0,
        "Height": 1.68,
        "Weight": 76.0,
        "fev1fvc_abs": 0.72,
        "fev1_pp": 71.0,
        "fvc_pp": 77.9,
        "tlc_pp": 95.5,
        "tlco_pp": 90.8,
    }
    # Default: TLC 95.5 >= 80 -> non_specific (not restrictive)
    default_out = build_patient_interpretation(
        raw_inputs=sample2_inputs,
        prob_ipf=0.01,
        screening_threshold=0.1868,
        confirmatory_threshold=0.3429,
        active_mode="screening",
        top_shap_features=[],
        artifacts=real_artifacts,
        rules=real_rules,
    )
    assert default_out["interpretation"]["physiology"]["pattern"] == "non_specific"

    # Override rule: restrictive_requires_tlc_when_present = False -> FVC 77.9 < 80 triggers restrictive
    custom_rules = copy.deepcopy(real_rules)
    custom_rules["physiology"]["restrictive_requires_tlc_when_present"] = False
    custom_out = build_patient_interpretation(
        raw_inputs=sample2_inputs,
        prob_ipf=0.01,
        screening_threshold=0.1868,
        confirmatory_threshold=0.3429,
        active_mode="screening",
        top_shap_features=[],
        artifacts=real_artifacts,
        rules=custom_rules,
    )
    assert custom_out["interpretation"]["physiology"]["pattern"] == "restrictive"


# ---------------------------------------------------------------------------
# 5. Optional LLM Rephrasing (Off by Default + Strict Validator + Fallback)
# ---------------------------------------------------------------------------


class _FakeAnthropicMessages:
    def __init__(self, reply_text: str = "", should_raise: bool = False):
        self.reply_text = reply_text
        self.should_raise = should_raise
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.should_raise:
            raise RuntimeError("Simulated upstream timeout")
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=self.reply_text)],
        )


class _FakeAnthropicClient:
    def __init__(self, reply_text: str = "", should_raise: bool = False):
        self.messages = _FakeAnthropicMessages(reply_text=reply_text, should_raise=should_raise)


def test_llm_rephrase_off_by_default(real_artifacts, real_rules):
    fake_client = _FakeAnthropicClient(reply_text="Should never be called.")
    out = build_patient_interpretation(
        raw_inputs={"sex": "M", "age": 66.0, "fev1fvc_abs": 0.83, "fev1_pp": 65.6, "fvc_pp": 60.5, "tlc_pp": 64.2, "tlco_pp": 40.1},
        prob_ipf=0.5369,
        screening_threshold=0.1868,
        confirmatory_threshold=0.3429,
        active_mode="screening",
        top_shap_features=[{"feature": "tlco_pp", "value": "40.1%", "shap_value": 1.37}],
        artifacts=real_artifacts,
        rules=real_rules,
        llm_client=fake_client,
    )
    assert out["interpretation_source"] == "deterministic"
    assert len(fake_client.messages.calls) == 0


def test_llm_rephrase_valid_output_passes_and_sends_only_deidentified_facts(
    monkeypatch, real_artifacts, real_rules
):
    monkeypatch.setenv("ENABLE_LLM_INTERPRETATION_REPHRASE", "true")
    valid_reply = (
        "Physiology: Restrictive pattern (TLC 64.2%, FVC 60.5%, FEV1/FVC 0.83) with moderately "
        "reduced gas transfer (TLCO 40.1%).\n"
        "Model result: Calibrated probability is 53.7% (0.5369), positive in both Screening "
        "(0.1868) and Confirmatory (0.3429) modes; 56.8% (50 of 88) of test patients in this bin had IPF.\n"
        "What pushed it: Reduced TLCO (40.1%) and age 66 pushed the score higher. IPF patients in "
        "this dataset are older (median 68.9 vs. 48.9 years).\n"
        "Caution: Lung function alone cannot separate IPF from other fibrosing ILDs "
        "(Screening false positives: i-NSIP 60.9%, CTD 35.0%, u-ILD 31.1%).\n"
        "For clinician consideration: HRCT pattern review, exposure and autoimmune history, and "
        "multidisciplinary discussion per ATS/ERS/JRS/ALAT guidelines (2018, 2022).\n"
        f"{AUTOMATED_INTERPRETATION_NOTICE}"
    )
    fake_client = _FakeAnthropicClient(reply_text=valid_reply)
    out = build_patient_interpretation(
        raw_inputs={
            "patient_name": "John Doe",  # Should never be forwarded into LLM prompt
            "sex": "M",
            "age": 66.0,
            "fev1fvc_abs": 0.83,
            "fev1_pp": 65.6,
            "fvc_pp": 60.5,
            "tlc_pp": 64.2,
            "tlco_pp": 40.1,
        },
        prob_ipf=0.5369,
        screening_threshold=0.1868,
        confirmatory_threshold=0.3429,
        active_mode="screening",
        top_shap_features=[
            {"feature": "tlco_pp", "value": "40.1%", "shap_value": 1.37},
            {"feature": "age", "value": "66 yrs", "shap_value": 1.35},
        ],
        artifacts=real_artifacts,
        rules=real_rules,
        llm_client=fake_client,
    )
    assert out["interpretation_source"] == "llm_rephrased"
    assert AUTOMATED_INTERPRETATION_NOTICE in out["interpretation_text"]
    assert len(fake_client.messages.calls) == 1
    sent_payload = json.dumps(fake_client.messages.calls[0]["messages"])
    assert "John Doe" not in sent_payload
    assert "patient_name" not in sent_payload


@pytest.mark.parametrize(
    "bad_reply,expected_reason_fragment",
    [
        (
            "Physiology: Restrictive pattern with TLCO 40.1%. Estimated 5-year survival is 77.3%.",
            "ungrounded number",
        ),
        (
            "Model result: Score 53.7% confirms an IPF diagnosis and rules in IPF.",
            "forbidden diagnosis/treatment",
        ),
        (
            "For clinician consideration: Start pirfenidone or nintedanib antifibrotic therapy promptly.",
            "forbidden diagnosis/treatment",
        ),
        (
            "Caution: Prescribe corticosteroids if i-NSIP 60.9% is suspected.",
            "forbidden diagnosis/treatment",
        ),
    ],
)
def test_llm_rephrase_validator_rejects_bad_outputs_and_falls_back(
    monkeypatch, bad_reply, expected_reason_fragment, real_artifacts, real_rules
):
    monkeypatch.setenv("ENABLE_LLM_INTERPRETATION_REPHRASE", "true")
    fake_client = _FakeAnthropicClient(reply_text=bad_reply)
    out = build_patient_interpretation(
        raw_inputs={"sex": "M", "age": 66.0, "fev1fvc_abs": 0.83, "fev1_pp": 65.6, "fvc_pp": 60.5, "tlc_pp": 64.2, "tlco_pp": 40.1},
        prob_ipf=0.5369,
        screening_threshold=0.1868,
        confirmatory_threshold=0.3429,
        active_mode="screening",
        top_shap_features=[{"feature": "tlco_pp", "value": "40.1%", "shap_value": 1.37}],
        artifacts=real_artifacts,
        rules=real_rules,
        llm_client=fake_client,
    )
    assert out["interpretation_source"] == "deterministic"
    assert expected_reason_fragment in out["interpretation"]["llm_fallback_reason"].lower()
    assert AUTOMATED_INTERPRETATION_NOTICE in out["interpretation_text"]
    assert "pirfenidone" not in out["interpretation_text"].lower()
    assert "77.3%" not in out["interpretation_text"]


def test_llm_rephrase_exception_falls_back_cleanly_via_api(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_INTERPRETATION_REPHRASE", "true")
    api_main.STATE["interpretation_llm_client_override"] = _FakeAnthropicClient(should_raise=True)

    res = client.post(
        "/predict-ipf",
        json={
            "sex": "M",
            "age": 66.0,
            "height": 1.71,
            "weight": 90.0,
            "fev1fvc_abs": 0.83,
            "fev1_pp": 65.6,
            "fvc_pp": 60.5,
            "tlc_pp": 64.2,
            "tlco_pp": 40.1,
            "operating_mode": "screening",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["interpretation_source"] == "deterministic"
    assert "llm call error" in data["interpretation"]["llm_fallback_reason"].lower()
    assert AUTOMATED_INTERPRETATION_NOTICE in data["interpretation_text"]
