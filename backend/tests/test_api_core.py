"""
Core Clinical API, Schema, Unit, Security, and Regression Tests (`backend/tests/test_api_core.py`).

ZERO `torch` or `torchvision` dependency:
All tests in this module MUST pass in a lightweight serving environment where `torch` is NOT installed.
"""

import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

import backend.api.main as api_main
from backend.api.main import app, reset_rate_limiter
from backend.src.config import (
    ARTIFACTS_JSON_PATH,
    EXPLORATORY_PROGRESSION_NOTE,
    LEAKAGE_AND_EXCLUDED_COLUMNS,
    MANDATORY_DISCLAIMER,
    REAL_ZENODO_DIAGNOSIS_GROUPS,
    REAL_ZENODO_MD5,
    ZENODO_CSV_PATH,
    check_sklearn_major_minor_match,
    compute_file_md5,
)
from backend.src.generate_sample_data import (
    ensure_osic_dataset,
    ensure_zenodo_dataset,
    main as demo_cli_main,
)
from backend.src.preprocessing import (
    build_preprocessor,
    load_raw_zenodo_dataframe,
    prepare_clinical_features_and_target,
    validate_fev1fvc_ratio,
    validate_height_meters,
    verify_case_number_uniqueness,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_rate_limit():
    reset_rate_limiter()


def test_real_zenodo_file_md5_diagnosis_groups_and_case_number_uniqueness():
    """Verify real Zenodo CSV MD5, 7 true diagnosis groups, and 6,808 unique Case_numbers."""
    assert ZENODO_CSV_PATH.exists(), f"Missing real Zenodo CSV at {ZENODO_CSV_PATH}"
    assert compute_file_md5(ZENODO_CSV_PATH) == REAL_ZENODO_MD5

    df, data_source = load_raw_zenodo_dataframe(ZENODO_CSV_PATH, demo=False)
    assert data_source == "real"
    assert len(df) == 6808
    assert int(df.isna().sum().sum()) == 0

    case_info = verify_case_number_uniqueness(df)
    assert case_info["is_unique"] is True
    assert case_info["unique_case_numbers"] == 6808

    observed_groups = set(df["diagnosis group"].astype(str).str.strip().unique())
    assert observed_groups == REAL_ZENODO_DIAGNOSIS_GROUPS


def test_no_silent_fallback_and_real_data_write_protection(tmp_path, monkeypatch):
    """Verify missing real data raises FileNotFoundError and real data is never overwritten."""
    with pytest.raises(FileNotFoundError):
        ensure_zenodo_dataset(tmp_path / "missing_zenodo.csv", demo=False)

    with pytest.raises(FileNotFoundError):
        ensure_osic_dataset(tmp_path / "missing_osic.csv", tmp_path / "dicom", demo=False)

    monkeypatch.setattr(sys, "argv", ["generate_sample_data.py"])
    with pytest.raises(RuntimeError, match="--demo"):
        demo_cli_main()

    _, src_tag = ensure_zenodo_dataset(ZENODO_CSV_PATH, demo=True)
    assert src_tag == "real"
    assert compute_file_md5(ZENODO_CSV_PATH) == REAL_ZENODO_MD5


def test_no_leakage_on_real_columns_and_age_formula():
    """Verify leakage columns are excluded and baseline age = age at event - survival time / 365.25."""
    df, _ = load_raw_zenodo_dataframe(ZENODO_CSV_PATH, demo=False)
    X, y, feat_cols, raw_groups = prepare_clinical_features_and_target(
        df, feature_set_name="pp_only"
    )
    assert int(y.sum()) == 757
    assert len(raw_groups) == 6808

    for leak_col in LEAKAGE_AND_EXCLUDED_COLUMNS + ["diagnosis group"]:
        assert leak_col not in X.columns
        assert leak_col not in feat_cols

    expected_age = float(df.loc[0, "age at event"]) - (
        float(df.loc[0, "survival time"]) / 365.25
    )
    assert np.isclose(X.loc[0, "age"], expected_age, atol=1e-4)


def test_explicit_unit_handling_edge_cases():
    """Verify no heuristic unit multiplication on edge cases (ratio=1.0, height=3.0)."""
    ratios = validate_fev1fvc_ratio(pd.Series([0.75, 1.0, 1.15]))
    assert np.allclose(ratios.values, [0.75, 1.0, 1.15])

    heights = validate_height_meters(pd.Series([1.65, 2.10, 3.0]))
    assert np.allclose(heights.values, [1.65, 2.10, 3.0])

    valid_payload = {
        "sex": "M",
        "age": 68,
        "height": 1.75,
        "weight": 80.0,
        "fev1fvc_abs": 1.0,
        "fev1_pp": 72.0,
        "fvc_pp": 68.0,
        "tlc_pp": 65.0,
        "tlco_pp": 48.0,
    }
    assert client.post("/predict-ipf", json=valid_payload).status_code == 200
    assert client.post("/predict-ipf", json={**valid_payload, "height": 3.0}).status_code == 422
    assert client.post("/predict-ipf", json={**valid_payload, "height": 174.0}).status_code == 422
    assert client.post("/predict-ipf", json={**valid_payload, "fev1fvc_abs": 82.0}).status_code == 422


def test_median_imputer_handles_missing_values():
    """Verify SimpleImputer(strategy='median') handles missing values cleanly."""
    feat_cols = ["sex", "Height", "Weight", "age", "fev1fvc_abs", "fev1_pp", "fvc_pp", "tlc_pp", "tlco_pp"]
    df_train = pd.DataFrame(
        {
            "sex": [1.0, 0.0, 1.0, np.nan],
            "Height": [1.75, 1.62, np.nan, 1.80],
            "Weight": [80.0, 65.0, 75.0, np.nan],
            "age": [68.0, 55.0, 72.0, 61.0],
            "fev1fvc_abs": [0.82, 0.78, np.nan, 0.85],
            "fev1_pp": [70.0, 88.0, 65.0, np.nan],
            "fvc_pp": [65.0, 90.0, np.nan, 72.0],
            "tlc_pp": [62.0, 92.0, 60.0, np.nan],
            "tlco_pp": [45.0, 80.0, np.nan, 55.0],
        }
    )
    transformed = build_preprocessor(feat_cols).fit_transform(df_train)
    assert transformed.shape == (4, 9)
    assert not np.isnan(transformed).any()


def test_sklearn_major_minor_version_tolerance_and_deployed_torch_free_model():
    """
    Verify:
    1. `check_sklearn_major_minor_match` passes when major.minor matches (e.g., 1.9.0 vs 1.9.1).
    2. `/health` succeeds with `status='ok'`, `data_source='real'`, and `sklearn_version_match=True`.
    3. The deployed clinical model is torch-free XGBoost (selected via 1-SE rule), and
       both `screening` and `confirmatory` modes work and return `sensitivity_disclosure`.
    """
    assert check_sklearn_major_minor_match("1.9.0", "1.9.1") is True
    assert check_sklearn_major_minor_match("1.9.0", "1.9.9") is True
    assert check_sklearn_major_minor_match("1.8.0", "1.9.1") is False

    health_res = client.get("/health")
    assert health_res.status_code == 200
    h_data = health_res.json()
    with open(ARTIFACTS_JSON_PATH, "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    assert h_data["status"] == "ok"
    assert h_data["clinical_model_loaded"] is True
    assert h_data["progression_model_loaded"] is True
    assert h_data["data_source"] == "real"
    assert h_data["sklearn_version_match"] is True
    expected_img_avail = artifacts["osic_progression_study"]["patients_with_dicoms"] >= 100
    assert h_data["imaging_model_available"] is expected_img_avail
    assert h_data["disclaimer"] == MANDATORY_DISCLAIMER

    assert artifacts["best_clinical_model"] == "XGBoost"
    assert artifacts["requires_torch_at_serving"] is False

    for m_name, m_info in artifacts["clinical_models"].items():
        assert m_info["threshold"] == m_info["oof_tuned_threshold"]

    payload = {
        "sex": "M",
        "age": 73,
        "height": 1.73,
        "weight": 79.0,
        "fev1fvc_abs": 0.87,
        "fev1_pp": 66.0,
        "fvc_pp": 59.0,
        "tlc_pp": 57.0,
        "tlco_pp": 38.0,
    }
    res_s = client.post("/predict-ipf", json={**payload, "operating_mode": "screening"})
    assert res_s.status_code == 200
    s_data = res_s.json()
    assert s_data["operating_mode"] == "screening"
    assert s_data["threshold"] == round(float(artifacts["screening_threshold"]), 4)
    assert "misses" in s_data["sensitivity_disclosure"].lower()
    assert len(s_data["top_5_shap_features"]) == 5

    res_c = client.post("/predict-ipf", json={**payload, "operating_mode": "confirmatory"})
    assert res_c.status_code == 200
    c_data = res_c.json()
    assert c_data["operating_mode"] == "confirmatory"
    assert c_data["threshold"] == round(float(artifacts["confirmatory_threshold"]), 4)
    assert c_data["threshold"] >= s_data["threshold"]


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("sex", "Unknown"),
        ("age", 12.0),
        ("age", 125.0),
        ("height", 0.5),
        ("height", 172.0),
        ("weight", 15.0),
        ("weight", 300.0),
        ("fev1fvc_abs", 0.05),
        ("fev1fvc_abs", 85.0),
        ("fev1_pp", 5.0),
        ("fev1_pp", 280.0),
        ("fvc_pp", 8.0),
        ("fvc_pp", 260.0),
        ("tlc_pp", 4.0),
        ("tlc_pp", 290.0),
        ("tlco_pp", 2.0),
        ("tlco_pp", 255.0),
    ],
)
def test_schema_validation_for_every_field(field_name, bad_value):
    base_valid = {
        "sex": "F",
        "age": 62.0,
        "height": 1.64,
        "weight": 66.0,
        "fev1fvc_abs": 0.80,
        "fev1_pp": 78.0,
        "fvc_pp": 75.0,
        "tlc_pp": 74.0,
        "tlco_pp": 60.0,
    }
    res = client.post("/predict-ipf", json={**base_valid, field_name: bad_value})
    assert res.status_code == 422


def test_predict_progression_clinical_only_and_upload_security_limits():
    """
    Verify clinical-only `/predict-progression` (torch-free), security rejection of `.joblib`
    model uploads (415), disallowed extensions (415), corrupted DICOM (400), and oversized uploads (413).
    """
    res = client.post(
        "/predict-progression",
        data={
            "age": "70",
            "sex": "M",
            "fvc": "2400",
            "percent": "64.0",
            "smoking_status": "Ex-smoker",
        },
    )
    assert res.status_code == 200
    data = res.json()
    with open(ARTIFACTS_JSON_PATH, "r", encoding="utf-8") as f:
        expected_img_avail = json.load(f)["imaging_model_available"]
    assert data["exploratory_banner"] == EXPLORATORY_PROGRESSION_NOTE
    assert data["imaging_model_available"] is expected_img_avail
    assert data["usable_as_standalone_predictor"] is False
    assert data["imaging_used"] is False
    assert data["extracted_ct_features"] is None
    assert data["disclaimer"] == MANDATORY_DISCLAIMER

    # Security policy: model files (.joblib / .pkl) must NEVER be accepted via API upload
    res_joblib = client.post(
        "/predict-progression",
        data={"age": "68", "sex": "M", "fvc": "2500", "percent": "66.0", "smoking_status": "Ex-smoker"},
        files={"ct_scan": ("malicious.joblib", io.BytesIO(b"pickle"), "application/octet-stream")},
    )
    assert res_joblib.status_code == 415
    assert "security policy" in res_joblib.json()["detail"].lower()

    # Disallowed extension (.txt) -> 415
    res_ext = client.post(
        "/predict-progression",
        data={"age": "68", "sex": "M", "fvc": "2500", "percent": "66.0", "smoking_status": "Ex-smoker"},
        files={"ct_scan": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert res_ext.status_code == 415

    # Corrupted DICOM -> 400
    res_corrupt = client.post(
        "/predict-progression",
        data={"age": "68", "sex": "M", "fvc": "2500", "percent": "66.0", "smoking_status": "Ex-smoker"},
        files={"ct_scan": ("corrupt.dcm", io.BytesIO(b"NOT_A_DICOM"), "application/dicom")},
    )
    assert res_corrupt.status_code == 400

    # Oversized upload -> 413
    orig_limit = api_main.MAX_CT_UPLOAD_BYTES
    try:
        api_main.MAX_CT_UPLOAD_BYTES = 1024
        res_large = client.post(
            "/predict-progression",
            data={"age": "68", "sex": "M", "fvc": "2500", "percent": "66.0", "smoking_status": "Ex-smoker"},
            files={"ct_scan": ("large.dcm", io.BytesIO(b"0" * 2048), "application/dicom")},
        )
        assert res_large.status_code == 413
    finally:
        api_main.MAX_CT_UPLOAD_BYTES = orig_limit


def test_metrics_and_eda_loaded_from_artifacts_json_not_hardcoded():
    """Regression test ensuring `/model-info` and `/eda` load directly from `model_artifacts.json`."""
    with open(ARTIFACTS_JSON_PATH, "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    res_info = client.get("/model-info")
    assert res_info.status_code == 200
    info_data = res_info.json()

    assert info_data["data_source"] == "real"
    assert info_data["best_model"] == artifacts["best_clinical_model"]
    assert info_data["requires_torch_at_serving"] is False
    assert info_data["imaging_model_available"] is artifacts["imaging_model_available"]
    assert info_data["sensitivity_disclosure"] == artifacts["sensitivity_disclosure"]

    osic_study = info_data["osic_progression_study"]
    assert osic_study["exploratory_banner"] == EXPLORATORY_PROGRESSION_NOTE
    assert "fvc_change_52w" in osic_study["target_variants"]
    assert "robust_theil_sen_slope" in osic_study["target_variants"]
    assert "modality_ablation" in osic_study
    if osic_study["imaging_model_available"]:
        assert osic_study["patients_with_dicoms"] >= 100
        assert osic_study["modality_ablation"]["status"] == "completed"
        assert "fvc_change_52w" in osic_study["modality_ablation"]["by_target"]
        assert "robust_theil_sen_slope" in osic_study["modality_ablation"]["by_target"]

    res_eda = client.get("/eda")
    assert res_eda.status_code == 200
    eda_data = res_eda.json()
    assert eda_data["total_rows"] == 6808
    assert eda_data["ipf_rows"] == 757
    assert {g["group"] for g in eda_data["diagnosis_group_counts"]} == REAL_ZENODO_DIAGNOSIS_GROUPS


def test_shap_plain_language_summary_deterministic_template():
    """Requirement D: Verify deterministic SHAP plain-language summary on Sample 1."""
    sample1_payload = {
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
    }
    res = client.post("/predict-ipf", json=sample1_payload)
    assert res.status_code == 200
    data = res.json()
    summary = data["shap_plain_summary"]
    assert "log-odds scale" in summary.lower()
    assert (
        "SHAP shows what the model used, not causes. "
        "Age partly reflects differences between IPF and non-IPF patients in this dataset."
    ) in summary
    assert "TLCO" in summary or "age" in summary.lower()


def test_predict_progression_multimodal_no_ct_security_and_missing_torch_503(monkeypatch):
    """
    Requirement C: Verify POST /predict-progression-multimodal:
    1. Without CT zip -> clinical_only returned, hrct_only=None, fusion=None, modality_used flags exact.
    2. Security: non-zip and zip containing .joblib -> HTTP 415.
    3. Missing torch / ResNet-18 weights -> clear HTTP 503 while /predict-ipf stays torch-free.
    """
    import zipfile

    # 1. Clinical-only (no CT uploaded) -> never dummy values!
    res = client.post(
        "/predict-progression-multimodal",
        data={"age": "66", "sex": "M", "fvc": "2600", "percent": "60.5", "smoking_status": "Ex-smoker"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["exploratory_badge"] == "Exploratory, not validated"
    assert data["modality_used"] == {"clinical": True, "hrct": False, "fusion": False}
    assert data["clinical_only"] is not None
    assert data["hrct_only"] is None
    assert data["fusion"] is None
    assert "requires ct series" in data["requires_ct_series_message"].lower()
    assert "hrct_only" in data["cv_ci_summary"]
    assert "fusion" in data["cv_ci_summary"]

    # 2. Non-zip extension -> HTTP 415
    res_bad_ext = client.post(
        "/predict-progression-multimodal",
        data={"age": "66", "sex": "M", "fvc": "2600", "percent": "60.5"},
        files={"ct_series_zip": ("scan.dcm", io.BytesIO(b"123"), "application/dicom")},
    )
    assert res_bad_ext.status_code == 415

    # 3. Zip containing forbidden .joblib file -> HTTP 415
    buf_bad = io.BytesIO()
    with zipfile.ZipFile(buf_bad, "w") as zf:
        zf.writestr("evil.joblib", b"pickle")
    buf_bad.seek(0)
    res_bad_zip = client.post(
        "/predict-progression-multimodal",
        data={"age": "66", "sex": "M", "fvc": "2600", "percent": "60.5"},
        files={"ct_series_zip": ("series.zip", buf_bad, "application/zip")},
    )
    assert res_bad_zip.status_code == 415
    assert "security policy" in res_bad_zip.json()["detail"].lower()

    # 4. Missing torch / ResNet-18 weights -> HTTP 503
    buf_dcm = io.BytesIO()
    with zipfile.ZipFile(buf_dcm, "w") as zf:
        zf.writestr("slice_001.dcm", b"dummy")
    buf_dcm.seek(0)

    def _raise_missing_torch(*_args, **_kwargs):
        raise RuntimeError("PyTorch and torchvision are required for ResNet-18 CT scoring.")

    monkeypatch.setattr(api_main, "extract_features_from_dicom_zip_bytes", _raise_missing_torch)
    res_503 = client.post(
        "/predict-progression-multimodal",
        data={"age": "66", "sex": "M", "fvc": "2600", "percent": "60.5"},
        files={"ct_series_zip": ("series.zip", buf_dcm, "application/zip")},
    )
    assert res_503.status_code == 503
    assert "pytorch" in res_503.json()["detail"].lower()


def test_osic_competition_benchmark_and_fusion_comparison_artifacts():
    """Requirements A & B: Verify Laplace LL benchmark & 6-method fusion table in /model-info."""
    res = client.get("/model-info")
    assert res.status_code == 200
    osic = res.json()["osic_progression_study"]
    if not osic["imaging_model_available"]:
        pytest.skip("CT cache < 100 patients")

    comp = osic["osic_competition_benchmark"]
    assert comp["num_patients"] >= 100
    for expected_model in (
        "Baseline (Constant Baseline FVC)",
        "Ridge",
        "ElasticNet",
        "Quantile-Regression NN (Tabular)",
        "Random Forest",
        "SVM (RBF SVR)",
        "XGBoost",
    ):
        assert expected_model in comp["models"]
        assert "laplace_ll_stats" in comp["models"][expected_model]

    fusion = osic["fusion_comparison"]
    assert fusion["status"] == "completed"
    assert fusion["selected_fusion_method"] in fusion["methods"]
    assert fusion.get("statistically_confirmed_method") == "stacking"
    assert fusion.get("bonferroni_comparisons_count") == 20
    assert "fusion_beats_clinical_only" not in fusion
    assert fusion["fusion_vs_clinical_status"] in ("statistically_confirmed", "nominal_only")
    assert "gain > 1 se" not in fusion["selection_rationale"].lower()
    assert "bonferroni" in fusion["selection_rationale"].lower()
    assert "clinically negligible" in fusion["selection_rationale"].lower()
    assert "paired_vs_clinical_only" in fusion
    for key in (
        "clinical_only",
        "hrct_only",
        "early_fusion",
        "late_fusion_mean",
        "weighted_late_fusion",
        "stacking",
    ):
        assert key in fusion["methods"]
        for metric_k in ("laplace_ll_stats", "rmse_stats", "mae_stats", "auc_stats"):
            assert metric_k in fusion["methods"][key]
        if key != "clinical_only":
            paired = fusion["paired_vs_clinical_only"][key]
            assert paired["status"] in ("statistically_confirmed", "nominal_only")
            for f_k in ("mean_diff", "ci_95_lower", "ci_95_upper", "p_value", "se_nadeau_bengio"):
                assert f_k in paired["laplace_ll"]


def test_model_info_exposes_extractor_comparison_section():
    """Verify GET /model-info exposes extractor_comparison with folds_sha256 and licenses."""
    res = client.get("/model-info")
    assert res.status_code == 200
    ec = res.json().get("extractor_comparison")
    assert ec is not None
    assert ec["deployed_model_changed"] is False
    assert len(ec["protocol"]["folds_sha256"]) == 64
    for key in ("resnet18_baseline", "med3d", "radimagenet", "ctclip", "resnet50_imagenet_control"):
        assert key in ec["extractors"]
        assert ec["extractors"][key]["license"]
    assert "CC BY-NC-SA" in ec["extractors"]["ctclip"]["license"]
