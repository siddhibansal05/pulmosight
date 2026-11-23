"""
PyTorch & Imaging Feature Extraction Unit Tests (`backend/tests/test_torch_features.py`).

Skipped cleanly when `torch` or `torchvision` is not installed (`pytest.importorskip`).
Tests:
1. `PyTorchMLPClassifier` training with internal validation split, early stopping, and pure-NumPy inference.
2. Full 3D DICOM volume feature extraction with pretrained `torchvision` ResNet-18 (512-D embeddings).
3. In-Fold PCA(15) 3-way Nested GroupKFold modality ablation (`Clinical-only` vs `Clinical + HU/Volume`
   vs `Clinical + ResNet-18 In-Fold PCA`) verifying 512 raw features are reduced inside each training fold.
4. Valid DICOM upload to `POST /predict-progression`.
"""

import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

# Skip entire test module cleanly if torch or torchvision is absent (Item 2)
pytest.importorskip("torch")
pytest.importorskip("torchvision")

from fastapi.testclient import TestClient
from sklearn.base import is_classifier

from backend.api.main import app, reset_rate_limiter
from backend.src.generate_sample_data import generate_demo_dicom_series
from backend.src.imaging_progression import (
    CLINICAL_PROGRESSION_FEATURES,
    HU_VOLUME_FEATURE_NAMES,
    RESNET_FEATURE_NAMES,
    evaluate_modality_ablation_nested_gkf,
    extract_ct_features_from_3d_slices,
    load_dicom_as_hu,
    segment_lungs_hu,
)
from backend.src.pytorch_mlp import PyTorchMLPClassifier

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_rate_limit():
    reset_rate_limiter()


def test_pytorch_mlp_trains_and_exports_torch_free_numpy_weights():
    """Verify PyTorchMLPClassifier is a valid sklearn classifier and exports NumPy serving layers."""
    clf = PyTorchMLPClassifier(hidden_dims=(16, 8), max_epochs=8, patience=3, random_state=42)
    assert is_classifier(clf) is True

    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 9)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0.8).astype(int)

    clf.fit(X, y)
    assert hasattr(clf, "numpy_layers_")
    assert not hasattr(clf, "model_")  # No PyTorch nn.Module retained for serialization!

    probs = clf.predict_proba(X[:10])
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_dicom_3d_pretrained_resnet18_and_valid_upload(tmp_path):
    """Verify 3D DICOM HU segmentation, pretrained ResNet-18 512-D embeddings, and valid upload."""
    dcm_dir = tmp_path / "PAT_TEST"
    generate_demo_dicom_series(dcm_dir, "PAT_TEST", num_slices=3, rows=256, cols=256)
    dcm_files = sorted(dcm_dir.glob("*.dcm"))

    slice_tuples = [load_dicom_as_hu(f) for f in dcm_files]
    mask0 = segment_lungs_hu(slice_tuples[0][0])
    assert mask0.sum() > 500

    feats_3d = extract_ct_features_from_3d_slices(slice_tuples, extract_cnn=True)
    assert feats_3d["num_slices_processed"] == 3.0
    assert feats_3d["lung_volume_ml"] > 50.0
    assert "resnet_emb_0" in feats_3d
    assert "resnet_emb_511" in feats_3d

    with open(dcm_files[0], "rb") as f:
        dcm_bytes = f.read()
    res = client.post(
        "/predict-progression",
        data={"age": "68", "sex": "M", "fvc": "2500", "percent": "66.0", "smoking_status": "Ex-smoker"},
        files={"ct_scan": ("slice1.dcm", io.BytesIO(dcm_bytes), "application/dicom")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["imaging_used"] is True
    assert isinstance(data["imaging_model_available"], bool)
    assert data["extracted_ct_features"]["lung_volume_ml"] > 0


def test_in_fold_pca_3way_modality_ablation_pipeline():
    """
    Item 4 verification: verify `evaluate_modality_ablation_nested_gkf` reduces 512-D ResNet
    embeddings with in-fold PCA (10–20 components) and compares Clinical-only vs.
    Clinical + HU/Volume vs. Clinical + ResNet-18 (In-Fold PCA) with 95% CIs.
    """
    rng = np.random.default_rng(42)
    n_pat = 30
    data = {
        "Patient": [f"PAT_{i:03d}" for i in range(n_pat)],
        "fvc_change_52w_observed_ml": rng.normal(-180.0, 90.0, size=n_pat),
        "rapid_decliner_52w": rng.binomial(1, 0.45, size=n_pat),
    }
    for col in CLINICAL_PROGRESSION_FEATURES + HU_VOLUME_FEATURE_NAMES + RESNET_FEATURE_NAMES:
        data[col] = rng.normal(0.0, 1.0, size=n_pat)

    df_mock = pd.DataFrame(data)
    ablation = evaluate_modality_ablation_nested_gkf(
        df_mock,
        target_reg_col="fvc_change_52w_observed_ml",
        target_cls_col="rapid_decliner_52w",
        n_pca_components=12,
        n_splits=3,
        n_repeats=1,
    )
    assert ablation["status"] == "completed"
    assert 2 <= ablation["pca_components_used"] <= 20
    assert "Clinical-only" in ablation["modalities"]
    assert "Clinical + HU/Volume" in ablation["modalities"]
    assert any("ResNet-18 (In-Fold PCA" in k for k in ablation["modalities"].keys())


def test_predict_progression_multimodal_with_real_dicom_zip(tmp_path):
    """Requirement C (with torch): Upload a valid .zip of 3 DICOM slices and verify all 3 scores."""
    import zipfile

    dcm_dir = tmp_path / "PAT_ZIP"
    generate_demo_dicom_series(dcm_dir, "PAT_ZIP", num_slices=3, rows=256, cols=256)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dcm_f in sorted(dcm_dir.glob("*.dcm")):
            zf.write(dcm_f, arcname=dcm_f.name)
    zip_buf.seek(0)

    res = client.post(
        "/predict-progression-multimodal",
        data={
            "age": "66",
            "sex": "M",
            "fvc": "2600",
            "percent": "60.5",
            "smoking_status": "Ex-smoker",
            "horizon_weeks": "52",
        },
        files={"ct_series_zip": ("ct_series.zip", zip_buf, "application/zip")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["modality_used"] == {"clinical": True, "hrct": True, "fusion": True}
    assert data["clinical_only"] is not None
    assert data["hrct_only"] is not None
    assert data["fusion"] is not None
    assert data["extracted_ct_features"]["num_slices_processed"] == 3.0
    assert data["requires_ct_series_message"] is None
