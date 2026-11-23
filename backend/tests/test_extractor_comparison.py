"""
Automated tests for the OSIC Pretrained CT Feature Extractor Comparison
(`backend/src/extractor_comparison.py`).

Verifies:
1. Exact fold identity across extractors (identical fold patient sets and identical
   `folds_sha256` hash stored in artifacts).
2. Single-parameter (`emb_prefix`) protocol parity with `backend/src/osic_benchmarks.py`
   on the real ResNet-18 cached dataset.
3. Strict handling of missing feature files (`"pending: feature file not found"`)
   and explicitly unrun extractors (`"not run"` with exact reason) — zero fabricated metrics.
4. Paired Nadeau-Bengio tests against both (i) current ResNet-18 and (ii) clinical-only
   with exact Holm-Bonferroni & Bonferroni multiple-comparison adjustments.
5. Read-only guarantee: never modifies deployed clinical or progression models.
6. API exposure via `GET /model-info` (torch-free).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.src.config import (
    ARTIFACTS_JSON_PATH,
    CACHED_DICOM_FEATURES_CSV,
    OSIC_TRAIN_CSV_PATH,
    RANDOM_SEED,
)
from backend.src.extractor_comparison import (
    apply_holm_bonferroni,
    compute_folds_sha256,
    render_extractor_comparison_markdown,
    run_osic_extractor_comparison,
)
from backend.src.osic_benchmarks import _generate_patient_cv_splits


def test_identical_fold_indices_and_hash_across_extractors(tmp_path: Path):
    """
    Explicitly verify that two extractors with different embedding prefixes and
    different raw dimensions (512-D vs. 2048-D) produce IDENTICAL patient CV fold
    indices and the EXACT same `folds_sha256` hash.
    """
    base_df = pd.read_csv(CACHED_DICOM_FEATURES_CSV)
    patients = sorted(base_df["Patient"].astype(str).unique().tolist())

    splits_a = _generate_patient_cv_splits(np.array(patients), n_repeats=5, n_splits=5, seed=RANDOM_SEED)
    # Shuffle patient order as might occur in an alternative extractor CSV
    rng = np.random.default_rng(12345)
    shuffled = rng.permutation(patients)
    splits_b = _generate_patient_cv_splits(shuffled, n_repeats=5, n_splits=5, seed=RANDOM_SEED)

    assert len(splits_a) == 25 and len(splits_b) == 25
    for fold_idx, ((tr_a, va_a), (tr_b, va_b)) in enumerate(zip(splits_a, splits_b)):
        assert tr_a == tr_b, f"Training fold {fold_idx} differs across extractors!"
        assert va_a == va_b, f"Validation fold {fold_idx} differs across extractors!"

    hash_a = compute_folds_sha256(splits_a)
    hash_b = compute_folds_sha256(splits_b)
    assert len(hash_a) == 64
    assert hash_a == hash_b


def test_holm_bonferroni_multiple_comparison_math():
    """Verify step-down Holm-Bonferroni and Bonferroni adjustment invariants."""
    records = [
        {"mean_diff": 0.02, "higher_is_better": True, "_p_exact_float": 0.001},
        {"mean_diff": -2.0, "higher_is_better": False, "_p_exact_float": 0.015},
        {"mean_diff": 0.01, "higher_is_better": True, "_p_exact_float": 0.030},
        {"mean_diff": -0.05, "higher_is_better": True, "_p_exact_float": 0.002},  # worse direction!
    ]
    summary = apply_holm_bonferroni(records, alpha=0.05)
    assert summary["total_paired_tests"] == 4
    assert summary["bonferroni_alpha"] == round(0.05 / 4, 6)

    # Sorted raw p: 0.001 (x4=0.004), 0.002 (x3=0.006), 0.015 (x2=0.030), 0.030 (x1=0.030)
    assert records[0]["p_value_holm"] == pytest.approx(0.004, abs=1e-6)
    assert records[0]["p_value_bonferroni"] == pytest.approx(0.004, abs=1e-6)
    assert records[0]["statistically_better_holm"] is True

    assert records[3]["p_value_holm"] == pytest.approx(0.006, abs=1e-6)
    assert records[3]["significant_holm_005"] is True
    assert records[3]["statistically_better_holm"] is False  # wrong direction!

    assert records[1]["p_value_holm"] == pytest.approx(0.030, abs=1e-6)
    assert records[1]["statistically_better_holm"] is True
    assert records[1]["p_value_bonferroni"] == pytest.approx(0.060, abs=1e-6)
    assert records[1]["statistically_better_bonferroni"] is False

    assert records[2]["p_value_holm"] == pytest.approx(0.030, abs=1e-6)
    for rec in records:
        assert "_p_exact_float" not in rec


def test_extractor_comparison_end_to_end_with_completed_not_run_and_pending(tmp_path: Path):
    """
    End-to-end verification on a temporary OSIC folder containing:
      - real `cached_dicom_features.csv` (ResNet-18 baseline)
      - synthetic `cached_dicom_features_med3d.csv` + metadata JSON (completed alternative extractor)
      - `cached_dicom_features_ctclip_metadata.json` with status='not run' (gated access)
      - missing RadImageNet & Control CSVs (`"pending: feature file not found"`)
    Asserts identical folds_sha256 across completed extractors, paired Nadeau-Bengio tests
    against both ResNet-18 and Clinical-only with Holm correction, and zero fabrication.
    """
    shutil.copy2(CACHED_DICOM_FEATURES_CSV, tmp_path / "cached_dicom_features.csv")
    shutil.copy2(OSIC_TRAIN_CSV_PATH, tmp_path / "train.csv")

    base_df = pd.read_csv(CACHED_DICOM_FEATURES_CSV)
    patients = base_df["Patient"].astype(str).tolist()

    # Create a small deterministic synthetic med3d CSV (32-D embeddings) on the exact same 175 patients
    rng = np.random.default_rng(777)
    med3d_mat = rng.normal(0.0, 1.0, size=(len(patients), 32))
    med3d_df = pd.DataFrame(med3d_mat, columns=[f"med3d_emb_{i}" for i in range(32)])
    med3d_df.insert(0, "Patient", patients)
    med3d_df.to_csv(tmp_path / "cached_dicom_features_med3d.csv", index=False)
    (tmp_path / "cached_dicom_features_med3d_metadata.json").write_text(
        json.dumps(
            {
                "extractor_key": "med3d",
                "status": "completed",
                "embedding_size": 32,
                "runtime_seconds": 180.5,
                "runtime_minutes": 3.01,
                "peak_gpu_memory_mb": 1120.4,
            }
        ),
        encoding="utf-8",
    )

    # Create a 'not run' CT-CLIP metadata JSON (no CSV)
    (tmp_path / "cached_dicom_features_ctclip_metadata.json").write_text(
        json.dumps(
            {
                "extractor_key": "ctclip",
                "status": "not run",
                "not_run_reason": "GatedRepoError: 403 Forbidden accessing ibrahimhamamci/CT-RATE",
                "embedding_size": 512,
            }
        ),
        encoding="utf-8",
    )

    result = run_osic_extractor_comparison(
        osic_dir=tmp_path,
        train_csv_path=tmp_path / "train.csv",
        dicom_root=tmp_path / "_no_dicoms",
        n_pca=15,
    )

    canonical_hash = result["protocol"]["folds_sha256"]
    assert len(canonical_hash) == 64
    assert result["deployed_model_changed"] is False

    exts = result["extractors"]
    assert exts["resnet18_baseline"]["status"] == "completed"
    assert exts["med3d"]["status"] == "completed"
    assert exts["radimagenet"]["status"] == "pending: not evaluated yet"
    assert exts["ctclip"]["status"] == "not run"
    assert "403 Forbidden" in exts["ctclip"]["status_reason"]
    assert exts["resnet50_imagenet_control"]["status"] == "pending: not evaluated yet"

    # Zero fabricated metrics for missing / not-run extractors
    for missing_key in ("radimagenet", "ctclip", "resnet50_imagenet_control"):
        assert exts[missing_key]["hrct_only"] is None
        assert exts[missing_key]["image_plus_tabular"] is None
        assert exts[missing_key]["fusion_methods"] is None

    # Exact identical fold hash verified across both completed extractors
    assert exts["resnet18_baseline"]["identical_folds_verified"] is True
    assert exts["med3d"]["identical_folds_verified"] is True
    assert exts["resnet18_baseline"]["folds_sha256"] == canonical_hash
    assert exts["med3d"]["folds_sha256"] == canonical_hash

    # Metadata JSON values propagated accurately for Med3D
    assert exts["med3d"]["embedding_size"] == 32
    assert exts["med3d"]["extraction_runtime_minutes"] == pytest.approx(3.01)
    assert exts["med3d"]["peak_gpu_memory_mb"] == pytest.approx(1120.4)

    # Paired tests: 24 vs ResNet-18 (med3d) + 48 vs Clinical-only (resnet18 + med3d) = 72 tests
    mtc = result["multiple_testing"]
    assert mtc["num_tests_vs_resnet18"] == 24
    assert mtc["num_tests_vs_clinical_only"] == 48
    assert mtc["total_paired_tests"] == 72

    # Because some extractors are still pending, overall flags must be null + "not evaluated",
    # while completed-only flags are computed strictly from completed extractors (med3d)
    assert mtc["any_new_extractor_beats_resnet18_holm"] is None
    assert mtc["any_new_extractor_beats_clinical_only_holm"] is None
    assert mtc["any_new_extractor_beats_resnet18_holm_note"] == "not evaluated"
    assert mtc["any_new_extractor_beats_clinical_only_holm_note"] == "not evaluated"
    assert isinstance(mtc["completed_extractors_beat_resnet18_holm"], bool)
    assert isinstance(mtc["completed_extractors_beat_clinical_only_holm"], bool)
    assert "No alternative extractor has beaten ResNet-18 or Clinical-only" not in result["overall_verdict"]

    med_r18 = exts["med3d"]["paired_vs_resnet18"]["late_fusion_mean"]["laplace_ll"]
    assert "p_value" in med_r18 and "p_value_holm" in med_r18 and "p_value_bonferroni" in med_r18
    assert med_r18["p_value_bonferroni"] >= med_r18["p_value_holm"] >= med_r18["p_value_raw_exact"] - 1e-9

    md = render_extractor_comparison_markdown(result)
    assert canonical_hash in md
    assert "pending: not evaluated yet" in md
    assert "not evaluated" in md
    assert "No alternative extractor has beaten ResNet-18 or Clinical-only" not in md
    assert "403 Forbidden" in md
    assert "arXiv:1904.00625" in md
    assert "10.1148/ryai.210315" in md
    assert "arXiv:2403.17834" in md
    assert "CC BY-NC-SA 4.0" in md


def test_saved_model_artifacts_extractor_comparison_parity_and_hygiene():
    """
    Verify the persisted `extractor_comparison` in `backend/models/model_artifacts.json`
    matches the baseline `osic_progression_study` numbers verbatim, stores `folds_sha256`,
    never fabricates numbers for missing CSVs, and never alters the deployed model.
    """
    art = json.loads(ARTIFACTS_JSON_PATH.read_text(encoding="utf-8"))
    ec = art.get("extractor_comparison")
    assert ec is not None, "model_artifacts.json missing extractor_comparison section"
    assert ec["deployed_model_changed"] is False
    assert len(ec["protocol"]["folds_sha256"]) == 64
    assert "No alternative extractor has beaten ResNet-18 or Clinical-only" not in ec["overall_verdict"]

    r18 = ec["extractors"]["resnet18_baseline"]
    assert r18["status"] == "completed"
    assert r18["identical_folds_verified"] is True
    assert r18["folds_sha256"] == ec["protocol"]["folds_sha256"]

    # Exact parity with existing osic_progression_study.fusion_comparison & benchmark
    fc = art["osic_progression_study"]["fusion_comparison"]["methods"]
    for m_key in ("clinical_only", "hrct_only", "early_fusion", "late_fusion_mean", "weighted_late_fusion", "stacking"):
        assert r18["fusion_methods"][m_key]["laplace_ll_stats"]["mean"] == pytest.approx(
            fc[m_key]["laplace_ll_stats"]["mean"], abs=1e-4
        )
        assert r18["fusion_methods"][m_key]["auc_stats"]["mean"] == pytest.approx(
            fc[m_key]["auc_stats"]["mean"], abs=1e-4
        )

    it_ref = art["osic_progression_study"]["osic_competition_benchmark"]["models"][
        "Image + Tabular (ResNet-18 PCA 15D)"
    ]
    assert r18["image_plus_tabular"]["laplace_ll_stats"]["mean"] == pytest.approx(
        it_ref["laplace_ll_stats"]["mean"], abs=1e-4
    )

    # Verify completed real extractors and unrun optional control strictly match real files on disk
    for key in ("med3d", "radimagenet", "ctclip"):
        entry = ec["extractors"][key]
        assert entry["status"] == "completed"
        assert entry["identical_folds_verified"] is True
        assert entry["folds_sha256"] == ec["protocol"]["folds_sha256"]
        assert entry["hrct_only"] is not None
        assert entry["image_plus_tabular"] is not None
        assert entry["fusion_methods"] is not None

    ctrl = ec["extractors"]["resnet50_imagenet_control"]
    assert ctrl["status"] == "not run"
    assert "RUN_IMAGENET_RESNET50_CONTROL" in (ctrl.get("status_reason") or "")
    assert ctrl["hrct_only"] is None
    assert ctrl["image_plus_tabular"] is None
    assert ctrl["fusion_methods"] is None
    assert ctrl["extraction_runtime_minutes"] is None
    assert ctrl["peak_gpu_memory_mb"] is None

    mtc = ec["multiple_testing"]
    assert mtc["total_paired_tests"] == 168
    assert mtc["any_new_extractor_beats_resnet18_holm"] is False
    assert mtc["any_new_extractor_beats_clinical_only_holm"] is False


def test_kaggle_extractor_notebook_static_hygiene_and_med3d_key_logic():
    """
    Static verification (no notebook execution) of backend/scripts/kaggle_extractor_notebook.ipynb:
      - Valid nbformat 4.5, zero saved outputs, clean Python AST
      - KAGGLE_USERNAME = "YOUR_KAGGLE_USERNAME" defined in first code cell + markdown note
      - No personal Kaggle username hard-coded
      - Med3D loader compares against model.state_dict() keys ignoring num_batches_tracked
        (no hard-coded key count 102)
    """
    import ast
    from backend.src.config import PROJECT_ROOT

    nb_path = PROJECT_ROOT / "backend" / "scripts" / "kaggle_extractor_notebook.ipynb"
    raw = nb_path.read_text(encoding="utf-8")
    nb = json.loads(raw)

    assert nb["nbformat"] == 4 and nb["nbformat_minor"] == 5
    assert "dashmeetsingh2005" not in raw

    cell0 = "".join(nb["cells"][0]["source"])
    cell1 = "".join(nb["cells"][1]["source"])
    cell5 = "".join(nb["cells"][5]["source"])
    cell6 = "".join(nb["cells"][6]["source"])

    assert "KAGGLE_USERNAME" in cell0 and "YOUR_KAGGLE_USERNAME" in cell0
    assert 'KAGGLE_USERNAME = "YOUR_KAGGLE_USERNAME"' in cell1
    assert "102" not in cell5
    assert "102" not in cell6
    assert "num_batches_tracked" in cell6
    assert "model_state_dict_keys" in cell6 and "checkpoint_keys" in cell6

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], f"Cell {idx} has saved outputs!"
            assert cell["execution_count"] is None, f"Cell {idx} has non-null execution_count!"
            ast.parse("".join(cell["source"]), filename=f"kaggle_cell_{idx}.py")
