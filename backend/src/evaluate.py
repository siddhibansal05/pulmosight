"""
Model Evaluation, Fair Common-Rule OOF Threshold Tuning (80% Sensitivity Screening Mode
& High-Specificity Confirmatory Mode), Bootstrap Confidence Intervals,
Per-Diagnosis-Group Error Analysis, & Plotting (`backend/src/evaluate.py`).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from backend.src.config import PLOTS_DIR, RANDOM_SEED, setup_logger
from backend.src.preprocessing import DIAGNOSIS_GROUP_DESCRIPTIONS

logger = setup_logger("evaluate")


def tune_threshold_for_target_sensitivity_oof(
    y_true_train: np.ndarray,
    oof_probs_train: np.ndarray,
    target_sensitivity: float = 0.80,
) -> float:
    """
    Unified fair threshold rule applied across ALL models (Requirement D1):
    Find the highest probability threshold on Out-of-Fold (OOF) training predictions
    that achieves at least `target_sensitivity` (e.g., 80% sensitivity on training OOF).
    Never uses the test set.
    """
    y_arr = np.asarray(y_true_train).astype(int)
    p_arr = np.asarray(oof_probs_train).astype(float)

    fpr, tpr, thresholds = roc_curve(y_arr, p_arr)
    valid_mask = (tpr >= target_sensitivity) & np.isfinite(thresholds)
    if np.any(valid_mask):
        # Highest threshold that achieves tpr >= target_sensitivity maximizes specificity at that sensitivity
        chosen = float(np.max(thresholds[valid_mask]))
    else:
        chosen = 0.50

    return round(float(np.clip(chosen, 0.02, 0.98)), 4)


def tune_confirmatory_threshold_oof(
    y_true_train: np.ndarray,
    oof_probs_train: np.ndarray,
    target_specificity: float = 0.92,
) -> float:
    """
    Tune a secondary 'Confirmatory Mode' (high-specificity) threshold strictly on
    Out-of-Fold (OOF) training predictions that achieves specificity >= `target_specificity`
    (default 92% specificity, i.e., FPR <= 0.08) while maximizing sensitivity/F1.
    Never uses the test set.
    """
    y_arr = np.asarray(y_true_train).astype(int)
    p_arr = np.asarray(oof_probs_train).astype(float)

    fpr, tpr, thresholds = roc_curve(y_arr, p_arr)
    max_fpr = 1.0 - target_specificity
    valid_mask = (fpr <= max_fpr) & (tpr > 0.10) & np.isfinite(thresholds)
    if np.any(valid_mask):
        # Lowest threshold that keeps FPR <= max_fpr maximizes sensitivity at >=92% specificity
        chosen = float(np.min(thresholds[valid_mask]))
    else:
        chosen = 0.50

    return round(float(np.clip(chosen, 0.05, 0.98)), 4)


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstraps: int = 400,
    seed: int = RANDOM_SEED,
) -> Dict[str, Dict[str, float]]:
    """
    Compute 95% non-parametric bootstrap confidence intervals on the held-out test set
    for ROC-AUC, Sensitivity (Recall), and Specificity.
    """
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    n = len(y_true_arr)

    aucs, sens_list, spec_list = [], [], []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yb = y_true_arr[idx]
        pb = y_prob_arr[idx]
        if len(np.unique(yb)) < 2:
            continue
        pred_b = (pb >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yb, pred_b, labels=[0, 1]).ravel()
        aucs.append(float(roc_auc_score(yb, pb)))
        sens_list.append(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0)
        spec_list.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0)

    def _ci_dict(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"lower": 0.0, "upper": 0.0}
        return {
            "lower": round(float(np.percentile(arr, 2.5)), 4),
            "upper": round(float(np.percentile(arr, 97.5)), 4),
        }

    return {
        "auc_ci_95": _ci_dict(aucs),
        "sensitivity_ci_95": _ci_dict(sens_list),
        "specificity_ci_95": _ci_dict(spec_list),
    }


def compute_per_diagnosis_group_error_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    diagnosis_groups: pd.Series,
    threshold: float,
) -> List[Dict[str, Any]]:
    """
    Analyze classification errors broken down by the true underlying Zenodo diagnosis group
    (`1_SAR`, `o-ILD`, `IPF`, `CTD`, `HP`, `u-ILD`, `i-NSIP`) on the held-out test set.
    """
    df_err = pd.DataFrame(
        {
            "group": diagnosis_groups.astype(str).str.strip().values,
            "y_true": np.asarray(y_true).astype(int),
            "y_prob": np.asarray(y_prob).astype(float),
        }
    )
    df_err["y_pred"] = (df_err["y_prob"] >= threshold).astype(int)

    report = []
    for grp, sub in df_err.groupby("group"):
        total = int(len(sub))
        pred_ipf = int((sub["y_pred"] == 1).sum())
        pred_non_ipf = total - pred_ipf
        mean_prob = round(float(sub["y_prob"].mean()), 4)

        if grp == "IPF":
            error_count = pred_non_ipf
            error_type = "False Negative (Missed IPF)"
        else:
            error_count = pred_ipf
            error_type = "False Positive (Misclassified as IPF)"

        error_rate_pct = round((error_count / total) * 100.0, 2) if total > 0 else 0.0
        report.append(
            {
                "group": str(grp),
                "full_name": DIAGNOSIS_GROUP_DESCRIPTIONS.get(str(grp), str(grp)),
                "is_ipf": grp == "IPF",
                "total_test_cases": total,
                "predicted_ipf_count": pred_ipf,
                "predicted_non_ipf_count": pred_non_ipf,
                "error_type": error_type,
                "error_count": error_count,
                "error_rate_pct": error_rate_pct,
                "mean_predicted_ipf_prob": mean_prob,
            }
        )

    report.sort(key=lambda x: (0 if x["is_ipf"] else 1, -x["error_rate_pct"]))
    return report


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute point metrics and confusion matrix at a specific operating threshold."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    y_pred = (y_prob_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
    sensitivity = float(recall_score(y_true_arr, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    accuracy = float(accuracy_score(y_true_arr, y_pred))

    return {
        "threshold": round(float(threshold), 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(sensitivity, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "accuracy": round(accuracy, 4),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "total": int(tp + fp + fn + tn),
        },
    }


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    confirmatory_threshold: Optional[float] = None,
    model_name: str = "Model",
    diagnosis_groups: Optional[pd.Series] = None,
    compute_ci: bool = True,
) -> Dict[str, Any]:
    """
    Compute comprehensive binary classification metrics on the held-out test set
    at the unified 80%-OOF-sensitivity screening threshold (and optional high-specificity
    confirmatory threshold), plus threshold-independent ROC-AUC and PR-AUC.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    primary_metrics = evaluate_at_threshold(y_true_arr, y_prob_arr, threshold)

    conf_thresh = confirmatory_threshold if confirmatory_threshold is not None else threshold
    confirmatory_metrics = evaluate_at_threshold(y_true_arr, y_prob_arr, conf_thresh)

    fpr_arr, tpr_arr, _ = roc_curve(y_true_arr, y_prob_arr)
    fpr_grid = np.array(
        [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 0.90, 1.0]
    )
    tpr_interp = np.interp(fpr_grid, fpr_arr, tpr_arr)
    tpr_interp[0] = 0.0
    tpr_interp[-1] = 1.0

    prec_arr, rec_arr, _ = precision_recall_curve(y_true_arr, y_prob_arr)
    rec_sorted = rec_arr[::-1]
    prec_sorted = prec_arr[::-1]
    recall_grid = np.linspace(0.0, 1.0, 15)
    prec_interp = np.interp(recall_grid, rec_sorted, prec_sorted)

    ci_info = (
        compute_bootstrap_ci(y_true_arr, y_prob_arr, threshold=threshold)
        if compute_ci
        else {}
    )
    conf_ci_info = (
        compute_bootstrap_ci(y_true_arr, y_prob_arr, threshold=conf_thresh)
        if compute_ci
        else {}
    )

    group_errors_screening = (
        compute_per_diagnosis_group_error_analysis(
            y_true_arr, y_prob_arr, diagnosis_groups, threshold=threshold
        )
        if diagnosis_groups is not None
        else []
    )
    group_errors_confirmatory = (
        compute_per_diagnosis_group_error_analysis(
            y_true_arr, y_prob_arr, diagnosis_groups, threshold=conf_thresh
        )
        if diagnosis_groups is not None
        else []
    )

    return {
        "model": model_name,
        "threshold_rule": "80% Target Sensitivity on Out-of-Fold (OOF) Training Predictions",
        "threshold": round(float(threshold), 4),
        "oof_tuned_threshold": round(float(threshold), 4),
        "confirmatory_threshold": round(float(conf_thresh), 4),
        "auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1": primary_metrics["f1"],
        "precision": primary_metrics["precision"],
        "recall": primary_metrics["recall"],
        "sensitivity": primary_metrics["sensitivity"],
        "specificity": primary_metrics["specificity"],
        "accuracy": primary_metrics["accuracy"],
        "brier_score": round(brier, 4),
        "bootstrap_ci_95": ci_info,
        "confusion_matrix": primary_metrics["confusion_matrix"],
        "operating_modes": {
            "screening": {
                "description": "High-Sensitivity Screening Mode (Tuned for 80% OOF Training Sensitivity)",
                **primary_metrics,
                "bootstrap_ci_95": ci_info,
                "per_group_error_analysis": group_errors_screening,
            },
            "confirmatory": {
                "description": "High-Specificity Confirmatory Mode (Tuned for ≥92% OOF Training Specificity)",
                **confirmatory_metrics,
                "bootstrap_ci_95": conf_ci_info,
                "per_group_error_analysis": group_errors_confirmatory,
            },
        },
        "per_group_error_analysis": group_errors_screening,
        "roc_curve": [
            {"fpr": round(float(f), 3), "tpr": round(float(t), 4)}
            for f, t in zip(fpr_grid, tpr_interp)
        ],
        "pr_curve": [
            {"recall": round(float(r), 3), "precision": round(float(p), 4)}
            for r, p in zip(recall_grid, prec_interp)
        ],
    }


def plot_roc_and_pr_curves(
    model_evals: Dict[str, Dict[str, Any]],
    save_path: Path = PLOTS_DIR / "roc_pr_curves.png",
) -> Path:
    """Save side-by-side ROC and Precision-Recall curves evaluated on the held-out test set."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=140)

    ax_roc = axes[0]
    for name, metrics in model_evals.items():
        pts = metrics["roc_curve"]
        fpr = [p["fpr"] for p in pts]
        tpr = [p["tpr"] for p in pts]
        ax_roc.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {metrics['auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Chance")
    ax_roc.set_title("Held-Out Test ROC Curve (Real Zenodo ILD)", fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)")
    ax_roc.legend(loc="lower right", fontsize=8.5)
    ax_roc.grid(alpha=0.25)

    ax_pr = axes[1]
    for name, metrics in model_evals.items():
        pts = metrics["pr_curve"]
        rec = [p["recall"] for p in pts]
        prec = [p["precision"] for p in pts]
        ax_pr.plot(rec, prec, lw=2.2, label=f"{name} (PR-AUC = {metrics['pr_auc']:.3f})")
    ax_pr.set_title("Held-Out Test Precision-Recall Curve (11.1% IPF)", fontweight="bold")
    ax_pr.set_xlabel("Recall (Sensitivity)")
    ax_pr.set_ylabel("Precision (PPV)")
    ax_pr.legend(loc="upper right", fontsize=8.5)
    ax_pr.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved ROC and PR curves to {save_path}")
    return save_path


def plot_calibration_curve_comparison(
    y_true_test: np.ndarray,
    prob_uncalibrated_test: np.ndarray,
    prob_calibrated_test: np.ndarray,
    model_name: str,
    save_path: Path = PLOTS_DIR / "calibration_curve.png",
) -> Dict[str, Any]:
    """
    Evaluate probability calibration strictly on the untouched 20% test set.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    prob_true_uncal, prob_pred_uncal = calibration_curve(
        y_true_test, prob_uncalibrated_test, n_bins=8, strategy="uniform"
    )
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_true_test, prob_calibrated_test, n_bins=8, strategy="uniform"
    )

    brier_uncal = round(float(brier_score_loss(y_true_test, prob_uncalibrated_test)), 4)
    brier_cal = round(float(brier_score_loss(y_true_test, prob_calibrated_test)), 4)

    fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=140)
    ax.plot([0, 1], [0, 1], "k--", lw=1.4, label="Perfectly Calibrated")
    ax.plot(
        prob_pred_uncal,
        prob_true_uncal,
        "s-",
        color="#f59e0b",
        lw=2,
        label=f"{model_name} Uncalibrated (Test Brier = {brier_uncal:.4f})",
    )
    ax.plot(
        prob_pred_cal,
        prob_true_cal,
        "o-",
        color="#0284c7",
        lw=2.4,
        label=f"{model_name} Calibrated (Test Brier = {brier_cal:.4f})",
    )
    ax.set_title(f"Untouched Test-Set Calibration Curve — {model_name}", fontweight="bold")
    ax.set_xlabel("Mean Predicted IPF Probability")
    ax.set_ylabel("Observed Fraction of IPF Cases")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved test-set calibration curve plot to {save_path}")

    return {
        "test_brier_uncalibrated": brier_uncal,
        "test_brier_calibrated": brier_cal,
        "methodology_note": (
            "CalibratedClassifierCV(cv=5) was fit on the training split (also used for tuning); "
            "Brier scores and calibration curve coordinates are evaluated strictly on the untouched 20% test set."
        ),
        "uncalibrated": [
            {"pred": round(float(p), 4), "true": round(float(t), 4)}
            for p, t in zip(prob_pred_uncal, prob_true_uncal)
        ],
        "calibrated": [
            {"pred": round(float(p), 4), "true": round(float(t), 4)}
            for p, t in zip(prob_pred_cal, prob_true_cal)
        ],
    }


def _wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    """Compute 95% Wilson score interval for a binomial proportion k / n."""
    if n <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def compute_calibration_reliability_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bin_edges: Optional[List[float]] = None,
    source_label: str = "untouched_test_set",
) -> Dict[str, Any]:
    """
    Compute empirical IPF rates per calibrated probability bin on held-out test
    (or out-of-fold) predictions so each patient's score can be grounded against
    actual observed IPF prevalence in the same probability bin.
    """
    edges = list(bin_edges) if bin_edges else [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0]
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(y_prob).astype(float)

    bins: List[Dict[str, Any]] = []
    n_bins = len(edges) - 1
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        is_last = i == n_bins - 1
        if is_last:
            mask = (p_arr >= lo) & (p_arr <= hi + 1e-12)
            pct_label = f"≥{int(round(lo * 100))}%" if hi >= 1.0 else f"{int(round(lo * 100))}%–{int(round(hi * 100))}%"
            bin_label = f"{lo:.2f}–{hi:.2f}"
        else:
            mask = (p_arr >= lo) & (p_arr < hi)
            pct_label = f"{int(round(lo * 100))}%–{int(round(hi * 100))}%"
            bin_label = f"{lo:.2f}–{hi:.2f}"

        n_bin = int(mask.sum())
        n_ipf = int(y_arr[mask].sum()) if n_bin > 0 else 0
        rate = (n_ipf / n_bin) if n_bin > 0 else None
        mean_pred = float(p_arr[mask].mean()) if n_bin > 0 else None
        ci_lo, ci_hi = _wilson_ci_95(n_ipf, n_bin) if n_bin > 0 else (0.0, 0.0)

        bins.append(
            {
                "bin_index": i,
                "bin_label": bin_label,
                "pct_range_label": pct_label,
                "prob_min": round(lo, 4),
                "prob_max": round(hi, 4),
                "inclusive_upper": is_last,
                "n_patients": n_bin,
                "n_ipf": n_ipf,
                "empirical_ipf_rate": round(float(rate), 4) if rate is not None else None,
                "empirical_ipf_pct": round(float(rate * 100.0), 1) if rate is not None else None,
                "mean_predicted_prob": round(mean_pred, 4) if mean_pred is not None else None,
                "wilson_ci_95_lower_pct": round(ci_lo * 100.0, 1) if n_bin > 0 else None,
                "wilson_ci_95_upper_pct": round(ci_hi * 100.0, 1) if n_bin > 0 else None,
            }
        )

    total_n = int(len(y_arr))
    total_ipf = int(y_arr.sum())
    return {
        "source": source_label,
        "description": (
            "Empirical IPF prevalence across calibrated probability bins computed on the "
            "untouched 20% Zenodo held-out test split (n=1,362; 151 IPF) for the deployed "
            "calibrated XGBoost classifier."
        ),
        "total_patients": total_n,
        "total_ipf": total_ipf,
        "overall_ipf_rate_pct": round((total_ipf / max(1, total_n)) * 100.0, 2),
        "observed_prob_min": round(float(p_arr.min()), 4) if total_n > 0 else 0.0,
        "observed_prob_max": round(float(p_arr.max()), 4) if total_n > 0 else 0.0,
        "bin_edges": [round(float(e), 4) for e in edges],
        "bins": bins,
    }
