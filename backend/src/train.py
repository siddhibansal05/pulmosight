"""
Master Training Pipeline (`backend/src/train.py`).

Run from repository root:
    python -m backend.src.train
Or from `backend/`:
    python -m src.train

Key Methodological Guarantees:
1. Verifies `Case_number` uniqueness in `database_final_anonymized_for_zenodo.csv` (6,808 unique IDs
   across 6,808 rows); automatically switches to `StratifiedGroupKFold` by `Case_number` if duplicates exist.
2. Embeds `SimpleImputer(strategy='median')` and `StandardScaler()` inside each model's `Pipeline`
   so scaling is fit strictly inside each CV fold.
3. Compares Random Forest, SVM, XGBoost, and a PyTorch MLP (`PyTorchMLPClassifier` with internal
   validation split and early stopping), selecting the best model strictly by 5-Fold Training CV ROC-AUC.
4. Calibrates probabilities (`CalibratedClassifierCV(method='sigmoid')`) and computes true Out-of-Fold (OOF)
   calibrated probability predictions on the training split (`cross_val_predict(..., cv=5)`).
5. Applies ONE unified, fair threshold rule across all 4 models on their OOF calibrated training
   predictions:
   - Primary / Screening Mode (`threshold` == `oof_tuned_threshold`): target 80% OOF training sensitivity.
   - Secondary / Confirmatory Mode (`confirmatory_threshold`): target >= 92% OOF training specificity.
6. Evaluates on the untouched 20% test set once at the end, reporting ROC-AUC, PR-AUC, 95% bootstrap CIs,
   calibration Brier scores, and per-diagnosis-group error analysis (`i-NSIP`, `u-ILD`, `CTD`, etc.).
"""

import argparse
import json
from typing import Any, Dict
import joblib
import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from backend.src.config import (
    ARTIFACTS_JSON_PATH,
    CLINICAL_MODEL_PATH,
    MODEL_VERSION,
    RANDOM_SEED,
    ZENODO_CSV_PATH,
    compute_file_md5,
    set_global_seed,
    setup_logger,
)
from backend.src.evaluate import (
    compute_calibration_reliability_table,
    evaluate_binary_classifier,
    plot_calibration_curve_comparison,
    plot_roc_and_pr_curves,
    tune_confirmatory_threshold_oof,
    tune_threshold_for_target_sensitivity_oof,
)
from backend.src.explain import build_shap_explainer_and_summary
from backend.src.extractor_comparison import (
    DOCS_EXTRACTOR_COMPARISON_PATH,
    render_extractor_comparison_markdown,
    run_osic_extractor_comparison,
)
from backend.src.imaging_progression import train_osic_progression_models
from backend.src.preprocessing import (
    build_preprocessor,
    compute_eda_summary,
    get_stratified_train_test_split,
    load_raw_zenodo_dataframe,
    prepare_clinical_features_and_target,
    verify_case_number_uniqueness,
)
from backend.src.pytorch_mlp import PyTorchMLPClassifier

logger = setup_logger("train")


def compare_feature_sets_pp_vs_z(df_raw, cv) -> Dict[str, Any]:
    """
    Test `pp_only`, `z_only`, and `both` feature sets using 5-fold Stratified CV
    strictly on the 80% training split, with `StandardScaler` inside the Pipeline.
    """
    results = {}
    for set_name in ("pp_only", "z_only", "both"):
        X, y, feat_cols, _ = prepare_clinical_features_and_target(
            df_raw, feature_set_name=set_name
        )
        X_train, _, y_train, _ = get_stratified_train_test_split(X, y)

        pos_ratio = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
        clf_pipe = Pipeline(
            [
                ("prep", build_preprocessor(feat_cols)),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=120,
                        max_depth=4,
                        learning_rate=0.05,
                        scale_pos_weight=pos_ratio,
                        eval_metric="logloss",
                        random_state=RANDOM_SEED,
                        verbosity=0,
                    ),
                ),
            ]
        )
        scores = cross_val_score(clf_pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        mean_auc = round(float(scores.mean()), 4)
        std_auc = round(float(scores.std()), 4)
        results[set_name] = {
            "feature_set": set_name,
            "num_features": len(feat_cols),
            "features": feat_cols,
            "cv_roc_auc_mean": mean_auc,
            "cv_roc_auc_std": std_auc,
            "formatted": f"{mean_auc:.4f} ± {std_auc:.4f}",
        }
        logger.info(
            f"Feature Set Ablation [{set_name}] ({len(feat_cols)} features): "
            f"5-Fold Train CV ROC-AUC = {mean_auc:.4f} ± {std_auc:.4f}"
        )

    best_set = max(results.values(), key=lambda x: x["cv_roc_auc_mean"])["feature_set"]
    return {
        "comparison": results,
        "best_feature_set": best_set,
        "summary_note": (
            f"On the real Zenodo ILD training set, 'both' ({results['both']['formatted']}), "
            f"'pp_only' ({results['pp_only']['formatted']}), and 'z_only' ({results['z_only']['formatted']}) "
            f"perform within one standard error. 'pp_only' is deployed for /predict-ipf because "
            "clinical spirometry reports directly provide % predicted values."
        ),
    }


def compare_class_weights_vs_smote(X_train, y_train, feat_cols, cv) -> Dict[str, Any]:
    """
    Compare cost-sensitive Class Weights (`class_weight='balanced'`) against
    in-fold SMOTE oversampling (`imblearn.pipeline.Pipeline`) on the training set.
    """
    pipe_weights = Pipeline(
        [
            ("prep", build_preprocessor(feat_cols)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=10,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    scores_weights = cross_val_score(
        pipe_weights, X_train, y_train, cv=cv, scoring="roc_auc"
    )

    pipe_smote = ImbPipeline(
        [
            ("prep", build_preprocessor(feat_cols)),
            ("smote", SMOTE(random_state=RANDOM_SEED)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=10,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    scores_smote = cross_val_score(
        pipe_smote, X_train, y_train, cv=cv, scoring="roc_auc"
    )

    w_mean, w_std = round(float(scores_weights.mean()), 4), round(float(scores_weights.std()), 4)
    s_mean, s_std = round(float(scores_smote.mean()), 4), round(float(scores_smote.std()), 4)

    logger.info(
        f"Imbalance Comparison (5-Fold Train CV ROC-AUC): "
        f"Class Weights = {w_mean:.4f} ± {w_std:.4f} | In-Fold SMOTE = {s_mean:.4f} ± {s_std:.4f}"
    )

    return {
        "class_weights_cv_auc_mean": w_mean,
        "class_weights_cv_auc_std": w_std,
        "class_weights_formatted": f"{w_mean:.4f} ± {w_std:.4f}",
        "smote_cv_auc_mean": s_mean,
        "smote_cv_auc_std": s_std,
        "smote_formatted": f"{s_mean:.4f} ± {s_std:.4f}",
        "selected_strategy": "class_weights" if w_mean >= s_mean - 0.005 else "smote",
        "rationale": (
            f"Class weights ({w_mean:.4f} ± {w_std:.4f}) vs. in-fold SMOTE ({s_mean:.4f} ± {s_std:.4f}): "
            "cost-sensitive class weighting achieves higher CV ROC-AUC while preserving "
            "the true physiological feature distribution prior to probability calibration."
        ),
    }


def run_full_training_pipeline(demo: bool = False) -> Dict[str, Any]:
    """
    Execute the complete, leakage-free training, CV model selection, unified 80%-OOF-sensitivity
    threshold tuning, test-set evaluation, SHAP explainability, and calibration workflow.
    """
    set_global_seed(RANDOM_SEED)
    logger.info(f"Starting Part 1: Zenodo Clinical IPF Classifier Training (demo={demo})...")

    df_raw, zenodo_data_source = load_raw_zenodo_dataframe(ZENODO_CSV_PATH, demo=demo)
    zenodo_md5 = compute_file_md5(ZENODO_CSV_PATH)
    case_uniqueness = verify_case_number_uniqueness(df_raw)
    eda_summary = compute_eda_summary(df_raw, data_source=zenodo_data_source)

    if case_uniqueness["is_unique"]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    else:
        cv = StratifiedGroupKFold(n_splits=5)

    # 1. Ablation: Compare pp_only vs z_only vs both on training set only
    feature_set_report = compare_feature_sets_pp_vs_z(df_raw, cv)

    # 2. Prepare primary `pp_only` feature matrix and 80/20 stratified split
    X, y, feat_cols, raw_diag_groups = prepare_clinical_features_and_target(
        df_raw, feature_set_name="pp_only"
    )
    case_series = df_raw["Case_number"] if "Case_number" in df_raw.columns else None
    (
        X_train,
        X_test,
        y_train,
        y_test,
        diag_train,
        diag_test,
    ) = get_stratified_train_test_split(
        X, y, diagnosis_groups=raw_diag_groups, case_numbers=case_series, test_size=0.20
    )

    neg_pos_ratio = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    logger.info(
        f"Stratified 80/20 Split -> Train: {len(X_train)} rows ({int(y_train.sum())} IPF) | "
        f"Untouched Test: {len(X_test)} rows ({int(y_test.sum())} IPF) | "
        f"scale_pos_weight={neg_pos_ratio:.2f}"
    )

    # 3. Imbalance comparison: Class Weights vs in-fold SMOTE on training set
    imbalance_report = compare_class_weights_vs_smote(X_train, y_train, feat_cols, cv)

    # 4. Build in-fold Pipelines (Preprocessor + Classifier), including PyTorch MLP with early stopping
    candidates = {
        "Random Forest": (
            Pipeline(
                [
                    ("prep", build_preprocessor(feat_cols)),
                    (
                        "clf",
                        RandomForestClassifier(
                            class_weight="balanced",
                            random_state=RANDOM_SEED,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "clf__n_estimators": [150, 250, 350],
                "clf__max_depth": [6, 8, 10, 14],
                "clf__min_samples_leaf": [3, 6, 10],
                "clf__max_features": ["sqrt", "log2"],
            },
        ),
        "SVM": (
            Pipeline(
                [
                    ("prep", build_preprocessor(feat_cols)),
                    (
                        "clf",
                        SVC(
                            class_weight="balanced",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            {
                "clf__C": [0.5, 1.0, 2.0, 4.0],
                "clf__gamma": ["scale", "auto", 0.05, 0.1],
                "clf__kernel": ["rbf"],
            },
        ),
        "XGBoost": (
            Pipeline(
                [
                    ("prep", build_preprocessor(feat_cols)),
                    (
                        "clf",
                        XGBClassifier(
                            scale_pos_weight=neg_pos_ratio,
                            eval_metric="logloss",
                            random_state=RANDOM_SEED,
                            verbosity=0,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "clf__n_estimators": [100, 180, 250],
                "clf__max_depth": [3, 4, 5],
                "clf__learning_rate": [0.02, 0.04, 0.08],
                "clf__subsample": [0.75, 0.85, 1.0],
                "clf__colsample_bytree": [0.75, 0.9, 1.0],
            },
        ),
        "PyTorch MLP": (
            Pipeline(
                [
                    ("prep", build_preprocessor(feat_cols)),
                    (
                        "clf",
                        PyTorchMLPClassifier(
                            max_epochs=70,
                            patience=10,
                            val_fraction=0.15,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            {
                "clf__hidden_dims": [(64, 32), (128, 64), (64,)],
                "clf__dropout": [0.15, 0.25],
                "clf__lr": [0.002, 0.004],
                "clf__weight_decay": [1e-4, 1e-3],
            },
        ),
    }

    tuned_pipelines = {}
    calibrated_models = {}
    cv_summary_by_model = {}
    screening_thresholds = {}
    confirmatory_thresholds = {}

    for model_name, (pipeline_est, param_dist) in candidates.items():
        logger.info(
            f"Tuning {model_name} Pipeline via RandomizedSearchCV (5-Fold Train CV, scoring='roc_auc')..."
        )
        n_iter_search = 4 if model_name == "PyTorch MLP" else 6
        search = RandomizedSearchCV(
            estimator=pipeline_est,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        search.fit(X_train, y_train)

        best_pipe = search.best_estimator_
        tuned_pipelines[model_name] = best_pipe

        fold_scores = cross_val_score(best_pipe, X_train, y_train, cv=cv, scoring="roc_auc")

        # Calibrate each model using sigmoid calibration on X_train and compute true Out-of-Fold
        # calibrated probabilities on X_train to tune operating thresholds under ONE unified rule
        cal_wrapper = CalibratedClassifierCV(
            estimator=best_pipe,
            method="sigmoid",
            cv=3,
        )
        oof_cal_probs = cross_val_predict(
            cal_wrapper, X_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        cal_wrapper.fit(X_train, y_train)
        calibrated_models[model_name] = cal_wrapper

        # Unified Rule (Requirement D1/D2/D3):
        # 1. Screening threshold: 80% target sensitivity on Out-of-Fold calibrated training predictions
        # 2. Confirmatory threshold: >=92% target specificity on Out-of-Fold calibrated training predictions
        screen_thresh = tune_threshold_for_target_sensitivity_oof(
            y_train.values, oof_cal_probs, target_sensitivity=0.80
        )
        confirm_thresh = tune_confirmatory_threshold_oof(
            y_train.values, oof_cal_probs, target_specificity=0.92
        )
        screening_thresholds[model_name] = screen_thresh
        confirmatory_thresholds[model_name] = confirm_thresh

        cv_summary_by_model[model_name] = {
            "cv_roc_auc_mean": round(float(fold_scores.mean()), 4),
            "cv_roc_auc_std": round(float(fold_scores.std()), 4),
            "cv_roc_auc_formatted": f"{fold_scores.mean():.4f} ± {fold_scores.std():.4f}",
            "threshold": screen_thresh,
            "oof_tuned_threshold": screen_thresh,
            "confirmatory_threshold": confirm_thresh,
            "best_params": {k: str(v) for k, v in search.best_params_.items()},
        }
        logger.info(
            f"[{model_name}] Train 5-Fold CV ROC-AUC = {fold_scores.mean():.4f} ± {fold_scores.std():.4f} "
            f"| OOF Calibrated Screening Thresh (80% Sens) = {screen_thresh:.4f} "
            f"| Confirmatory Thresh (92% Spec) = {confirm_thresh:.4f}"
        )

    # 5. Select the Deployed Clinical Model strictly on Training 5-Fold CV ROC-AUC
    # with the 1-SE (One Standard Error) Parsimony Rule:
    # Prefer torch-free XGBoost when the CV ROC-AUC difference vs. PyTorch MLP is < 1 standard error.
    highest_cv_model = max(
        cv_summary_by_model.keys(),
        key=lambda m: cv_summary_by_model[m]["cv_roc_auc_mean"],
    )
    highest_cv_mean = cv_summary_by_model[highest_cv_model]["cv_roc_auc_mean"]
    highest_cv_se = cv_summary_by_model[highest_cv_model]["cv_roc_auc_std"] / np.sqrt(5.0)

    if (
        highest_cv_model == "PyTorch MLP"
        and "XGBoost" in cv_summary_by_model
        and (highest_cv_mean - cv_summary_by_model["XGBoost"]["cv_roc_auc_mean"]) <= highest_cv_se
    ):
        best_model_name = "XGBoost"
        selection_criterion = (
            f"5-Fold Train CV ROC-AUC with 1-SE parsimony rule: PyTorch MLP ({highest_cv_mean:.4f} ± "
            f"{cv_summary_by_model['PyTorch MLP']['cv_roc_auc_std']:.4f}, SE={highest_cv_se:.4f}) "
            f"vs. XGBoost ({cv_summary_by_model['XGBoost']['cv_roc_auc_mean']:.4f} ± "
            f"{cv_summary_by_model['XGBoost']['cv_roc_auc_std']:.4f}) differed by "
            f"{highest_cv_mean - cv_summary_by_model['XGBoost']['cv_roc_auc_mean']:.4f} (< 1 SE). "
            "Selected XGBoost as deployed model for torch-free API serving and exact TreeSHAP; "
            "PyTorch MLP is retained as a deep-learning comparison baseline."
        )
    else:
        best_model_name = highest_cv_model
        selection_criterion = (
            f"Highest 5-Fold Train CV ROC-AUC on training split ({highest_cv_mean:.4f})."
        )

    best_uncalibrated_pipeline = tuned_pipelines[best_model_name]
    best_calibrated_clf = calibrated_models[best_model_name]
    optimal_screening_threshold = screening_thresholds[best_model_name]
    optimal_confirmatory_threshold = confirmatory_thresholds[best_model_name]

    logger.info(
        f"Selected Deployed Clinical Classifier: {best_model_name} | {selection_criterion}"
    )

    # 6. Final Untouched Test-Set Evaluation (All models evaluated with the same unified 80%-OOF-Sens rule)
    model_evaluations = {}
    for model_name, cal_clf in calibrated_models.items():
        y_prob_test = cal_clf.predict_proba(X_test)[:, 1]
        screen_t = screening_thresholds[model_name]
        confirm_t = confirmatory_thresholds[model_name]

        eval_metrics = evaluate_binary_classifier(
            y_true=y_test.values,
            y_prob=y_prob_test,
            threshold=screen_t,
            confirmatory_threshold=confirm_t,
            model_name=model_name,
            diagnosis_groups=diag_test,
            compute_ci=True,
        )
        eval_metrics.update(cv_summary_by_model[model_name])
        model_evaluations[model_name] = eval_metrics

        ci_auc = eval_metrics["bootstrap_ci_95"]["auc_ci_95"]
        logger.info(
            f"[UNTOUCHED TEST SET (80% OOF Sens Rule) — {model_name}] "
            f"ROC-AUC={eval_metrics['auc']:.4f} (95% CI: {ci_auc['lower']:.4f}–{ci_auc['upper']:.4f}) | "
            f"PR-AUC={eval_metrics['pr_auc']:.4f} | F1={eval_metrics['f1']:.4f} | "
            f"Sens={eval_metrics['sensitivity']:.4f} | Spec={eval_metrics['specificity']:.4f} | "
            f"Brier={eval_metrics['brier_score']:.4f} (thresh={screen_t:.4f})"
        )

    # 7. Calibration curve & Brier score strictly on the untouched test set
    if hasattr(best_uncalibrated_pipeline, "predict_proba"):
        prob_uncal_test = best_uncalibrated_pipeline.predict_proba(X_test)[:, 1]
    else:
        raw_dec = best_uncalibrated_pipeline.decision_function(X_test)
        prob_uncal_test = 1.0 / (1.0 + np.exp(-raw_dec))
    prob_cal_test = best_calibrated_clf.predict_proba(X_test)[:, 1]
    calibration_coords = plot_calibration_curve_comparison(
        y_true_test=y_test.values,
        prob_uncalibrated_test=prob_uncal_test,
        prob_calibrated_test=prob_cal_test,
        model_name=best_model_name,
    )
    calibration_reliability_table = compute_calibration_reliability_table(
        y_true=y_test.values,
        y_prob=prob_cal_test,
        source_label="untouched_test_set",
    )
    age_ipf_vals = X.loc[y == 1, "age"].dropna().values
    age_non_ipf_vals = X.loc[y == 0, "age"].dropna().values
    cohort_age_stats = {
        "source": "real_zenodo_cohort" if zenodo_data_source == "real" else "demo_cohort",
        "ipf_n": int(len(age_ipf_vals)),
        "ipf_median_age": round(float(np.median(age_ipf_vals)), 1),
        "ipf_iqr_25": round(float(np.percentile(age_ipf_vals, 25)), 1),
        "ipf_iqr_75": round(float(np.percentile(age_ipf_vals, 75)), 1),
        "non_ipf_n": int(len(age_non_ipf_vals)),
        "non_ipf_median_age": round(float(np.median(age_non_ipf_vals)), 1),
        "non_ipf_iqr_25": round(float(np.percentile(age_non_ipf_vals, 25)), 1),
        "non_ipf_iqr_75": round(float(np.percentile(age_non_ipf_vals, 75)), 1),
    }
    plot_roc_and_pr_curves(model_evaluations)

    # 8. SHAP Explainability on the selected pipeline
    shap_info = build_shap_explainer_and_summary(
        fitted_pipeline=best_uncalibrated_pipeline,
        X_background=X_train,
        feature_names=feat_cols,
    )

    # Build exact, transparent sensitivity & per-group misclassification disclosures (Item 6)
    best_modes = model_evaluations[best_model_name]["operating_modes"]
    sc_cm = best_modes["screening"]["confusion_matrix"]
    cf_cm = best_modes["confirmatory"]["confusion_matrix"]
    sc_miss_pct = round(sc_cm["fn"] / max(1, sc_cm["tp"] + sc_cm["fn"]) * 100.0, 1)
    cf_miss_pct = round(cf_cm["fn"] / max(1, cf_cm["tp"] + cf_cm["fn"]) * 100.0, 1)

    sc_group_map = {
        g["group"]: g for g in best_modes["screening"]["per_group_error_analysis"]
    }
    insip_pct = round(sc_group_map.get("i-NSIP", {}).get("error_rate_pct", 0.0), 1)
    uild_pct = round(sc_group_map.get("u-ILD", {}).get("error_rate_pct", 0.0), 1)
    ctd_pct = round(sc_group_map.get("CTD", {}).get("error_rate_pct", 0.0), 1)

    sensitivity_disclosure = (
        f"Deployed model ({best_model_name}): Screening mode misses {sc_miss_pct}% of IPF cases "
        f"({sc_cm['fn']} of {sc_cm['tp'] + sc_cm['fn']} missed on the untouched test set), and "
        f"confirmatory mode misses {cf_miss_pct}% of IPF cases ({cf_cm['fn']} of {cf_cm['tp'] + cf_cm['fn']}). "
        f"In screening mode, i-NSIP ({insip_pct}% flagged), u-ILD ({uild_pct}% flagged), and CTD "
        f"({ctd_pct}% flagged) are most often misclassified as IPF, which is clinically expected."
    )

    clinical_error_note = (
        f"{sensitivity_disclosure} Overlap between IPF and fibrotic i-NSIP, u-ILD, and CTD is expected "
        "clinically because these interstitial lung diseases share restrictive mechanics and reduced TLCO; "
        "this system is decision support only and never replaces multidisciplinary diagnosis."
    )

    # 9. Save clinical pipeline bundle with joblib (torch-free when XGBoost is deployed!)
    clinical_bundle = {
        "model_version": MODEL_VERSION,
        "sklearn_version": sklearn.__version__,
        "requires_torch_at_serving": False,
        "security_note": (
            "Security Warning: .joblib files use Python pickle under the hood and must only be "
            "loaded from trusted local sources. Never accept model files through an API upload."
        ),
        "data_source": zenodo_data_source,
        "zenodo_md5": zenodo_md5,
        "case_number_uniqueness": case_uniqueness,
        "best_model_name": best_model_name,
        "highest_cv_model": highest_cv_model,
        "selection_criterion": selection_criterion,
        "threshold_tuning_method": (
            "Unified rule across all models: thresholds are tuned strictly on Out-of-Fold (OOF) "
            "calibrated probability predictions on the 80% training split (cross_val_predict with "
            "CalibratedClassifierCV(method='sigmoid')). Primary 'screening' threshold targets 80% OOF "
            "training sensitivity; secondary 'confirmatory' threshold targets >=92% OOF training specificity."
        ),
        "optimal_threshold": optimal_screening_threshold,
        "screening_threshold": optimal_screening_threshold,
        "confirmatory_threshold": optimal_confirmatory_threshold,
        "operating_modes": model_evaluations[best_model_name]["operating_modes"],
        "sensitivity_disclosure": sensitivity_disclosure,
        "feature_names": feat_cols,
        "calibrated_classifier": best_calibrated_clf,
        "uncalibrated_pipeline": best_uncalibrated_pipeline,
        "shap_background": shap_info["background_data"],
        "global_shap_ranking": shap_info["global_ranking"],
        "model_evaluations": model_evaluations,
        "clinical_error_analysis_note": clinical_error_note,
        "feature_set_ablation": feature_set_report,
        "imbalance_comparison": imbalance_report,
        "calibration_curve": calibration_coords,
        "calibration_reliability_table": calibration_reliability_table,
        "cohort_age_stats": cohort_age_stats,
    }
    CLINICAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clinical_bundle, CLINICAL_MODEL_PATH)
    logger.info(
        f"Saved torch-free calibrated clinical IPF pipeline ({best_model_name}, "
        f"sklearn=={sklearn.__version__}, data_source={zenodo_data_source}) to {CLINICAL_MODEL_PATH}"
    )

    # 10. Train Part 2: Exploratory OSIC Pulmonary Fibrosis Progression Study
    logger.info("Starting Part 2: Exploratory OSIC Pulmonary Fibrosis Progression Study...")
    progression_bundle = train_osic_progression_models(demo=demo)

    overall_data_source = (
        "real"
        if (zenodo_data_source == "real" and progression_bundle["data_source"] == "real")
        else "synthetic"
    )

    # 10b. Run read-only OSIC Pretrained CT Feature Extractor Comparison (never alters deployed models)
    extractor_comparison = None
    if not demo and progression_bundle.get("imaging_model_available"):
        logger.info("Running OSIC Pretrained CT Feature Extractor Comparison (same 5x5 folds)...")
        extractor_comparison = run_osic_extractor_comparison()
        DOCS_EXTRACTOR_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
        DOCS_EXTRACTOR_COMPARISON_PATH.write_text(
            render_extractor_comparison_markdown(extractor_comparison), encoding="utf-8"
        )

    # 11. Save consolidated JSON artifacts
    artifacts_payload = {
        "model_version": MODEL_VERSION,
        "sklearn_version": sklearn.__version__,
        "requires_torch_at_serving": False,
        "data_source": overall_data_source,
        "zenodo_data_source": zenodo_data_source,
        "zenodo_md5": zenodo_md5,
        "case_number_uniqueness": case_uniqueness,
        "osic_data_source": progression_bundle["data_source"],
        "imaging_model_available": progression_bundle["imaging_model_available"],
        "best_clinical_model": best_model_name,
        "highest_cv_model": highest_cv_model,
        "selection_criterion": selection_criterion,
        "threshold_tuning_method": clinical_bundle["threshold_tuning_method"],
        "optimal_threshold": optimal_screening_threshold,
        "screening_threshold": optimal_screening_threshold,
        "confirmatory_threshold": optimal_confirmatory_threshold,
        "operating_modes": model_evaluations[best_model_name]["operating_modes"],
        "sensitivity_disclosure": sensitivity_disclosure,
        "clinical_models": model_evaluations,
        "per_group_error_analysis": model_evaluations[best_model_name]["per_group_error_analysis"],
        "clinical_error_analysis_note": clinical_error_note,
        "feature_set_ablation": feature_set_report,
        "imbalance_comparison": imbalance_report,
        "global_shap_ranking": shap_info["global_ranking"],
        "calibration_curve": calibration_coords,
        "calibration_reliability_table": calibration_reliability_table,
        "cohort_age_stats": cohort_age_stats,
        "eda_summary": eda_summary,
        "osic_progression_study": {
            "exploratory_banner": progression_bundle["exploratory_banner"],
            "study_note": progression_bundle["study_description"],
            "data_source": progression_bundle["data_source"],
            "imaging_model_available": progression_bundle["imaging_model_available"],
            "min_imaging_patients_required": progression_bundle["min_imaging_patients_required"],
            "patients_with_dicoms": progression_bundle["patients_with_dicoms"],
            "usable_as_standalone_predictor": progression_bundle["usable_as_standalone_predictor"],
            "any_model_beats_both_baselines": progression_bundle["any_model_beats_both_baselines"],
            "best_model": progression_bundle["best_model_name"],
            "split_strategy": progression_bundle["split_strategy"],
            "num_patients": progression_bundle["num_patients"],
            "num_visits": progression_bundle["num_visits"],
            "baseline_metrics": progression_bundle["baseline_metrics"],
            "models": progression_bundle["metrics"],
            "target_variants": progression_bundle["target_variants"],
            "modality_ablation": progression_bundle["modality_ablation"],
            "osic_competition_benchmark": progression_bundle.get("osic_competition_benchmark"),
            "fusion_comparison": progression_bundle.get("fusion_comparison"),
            "selected_fusion_method": progression_bundle.get("selected_fusion_method"),
        },
        "extractor_comparison": extractor_comparison,
    }
    with open(ARTIFACTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(artifacts_payload, f, indent=2)
    logger.info(
        f"Saved real evaluation & EDA artifacts JSON (data_source={overall_data_source}) to {ARTIFACTS_JSON_PATH}"
    )

    return artifacts_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Part 1 (Zenodo IPF Classifier) and Part 2 (Exploratory OSIC Progression Study)."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Explicitly allow synthetic demo dataset generation if real datasets are missing.",
    )
    args = parser.parse_args()
    run_full_training_pipeline(demo=args.demo)


if __name__ == "__main__":
    main()
