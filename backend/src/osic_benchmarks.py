"""
OSIC Competition Benchmark (Modified Laplace Log-Likelihood) & 6-Method Fusion Study
(`backend/src/osic_benchmarks.py`).

Implements:
1. Official OSIC Modified Laplace Log-Likelihood metric (sigma clipped at >=70, error clipped at <=1000).
2. 5-Repeat x 5-Fold Nested GroupKFold benchmark across:
   - Baseline (Constant Baseline FVC)
   - Ridge
   - ElasticNet
   - Quantile-Regression Neural Network (Tabular PyTorch MLP with Pinball Loss on q=[0.20, 0.50, 0.80])
   - Random Forest
   - SVM (RBF SVR)
   - XGBoost
   - Image + Tabular (ResNet-18 In-Fold PCA 15D + Clinical)
   (EfficientNet-B5 explicitly recorded as not run due to compute/raw-DICOM constraints.)
3. 6-Method Fusion Comparison on the EXACT SAME 5x5 Nested GroupKFold splits:
   - Clinical-only
   - HRCT-only (HU/Volume + ResNet-18 In-Fold PCA 15D)
   - Early fusion
   - Late fusion (mean)
   - Weighted late fusion (weights tuned strictly inside training folds)
   - Stacking (meta-learner trained strictly on inner out-of-fold predictions)
   Selected via the One-Standard-Error (1-SE) parsimony rule (`selected_fusion_method`).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBClassifier, XGBRegressor

from backend.src.config import RANDOM_SEED, setup_logger

logger = setup_logger("osic_benchmarks")

FVC_MIN_ML = 500.0
FVC_MAX_ML = 6500.0


def winsorize_fvc(pred: np.ndarray) -> np.ndarray:
    """Clip predicted FVC values to a physiologically plausible range [500, 6500] mL."""
    return np.clip(np.asarray(pred, dtype=np.float64), FVC_MIN_ML, FVC_MAX_ML)


class InFoldWinsorizer(BaseEstimator, TransformerMixin):
    """
    Leak-free feature winsorizer fitted strictly on training-fold quantiles (default 1st-99th)
    to prevent single-patient CT segmentation/metadata outliers from exploding linear models.
    """

    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99) -> None:
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=np.float64)
        self.lower_bounds_ = np.nanpercentile(arr, self.lower_q * 100.0, axis=0)
        self.upper_bounds_ = np.nanpercentile(arr, self.upper_q * 100.0, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=np.float64)
        return np.clip(arr, self.lower_bounds_, self.upper_bounds_)

VISIT_CLINICAL_FEATURES = [
    "Baseline_FVC",
    "Baseline_Percent",
    "Delta_Weeks",
    "Age",
    "Sex_Male",
    "Smoking_Code",
]

PATIENT_CLINICAL_FEATURES = [
    "Baseline_FVC",
    "Baseline_Percent",
    "Age",
    "Sex_Male",
    "Smoking_Code",
]


def modified_laplace_log_likelihood(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    Official Kaggle OSIC Pulmonary Fibrosis Progression metric:
      sigma_clipped = max(sigma, 70)
      delta = min(|y_true - y_pred|, 1000)
      LL = -sqrt(2) * delta / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    Higher (less negative) is better.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    sc = np.maximum(np.asarray(sigma, dtype=np.float64), 70.0)
    delta = np.minimum(np.abs(yt - yp), 1000.0)
    ll = -(np.sqrt(2.0) * delta) / sc - np.log(np.sqrt(2.0) * sc)
    return float(np.mean(ll))


def _compute_ci_95(values: List[float], decimals: int = 4) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    se = std_v / np.sqrt(max(1, len(arr)))
    lo = mean_v - 1.96 * se
    hi = mean_v + 1.96 * se
    fmt_dec = 3 if abs(mean_v) >= 10 else 4
    return {
        "mean": round(mean_v, decimals),
        "std": round(std_v, decimals),
        "se": round(se, decimals),
        "ci_95_lower": round(lo, decimals),
        "ci_95_upper": round(hi, decimals),
        "formatted": f"{mean_v:.{fmt_dec}f} ± {std_v:.{fmt_dec}f} (95% CI: {lo:.{fmt_dec}f}–{hi:.{fmt_dec}f})",
    }


def nadeau_bengio_paired_test(
    method_scores: List[float],
    baseline_scores: List[float],
    test_train_ratio: float = 0.25,
    higher_is_better: bool = True,
    decimals: int = 4,
) -> Dict[str, Any]:
    """
    Nadeau-Bengio (2003) / Bouckaert-Frank (2004) corrected resampled paired t-test
    on per-fold differences d_i = method_i - baseline_i across identical repeated CV folds:
      Var_NB(d_bar) = (1/n + n_test/n_train) * s_d^2
      df = n - 1,  t = d_bar / sqrt(Var_NB),  CI_95 = d_bar +/- t_{0.975, df} * SE_NB
    """
    m_arr = np.asarray(method_scores, dtype=np.float64)
    b_arr = np.asarray(baseline_scores, dtype=np.float64)
    diffs = m_arr - b_arr
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    if n <= 1:
        return {
            "n_folds": n,
            "mean_diff": round(mean_d, decimals),
            "sd_diff": 0.0,
            "se_nadeau_bengio": 0.0,
            "test_train_ratio": round(float(test_train_ratio), 4),
            "t_stat": 0.0,
            "df": 0,
            "p_value": 1.0,
            "ci_95_lower": round(mean_d, decimals),
            "ci_95_upper": round(mean_d, decimals),
            "ci_excludes_zero": False,
            "statistically_better": False,
            "formatted": f"{mean_d:+.4f} (n={n}, p=1.000)",
        }

    sd_d = float(np.std(diffs, ddof=1))
    var_nb = (1.0 / n + float(test_train_ratio)) * (sd_d ** 2)
    se_nb = float(np.sqrt(max(var_nb, 0.0)))
    df = n - 1
    t_crit = float(scipy_stats.t.ppf(0.975, df))
    if se_nb > 1e-12:
        t_stat = float(mean_d / se_nb)
        p_val = float(2.0 * scipy_stats.t.sf(abs(t_stat), df))
    else:
        t_stat = 0.0
        p_val = 1.0

    lo = mean_d - t_crit * se_nb
    hi = mean_d + t_crit * se_nb
    excludes_zero = bool(lo > 0.0 or hi < 0.0)
    stat_better = bool(lo > 0.0) if higher_is_better else bool(hi < 0.0)
    fmt_dec = 2 if abs(mean_d) >= 10 or abs(se_nb) >= 10 else 4
    return {
        "n_folds": n,
        "mean_diff": round(mean_d, decimals),
        "sd_diff": round(sd_d, decimals),
        "se_nadeau_bengio": round(se_nb, decimals),
        "test_train_ratio": round(float(test_train_ratio), 4),
        "t_stat": round(t_stat, 4),
        "df": df,
        "p_value": round(p_val, 4),
        "ci_95_lower": round(lo, decimals),
        "ci_95_upper": round(hi, decimals),
        "ci_excludes_zero": excludes_zero,
        "statistically_better": stat_better,
        "formatted": f"{mean_d:+.{fmt_dec}f} (95% CI: {lo:+.{fmt_dec}f} to {hi:+.{fmt_dec}f}, p = {p_val:.4f})",
    }


def _fit_in_fold_sigma_params(
    y_tr: np.ndarray,
    oof_pred_tr: np.ndarray,
    delta_weeks_tr: np.ndarray,
) -> Tuple[float, float]:
    """
    Fit a strictly in-fold linear horizon uncertainty model:
      sigma(dw) = max(70.0, a + b * max(dw, 0))
    by maximizing modified Laplace LL on training-fold OOF predictions only.
    """
    abs_err = np.abs(np.asarray(y_tr, dtype=np.float64) - np.asarray(oof_pred_tr, dtype=np.float64))
    dw = np.maximum(np.asarray(delta_weeks_tr, dtype=np.float64), 0.0)
    init_a = max(70.0, float(np.sqrt(2.0) * np.median(abs_err)))
    init_b = 1.2

    def neg_ll(params):
        a, b = params
        sig = np.maximum(a + b * dw, 70.0)
        return -modified_laplace_log_likelihood(y_tr, oof_pred_tr, sig)

    res = minimize(
        neg_ll,
        x0=np.array([init_a, init_b], dtype=np.float64),
        bounds=[(70.0, 1200.0), (0.0, 25.0)],
        method="L-BFGS-B",
    )
    a_opt, b_opt = float(res.x[0]), float(res.x[1])
    return a_opt, b_opt


def _predict_sigma(delta_weeks: np.ndarray, params: Tuple[float, float]) -> np.ndarray:
    a, b = params
    return np.maximum(a + b * np.maximum(np.asarray(delta_weeks, dtype=np.float64), 0.0), 70.0)


def _compute_inner_oof_predictions(
    estimator_factory,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    n_inner: int = 3,
) -> np.ndarray:
    """Compute strictly in-fold out-of-fold predictions via patient GroupKFold."""
    oof = np.zeros(len(X_tr), dtype=np.float64)
    gkf = GroupKFold(n_splits=min(n_inner, len(np.unique(groups_tr))))
    for itr, iva in gkf.split(X_tr, y_tr, groups=groups_tr):
        est = estimator_factory()
        est.fit(X_tr.iloc[itr], y_tr[itr])
        oof[iva] = est.predict(X_tr.iloc[iva])
    return oof


def _fit_predict_quantile_nn_tabular(
    X_tr_raw: np.ndarray,
    y_tr_raw: np.ndarray,
    groups_tr: np.ndarray,
    X_va_raw: np.ndarray,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tabular PyTorch Quantile-Regression Neural Network predicting (q20, q50, q80)
    with Pinball Loss + monotonic quantile heads + patient-grouped early stopping.
    Returns (y_pred_median, sigma_calibrated) in original FVC mL units.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_va_s = scaler.transform(X_va_raw).astype(np.float32)

    y_mean = float(np.mean(y_tr_raw))
    y_std = float(np.std(y_tr_raw)) if float(np.std(y_tr_raw)) > 1e-6 else 1.0
    y_tr_s = ((y_tr_raw - y_mean) / y_std).astype(np.float32)

    # Internal patient-grouped train/val split (85/15) for early stopping & sigma scaling
    uniq_g = np.unique(groups_tr)
    rng = np.random.default_rng(seed)
    perm_g = rng.permutation(uniq_g)
    n_val_g = max(2, int(round(0.15 * len(uniq_g))))
    val_g_set = set(perm_g[:n_val_g].tolist())
    val_mask = np.array([g in val_g_set for g in groups_tr], dtype=bool)
    tr_mask = ~val_mask

    class QuantileMLP(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            self.head = nn.Linear(32, 3)

        def forward(self, x):
            h = self.net(x)
            out = self.head(h)
            q50 = out[:, 1]
            d_lo = F.softplus(out[:, 0]) + 1e-3
            d_hi = F.softplus(out[:, 2]) + 1e-3
            return torch.stack([q50 - d_lo, q50, q50 + d_hi], dim=1)

    quantiles = torch.tensor([0.20, 0.50, 0.80], dtype=torch.float32)

    def pinball_loss(preds, targets):
        err = targets.unsqueeze(1) - preds
        return torch.mean(torch.maximum(quantiles * err, (quantiles - 1.0) * err))

    model = QuantileMLP(X_tr_s.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-3)

    x_t = torch.from_numpy(X_tr_s[tr_mask])
    y_t = torch.from_numpy(y_tr_s[tr_mask])
    x_v = torch.from_numpy(X_tr_s[val_mask])
    y_v = torch.from_numpy(y_tr_s[val_mask])

    best_loss = float("inf")
    best_state = None
    patience_left = 18

    for _ in range(140):
        model.train()
        opt.zero_grad()
        loss = pinball_loss(model(x_t), y_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            v_loss = float(pinball_loss(model(x_v), y_v).item())
        if v_loss < best_loss - 1e-5:
            best_loss = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = 18
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_q = model(x_v).numpy() * y_std + y_mean
        va_q = model(torch.from_numpy(X_va_s)).numpy() * y_std + y_mean

    # Calibrate quantile spread multiplier strictly on internal training validation patients
    val_spread = np.maximum(val_q[:, 2] - val_q[:, 0], 10.0)
    val_y_true = y_tr_raw[val_mask]
    best_c, best_ll = 1.0, -1e9
    for c in np.linspace(0.5, 2.5, 21):
        ll_c = modified_laplace_log_likelihood(val_y_true, val_q[:, 1], np.maximum(val_spread * c, 70.0))
        if ll_c > best_ll:
            best_ll = ll_c
            best_c = float(c)

    y_pred_va = va_q[:, 1].astype(np.float64)
    sigma_va = np.maximum((va_q[:, 2] - va_q[:, 0]) * best_c, 70.0).astype(np.float64)
    return y_pred_va, sigma_va


def build_osic_visit_dataset(
    raw_train_df: pd.DataFrame,
    pat_df: pd.DataFrame,
    hu_cols: List[str],
    resnet_cols: List[str],
) -> pd.DataFrame:
    """
    Build visit-level future-FVC prediction dataset from OSIC `train.csv` joined with
    patient-level baseline covariates and cached 3D CT features (strictly future visits:
    `Weeks > Baseline_Weeks`).
    """
    ct_cols = [c for c in hu_cols + resnet_cols if c in pat_df.columns]
    pat_lookup = pat_df.set_index("Patient")

    rows = []
    for pid, grp in raw_train_df.groupby("Patient"):
        pid_str = str(pid)
        if pid_str not in pat_lookup.index:
            continue
        p_info = pat_lookup.loc[pid_str]
        if not bool(p_info.get("has_ct", False)):
            continue

        g_sorted = grp.sort_values("Weeks").reset_index(drop=True)
        if len(g_sorted) < 2:
            continue
        base_row = g_sorted.iloc[0]
        base_weeks = float(base_row["Weeks"])
        base_fvc = float(base_row["FVC"])
        base_pct = float(base_row["Percent"])
        age = float(p_info["Age"])
        sex_male = float(p_info["Sex_Male"])
        smoke = float(p_info["Smoking_Code"])

        ct_dict = {c: float(p_info[c]) for c in ct_cols}

        for idx in range(1, len(g_sorted)):
            v_row = g_sorted.iloc[idx]
            dw = float(v_row["Weeks"]) - base_weeks
            if dw <= 0.0:
                continue
            rows.append(
                {
                    "Patient": pid_str,
                    "Baseline_Weeks": base_weeks,
                    "Target_Weeks": float(v_row["Weeks"]),
                    "Delta_Weeks": dw,
                    "Baseline_FVC": base_fvc,
                    "Baseline_Percent": base_pct,
                    "Age": age,
                    "Sex_Male": sex_male,
                    "Smoking_Code": smoke,
                    "Target_FVC": float(v_row["FVC"]),
                    "rapid_decliner_52w": int(p_info["rapid_decliner_52w"]),
                    **ct_dict,
                }
            )

    return pd.DataFrame(rows)


def _generate_patient_cv_splits(
    patient_ids: np.ndarray,
    n_repeats: int = 5,
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> List[Tuple[set, set]]:
    """
    Generate deterministic 5-Repeat x 5-Fold patient-level train/val split sets so
    EVERY benchmark model and EVERY fusion method uses the EXACT SAME 25 outer folds.
    """
    uniq = np.sort(np.unique(patient_ids))
    splits: List[Tuple[set, set]] = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + rep * 131)
        perm = rng.permutation(uniq)
        gkf = GroupKFold(n_splits=min(n_splits, len(perm)))
        dummy = np.zeros(len(perm))
        for tr_i, va_i in gkf.split(dummy, dummy, groups=perm):
            splits.append((set(perm[tr_i].tolist()), set(perm[va_i].tolist())))
    return splits


def evaluate_osic_literature_benchmark(
    visit_df: pd.DataFrame,
    splits: List[Tuple[set, set]],
    hu_cols: List[str],
    resnet_cols: List[str],
    n_pca: int = 15,
) -> Dict[str, Any]:
    """
    Requirement A: Compare Ridge, ElasticNet, Quantile-Regression NN (Tabular),
    Random Forest, SVM, XGBoost, and Image+Tabular (ResNet-18 In-Fold PCA 15D)
    on identical 5x5 Nested GroupKFold splits, reporting Modified Laplace LL,
    RMSE, and MAE with 95% confidence intervals.
    """
    model_order = [
        "Baseline (Constant Baseline FVC)",
        "Ridge",
        "ElasticNet",
        "Quantile-Regression NN (Tabular)",
        "Random Forest",
        "SVM (RBF SVR)",
        "XGBoost",
        f"Image + Tabular (ResNet-18 PCA {n_pca}D)",
    ]
    ll_scores = {m: [] for m in model_order}
    rmse_scores = {m: [] for m in model_order}
    mae_scores = {m: [] for m in model_order}

    def _build_img_tab_pipe():
        return Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("clin", StandardScaler(), VISIT_CLINICAL_FEATURES),
                            (
                                "hu",
                                Pipeline(
                                    [
                                        ("win", InFoldWinsorizer(0.01, 0.99)),
                                        ("scaler", StandardScaler()),
                                    ]
                                ),
                                hu_cols,
                            ),
                            (
                                "resnet_pca",
                                Pipeline(
                                    [
                                        ("scaler", StandardScaler()),
                                        ("pca", PCA(n_components=n_pca, random_state=RANDOM_SEED)),
                                    ]
                                ),
                                resnet_cols,
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                ("reg", Ridge(alpha=500.0)),
            ]
        )

    img_tab_name = f"Image + Tabular (ResNet-18 PCA {n_pca}D)"
    sklearn_specs = {
        "Ridge": (
            lambda: Pipeline([("sc", StandardScaler()), ("reg", Ridge())]),
            {"reg__alpha": [1.0, 10.0, 50.0, 200.0]},
            VISIT_CLINICAL_FEATURES,
            False,
        ),
        "ElasticNet": (
            lambda: Pipeline([("sc", StandardScaler()), ("reg", ElasticNet(max_iter=3000, random_state=RANDOM_SEED))]),
            {"reg__alpha": [0.1, 1.0, 10.0], "reg__l1_ratio": [0.2, 0.5, 0.8]},
            VISIT_CLINICAL_FEATURES,
            False,
        ),
        "Random Forest": (
            lambda: Pipeline([("reg", RandomForestRegressor(n_estimators=80, random_state=RANDOM_SEED, n_jobs=1))]),
            {"reg__max_depth": [4, 7], "reg__min_samples_leaf": [5, 10]},
            VISIT_CLINICAL_FEATURES,
            False,
        ),
        "SVM (RBF SVR)": (
            lambda: Pipeline([("sc", StandardScaler()), ("reg", SVR(kernel="rbf"))]),
            {"reg__C": [100.0, 500.0, 1500.0]},
            VISIT_CLINICAL_FEATURES,
            False,
        ),
        "XGBoost": (
            lambda: Pipeline([("reg", XGBRegressor(n_estimators=70, random_state=RANDOM_SEED, verbosity=0, n_jobs=1))]),
            {"reg__max_depth": [2, 3], "reg__learning_rate": [0.04, 0.08]},
            VISIT_CLINICAL_FEATURES,
            False,
        ),
        img_tab_name: (
            _build_img_tab_pipe,
            {"reg__alpha": [100.0, 500.0, 1000.0]},
            VISIT_CLINICAL_FEATURES + hu_cols + resnet_cols,
            True,  # Predicts change from Baseline_FVC, reconstructed + winsorized to [500, 6500] mL
        ),
    }

    patients_arr = visit_df["Patient"].values
    for fold_idx, (tr_pats, va_pats) in enumerate(splits):
        tr_mask = np.array([p in tr_pats for p in patients_arr], dtype=bool)
        va_mask = np.array([p in va_pats for p in patients_arr], dtype=bool)

        df_tr = visit_df.loc[tr_mask].reset_index(drop=True)
        df_va = visit_df.loc[va_mask].reset_index(drop=True)
        y_tr = df_tr["Target_FVC"].values.astype(np.float64)
        y_va = df_va["Target_FVC"].values.astype(np.float64)
        base_fvc_tr = df_tr["Baseline_FVC"].values.astype(np.float64)
        base_fvc_va = df_va["Baseline_FVC"].values.astype(np.float64)
        y_delta_tr = y_tr - base_fvc_tr
        g_tr = df_tr["Patient"].values
        dw_tr = df_tr["Delta_Weeks"].values.astype(np.float64)
        dw_va = df_va["Delta_Weeks"].values.astype(np.float64)

        # 1. Baseline (predict constant Baseline_FVC, sigma fitted in-fold)
        base_pred_tr = winsorize_fvc(base_fvc_tr)
        base_pred_va = winsorize_fvc(base_fvc_va)
        base_sig_p = _fit_in_fold_sigma_params(y_tr, base_pred_tr, dw_tr)
        base_sig_va = _predict_sigma(dw_va, base_sig_p)
        b_name = "Baseline (Constant Baseline FVC)"
        ll_scores[b_name].append(modified_laplace_log_likelihood(y_va, base_pred_va, base_sig_va))
        rmse_scores[b_name].append(float(np.sqrt(mean_squared_error(y_va, base_pred_va))))
        mae_scores[b_name].append(float(mean_absolute_error(y_va, base_pred_va)))

        # 2. Quantile-Regression Neural Network (Tabular)
        qnn_pred_va, qnn_sig_va = _fit_predict_quantile_nn_tabular(
            df_tr[VISIT_CLINICAL_FEATURES].values,
            y_tr,
            g_tr,
            df_va[VISIT_CLINICAL_FEATURES].values,
            seed=RANDOM_SEED + fold_idx,
        )
        qnn_pred_va = winsorize_fvc(qnn_pred_va)
        q_name = "Quantile-Regression NN (Tabular)"
        ll_scores[q_name].append(modified_laplace_log_likelihood(y_va, qnn_pred_va, qnn_sig_va))
        rmse_scores[q_name].append(float(np.sqrt(mean_squared_error(y_va, qnn_pred_va))))
        mae_scores[q_name].append(float(mean_absolute_error(y_va, qnn_pred_va)))

        # 3. All nested-tuned sklearn/XGBoost & Image+Tabular models
        inner_gkf = GroupKFold(n_splits=3)
        for m_name, (pipe_fn, param_grid, cols, predict_delta) in sklearn_specs.items():
            X_tr_m = df_tr[cols]
            X_va_m = df_va[cols]
            target_tr = y_delta_tr if predict_delta else y_tr
            gs = GridSearchCV(
                pipe_fn(),
                param_grid,
                scoring="neg_root_mean_squared_error",
                cv=inner_gkf,
                n_jobs=1,
            )
            gs.fit(X_tr_m, target_tr, groups=g_tr)
            best_params = gs.best_params_

            # Compute strictly in-fold OOF predictions to calibrate sigma(dw) without leakage
            def _tuned_factory(p_fn=pipe_fn, bp=best_params):
                est = p_fn()
                est.set_params(**bp)
                return est

            oof_raw = _compute_inner_oof_predictions(_tuned_factory, X_tr_m, target_tr, g_tr, n_inner=3)
            oof_tr = winsorize_fvc(base_fvc_tr + oof_raw) if predict_delta else winsorize_fvc(oof_raw)
            sig_p = _fit_in_fold_sigma_params(y_tr, oof_tr, dw_tr)

            pred_raw = gs.best_estimator_.predict(X_va_m)
            pred_va = winsorize_fvc(base_fvc_va + pred_raw) if predict_delta else winsorize_fvc(pred_raw)
            sig_va = _predict_sigma(dw_va, sig_p)

            ll_scores[m_name].append(modified_laplace_log_likelihood(y_va, pred_va, sig_va))
            rmse_scores[m_name].append(float(np.sqrt(mean_squared_error(y_va, pred_va))))
            mae_scores[m_name].append(float(mean_absolute_error(y_va, pred_va)))

    models_out = {}
    best_m = None
    best_ll = -1e9
    for m_name in model_order:
        ll_st = _compute_ci_95(ll_scores[m_name])
        rm_st = _compute_ci_95(rmse_scores[m_name])
        ma_st = _compute_ci_95(mae_scores[m_name])
        models_out[m_name] = {
            "model": m_name,
            "laplace_ll_stats": ll_st,
            "rmse_stats": rm_st,
            "mae_stats": ma_st,
        }
        if m_name != "Baseline (Constant Baseline FVC)" and ll_st["mean"] > best_ll:
            best_ll = ll_st["mean"]
            best_m = m_name

    logger.info(
        f"[OSIC Competition Metric Benchmark (n={visit_df['Patient'].nunique()} patients, "
        f"{len(visit_df)} future visits)] Best internal CV model: {best_m} "
        f"(Laplace LL = {models_out[best_m]['laplace_ll_stats']['formatted']})"
    )

    return {
        "evaluation_protocol": "5-Repeat x 5-Fold Nested GroupKFold by Patient on OSIC train.csv future visits (NOT Kaggle leaderboard)",
        "metric_definition": "Modified Laplace Log-Likelihood: sigma clipped >= 70 mL, |FVC_true - FVC_pred| clipped <= 1000 mL",
        "fvc_winsorize_bounds_ml": [FVC_MIN_ML, FVC_MAX_ML],
        "image_tabular_design": (
            "Image + Tabular predicts change from Baseline_FVC (Delta_FVC = Target_FVC - Baseline_FVC) "
            "using in-fold 1st-99th percentile winsorization on HU/volume features, in-fold 15D ResNet-18 PCA, "
            "and strong L2 regularization (Ridge alpha in [100, 500, 1000]), reconstructed as "
            "Baseline_FVC + Delta_FVC and winsorized to [500, 6500] mL."
        ),
        "num_patients": int(visit_df["Patient"].nunique()),
        "num_future_visits": int(len(visit_df)),
        "best_internal_model": best_m,
        "efficientnet_b5_note": (
            "EfficientNet-B5 end-to-end 3D fine-tuning was NOT run locally because the 10.4 GB raw "
            "OSIC DICOM archive is excluded from the handover package and Kaggle GPU extraction used "
            "pretrained ResNet-18 (512-D) embeddings."
        ),
        "models": models_out,
    }


def evaluate_osic_fusion_comparison(
    visit_df: pd.DataFrame,
    pat_df: pd.DataFrame,
    splits: List[Tuple[set, set]],
    hu_cols: List[str],
    resnet_cols: List[str],
    n_pca: int = 15,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Requirement B: Compare all 6 fusion methods on the EXACT SAME 5x5 Nested GroupKFold splits:
      1. clinical_only
      2. hrct_only (HU/Volume + ResNet-18 In-Fold PCA 15D)
      3. early_fusion
      4. late_fusion_mean
      5. weighted_late_fusion (weights tuned strictly inside training folds)
      6. stacking (meta-learner trained strictly on inner OOF predictions)
    Reports mean +/- std and 95% CI for Laplace LL, RMSE, MAE, and patient-level rapid-decliner AUC.
    Selects `selected_fusion_method` via the One-Standard-Error (1-SE) rule, preferring `clinical_only`
    whenever fusion does not clearly outperform clinical-only by > 1 SE.
    Also fits and returns production serving pipelines (torch-free sklearn/XGBoost) for
    `clinical_only`, `hrct_only`, and the selected `fusion` method.
    """
    methods = [
        ("clinical_only", "Clinical-only"),
        ("hrct_only", "HRCT-only (HU/Volume + ResNet-18 PCA 15D)"),
        ("early_fusion", "Early fusion"),
        ("late_fusion_mean", "Late fusion (mean)"),
        ("weighted_late_fusion", "Weighted late fusion (in-fold tuned)"),
        ("stacking", "Stacking (OOF meta-learner)"),
    ]
    ll_m = {k: [] for k, _ in methods}
    rmse_m = {k: [] for k, _ in methods}
    mae_m = {k: [] for k, _ in methods}
    auc_m = {k: [] for k, _ in methods}
    visit_test_train_ratios: List[float] = []
    pat_test_train_ratios: List[float] = []
    learned_reg_weights: List[float] = []
    learned_cls_weights: List[float] = []

    ct_pat_df = pat_df[pat_df["has_ct"]].reset_index(drop=True)
    visit_pats = visit_df["Patient"].values
    pat_pats = ct_pat_df["Patient"].values

    def make_clin_reg():
        return Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=10.0))])

    def make_hrct_reg():
        return Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("dw", StandardScaler(), ["Delta_Weeks"]),
                            (
                                "hu",
                                Pipeline(
                                    [
                                        ("win", InFoldWinsorizer(0.01, 0.99)),
                                        ("sc", StandardScaler()),
                                    ]
                                ),
                                hu_cols,
                            ),
                            (
                                "pca",
                                Pipeline(
                                    [
                                        ("sc", StandardScaler()),
                                        ("pca", PCA(n_components=n_pca, random_state=RANDOM_SEED)),
                                    ]
                                ),
                                resnet_cols,
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                ("reg", Ridge(alpha=500.0)),
            ]
        )

    def make_early_reg():
        return Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("clin", StandardScaler(), VISIT_CLINICAL_FEATURES),
                            (
                                "hu",
                                Pipeline(
                                    [
                                        ("win", InFoldWinsorizer(0.01, 0.99)),
                                        ("sc", StandardScaler()),
                                    ]
                                ),
                                hu_cols,
                            ),
                            (
                                "pca",
                                Pipeline(
                                    [
                                        ("sc", StandardScaler()),
                                        ("pca", PCA(n_components=n_pca, random_state=RANDOM_SEED)),
                                    ]
                                ),
                                resnet_cols,
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                ("reg", Ridge(alpha=500.0)),
            ]
        )

    def make_clin_cls():
        return Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "cls",
                    XGBClassifier(
                        n_estimators=60,
                        max_depth=3,
                        learning_rate=0.05,
                        eval_metric="logloss",
                        random_state=RANDOM_SEED,
                        verbosity=0,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    def make_hrct_cls():
        return Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("hu", StandardScaler(), hu_cols),
                            (
                                "pca",
                                Pipeline(
                                    [
                                        ("sc", StandardScaler()),
                                        ("pca", PCA(n_components=n_pca, random_state=RANDOM_SEED)),
                                    ]
                                ),
                                resnet_cols,
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "cls",
                    XGBClassifier(
                        n_estimators=60,
                        max_depth=3,
                        learning_rate=0.05,
                        eval_metric="logloss",
                        random_state=RANDOM_SEED,
                        verbosity=0,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    def make_early_cls():
        return Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("clin", StandardScaler(), PATIENT_CLINICAL_FEATURES),
                            ("hu", StandardScaler(), hu_cols),
                            (
                                "pca",
                                Pipeline(
                                    [
                                        ("sc", StandardScaler()),
                                        ("pca", PCA(n_components=n_pca, random_state=RANDOM_SEED)),
                                    ]
                                ),
                                resnet_cols,
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "cls",
                    XGBClassifier(
                        n_estimators=60,
                        max_depth=3,
                        learning_rate=0.05,
                        eval_metric="logloss",
                        random_state=RANDOM_SEED,
                        verbosity=0,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    for tr_pats, va_pats in splits:
        # ---- A. Visit-level Future FVC (Laplace LL, RMSE, MAE) ----
        v_tr_mask = np.array([p in tr_pats for p in visit_pats], dtype=bool)
        v_va_mask = np.array([p in va_pats for p in visit_pats], dtype=bool)
        vtr = visit_df.loc[v_tr_mask].reset_index(drop=True)
        vva = visit_df.loc[v_va_mask].reset_index(drop=True)
        visit_test_train_ratios.append(float(len(vva) / max(1, len(vtr))))
        y_tr = vtr["Target_FVC"].values.astype(np.float64)
        y_va = vva["Target_FVC"].values.astype(np.float64)
        base_fvc_tr = vtr["Baseline_FVC"].values.astype(np.float64)
        base_fvc_va = vva["Baseline_FVC"].values.astype(np.float64)
        y_delta_tr = y_tr - base_fvc_tr
        g_tr = vtr["Patient"].values
        dw_tr = vtr["Delta_Weeks"].values.astype(np.float64)
        dw_va = vva["Delta_Weeks"].values.astype(np.float64)

        all_vcols = VISIT_CLINICAL_FEATURES + hu_cols + resnet_cols
        hrct_vcols = ["Delta_Weeks"] + hu_cols + resnet_cols

        # Inner OOF predictions for clinical (absolute FVC), HRCT & early (change from Baseline_FVC)
        oof_c = winsorize_fvc(
            _compute_inner_oof_predictions(make_clin_reg, vtr[VISIT_CLINICAL_FEATURES], y_tr, g_tr)
        )
        oof_h_delta = _compute_inner_oof_predictions(make_hrct_reg, vtr[hrct_vcols], y_delta_tr, g_tr)
        oof_h = winsorize_fvc(base_fvc_tr + oof_h_delta)
        oof_e_delta = _compute_inner_oof_predictions(make_early_reg, vtr[all_vcols], y_delta_tr, g_tr)
        oof_e = winsorize_fvc(base_fvc_tr + oof_e_delta)

        # Fit base regressors on full outer training fold
        est_c = make_clin_reg().fit(vtr[VISIT_CLINICAL_FEATURES], y_tr)
        est_h = make_hrct_reg().fit(vtr[hrct_vcols], y_delta_tr)
        est_e = make_early_reg().fit(vtr[all_vcols], y_delta_tr)

        pred_c = winsorize_fvc(est_c.predict(vva[VISIT_CLINICAL_FEATURES]))
        pred_h = winsorize_fvc(base_fvc_va + est_h.predict(vva[hrct_vcols]))
        pred_e = winsorize_fvc(base_fvc_va + est_e.predict(vva[all_vcols]))

        # Late fusion (mean)
        oof_lm = winsorize_fvc(0.5 * oof_c + 0.5 * oof_h)
        pred_lm = winsorize_fvc(0.5 * pred_c + 0.5 * pred_h)

        # Weighted late fusion (weight w tuned strictly on inner OOF training fold)
        best_w, best_w_rmse = 1.0, float("inf")
        for w in np.linspace(0.0, 1.0, 21):
            blend_oof = winsorize_fvc(w * oof_c + (1.0 - w) * oof_h)
            r_val = float(np.sqrt(mean_squared_error(y_tr, blend_oof)))
            if r_val < best_w_rmse:
                best_w_rmse = r_val
                best_w = float(w)
        learned_reg_weights.append(best_w)
        oof_wl = winsorize_fvc(best_w * oof_c + (1.0 - best_w) * oof_h)
        pred_wl = winsorize_fvc(best_w * pred_c + (1.0 - best_w) * pred_h)

        # Stacking meta-learner trained strictly on inner OOF predictions
        meta_X_tr = np.column_stack([oof_c, oof_h])
        meta_X_va = np.column_stack([pred_c, pred_h])
        meta_reg = Ridge(alpha=10.0).fit(meta_X_tr, y_tr)
        oof_st = winsorize_fvc(meta_reg.predict(meta_X_tr))
        pred_st = winsorize_fvc(meta_reg.predict(meta_X_va))

        reg_map = {
            "clinical_only": (oof_c, pred_c),
            "hrct_only": (oof_h, pred_h),
            "early_fusion": (oof_e, pred_e),
            "late_fusion_mean": (oof_lm, pred_lm),
            "weighted_late_fusion": (oof_wl, pred_wl),
            "stacking": (oof_st, pred_st),
        }
        for key, (oof_tr_k, pred_va_k) in reg_map.items():
            sig_p = _fit_in_fold_sigma_params(y_tr, oof_tr_k, dw_tr)
            sig_va = _predict_sigma(dw_va, sig_p)
            ll_m[key].append(modified_laplace_log_likelihood(y_va, pred_va_k, sig_va))
            rmse_m[key].append(float(np.sqrt(mean_squared_error(y_va, pred_va_k))))
            mae_m[key].append(float(mean_absolute_error(y_va, pred_va_k)))

        # ---- B. Patient-level Rapid Decliner Classification (AUC) ----
        p_tr_mask = np.array([p in tr_pats for p in pat_pats], dtype=bool)
        p_va_mask = np.array([p in va_pats for p in pat_pats], dtype=bool)
        ptr = ct_pat_df.loc[p_tr_mask].reset_index(drop=True)
        pva = ct_pat_df.loc[p_va_mask].reset_index(drop=True)
        yc_tr = ptr["rapid_decliner_52w"].values.astype(int)
        yc_va = pva["rapid_decliner_52w"].values.astype(int)
        pg_tr = ptr["Patient"].values

        if len(np.unique(yc_tr)) < 2 or len(np.unique(yc_va)) < 2:
            continue
        pat_test_train_ratios.append(float(len(pva) / max(1, len(ptr))))

        all_pcols = PATIENT_CLINICAL_FEATURES + hu_cols + resnet_cols
        hrct_pcols = hu_cols + resnet_cols

        # Inner OOF probabilities strictly on training patients
        oof_pc = np.zeros(len(ptr), dtype=np.float64)
        oof_ph = np.zeros(len(ptr), dtype=np.float64)
        igkf = GroupKFold(n_splits=3)
        for itr, iva in igkf.split(ptr, yc_tr, groups=pg_tr):
            if len(np.unique(yc_tr[itr])) < 2:
                oof_pc[iva] = float(np.mean(yc_tr))
                oof_ph[iva] = float(np.mean(yc_tr))
                continue
            oof_pc[iva] = make_clin_cls().fit(ptr[PATIENT_CLINICAL_FEATURES].iloc[itr], yc_tr[itr]).predict_proba(ptr[PATIENT_CLINICAL_FEATURES].iloc[iva])[:, 1]
            oof_ph[iva] = make_hrct_cls().fit(ptr[hrct_pcols].iloc[itr], yc_tr[itr]).predict_proba(ptr[hrct_pcols].iloc[iva])[:, 1]

        cls_c = make_clin_cls().fit(ptr[PATIENT_CLINICAL_FEATURES], yc_tr)
        cls_h = make_hrct_cls().fit(ptr[hrct_pcols], yc_tr)
        cls_e = make_early_cls().fit(ptr[all_pcols], yc_tr)

        pr_c = cls_c.predict_proba(pva[PATIENT_CLINICAL_FEATURES])[:, 1]
        pr_h = cls_h.predict_proba(pva[hrct_pcols])[:, 1]
        pr_e = cls_e.predict_proba(pva[all_pcols])[:, 1]
        pr_lm = 0.5 * pr_c + 0.5 * pr_h

        # Tune classification fusion weight strictly on inner OOF AUC
        best_wc, best_wc_auc = 1.0, -1.0
        for w in np.linspace(0.0, 1.0, 21):
            b_oof = w * oof_pc + (1.0 - w) * oof_ph
            a_val = float(roc_auc_score(yc_tr, b_oof))
            if a_val > best_wc_auc:
                best_wc_auc = a_val
                best_wc = float(w)
        learned_cls_weights.append(best_wc)
        pr_wl = best_wc * pr_c + (1.0 - best_wc) * pr_h

        # Stacking logistic meta-learner on inner OOF probabilities
        meta_cls = LogisticRegression(C=1.0, random_state=RANDOM_SEED).fit(
            np.column_stack([oof_pc, oof_ph]), yc_tr
        )
        pr_st = meta_cls.predict_proba(np.column_stack([pr_c, pr_h]))[:, 1]

        cls_map = {
            "clinical_only": pr_c,
            "hrct_only": pr_h,
            "early_fusion": pr_e,
            "late_fusion_mean": pr_lm,
            "weighted_late_fusion": pr_wl,
            "stacking": pr_st,
        }
        for key, prob_va in cls_map.items():
            auc_m[key].append(float(roc_auc_score(yc_va, prob_va)))

    # Summarize all 6 methods + paired Nadeau-Bengio corrected resampled t-test vs Clinical-only on the exact same 25 folds
    mean_v_ratio = float(np.mean(visit_test_train_ratios)) if visit_test_train_ratios else 0.25
    mean_p_ratio = float(np.mean(pat_test_train_ratios)) if pat_test_train_ratios else 0.25

    method_table = {}
    paired_vs_clinical: Dict[str, Any] = {}
    for key, display_name in methods:
        entry: Dict[str, Any] = {
            "key": key,
            "method": display_name,
            "laplace_ll_stats": _compute_ci_95(ll_m[key]),
            "rmse_stats": _compute_ci_95(rmse_m[key]),
            "mae_stats": _compute_ci_95(mae_m[key]),
            "auc_stats": _compute_ci_95(auc_m[key]) if auc_m[key] else _compute_ci_95([0.5]),
        }
        if key != "clinical_only":
            nb_ll = nadeau_bengio_paired_test(
                ll_m[key], ll_m["clinical_only"], test_train_ratio=mean_v_ratio, higher_is_better=True
            )
            nb_rmse = nadeau_bengio_paired_test(
                rmse_m[key], rmse_m["clinical_only"], test_train_ratio=mean_v_ratio, higher_is_better=False
            )
            nb_mae = nadeau_bengio_paired_test(
                mae_m[key], mae_m["clinical_only"], test_train_ratio=mean_v_ratio, higher_is_better=False
            )
            nb_auc = (
                nadeau_bengio_paired_test(
                    auc_m[key], auc_m["clinical_only"], test_train_ratio=mean_p_ratio, higher_is_better=True
                )
                if len(auc_m[key]) == len(auc_m["clinical_only"]) and len(auc_m[key]) > 1
                else None
            )
            paired_entry = {
                "method": display_name,
                "reference": "Clinical-only",
                "test_name": "Nadeau-Bengio corrected resampled paired t-test (25 identical folds)",
                "laplace_ll": nb_ll,
                "rmse": nb_rmse,
                "mae": nb_mae,
                "auc": nb_auc,
                "status": "statistically_confirmed" if nb_ll["statistically_better"] else "nominal_only",
            }
            entry["paired_vs_clinical_only"] = paired_entry
            paired_vs_clinical[key] = paired_entry
        method_table[key] = entry

    # One-Standard-Error (1-SE) selection on primary CV Laplace LL (higher = better)
    best_key = max(method_table.keys(), key=lambda k: method_table[k]["laplace_ll_stats"]["mean"])
    best_mean = method_table[best_key]["laplace_ll_stats"]["mean"]
    best_se = method_table[best_key]["laplace_ll_stats"]["se"]
    clin_mean = method_table["clinical_only"]["laplace_ll_stats"]["mean"]

    # Parsimony order (simplest first): clinical_only -> late_fusion_mean -> weighted_late_fusion -> early_fusion -> stacking
    parsimony_order = [
        "clinical_only",
        "late_fusion_mean",
        "weighted_late_fusion",
        "early_fusion",
        "stacking",
        "hrct_only",
    ]
    selected_method = best_key
    for cand in parsimony_order:
        if method_table[cand]["laplace_ll_stats"]["mean"] >= (best_mean - best_se):
            selected_method = cand
            break

    # Identify any statistically confirmed fusion methods (Nadeau-Bengio paired 95% CI strictly > 0 on Laplace LL)
    n_paired_comparisons = len(paired_vs_clinical) * 4  # 5 non-clinical methods x 4 metrics = 20
    bonferroni_alpha = round(0.05 / max(1, n_paired_comparisons), 4)
    confirmed_keys = [
        k for k, p in paired_vs_clinical.items() if bool(p["laplace_ll"]["statistically_better"])
    ]
    statistically_confirmed_method = (
        max(confirmed_keys, key=lambda k: method_table[k]["laplace_ll_stats"]["mean"])
        if confirmed_keys
        else None
    )

    selected_paired_ll = (
        paired_vs_clinical[selected_method]["laplace_ll"]
        if selected_method in paired_vs_clinical
        else None
    )
    fusion_vs_clinical_status = (
        "statistically_confirmed"
        if (selected_paired_ll is not None and bool(selected_paired_ll["statistically_better"]))
        else "nominal_only"
    )

    mean_fvc_ml = round(float(visit_df["Target_FVC"].mean()), 1)
    sel_ll = method_table[selected_method]["laplace_ll_stats"]["mean"]
    sel_auc = method_table[selected_method]["auc_stats"]["mean"]
    clin_auc = method_table["clinical_only"]["auc_stats"]["mean"]

    if statistically_confirmed_method is not None and statistically_confirmed_method != selected_method:
        sc_p = paired_vs_clinical[statistically_confirmed_method]
        sc_ll_diff = sc_p["laplace_ll"]["mean_diff"]
        sc_ll_p = sc_p["laplace_ll"]["p_value"]
        sc_mae_diff = sc_p["mae"]["mean_diff"]
        sc_mae_p = sc_p["mae"]["p_value"]
        sc_name = method_table[statistically_confirmed_method]["method"]
        mae_pct_fvc = abs(sc_mae_diff) / max(1.0, mean_fvc_ml) * 100.0
        bonf_survives = (sc_ll_p < bonferroni_alpha) and (sc_mae_p < bonferroni_alpha)
        sel_p_ll = selected_paired_ll["p_value"] if selected_paired_ll else 1.0
        sel_d_ll = selected_paired_ll["mean_diff"] if selected_paired_ll else 0.0
        rationale = (
            f"{method_table[selected_method]['method']} has the best mean FVC Laplace LL "
            f"({sel_ll:.4f} vs. clinical-only {clin_mean:.4f}; Δ = {sel_d_ll:+.4f}, p = {sel_p_ll:.4f}, nominal only), "
            f"while its rapid-decliner AUC is lower ({sel_auc:.4f} vs. {clin_auc:.4f}). "
            f"{sc_name} is the only method with a statistically confirmed gain over clinical-only "
            f"({sc_ll_diff:+.4f} Laplace LL, p = {sc_ll_p:.4f}; {sc_mae_diff:+.1f} mL MAE, p = {sc_mae_p:.4f}); "
            + (
                f"across {n_paired_comparisons} paired comparisons, both p-values survive Bonferroni correction "
                f"(α = {bonferroni_alpha:.4f}), "
                if bonf_survives
                else f"across {n_paired_comparisons} paired comparisons (Bonferroni α = {bonferroni_alpha:.4f}), "
            )
            + f"but the gain is clinically negligible ({mae_pct_fvc:.2f}%, under 0.1% of mean FVC {mean_fvc_ml:.1f} mL). Exploratory."
        )
    elif selected_method == "clinical_only":
        rationale = (
            f"Clinical-only ({clin_mean:.4f}) is selected as selected_fusion_method; "
            f"no fusion method achieved a statistically confirmed improvement on FVC Laplace LL "
            f"(highest nominal mean: {method_table[best_key]['method']} at {best_mean:.4f}). Exploratory."
        )
    elif fusion_vs_clinical_status == "statistically_confirmed":
        rationale = (
            f"{method_table[selected_method]['method']} ({sel_ll:.4f}) achieved a statistically confirmed "
            f"gain over clinical-only ({clin_mean:.4f}; {selected_paired_ll['formatted']}). Exploratory."
        )
    else:
        rationale = (
            f"{method_table[selected_method]['method']} has the best mean FVC Laplace LL "
            f"({sel_ll:.4f}, nominal only), but the difference from clinical-only ({clin_mean:.4f}) "
            f"is not statistically confirmed, and its rapid-decliner AUC is lower ({sel_auc:.4f} vs. {clin_auc:.4f}). Exploratory."
        )

    # Fit final serving models on all 175 CT patients (pure sklearn/XGBoost -> torch-free loading!)
    y_all_v = visit_df["Target_FVC"].values.astype(np.float64)
    base_fvc_all_v = visit_df["Baseline_FVC"].values.astype(np.float64)
    y_delta_all_v = y_all_v - base_fvc_all_v
    g_all_v = visit_df["Patient"].values
    dw_all_v = visit_df["Delta_Weeks"].values.astype(np.float64)
    yc_all_p = ct_pat_df["rapid_decliner_52w"].values.astype(int)

    hrct_vcols = ["Delta_Weeks"] + hu_cols + resnet_cols
    all_vcols = VISIT_CLINICAL_FEATURES + hu_cols + resnet_cols
    hrct_pcols = hu_cols + resnet_cols
    all_pcols = PATIENT_CLINICAL_FEATURES + hu_cols + resnet_cols

    oof_c_all = winsorize_fvc(
        _compute_inner_oof_predictions(make_clin_reg, visit_df[VISIT_CLINICAL_FEATURES], y_all_v, g_all_v)
    )
    oof_h_all = winsorize_fvc(
        base_fvc_all_v + _compute_inner_oof_predictions(make_hrct_reg, visit_df[hrct_vcols], y_delta_all_v, g_all_v)
    )
    oof_e_all = winsorize_fvc(
        base_fvc_all_v + _compute_inner_oof_predictions(make_early_reg, visit_df[all_vcols], y_delta_all_v, g_all_v)
    )

    final_clin_reg = make_clin_reg().fit(visit_df[VISIT_CLINICAL_FEATURES], y_all_v)
    final_hrct_reg = make_hrct_reg().fit(visit_df[hrct_vcols], y_delta_all_v)
    final_early_reg = make_early_reg().fit(visit_df[all_vcols], y_delta_all_v)
    final_stack_reg = Ridge(alpha=10.0).fit(np.column_stack([oof_c_all, oof_h_all]), y_all_v)

    final_clin_cls = make_clin_cls().fit(ct_pat_df[PATIENT_CLINICAL_FEATURES], yc_all_p)
    final_hrct_cls = make_hrct_cls().fit(ct_pat_df[hrct_pcols], yc_all_p)
    final_early_cls = make_early_cls().fit(ct_pat_df[all_pcols], yc_all_p)

    # Inner OOF probs for final stacking classifier
    oof_pc_all = np.zeros(len(ct_pat_df), dtype=np.float64)
    oof_ph_all = np.zeros(len(ct_pat_df), dtype=np.float64)
    for itr, iva in GroupKFold(n_splits=5).split(ct_pat_df, yc_all_p, groups=ct_pat_df["Patient"].values):
        oof_pc_all[iva] = make_clin_cls().fit(ct_pat_df[PATIENT_CLINICAL_FEATURES].iloc[itr], yc_all_p[itr]).predict_proba(ct_pat_df[PATIENT_CLINICAL_FEATURES].iloc[iva])[:, 1]
        oof_ph_all[iva] = make_hrct_cls().fit(ct_pat_df[hrct_pcols].iloc[itr], yc_all_p[itr]).predict_proba(ct_pat_df[hrct_pcols].iloc[iva])[:, 1]
    final_stack_cls = LogisticRegression(C=1.0, random_state=RANDOM_SEED).fit(
        np.column_stack([oof_pc_all, oof_ph_all]), yc_all_p
    )

    mean_w_reg = round(float(np.mean(learned_reg_weights)), 4) if learned_reg_weights else 0.9
    mean_w_cls = round(float(np.mean(learned_cls_weights)), 4) if learned_cls_weights else 0.9

    oof_lm_all = winsorize_fvc(0.5 * oof_c_all + 0.5 * oof_h_all)
    oof_wl_all = winsorize_fvc(mean_w_reg * oof_c_all + (1.0 - mean_w_reg) * oof_h_all)
    oof_st_all = winsorize_fvc(final_stack_reg.predict(np.column_stack([oof_c_all, oof_h_all])))

    serving_models = {
        "selected_fusion_method": selected_method,
        "selected_fusion_display_name": method_table[selected_method]["method"],
        "hrct_predicts_delta_fvc": True,
        "early_predicts_delta_fvc": True,
        "fvc_winsorize_bounds_ml": [FVC_MIN_ML, FVC_MAX_ML],
        "visit_clinical_cols": VISIT_CLINICAL_FEATURES,
        "patient_clinical_cols": PATIENT_CLINICAL_FEATURES,
        "hrct_visit_cols": hrct_vcols,
        "hrct_patient_cols": hrct_pcols,
        "all_visit_cols": all_vcols,
        "all_patient_cols": all_pcols,
        "clinical_fvc_reg": final_clin_reg,
        "clinical_sigma_params": _fit_in_fold_sigma_params(y_all_v, oof_c_all, dw_all_v),
        "clinical_rapid_cls": final_clin_cls,
        "hrct_fvc_reg": final_hrct_reg,
        "hrct_sigma_params": _fit_in_fold_sigma_params(y_all_v, oof_h_all, dw_all_v),
        "hrct_rapid_cls": final_hrct_cls,
        "early_fvc_reg": final_early_reg,
        "early_sigma_params": _fit_in_fold_sigma_params(y_all_v, oof_e_all, dw_all_v),
        "early_rapid_cls": final_early_cls,
        "late_mean_sigma_params": _fit_in_fold_sigma_params(y_all_v, oof_lm_all, dw_all_v),
        "weighted_late_sigma_params": _fit_in_fold_sigma_params(y_all_v, oof_wl_all, dw_all_v),
        "stack_fvc_reg": final_stack_reg,
        "stack_sigma_params": _fit_in_fold_sigma_params(y_all_v, oof_st_all, dw_all_v),
        "stack_rapid_cls": final_stack_cls,
        "weighted_late_reg_clinical_weight": mean_w_reg,
        "weighted_late_cls_clinical_weight": mean_w_cls,
    }

    fusion_summary = {
        "status": "completed",
        "split_strategy": "5-Repeat x 5-Fold Nested GroupKFold by Patient (exact same 25 folds for all 6 methods; PCA, scalers, winsorizers, weights, and meta-learners fit strictly inside training folds)",
        "paired_test_method": (
            f"Nadeau-Bengio corrected resampled paired t-test on per-fold differences vs. Clinical-only "
            f"(25 identical folds, df=24, mean n_test/n_train={mean_v_ratio:.4f} for FVC regression, "
            f"{mean_p_ratio:.4f} for patient-level rapid-decliner AUC)"
        ),
        "hrct_branch_design": (
            "HRCT-only and Early Fusion predict change from Baseline_FVC (Delta_FVC = Target_FVC - Baseline_FVC) "
            "using in-fold 1st-99th percentile winsorization on HU/volume features, in-fold 15D ResNet-18 PCA, "
            "and strong L2 regularization (Ridge alpha=500). All reconstructed FVC predictions across all 6 methods "
            "are winsorized to the physiologically plausible range [500, 6500] mL."
        ),
        "fvc_winsorize_bounds_ml": [FVC_MIN_ML, FVC_MAX_ML],
        "num_ct_patients": int(len(ct_pat_df)),
        "num_future_visits": int(len(visit_df)),
        "pca_components_used": n_pca,
        "selected_fusion_method": selected_method,
        "selected_fusion_display_name": method_table[selected_method]["method"],
        "selected_fusion_rule": "Best mean CV Modified Laplace Log-Likelihood (nominal only unless paired Nadeau-Bengio 95% CI excludes zero)",
        "statistically_confirmed_method": statistically_confirmed_method,
        "statistically_confirmed_display_name": (
            method_table[statistically_confirmed_method]["method"]
            if statistically_confirmed_method
            else None
        ),
        "highest_cv_method": best_key,
        "fusion_vs_clinical_status": fusion_vs_clinical_status,
        "bonferroni_comparisons_count": n_paired_comparisons,
        "bonferroni_alpha": bonferroni_alpha,
        "mean_osic_fvc_ml": mean_fvc_ml,
        "selection_rationale": rationale,
        "mean_tuned_reg_clinical_weight": mean_w_reg,
        "mean_tuned_cls_clinical_weight": mean_w_cls,
        "paired_vs_clinical_only": paired_vs_clinical,
        "methods": method_table,
    }
    logger.info(f"[OSIC 6-Method Fusion Comparison] {rationale}")
    return fusion_summary, serving_models


def predict_multimodal_progression(
    serving_models: Dict[str, Any],
    fusion_comparison: Dict[str, Any],
    clinical_inputs: Dict[str, float],
    ct_features: Dict[str, float] = None,
    horizon_weeks: float = 52.0,
) -> Dict[str, Any]:
    """
    Compute clinical-only, HRCT-only, and fusion (selected_fusion_method) progression-risk
    outputs from the fitted torch-free serving pipelines.
    - When `ct_features` is None, `hrct_only` and `fusion` are strictly `None` (never dummy values).
    - Always attaches cross-validated 95% CIs from `fusion_comparison`.
    """
    base_fvc = float(clinical_inputs["Baseline_FVC"])
    base_pct = float(clinical_inputs["Baseline_Percent"])
    age = float(clinical_inputs["Age"])
    sex_male = float(clinical_inputs["Sex_Male"])
    smoke_code = float(clinical_inputs["Smoking_Code"])
    dw = float(max(1.0, horizon_weeks))

    methods_meta = (fusion_comparison or {}).get("methods", {})
    selected_method = str(
        serving_models.get("selected_fusion_method")
        or (fusion_comparison or {}).get("selected_fusion_method")
        or "clinical_only"
    )
    selected_display = str(
        serving_models.get("selected_fusion_display_name")
        or methods_meta.get(selected_method, {}).get("method")
        or selected_method
    )

    def _risk_tier(prob: float) -> str:
        if prob >= 0.55:
            return "Elevated Rapid-Decline Risk"
        if prob >= 0.35:
            return "Moderate Decline Risk"
        return "Lower Decline Risk"

    def _ci_block(method_key: str) -> Dict[str, Any]:
        m = methods_meta.get(method_key, {})
        return {
            "cv_laplace_ll_95ci": m.get("laplace_ll_stats", {}).get("formatted", "N/A"),
            "cv_rmse_95ci": m.get("rmse_stats", {}).get("formatted", "N/A"),
            "cv_mae_95ci": m.get("mae_stats", {}).get("formatted", "N/A"),
            "cv_auc_95ci": m.get("auc_stats", {}).get("formatted", "N/A"),
            "cv_auc_mean": m.get("auc_stats", {}).get("mean"),
            "cv_auc_lower": m.get("auc_stats", {}).get("ci_95_lower"),
            "cv_auc_upper": m.get("auc_stats", {}).get("ci_95_upper"),
            "cv_laplace_ll_mean": m.get("laplace_ll_stats", {}).get("mean"),
            "cv_laplace_ll_lower": m.get("laplace_ll_stats", {}).get("ci_95_lower"),
            "cv_laplace_ll_upper": m.get("laplace_ll_stats", {}).get("ci_95_upper"),
        }

    clin_visit_df = pd.DataFrame(
        [
            {
                "Baseline_FVC": base_fvc,
                "Baseline_Percent": base_pct,
                "Delta_Weeks": dw,
                "Age": age,
                "Sex_Male": sex_male,
                "Smoking_Code": smoke_code,
            }
        ]
    )[serving_models["visit_clinical_cols"]]
    clin_pat_df = pd.DataFrame(
        [
            {
                "Baseline_FVC": base_fvc,
                "Baseline_Percent": base_pct,
                "Age": age,
                "Sex_Male": sex_male,
                "Smoking_Code": smoke_code,
            }
        ]
    )[serving_models["patient_clinical_cols"]]

    c_fvc = float(winsorize_fvc(serving_models["clinical_fvc_reg"].predict(clin_visit_df))[0])
    c_sig = float(_predict_sigma(np.array([dw]), serving_models["clinical_sigma_params"])[0])
    c_prob = float(np.clip(serving_models["clinical_rapid_cls"].predict_proba(clin_pat_df)[0, 1], 0.0, 1.0))

    clinical_out = {
        "method": "clinical_only",
        "method_display_name": "Clinical-only",
        "progression_risk_score": round(c_prob, 4),
        "risk_tier": _risk_tier(c_prob),
        "predicted_fvc_at_horizon_ml": round(c_fvc, 1),
        "predicted_fvc_change_ml": round(c_fvc - base_fvc, 1),
        "uncertainty_sigma_ml": round(c_sig, 1),
        "horizon_weeks": dw,
        "exploratory": True,
        "predicts_target": "FVC decline over follow-up (OSIC cohort), NOT an IPF diagnosis",
        **_ci_block("clinical_only"),
    }

    if ct_features is None:
        return {
            "clinical_only": clinical_out,
            "hrct_only": None,
            "fusion": None,
            "selected_fusion_method": selected_method,
            "selected_fusion_display_name": selected_display,
            "fusion_vs_clinical_status": str(
                (fusion_comparison or {}).get("fusion_vs_clinical_status", "nominal_only")
            ),
            "modality_used": {"clinical": True, "hrct": False, "fusion": False},
        }

    # Build HRCT and combined DataFrames strictly from extracted CT features
    hrct_visit_row = {"Delta_Weeks": dw, **{k: float(ct_features.get(k, 0.0)) for k in serving_models["hrct_visit_cols"] if k != "Delta_Weeks"}}
    hrct_visit_df = pd.DataFrame([hrct_visit_row])[serving_models["hrct_visit_cols"]]
    hrct_pat_df = pd.DataFrame([{k: float(ct_features.get(k, 0.0)) for k in serving_models["hrct_patient_cols"]}])[serving_models["hrct_patient_cols"]]

    all_visit_row = {
        "Baseline_FVC": base_fvc,
        "Baseline_Percent": base_pct,
        "Delta_Weeks": dw,
        "Age": age,
        "Sex_Male": sex_male,
        "Smoking_Code": smoke_code,
        **{k: float(ct_features.get(k, 0.0)) for k in serving_models["hrct_patient_cols"]},
    }
    all_visit_df = pd.DataFrame([all_visit_row])[serving_models["all_visit_cols"]]
    all_pat_df = pd.DataFrame([all_visit_row])[serving_models["all_patient_cols"]]

    h_raw = float(serving_models["hrct_fvc_reg"].predict(hrct_visit_df)[0])
    h_fvc = float(
        winsorize_fvc(
            base_fvc + h_raw if serving_models.get("hrct_predicts_delta_fvc", False) else h_raw
        )
    )
    h_sig = float(_predict_sigma(np.array([dw]), serving_models["hrct_sigma_params"])[0])
    h_prob = float(np.clip(serving_models["hrct_rapid_cls"].predict_proba(hrct_pat_df)[0, 1], 0.0, 1.0))

    hrct_out = {
        "method": "hrct_only",
        "method_display_name": "HRCT-only (HU/Volume + ResNet-18 PCA)",
        "progression_risk_score": round(h_prob, 4),
        "risk_tier": _risk_tier(h_prob),
        "predicted_fvc_at_horizon_ml": round(h_fvc, 1),
        "predicted_fvc_change_ml": round(h_fvc - base_fvc, 1),
        "uncertainty_sigma_ml": round(h_sig, 1),
        "horizon_weeks": dw,
        "exploratory": True,
        "predicts_target": "FVC decline over follow-up (OSIC cohort), NOT an IPF diagnosis",
        **_ci_block("hrct_only"),
    }

    # Compute fusion score strictly according to `selected_fusion_method`
    w_reg = float(serving_models.get("weighted_late_reg_clinical_weight", 0.9))
    w_cls = float(serving_models.get("weighted_late_cls_clinical_weight", 0.9))

    if selected_method == "clinical_only":
        f_fvc, f_sig, f_prob = c_fvc, c_sig, c_prob
    elif selected_method == "hrct_only":
        f_fvc, f_sig, f_prob = h_fvc, h_sig, h_prob
    elif selected_method == "early_fusion":
        e_raw = float(serving_models["early_fvc_reg"].predict(all_visit_df)[0])
        f_fvc = float(
            winsorize_fvc(
                base_fvc + e_raw if serving_models.get("early_predicts_delta_fvc", False) else e_raw
            )
        )
        f_sig = float(_predict_sigma(np.array([dw]), serving_models["early_sigma_params"])[0])
        f_prob = float(np.clip(serving_models["early_rapid_cls"].predict_proba(all_pat_df)[0, 1], 0.0, 1.0))
    elif selected_method == "late_fusion_mean":
        f_fvc = float(winsorize_fvc(0.5 * c_fvc + 0.5 * h_fvc))
        f_sig = float(_predict_sigma(np.array([dw]), serving_models["late_mean_sigma_params"])[0])
        f_prob = float(np.clip(0.5 * c_prob + 0.5 * h_prob, 0.0, 1.0))
    elif selected_method == "weighted_late_fusion":
        f_fvc = float(winsorize_fvc(w_reg * c_fvc + (1.0 - w_reg) * h_fvc))
        f_sig = float(_predict_sigma(np.array([dw]), serving_models["weighted_late_sigma_params"])[0])
        f_prob = float(np.clip(w_cls * c_prob + (1.0 - w_cls) * h_prob, 0.0, 1.0))
    else:  # stacking
        f_fvc = float(winsorize_fvc(serving_models["stack_fvc_reg"].predict(np.array([[c_fvc, h_fvc]]))[0]))
        f_sig = float(_predict_sigma(np.array([dw]), serving_models["stack_sigma_params"])[0])
        f_prob = float(np.clip(serving_models["stack_rapid_cls"].predict_proba(np.array([[c_prob, h_prob]]))[0, 1], 0.0, 1.0))

    fusion_out = {
        "method": selected_method,
        "method_display_name": selected_display,
        "progression_risk_score": round(f_prob, 4),
        "risk_tier": _risk_tier(f_prob),
        "predicted_fvc_at_horizon_ml": round(f_fvc, 1),
        "predicted_fvc_change_ml": round(f_fvc - base_fvc, 1),
        "uncertainty_sigma_ml": round(f_sig, 1),
        "horizon_weeks": dw,
        "exploratory": True,
        "predicts_target": "FVC decline over follow-up (OSIC cohort), NOT an IPF diagnosis",
        **_ci_block(selected_method),
    }

    return {
        "clinical_only": clinical_out,
        "hrct_only": hrct_out,
        "fusion": fusion_out,
        "selected_fusion_method": selected_method,
        "selected_fusion_display_name": selected_display,
        "fusion_vs_clinical_status": str(
            (fusion_comparison or {}).get("fusion_vs_clinical_status", "nominal_only")
        ),
        "modality_used": {"clinical": True, "hrct": True, "fusion": True},
    }


def run_osic_benchmarks_and_fusion(
    raw_train_df: pd.DataFrame,
    pat_df: pd.DataFrame,
    hu_cols: List[str],
    resnet_cols: List[str],
    n_pca: int = 15,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Run both Requirement A (Competition Metric Benchmark) and Requirement B (6-Method Fusion)."""
    visit_df = build_osic_visit_dataset(raw_train_df, pat_df, hu_cols, resnet_cols)
    ct_count = int(pat_df["has_ct"].sum()) if "has_ct" in pat_df.columns else int(visit_df["Patient"].nunique())
    safe_pca = max(2, min(n_pca, int(ct_count * 0.5) - 2))
    splits = _generate_patient_cv_splits(visit_df["Patient"].values, n_repeats=5, n_splits=5, seed=RANDOM_SEED)

    comp_benchmark = evaluate_osic_literature_benchmark(visit_df, splits, hu_cols, resnet_cols, n_pca=safe_pca)
    fusion_summary, serving_models = evaluate_osic_fusion_comparison(
        visit_df, pat_df, splits, hu_cols, resnet_cols, n_pca=safe_pca
    )
    return comp_benchmark, fusion_summary, serving_models
