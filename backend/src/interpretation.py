"""
Deterministic Clinical Result Interpretation Engine + Optional Validated LLM Rephrasing
(`backend/src/interpretation.py`).

Generates an evidence-grounded, five-part patient result summary strictly from:
  1. The patient's supplied demographic and pulmonary function values,
  2. Configurable physiology and guideline rules in `backend/config/interpretation_rules.json`,
  3. The calibrated model output and dual operating-mode thresholds, and
  4. Real held-out test-set statistics stored in `backend/models/model_artifacts.json`
     (reliability bin rates, cohort age medians, and per-diagnosis-group misclassification rates).

Five structured parts returned as `interpretation` (dict) and `interpretation_text` (str):
  (a) physiology               — ventilatory pattern & gas-transfer (TLCO) severity
  (b) model_result             — calibrated probability, both operating modes, & test-set reliability bin rate
  (c) what_pushed_it           — top drivers in plain words (+ real cohort age medians if age is a top driver)
  (d) caution                  — stored i-NSIP / u-ILD / CTD misclassification rates & PFT-only overlap warning
  (e) clinician_considerations — generic ATS/ERS/JRS/ALAT guideline items (never diagnosis or treatment advice)

Optional LLM rephrasing:
  - Controlled by environment flag `ENABLE_LLM_INTERPRETATION_REPHRASE` (default OFF / false).
  - Reuses `backend/agent/orchestrator.py` Anthropic client and sends ONLY de-identified structured facts.
  - Post-generation validator checks:
      (1) Every number in the LLM text appears in the supplied input facts, and
      (2) Zero diagnosis or treatment wording is present.
  - Automatically falls back to the deterministic `interpretation_text` if validation fails or on any error.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.src.config import (
    AUTOMATED_INTERPRETATION_NOTICE,
    INTERPRETATION_RULES_PATH,
    setup_logger,
)

logger = setup_logger("interpretation")

_RULES_CACHE: Optional[Dict[str, Any]] = None


def load_interpretation_rules(
    rules_path: Optional[Path] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Load configurable physiology and guideline rules from JSON."""
    global _RULES_CACHE
    path = rules_path or INTERPRETATION_RULES_PATH
    if use_cache and rules_path is None and _RULES_CACHE is not None:
        return _RULES_CACHE
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if rules_path is None:
        _RULES_CACHE = data
    return data


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, str) and not val.strip():
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _fmt_num(val: float, decimals: int = 1) -> str:
    if decimals == 0 or (decimals == 1 and abs(val - round(val)) < 0.05):
        return f"{round(val):.0f}" if decimals == 0 else f"{val:.1f}"
    return f"{val:.{decimals}f}"


def _join_phrases(phrases: List[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def _classify_tlco(
    tlco_pp: Optional[float],
    rules_phys: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify TLCO % predicted into ATS/ERS gas-transfer severity bands."""
    if tlco_pp is None:
        return {
            "key": "missing",
            "label": "not available (TLCO % predicted missing)",
            "value_pp": None,
            "borderline": False,
            "borderline_note": None,
        }

    bands = rules_phys["tlco_severity_bands"]
    margin = float(rules_phys.get("tlco_borderline_margin_pp", 3.0))
    matched = None
    for b in bands:
        lo = b.get("min_pp")
        hi = b.get("max_pp")
        in_lo = True if lo is None else (tlco_pp >= float(lo))
        in_hi = True if hi is None else (tlco_pp < float(hi))
        if in_lo and in_hi:
            matched = b
            break
    if matched is None:
        matched = bands[-1]

    # Check proximity to band cutoffs (80%, 60%, 40%)
    borderline = False
    borderline_note = None
    for cutoff, adj_label in ((80.0, "80% normal cutoff"), (60.0, "60% mild/moderate cutoff"), (40.0, "40% moderate/severe cutoff")):
        if abs(tlco_pp - cutoff) <= margin:
            borderline = True
            borderline_note = f"near the {adj_label}"
            break

    return {
        "key": matched["key"],
        "label": matched["label"],
        "value_pp": round(tlco_pp, 1),
        "borderline": borderline,
        "borderline_note": borderline_note,
    }


def analyze_physiology(
    raw_inputs: Dict[str, Any],
    rules: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Part (a) Physiology: rule-based ventilatory pattern recognition and gas-transfer severity
    strictly from patient inputs and `backend/config/interpretation_rules.json`.
    """
    r_phys = rules["physiology"]
    vol_t = r_phys["volume_thresholds"]
    rat_t = r_phys["ratio_thresholds"]
    ext_bounds = r_phys.get("extreme_value_bounds", {})

    tlc_cut = float(vol_t["tlc_reduced_below_pp"])
    fvc_cut = float(vol_t["fvc_reduced_below_pp"])
    fev1_cut = float(vol_t["fev1_reduced_below_pp"])
    tlc_high = float(vol_t.get("tlc_elevated_above_pp", 120.0))
    vol_margin = float(vol_t.get("borderline_margin_pp", 5.0))
    ratio_cut = float(rat_t["fev1fvc_preserved_at_or_above"])
    ratio_margin = float(rat_t.get("fev1fvc_borderline_margin", 0.03))
    req_tlc = bool(r_phys.get("restrictive_requires_tlc_when_present", True))

    age = _safe_float(raw_inputs.get("age"))
    fev1fvc = _safe_float(raw_inputs.get("fev1fvc_abs"))
    fev1_pp = _safe_float(raw_inputs.get("fev1_pp"))
    fvc_pp = _safe_float(raw_inputs.get("fvc_pp"))
    tlc_pp = _safe_float(raw_inputs.get("tlc_pp"))
    tlco_pp = _safe_float(raw_inputs.get("tlco_pp"))

    field_values = {
        "age": age,
        "fev1fvc_abs": fev1fvc,
        "fev1_pp": fev1_pp,
        "fvc_pp": fvc_pp,
        "tlc_pp": tlc_pp,
        "tlco_pp": tlco_pp,
    }
    missing_fields = [k for k in ("fev1fvc_abs", "fev1_pp", "fvc_pp", "tlc_pp", "tlco_pp") if field_values[k] is None]

    extreme_warnings: List[str] = []
    friendly_names = {
        "age": ("Age", "years"),
        "fev1fvc_abs": ("FEV1/FVC ratio", ""),
        "fev1_pp": ("FEV1", "% predicted"),
        "fvc_pp": ("FVC", "% predicted"),
        "tlc_pp": ("TLC", "% predicted"),
        "tlco_pp": ("TLCO", "% predicted"),
    }
    for fname, val in field_values.items():
        if val is None or fname not in ext_bounds:
            continue
        b = ext_bounds[fname]
        label, unit = friendly_names[fname]
        val_fmt = f"{val:.2f}" if fname == "fev1fvc_abs" else f"{val:.1f}"
        unit_suffix = f" {unit}" if unit else ""
        if b.get("warn_below") is not None and val < float(b["warn_below"]):
            extreme_warnings.append(
                f"{label} ({val_fmt}{unit_suffix}) is unusually low; verify measurement and units."
            )
        elif b.get("warn_above") is not None and val > float(b["warn_above"]):
            extreme_warnings.append(
                f"{label} ({val_fmt}{unit_suffix}) is unusually high; verify measurement and units."
            )

    # Ratio classification
    if fev1fvc is None:
        ratio_status = "missing"
        ratio_borderline = False
    elif fev1fvc >= ratio_cut:
        ratio_status = "preserved"
        ratio_borderline = abs(fev1fvc - ratio_cut) <= ratio_margin
    else:
        ratio_status = "reduced"
        ratio_borderline = abs(fev1fvc - ratio_cut) <= ratio_margin

    # Volume classification
    tlc_reduced = (tlc_pp < tlc_cut) if tlc_pp is not None else None
    tlc_elevated = (tlc_pp > tlc_high) if tlc_pp is not None else False
    fvc_reduced = (fvc_pp < fvc_cut) if fvc_pp is not None else None
    fev1_reduced = (fev1_pp < fev1_cut) if fev1_pp is not None else None

    tlc_borderline = (tlc_pp is not None) and (abs(tlc_pp - tlc_cut) <= vol_margin)
    fvc_borderline = (fvc_pp is not None) and (abs(fvc_pp - fvc_cut) <= vol_margin)

    # Restriction determination per ATS/ERS (TLC preferred when present; FVC fallback if TLC absent)
    used_fvc_proxy = False
    if req_tlc:
        if tlc_pp is not None:
            volume_reduced: Optional[bool] = tlc_reduced
        elif fvc_pp is not None:
            volume_reduced = fvc_reduced
            used_fvc_proxy = True
        else:
            volume_reduced = None
    else:
        if tlc_pp is not None or fvc_pp is not None:
            volume_reduced = bool(tlc_reduced) or bool(fvc_reduced)
        else:
            volume_reduced = None

    # Ventilatory pattern assignment
    if ratio_status == "missing" or volume_reduced is None:
        pattern = "indeterminate"
        pattern_label = "Indeterminate ventilatory pattern (incomplete spirometry or lung volume inputs)"
    elif ratio_status == "reduced":
        if volume_reduced:
            pattern = "mixed"
            pattern_label = "Mixed (obstructive and restrictive) ventilatory pattern"
        else:
            pattern = "obstructive"
            pattern_label = "Obstructive ventilatory pattern"
    else:
        # ratio preserved (>= 0.70)
        if volume_reduced:
            pattern = "restrictive"
            pattern_label = "Restrictive ventilatory pattern"
        elif bool(fvc_reduced) or bool(fev1_reduced):
            pattern = "non_specific"
            pattern_label = (
                "Non-specific ventilatory pattern (preserved FEV1/FVC and normal TLC, with reduced spirometry)"
            )
        else:
            pattern = "normal"
            pattern_label = "Normal ventilatory pattern"

    # Gas transfer (TLCO)
    tlco_info = _classify_tlco(tlco_pp, r_phys)
    isolated_gas_transfer_deficit = (
        pattern in ("normal", "non_specific") and tlco_info["key"] in ("mild", "moderate", "severe")
    )

    # Construct grounded physiology narrative
    detail_bits: List[str] = []
    if tlc_pp is not None:
        tlc_word = "reduced" if tlc_reduced else ("elevated" if tlc_elevated else "normal")
        detail_bits.append(f"TLC {tlc_pp:.1f}% predicted ({tlc_word}, cutoff {tlc_cut:.0f}%)")
    if fvc_pp is not None:
        fvc_word = "reduced" if fvc_reduced else "normal"
        detail_bits.append(f"FVC {fvc_pp:.1f}% predicted ({fvc_word}, cutoff {fvc_cut:.0f}%)")
    if fev1_pp is not None:
        fev1_word = "reduced" if fev1_reduced else "normal"
        detail_bits.append(f"FEV1 {fev1_pp:.1f}% predicted ({fev1_word})")
    if fev1fvc is not None:
        rat_word = "preserved" if ratio_status == "preserved" else "reduced"
        detail_bits.append(f"FEV1/FVC {fev1fvc:.2f} ({rat_word}, cutoff {ratio_cut:.2f})")

    sentences: List[str] = []
    if pattern == "restrictive":
        proxy_note = " based on FVC because TLC was not provided" if used_fvc_proxy else ""
        sentences.append(
            f"Restrictive ventilatory pattern{proxy_note}: {'; '.join(detail_bits)}."
        )
    elif pattern == "obstructive":
        hyper_note = " with elevated TLC consistent with hyperinflation" if tlc_elevated else ""
        sentences.append(
            f"Obstructive ventilatory pattern{hyper_note}: {'; '.join(detail_bits)}."
        )
    elif pattern == "mixed":
        proxy_note = " (using FVC because TLC was not provided)" if used_fvc_proxy else ""
        sentences.append(
            f"Mixed (obstructive and restrictive) ventilatory pattern{proxy_note}: {'; '.join(detail_bits)}."
        )
    elif pattern == "normal":
        sentences.append(
            f"Normal ventilatory pattern: {'; '.join(detail_bits)}."
        )
    elif pattern == "non_specific":
        sentences.append(
            f"Non-specific ventilatory pattern (normal TLC rules out lung-volume restriction, "
            f"while spirometry is reduced with preserved ratio): {'; '.join(detail_bits)}."
        )
    else:
        avail = "; ".join(detail_bits) if detail_bits else "no complete PFT values provided"
        sentences.append(
            f"Indeterminate ventilatory pattern due to missing inputs ({', '.join(missing_fields)}): {avail}."
        )

    # Gas-transfer sentence
    if tlco_pp is not None:
        b_suffix = f" ({tlco_info['borderline_note']})" if tlco_info["borderline_note"] else ""
        iso_prefix = "Isolated " if isolated_gas_transfer_deficit else ""
        tlco_desc = tlco_info["label"]
        if iso_prefix:
            tlco_desc = f"{iso_prefix}{tlco_desc} despite preserved lung volumes"
        else:
            tlco_desc = tlco_desc[0].upper() + tlco_desc[1:]
        sentences.append(f"Gas transfer: {tlco_desc} at TLCO {tlco_pp:.1f}% predicted{b_suffix}.")
    else:
        sentences.append("Gas transfer: TLCO % predicted was not provided.")

    # Borderline spirometry / volume callout
    borderline_notes: List[str] = []
    if ratio_borderline and fev1fvc is not None:
        borderline_notes.append(f"FEV1/FVC ({fev1fvc:.2f}) is borderline near {ratio_cut:.2f}")
    if tlc_borderline and tlc_pp is not None:
        borderline_notes.append(f"TLC ({tlc_pp:.1f}%) is borderline near {tlc_cut:.0f}%")
    if fvc_borderline and fvc_pp is not None:
        borderline_notes.append(f"FVC ({fvc_pp:.1f}%) is borderline near {fvc_cut:.0f}%")
    if borderline_notes:
        sentences.append(
            f"Borderline measurements: {'; '.join(borderline_notes)} "
            "(percent-predicted cutoffs are approximations without GLI z-scores/LLN)."
        )

    if extreme_warnings:
        sentences.append("Value check: " + " ".join(extreme_warnings))

    return {
        "pattern": pattern,
        "pattern_label": pattern_label,
        "ratio_status": ratio_status,
        "tlc_reduced": tlc_reduced,
        "tlc_elevated": tlc_elevated,
        "fvc_reduced": fvc_reduced,
        "fev1_reduced": fev1_reduced,
        "used_fvc_proxy_for_restriction": used_fvc_proxy,
        "gas_transfer_severity": tlco_info["key"],
        "gas_transfer_label": tlco_info["label"],
        "isolated_gas_transfer_deficit": isolated_gas_transfer_deficit,
        "missing_fields": missing_fields,
        "borderline_flags": borderline_notes,
        "extreme_value_warnings": extreme_warnings,
        "values": {
            "fev1fvc_abs": round(fev1fvc, 2) if fev1fvc is not None else None,
            "fev1_pp": round(fev1_pp, 1) if fev1_pp is not None else None,
            "fvc_pp": round(fvc_pp, 1) if fvc_pp is not None else None,
            "tlc_pp": round(tlc_pp, 1) if tlc_pp is not None else None,
            "tlco_pp": round(tlco_pp, 1) if tlco_pp is not None else None,
        },
        "thresholds_applied": {
            "tlc_reduced_below_pp": tlc_cut,
            "fvc_reduced_below_pp": fvc_cut,
            "fev1_reduced_below_pp": fev1_cut,
            "fev1fvc_preserved_at_or_above": ratio_cut,
            "tlco_normal_min_pp": 80.0,
            "tlco_mild_min_pp": 60.0,
            "tlco_moderate_min_pp": 40.0,
        },
        "summary": " ".join(sentences),
    }


def lookup_reliability_bin(
    prob: float,
    reliability_table: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the held-out test reliability bin containing `prob`."""
    if not reliability_table or not reliability_table.get("bins"):
        return None
    p = max(0.0, min(1.0, float(prob)))
    bins = reliability_table["bins"]
    for idx, b in enumerate(bins):
        lo = float(b["prob_min"])
        hi = float(b["prob_max"])
        is_last = idx == len(bins) - 1 or bool(b.get("inclusive_upper", False))
        if is_last:
            if lo <= p <= hi + 1e-12:
                return b
        else:
            if lo <= p < hi:
                return b
    return bins[-1]


def analyze_model_result(
    prob_ipf: float,
    screening_threshold: float,
    confirmatory_threshold: float,
    active_mode: str,
    artifacts: Dict[str, Any],
    rules: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Part (b) Model result: calibrated probability, Screening & Confirmatory thresholds and calls,
    borderline/conflict detection, and empirical IPF rate in the matching test-set score bin.
    """
    prob = round(float(prob_ipf), 4)
    prob_pct = round(prob * 100.0, 1)
    s_thresh = round(float(screening_threshold), 4)
    c_thresh = round(float(confirmatory_threshold), 4)
    margin = float(rules["model_result"].get("borderline_probability_margin", 0.05))

    s_pos = prob >= s_thresh
    c_pos = prob >= c_thresh
    s_call = "Positive (IPF)" if s_pos else "Negative (Non-IPF)"
    c_call = "Positive (IPF)" if c_pos else "Negative (Non-IPF)"
    modes_disagree = s_pos != c_pos

    near_screening = abs(prob - s_thresh) <= margin
    near_confirmatory = abs(prob - c_thresh) <= margin
    is_borderline = modes_disagree or near_screening or near_confirmatory

    rel_table = artifacts.get("calibration_reliability_table")
    matched_bin = lookup_reliability_bin(prob, rel_table)
    obs_max = rel_table.get("observed_prob_max") if rel_table else None

    sentences = [
        f"Calibrated IPF probability is {prob_pct:.1f}% ({prob:.4f}).",
        f"Screening mode (threshold {s_thresh:.4f}): {s_call}; "
        f"Confirmatory mode (threshold {c_thresh:.4f}): {c_call}.",
    ]
    if modes_disagree:
        sentences.append(
            "Operating modes conflict: this score exceeds the Screening threshold "
            f"({s_thresh:.4f}) but remains below the Confirmatory threshold ({c_thresh:.4f}), "
            "indicating an intermediate/borderline risk zone."
        )
    elif is_borderline:
        nearest_name = (
            f"Screening threshold ({s_thresh:.4f})"
            if abs(prob - s_thresh) <= abs(prob - c_thresh)
            else f"Confirmatory threshold ({c_thresh:.4f})"
        )
        sentences.append(
            f"Borderline score: probability is within {margin:.2f} of the {nearest_name}."
        )

    bin_info: Optional[Dict[str, Any]] = None
    if matched_bin is not None and matched_bin.get("n_patients", 0) > 0:
        emp_pct = float(matched_bin["empirical_ipf_pct"])
        n_ipf = int(matched_bin["n_ipf"])
        n_tot = int(matched_bin["n_patients"])
        bin_range = matched_bin["pct_range_label"]
        ci_lo = matched_bin.get("wilson_ci_95_lower_pct")
        ci_hi = matched_bin.get("wilson_ci_95_upper_pct")
        ci_text = f" (95% CI {ci_lo:.1f}%–{ci_hi:.1f}%)" if (ci_lo is not None and ci_hi is not None) else ""
        extra_max = ""
        if obs_max is not None and prob > float(obs_max):
            extra_max = (
                f" Note: this score exceeds the highest calibrated score observed in the "
                f"held-out test set ({float(obs_max):.4f})."
            )
        sentences.append(
            f"In the held-out test set, {emp_pct:.1f}% ({n_ipf} of {n_tot}){ci_text} of patients "
            f"whose calibrated score fell in the {bin_range} bin ({matched_bin['bin_label']}) had IPF.{extra_max}"
        )
        bin_info = {
            "bin_label": matched_bin["bin_label"],
            "pct_range_label": bin_range,
            "n_ipf": n_ipf,
            "n_patients": n_tot,
            "empirical_ipf_pct": emp_pct,
            "wilson_ci_95_lower_pct": ci_lo,
            "wilson_ci_95_upper_pct": ci_hi,
            "above_observed_test_max": bool(obs_max is not None and prob > float(obs_max)),
        }
    else:
        sentences.append(
            "Held-out test reliability bin data is unavailable for this score range."
        )

    return {
        "ipf_probability": prob,
        "ipf_probability_pct": prob_pct,
        "active_mode": active_mode,
        "screening_threshold": s_thresh,
        "screening_positive": s_pos,
        "screening_call": s_call,
        "confirmatory_threshold": c_thresh,
        "confirmatory_positive": c_pos,
        "confirmatory_call": c_call,
        "modes_disagree": modes_disagree,
        "is_borderline": is_borderline,
        "reliability_bin": bin_info,
        "summary": " ".join(sentences),
    }


def _describe_driver_plain(item: Dict[str, Any], raw_inputs: Dict[str, Any]) -> str:
    feat = str(item.get("feature", "")).strip()
    val_str = str(item.get("value", "")).strip()

    if feat == "sex":
        raw_sex = str(raw_inputs.get("sex", val_str)).strip().upper()
        return "male sex" if raw_sex.startswith("M") else "female sex"
    if feat == "age":
        a = _safe_float(raw_inputs.get("age"))
        if a is not None:
            return f"age {a:.0f} years" if abs(a - round(a)) < 0.05 else f"age {a:.1f} years"
        return f"age ({val_str})"
    if feat == "tlco_pp":
        v = _safe_float(raw_inputs.get("tlco_pp"))
        if v is not None:
            qual = "severely reduced " if v < 40 else "moderately reduced " if v < 60 else "mildly reduced " if v < 80 else "preserved "
            return f"{qual}TLCO ({v:.1f}%)"
        return f"TLCO ({val_str})"
    if feat == "tlc_pp":
        v = _safe_float(raw_inputs.get("tlc_pp"))
        if v is not None:
            qual = "reduced " if v < 80 else "elevated " if v > 120 else "normal "
            return f"{qual}TLC ({v:.1f}%)"
        return f"TLC ({val_str})"
    if feat == "fvc_pp":
        v = _safe_float(raw_inputs.get("fvc_pp"))
        if v is not None:
            qual = "reduced " if v < 80 else "preserved "
            return f"{qual}FVC ({v:.1f}%)"
        return f"FVC ({val_str})"
    if feat == "fev1_pp":
        v = _safe_float(raw_inputs.get("fev1_pp"))
        if v is not None:
            qual = "reduced " if v < 80 else "preserved "
            return f"{qual}FEV1 ({v:.1f}%)"
        return f"FEV1 ({val_str})"
    if feat == "fev1fvc_abs":
        v = _safe_float(raw_inputs.get("fev1fvc_abs"))
        if v is not None:
            qual = "preserved " if v >= 0.70 else "reduced "
            return f"{qual}FEV1/FVC ratio ({v:.2f})"
        return f"FEV1/FVC ratio ({val_str})"
    if feat == "Height":
        v = _safe_float(raw_inputs.get("Height", raw_inputs.get("height")))
        if v is not None:
            return f"height ({v:.2f} m)"
        return f"height ({val_str})"
    if feat == "Weight":
        v = _safe_float(raw_inputs.get("Weight", raw_inputs.get("weight")))
        if v is not None:
            return f"weight ({v:.1f} kg)"
        return f"weight ({val_str})"
    disp = str(item.get("display_name") or feat)
    return f"{disp} ({val_str})" if val_str else disp


def analyze_drivers(
    top_shap_features: List[Dict[str, Any]],
    raw_inputs: Dict[str, Any],
    artifacts: Dict[str, Any],
    rules: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Part (c) What pushed it: top contributors in plain words with direction.
    If age is among the top drivers, appends one sentence quoting real IPF vs. Non-IPF
    age medians from the Zenodo dataset stored in `model_artifacts.json`.
    """
    r_drv = rules["drivers"]
    max_pos = int(r_drv.get("max_positive_drivers", 3))
    max_neg = int(r_drv.get("max_negative_drivers", 2))
    age_rank_cutoff = int(r_drv.get("age_driver_rank_cutoff", 3))
    min_abs = float(r_drv.get("min_abs_shap", 0.03))

    ordered = sorted(
        list(top_shap_features or []),
        key=lambda x: abs(float(x.get("shap_value", 0.0))),
        reverse=True,
    )
    meaningful = [x for x in ordered if abs(float(x.get("shap_value", 0.0))) >= min_abs]

    pos_items = [x for x in meaningful if float(x.get("shap_value", 0.0)) > 0][:max_pos]
    neg_items = [x for x in meaningful if float(x.get("shap_value", 0.0)) < 0][:max_neg]

    pos_phrases = [_describe_driver_plain(x, raw_inputs) for x in pos_items]
    neg_phrases = [_describe_driver_plain(x, raw_inputs) for x in neg_items]

    sentences: List[str] = []
    if pos_phrases and neg_phrases:
        lead = (
            f"Pushed score toward IPF: {_join_phrases(pos_phrases)}; "
            f"pushed score away from IPF: {_join_phrases(neg_phrases)}."
        )
        sentences.append(lead)
    elif pos_phrases:
        sentences.append(
            f"Pushed score toward IPF: {_join_phrases(pos_phrases)}, with no major opposing features."
        )
    elif neg_phrases:
        sentences.append(
            f"Pushed score away from IPF: {_join_phrases(neg_phrases)}, with no major features raising IPF risk."
        )
    else:
        sentences.append(
            "No individual input feature produced a large directional shift away from the cohort baseline."
        )

    # Check whether age is a top driver (by |SHAP| rank)
    age_is_top_driver = any(
        str(x.get("feature", "")).strip().lower() == "age"
        for x in meaningful[:age_rank_cutoff]
    )

    age_stats = artifacts.get("cohort_age_stats") or {}
    eda = artifacts.get("eda_summary") or {}
    ipf_med = age_stats.get("ipf_median_age", eda.get("median_age_ipf"))
    non_ipf_med = age_stats.get("non_ipf_median_age", eda.get("median_age_non_ipf"))

    age_context_sentence: Optional[str] = None
    if age_is_top_driver and ipf_med is not None and non_ipf_med is not None:
        age_context_sentence = (
            f"Age context: IPF patients in this dataset are older than non-IPF patients "
            f"(median age {float(ipf_med):.1f} vs. {float(non_ipf_med):.1f} years)."
        )
        sentences.append(age_context_sentence)

    return {
        "toward_ipf": [
            {
                "feature": x.get("feature"),
                "label": _describe_driver_plain(x, raw_inputs),
                "direction": "toward_ipf",
                "shap_value": round(float(x.get("shap_value", 0.0)), 4),
            }
            for x in pos_items
        ],
        "away_from_ipf": [
            {
                "feature": x.get("feature"),
                "label": _describe_driver_plain(x, raw_inputs),
                "direction": "away_from_ipf",
                "shap_value": round(float(x.get("shap_value", 0.0)), 4),
            }
            for x in neg_items
        ],
        "age_is_top_driver": age_is_top_driver,
        "cohort_age_medians": (
            {
                "ipf_median_age": round(float(ipf_med), 1),
                "non_ipf_median_age": round(float(non_ipf_med), 1),
            }
            if (ipf_med is not None and non_ipf_med is not None)
            else None
        ),
        "age_context_sentence": age_context_sentence,
        "summary": " ".join(sentences),
    }


def analyze_caution(
    model_result: Dict[str, Any],
    artifacts: Dict[str, Any],
    rules: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Part (d) Caution: quote stored per-group test misclassification rates (i-NSIP, u-ILD, CTD)
    from `model_artifacts.json` and state that lung function alone cannot separate IPF from
    other fibrosing ILDs.
    """
    r_caut = rules["caution"]
    target_groups = list(r_caut.get("highlighted_misclassification_groups", ["i-NSIP", "u-ILD", "CTD"]))
    overlap_stmt = str(
        r_caut.get(
            "lung_function_overlap_statement",
            "Lung function alone cannot separate IPF from other fibrosing interstitial lung diseases.",
        )
    )

    op_modes = artifacts.get("operating_modes") or {}
    sc_groups = {
        g["group"]: g
        for g in (
            op_modes.get("screening", {}).get("per_group_error_analysis")
            or artifacts.get("per_group_error_analysis")
            or []
        )
    }
    cf_groups = {
        g["group"]: g
        for g in (op_modes.get("confirmatory", {}).get("per_group_error_analysis") or [])
    }

    group_rates: List[Dict[str, Any]] = []
    sc_phrases: List[str] = []
    cf_phrases: List[str] = []
    for grp in target_groups:
        sg = sc_groups.get(grp)
        cg = cf_groups.get(grp)
        if sg is not None:
            sc_pct = round(float(sg["error_rate_pct"]), 1)
            sc_err = int(sg["error_count"])
            sc_tot = int(sg["total_test_cases"])
            sc_phrases.append(f"{grp} {sc_pct:.1f}% ({sc_err}/{sc_tot})")
        else:
            sc_pct = None
            sc_err = None
            sc_tot = None

        if cg is not None:
            cf_pct = round(float(cg["error_rate_pct"]), 1)
            cf_err = int(cg["error_count"])
            cf_tot = int(cg["total_test_cases"])
            cf_phrases.append(f"{grp} {cf_pct:.1f}% ({cf_err}/{cf_tot})")
        else:
            cf_pct = None
            cf_err = None
            cf_tot = None

        group_rates.append(
            {
                "group": grp,
                "screening_misclassification_pct": sc_pct,
                "screening_error_count": sc_err,
                "screening_total_cases": sc_tot,
                "confirmatory_misclassification_pct": cf_pct,
                "confirmatory_error_count": cf_err,
                "confirmatory_total_cases": cf_tot,
            }
        )

    # Missed IPF rates (false negatives) from stored test confusion matrices
    sc_ipf = sc_groups.get("IPF")
    cf_ipf = cf_groups.get("IPF")
    sc_fn_pct = round(float(sc_ipf["error_rate_pct"]), 1) if sc_ipf else None
    cf_fn_pct = round(float(cf_ipf["error_rate_pct"]), 1) if cf_ipf else None

    sentences = [overlap_stmt]
    if sc_phrases:
        if cf_phrases:
            sentences.append(
                f"In held-out testing, false-positive rates among fibrosing mimics were "
                f"{', '.join(sc_phrases)} in Screening mode and "
                f"{', '.join(cf_phrases)} in Confirmatory mode."
            )
        else:
            sentences.append(
                f"In held-out testing (Screening mode), false-positive rates among fibrosing mimics were "
                f"{', '.join(sc_phrases)}."
            )

    if (not model_result["screening_positive"] or not model_result["confirmatory_positive"]) and sc_fn_pct is not None:
        if cf_fn_pct is not None:
            sentences.append(
                f"A negative call does not exclude IPF: Screening mode missed {sc_fn_pct:.1f}% "
                f"({sc_ipf['error_count']}/{sc_ipf['total_test_cases']}) and Confirmatory mode missed "
                f"{cf_fn_pct:.1f}% ({cf_ipf['error_count']}/{cf_ipf['total_test_cases']}) of test IPF cases."
            )
        else:
            sentences.append(
                f"A negative call does not exclude IPF: Screening mode missed {sc_fn_pct:.1f}% "
                f"({sc_ipf['error_count']}/{sc_ipf['total_test_cases']}) of test IPF cases."
            )

    return {
        "lung_function_overlap_statement": overlap_stmt,
        "group_misclassification_rates": group_rates,
        "screening_missed_ipf_pct": sc_fn_pct,
        "confirmatory_missed_ipf_pct": cf_fn_pct,
        "summary": " ".join(sentences),
    }


def build_clinician_considerations(rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Part (e) For clinician consideration: generic ATS/ERS/JRS/ALAT IPF guideline checklist
    (never a diagnosis or treatment advice).
    """
    r_cc = rules["clinician_considerations"]
    items = list(r_cc["items"])
    ref = str(r_cc["guideline_reference"])
    summary = (
        f"Per {ref}, standard evaluation steps for suspected fibrosing ILD include: "
        + "; ".join(items)
        + "."
    )
    return {
        "guideline_reference": ref,
        "items": items,
        "summary": summary,
    }


def format_deterministic_interpretation_text(structured: Dict[str, Any]) -> str:
    """Format the 5-part structured interpretation into concise plain-language text."""
    lines = [
        f"Physiology: {structured['physiology']['summary']}",
        f"Model result: {structured['model_result']['summary']}",
        f"What pushed it: {structured['what_pushed_it']['summary']}",
        f"Caution: {structured['caution']['summary']}",
        f"For clinician consideration: {structured['clinician_considerations']['summary']}",
        structured["notice"],
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional LLM Rephrasing (Environment-Gated, Off by Default) & Strict Validator
# ---------------------------------------------------------------------------

_NUMBER_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])([+-]?\d+(?:\.\d+)?)(%?)(?![A-Za-z0-9_])")


def is_llm_rephrase_enabled(rules: Optional[Dict[str, Any]] = None) -> bool:
    """Check whether optional LLM rephrasing is enabled via environment variable (default OFF)."""
    env_flag = (
        (rules or {}).get("llm_rephrase", {}).get("env_flag")
        or "ENABLE_LLM_INTERPRETATION_REPHRASE"
    )
    raw = os.getenv(env_flag, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _extract_numeric_tokens(text: str) -> List[Tuple[str, float]]:
    """Extract all standalone numeric tokens (including percentages) from text."""
    results: List[Tuple[str, float]] = []
    for match in _NUMBER_TOKEN_RE.finditer(text):
        num_str = match.group(1)
        pct_str = match.group(2)
        try:
            val = abs(float(num_str))
        except ValueError:
            continue
        results.append((num_str + pct_str, val))
    return results


def _collect_allowed_numbers(obj: Any, bucket: Set[float]) -> None:
    """Recursively collect all numeric values appearing in structured facts or their text."""
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if not (math.isnan(float(obj)) or math.isinf(float(obj))):
            val = abs(float(obj))
            bucket.add(val)
            # Also allow exact rounded 1dp / 2dp / 4dp / int representations of stored floats
            bucket.add(round(val, 4))
            bucket.add(round(val, 2))
            bucket.add(round(val, 1))
            if abs(val - round(val)) < 1e-6:
                bucket.add(float(round(val)))
        return
    if isinstance(obj, str):
        for _, v in _extract_numeric_tokens(obj):
            bucket.add(v)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_allowed_numbers(v, bucket)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_allowed_numbers(item, bucket)


def validate_llm_rephrased_interpretation(
    candidate_text: str,
    structured_facts: Dict[str, Any],
    deterministic_text: str,
    rules: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Strictly validate LLM-rephrased text before returning it:
      1. Must be non-empty.
      2. Every number appearing in `candidate_text` MUST exist in the supplied input facts.
      3. Must contain ZERO diagnosis assertions or treatment recommendations.
    Returns `(is_valid, reasons)`.
    """
    reasons: List[str] = []
    text = (candidate_text or "").strip()
    if not text:
        return False, ["LLM returned empty text."]

    # 1. Exact number conservation check (exclude internal metadata like rules_version)
    allowed_numbers: Set[float] = set()
    facts_for_numbers = {
        k: v
        for k, v in structured_facts.items()
        if k not in ("rules_version", "clinical_review_note", "source", "llm_fallback_reason")
    }
    _collect_allowed_numbers(facts_for_numbers, allowed_numbers)
    _collect_allowed_numbers(deterministic_text, allowed_numbers)

    candidate_nums = _extract_numeric_tokens(text)
    ungrounded: List[str] = []
    for raw_tok, val in candidate_nums:
        if not any(abs(val - ref) <= 1e-6 for ref in allowed_numbers):
            ungrounded.append(raw_tok)
    if ungrounded:
        reasons.append(
            f"Ungrounded number(s) not present in supplied facts: {sorted(set(ungrounded))}"
        )

    # 2. Forbidden diagnosis / treatment wording check
    forbidden_patterns = rules.get("llm_rephrase", {}).get(
        "forbidden_diagnosis_and_treatment_patterns", []
    )
    for pat in forbidden_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            reasons.append(f"Forbidden diagnosis/treatment wording matched: '{m.group(0)}'")

    return len(reasons) == 0, reasons


def _rephrase_with_llm(
    structured_facts: Dict[str, Any],
    deterministic_text: str,
    rules: Dict[str, Any],
    llm_client: Optional[Any] = None,
    model_name: Optional[str] = None,
) -> Tuple[str, str, Optional[str]]:
    """
    Optional LLM rephrasing reusing `backend/agent/orchestrator.py`.
    Sends ONLY de-identified structured facts (no PHI).
    Falls back to `deterministic_text` on any error or validation failure.

    Returns `(final_text, source, fallback_reason)`.
    """
    from backend.agent.orchestrator import (
        DEFAULT_ANTHROPIC_MODEL,
        _create_default_anthropic_client,
    )

    if llm_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            return (
                deterministic_text,
                "deterministic",
                "ANTHROPIC_API_KEY not set; using deterministic interpretation.",
            )
        client = _create_default_anthropic_client(api_key)
    else:
        client = llm_client

    chosen_model = (
        model_name
        or os.getenv("ANTHROPIC_MODEL", "").strip()
        or DEFAULT_ANTHROPIC_MODEL
    )
    notice = structured_facts["notice"]

    # Build strictly de-identified payload (structured numeric/categorical facts only)
    deidentified_facts = {
        "physiology": structured_facts["physiology"],
        "model_result": structured_facts["model_result"],
        "what_pushed_it": structured_facts["what_pushed_it"],
        "caution": structured_facts["caution"],
        "clinician_considerations": structured_facts["clinician_considerations"],
        "notice": notice,
    }

    system_prompt = (
        "You are a clinical decision support text editor. Rephrase ONLY the supplied de-identified "
        "structured JSON facts into a clear five-part clinical result summary labeled: "
        "'Physiology:', 'Model result:', 'What pushed it:', 'Caution:', and 'For clinician consideration:'. "
        "STRICT RULES:\n"
        "1. Use ONLY the supplied facts. Do NOT add medical claims, risk factors, or external statistics.\n"
        "2. Do NOT introduce ANY number that is not explicitly present in the supplied JSON.\n"
        "3. NEVER make a diagnosis, rule in/out IPF, or recommend any drug, medication, or treatment.\n"
        f"4. End verbatim with: {notice}"
    )
    user_prompt = (
        "Rephrase these de-identified facts clearly and concisely without changing or adding any numbers:\n"
        + json.dumps(deidentified_facts, indent=2)
    )

    try:
        resp = client.messages.create(
            model=chosen_model,
            max_tokens=700,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        blocks = list(getattr(resp, "content", []) or [])
        text_parts = [
            (b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
            for b in blocks
            if (b.get("type") if isinstance(b, dict) else getattr(b, "type", "")) == "text"
        ]
        candidate = "\n".join(t.strip() for t in text_parts if t and t.strip()).strip()
    except Exception as exc:
        logger.warning(f"LLM interpretation rephrasing failed ({exc}); falling back to deterministic text.")
        return deterministic_text, "deterministic", f"LLM call error: {exc}"

    is_valid, reasons = validate_llm_rephrased_interpretation(
        candidate_text=candidate,
        structured_facts=deidentified_facts,
        deterministic_text=deterministic_text,
        rules=rules,
    )
    if not is_valid:
        reason_str = "; ".join(reasons)
        logger.warning(f"LLM interpretation failed validation ({reason_str}); falling back to deterministic text.")
        return deterministic_text, "deterministic", f"Validation failed: {reason_str}"

    if notice not in candidate:
        candidate = f"{candidate}\n{notice}"
    return candidate, "llm_rephrased", None


def build_patient_interpretation(
    raw_inputs: Dict[str, Any],
    prob_ipf: float,
    screening_threshold: float,
    confirmatory_threshold: float,
    active_mode: str,
    top_shap_features: List[Dict[str, Any]],
    artifacts: Dict[str, Any],
    rules: Optional[Dict[str, Any]] = None,
    enable_llm_rephrase: Optional[bool] = None,
    llm_client: Optional[Any] = None,
    llm_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entrypoint: build the deterministic 5-part structured interpretation plus
    plain-language text (with optional env-gated, post-validated LLM rephrasing).
    """
    active_rules = rules if rules is not None else load_interpretation_rules()
    notice = str(active_rules.get("mandatory_notice", AUTOMATED_INTERPRETATION_NOTICE))

    physiology = analyze_physiology(raw_inputs, active_rules)
    model_result = analyze_model_result(
        prob_ipf=prob_ipf,
        screening_threshold=screening_threshold,
        confirmatory_threshold=confirmatory_threshold,
        active_mode=active_mode,
        artifacts=artifacts,
        rules=active_rules,
    )
    what_pushed_it = analyze_drivers(
        top_shap_features=top_shap_features,
        raw_inputs=raw_inputs,
        artifacts=artifacts,
        rules=active_rules,
    )
    caution = analyze_caution(
        model_result=model_result,
        artifacts=artifacts,
        rules=active_rules,
    )
    clinician_considerations = build_clinician_considerations(active_rules)

    structured: Dict[str, Any] = {
        "rules_version": active_rules.get("rules_version", "1.0.0"),
        "physiology": physiology,
        "model_result": model_result,
        "what_pushed_it": what_pushed_it,
        "caution": caution,
        "clinician_considerations": clinician_considerations,
        "notice": notice,
        "clinical_review_note": active_rules.get("clinical_review_note", ""),
    }

    deterministic_text = format_deterministic_interpretation_text(structured)
    final_text = deterministic_text
    source = "deterministic"
    fallback_reason: Optional[str] = None

    use_llm = (
        enable_llm_rephrase
        if enable_llm_rephrase is not None
        else is_llm_rephrase_enabled(active_rules)
    )
    if use_llm:
        final_text, source, fallback_reason = _rephrase_with_llm(
            structured_facts=structured,
            deterministic_text=deterministic_text,
            rules=active_rules,
            llm_client=llm_client,
            model_name=llm_model_name,
        )

    structured["source"] = source
    if fallback_reason is not None:
        structured["llm_fallback_reason"] = fallback_reason

    return {
        "interpretation": structured,
        "interpretation_text": final_text,
        "interpretation_source": source,
    }
