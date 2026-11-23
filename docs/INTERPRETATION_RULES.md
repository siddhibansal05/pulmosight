# PulmoSight CDSS — Clinical Result Interpretation Rules & Reference Thresholds

> **Mandatory Notice:** *Automated interpretation, decision support only. Verify clinically.*  
> **Clinician Review Requirement:** *All threshold rules and text templates documented below should be reviewed and approved by a qualified clinician before any real clinical use.*

---

## 1. Purpose & Evidence-Grounding Policy

The PulmoSight result interpretation engine (`backend/src/interpretation.py`) generates a five-part **"Interpretation of this result"** summary returned by `POST /predict-ipf` (as structured `interpretation` and plain-language `interpretation_text`), rendered under the SHAP chart on Page 2, and included on Page 2 of the downloadable PDF report.

Every statement produced by the engine is strictly grounded in one of four traceable sources:
1. The patient's supplied demographic and pulmonary function values,
2. Configurable physiological and guideline rules stored in [`backend/config/interpretation_rules.json`](../backend/config/interpretation_rules.json),
3. The calibrated model output and dual out-of-fold (OOF) operating thresholds, and
4. Real held-out test-set statistics stored in [`backend/models/model_artifacts.json`](../backend/models/model_artifacts.json) (empirical reliability bin rates, Zenodo cohort age medians, and per-diagnosis-group misclassification rates).

The engine **never** invents medical claims, **never** issues a clinical diagnosis, and **never** gives medication or treatment advice.

---

## 2. Approximation Notice (Percent Predicted vs. GLI LLN / Z-Scores)

> **Important Approximation Caveat:** Modern ERS/ATS technical standards (Stanojevic et al., *Eur Respir J* 2022) recommend defining abnormality using age-, sex-, and height-specific **lower limits of normal (LLN, 5th percentile, z-score < −1.645)** rather than fixed cutoffs. Because routine clinical referrals and `/predict-ipf` inputs supply **percent-predicted (`% pred`) values and an absolute `FEV1/FVC` decimal ratio**, the fixed thresholds below (`80% predicted` for volumes/TLCO and `0.70` for `FEV1/FVC`) are **documented approximations** when only percent predicted is available. Fixed cutoffs can over-classify airflow obstruction in older adults and under-classify impairment in younger adults; borderline values are explicitly flagged in the output.

---

## 3. Rule-by-Rule Threshold Specification (`backend/config/interpretation_rules.json`)

### Part (a): Physiology — Ventilatory Pattern & Gas-Transfer Severity

#### 3.1 Spirometry & Lung Volume Cutoffs
| Parameter | Config Key (`interpretation_rules.json`) | Threshold | Classification | Borderline Margin | Citation |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **FEV1/FVC Ratio** | `ratio_thresholds.fev1fvc_preserved_at_or_above` | `>= 0.70` | Ratio preserved | `±0.03` (`0.67–0.73`) | Pellegrino et al. (2005); Stanojevic et al. (2022) (fixed-ratio approximation) |
| **FEV1/FVC Ratio** | `ratio_thresholds.fev1fvc_preserved_at_or_above` | `< 0.70` | Ratio reduced (airflow obstruction) | `±0.03` | Pellegrino et al. (2005); Stanojevic et al. (2022) |
| **TLC % Predicted** | `volume_thresholds.tlc_reduced_below_pp` | `< 80.0%` | Reduced lung volume (restriction criterion when TLC present) | `±5.0%` (`75%–85%`) | Pellegrino et al. (2005); Stanojevic et al. (2022) |
| **TLC % Predicted** | `volume_thresholds.tlc_elevated_above_pp` | `> 120.0%` | Elevated TLC (hyperinflation note if obstructive) | — | Pellegrino et al. (2005) |
| **FVC % Predicted** | `volume_thresholds.fvc_reduced_below_pp` | `< 80.0%` | Reduced FVC (fallback volume criterion only if TLC missing; flagged as non-specific if TLC ≥ 80%) | `±5.0%` (`75%–85%`) | Pellegrino et al. (2005); Stanojevic et al. (2022) |
| **FEV1 % Predicted** | `volume_thresholds.fev1_reduced_below_pp` | `< 80.0%` | Reduced FEV1 | — | Pellegrino et al. (2005) |

#### 3.2 Ventilatory Pattern Decision Table (`restrictive_requires_tlc_when_present: true`)
| Assigned Pattern | `FEV1/FVC` | Volume Criterion (`TLC` preferred; `FVC` fallback if `TLC` absent) | Notes & Citation |
| :--- | :---: | :--- | :--- |
| **`restrictive`** | `>= 0.70` | `TLC < 80%` (or `TLC` missing and `FVC < 80%`) | Classic restrictive physiology (Pellegrino et al., 2005; Stanojevic et al., 2022). |
| **`obstructive`** | `< 0.70` | `TLC >= 80%` (or `TLC` missing and `FVC >= 80%`) | Airflow obstruction without reduced TLC; notes hyperinflation if `TLC > 120%`. |
| **`mixed`** | `< 0.70` | `TLC < 80%` (or `TLC` missing and `FVC < 80%`) | Combined obstructive and restrictive defect (Pellegrino et al., 2005). |
| **`normal`** | `>= 0.70` | `TLC >= 80%`, `FVC >= 80%`, and `FEV1 >= 80%` | Preserved ventilatory mechanics (isolated gas-transfer impairment still checked via TLCO). |
| **`non_specific`** | `>= 0.70` | `TLC >= 80%` AND (`FVC < 80%` or `FEV1 < 80%`) | Per ATS/ERS 2005/2022, normal TLC excludes true lung-volume restriction despite mildly reduced spirometry. Configurable via `restrictive_requires_tlc_when_present`. |
| **`indeterminate`** | Missing or both `TLC` & `FVC` missing | Incomplete input | Explicitly lists missing fields rather than guessing. |

#### 3.3 Gas-Transfer (`TLCO % Predicted`) Severity Bands
| Band Key | Range (`TLCO % pred`) | Label | Borderline Cutoff Check | Citation |
| :--- | :---: | :--- | :---: | :--- |
| `normal` | `>= 80.0%` | Normal (preserved gas transfer) | Within `±3.0%` of `80%` | Pellegrino et al. (*Eur Respir J* 2005) |
| `mild` | `60.0%` to `< 80.0%` | Mildly reduced gas transfer | Within `±3.0%` of `80%` or `60%` | Pellegrino et al. (*Eur Respir J* 2005) |
| `moderate` | `40.0%` to `< 60.0%` | Moderately reduced gas transfer | Within `±3.0%` of `60%` or `40%` | Pellegrino et al. (*Eur Respir J* 2005) |
| `severe` | `< 40.0%` | Severely reduced gas transfer | Within `±3.0%` of `40%` | Pellegrino et al. (*Eur Respir J* 2005) |
| `missing` | Absent / `NaN` | Not available | — | Edge-case handling |

When `pattern` is `normal` or `non_specific` and `TLCO < 80%`, the engine explicitly notes an **isolated gas-transfer deficit despite preserved lung volumes**.

#### 3.4 Extreme Value Sanity Checks (`extreme_value_bounds`)
| Parameter | Warning Flag Below | Warning Flag Above | Action |
| :--- | :---: | :---: | :--- |
| `age` | `< 25` years | `> 90` years | Appends verification warning to physiology text |
| `fev1fvc_abs` | `< 0.35` | `> 1.00` | Appends unit/measurement verification warning |
| `fev1_pp`, `fvc_pp`, `tlco_pp` | `< 25.0%` | `> 150.0%` | Appends unit/measurement verification warning |
| `tlc_pp` | `< 30.0%` | `> 150.0%` | Appends unit/measurement verification warning |

---

### Part (b): Model Result & Empirical Test-Set Reliability Table

Reports the calibrated IPF probability, the positive/negative call in **both** OOF-tuned operating modes:
- **Screening Mode (`threshold = 0.1868`):** tuned for 80% OOF training sensitivity (`Positive (IPF)` if `prob >= 0.1868`, else `Negative (Non-IPF)`).
- **Confirmatory Mode (`threshold = 0.3429`):** tuned for ≥92% OOF training specificity (`Positive (IPF)` if `prob >= 0.3429`, else `Negative (Non-IPF)`).
- **Conflict / Borderline Flagging:** Explicitly flags **conflicting operating modes** whenever Screening is Positive and Confirmatory is Negative (`0.1868 <= prob < 0.3429`), and flags **borderline scores** within `±0.05` (`borderline_probability_margin`) of either threshold.

#### Held-Out Test Calibration Reliability Table (`model_artifacts.json` — Calibrated XGBoost, `n = 1,362`, `151` IPF)
| Calibrated Probability Bin | Held-Out Test Patients (`n`) | Observed IPF (`k`) | Empirical IPF Rate | 95% Wilson CI | Mean Predicted Probability |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `0%–5%` (`0.00–0.05`) | `931` | `10` | **`1.1%`** | `0.6%–2.0%` | `0.0141` |
| `5%–10%` (`0.05–0.10`) | `84` | `8` | **`9.5%`** | `4.9%–17.7%` | `0.0701` |
| `10%–20%` (`0.10–0.20`) | `85` | `18` | **`21.2%`** | `13.8%–31.0%` | `0.1451` |
| `20%–35%` (`0.20–0.35`) | `93` | `25` | **`26.9%`** | `18.9%–36.7%` | `0.2643` |
| `35%–50%` (`0.35–0.50`) | `81` | `40` | **`49.4%`** | `38.8%–60.0%` | `0.4300` |
| `≥50%` (`0.50–1.00`) | `88` | `50` | **`56.8%`** | `46.4%–66.7%` | `0.5547` |

*Note: Maximum calibrated probability observed on the held-out test set is `0.6359`. Scores above `0.6359` carry an explicit note that the score exceeds the highest held-out test observation.*

---

### Part (c): What Pushed It (Top Contributors & Cohort Age Context)

- Lists up to `3` features pushing the score toward IPF (`shap_value > 0`) and up to `2` features pushing the score away from IPF (`shap_value < 0`), filtering out negligible contributions (`|shap_value| < 0.03`), expressed strictly in plain words with the patient's own values.
- **Age Driver Context Rule (`age_driver_rank_cutoff = 3`):** Whenever `age` ranks among the top 3 features by `|SHAP|`, the engine appends one factual sentence citing the real Zenodo cohort medians stored in `model_artifacts.json`:
  - **IPF median age:** `68.9 years` (IQR `62.1–75.5`, `n = 757`)
  - **Non-IPF median age:** `48.9 years` (IQR `36.9–60.0`, `n = 6,051`)

---

### Part (d): Caution & Per-Group Misclassification Rates

- Always states: **"Lung function alone cannot separate IPF from other fibrosing interstitial lung diseases."**
- Quotes real held-out test false-positive rates from `model_artifacts.json` for key fibrosing mimics:
  - **Screening Mode (`threshold = 0.1868`):** `i-NSIP 60.9% (14/23)`, `u-ILD 31.1% (14/45)`, `CTD 35.0% (42/120)`.
  - **Confirmatory Mode (`threshold = 0.3429`):** `i-NSIP 39.1% (9/23)`, `u-ILD 24.4% (11/45)`, `CTD 17.5% (21/120)`.
- Whenever either operating mode returns **Negative (Non-IPF)**, explicitly reminds that a negative score does not exclude IPF: Screening mode missed **`23.8%` (`36/151`)** and Confirmatory mode missed **`39.1%` (`59/151`)** of held-out test IPF cases.

---

### Part (e): For Clinician Consideration (Guideline-Grounded Checklist)

Derived directly from the **ATS/ERS/JRS/ALAT Clinical Practice Guidelines for Diagnosis of IPF** (Raghu et al., 2018; 2022 update). Strictly generic evaluation context — **never a diagnosis, never drug or treatment advice**:
1. **HRCT pattern review** (evaluating UIP, probable UIP, indeterminate, or alternative patterns)
2. **Exposure and autoimmune history** (environmental/occupational exposures, medications, and connective tissue disease evaluation)
3. **Multidisciplinary discussion (MDD)** integrating clinical, radiologic, and laboratory findings

---

## 4. Optional LLM Rephrasing & Post-Generation Guardrails

- **Disabled by default.** Controlled by environment variable `ENABLE_LLM_INTERPRETATION_REPHRASE=true` (reuses `backend/agent/orchestrator.py` and `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL`).
- **De-identification:** Sends **only** the de-identified structured numeric/categorical facts dictionary (never names, IDs, MRNs, or free text).
- **Mandatory Post-Generation Validation (`validate_llm_rephrased_interpretation`):**
  1. **Exact Number Conservation:** Every number appearing in the LLM output must exist in the supplied input facts; any invented or ungrounded number triggers rejection.
  2. **Zero Diagnosis / Treatment Wording:** Regex checks (`forbidden_diagnosis_and_treatment_patterns` in `interpretation_rules.json`) reject any diagnostic confirmation/exclusion phrasing (`"diagnosed with IPF"`, `"patient has IPF"`, `"rules out IPF"`, etc.) or therapeutic wording (`"prescribe"`, `"pirfenidone"`, `"nintedanib"`, `"antifibrotic"`, `"corticosteroids"`, `"start therapy"`, etc.).
  3. **Automatic Fallback:** Any validation failure, missing API key, or API error falls back automatically to the deterministic `interpretation_text` (`interpretation_source: "deterministic"`).

---

## 5. Primary References

1. **Pellegrino R, Viegi G, Brusasco V, et al.** *Interpretative strategies for lung function tests.* Eur Respir J. 2005;26(5):948–968. doi:`10.1183/09031936.05.00035205`.
2. **Stanojevic S, Kaminsky DA, Miller MR, et al.** *ERS/ATS technical standard on interpretive strategies for routine lung function tests.* Eur Respir J. 2022;60(1):2101499. doi:`10.1183/13993003.01499-2021`.
3. **Raghu G, Remy-Jardin M, Myers JL, et al.** *Diagnosis of Idiopathic Pulmonary Fibrosis. An Official ATS/ERS/JRS/ALAT Clinical Practice Guideline.* Am J Respir Crit Care Med. 2018;198(5):e44–e68. doi:`10.1164/rccm.201807-1255ST`.
4. **Raghu G, Remy-Jardin M, Richeldi L, et al.** *Idiopathic Pulmonary Fibrosis (an Update) and Progressive Pulmonary Fibrosis in Adults: An Official ATS/ERS/JRS/ALAT Clinical Practice Guideline.* Am J Respir Crit Care Med. 2022;205(9):e18–e47. doi:`10.1164/rccm.202202-0399ST`.

---

> **Mandatory Notice:** *Automated interpretation, decision support only. Verify clinically.*  
> *These rules must be reviewed by a qualified clinician before any real clinical use.*
