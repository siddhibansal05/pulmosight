/**
 * Deterministic Plain-Language SHAP Summary Generator (`frontend/src/utils/shapSummary.js`).
 *
 * Mirrors `backend/src/explain.py::build_shap_plain_language_summary` (NO LLM).
 * Used as a client-side fallback when `shap_plain_summary` is not returned by the API.
 */

export const SHAP_MANDATORY_CAVEAT =
  'SHAP shows what the model used, not causes. Age partly reflects differences between IPF and non-IPF patients in this dataset.';

function formatFeaturePhrase(item, patientData = {}) {
  const rawName = String(item.rawFeature || item.feature || '').toLowerCase();
  const valStr = String(item.value || '');

  if (rawName === 'sex' || rawName.includes('biological sex')) {
    const isMale =
      String(patientData.sex || valStr).trim().toUpperCase().startsWith('M');
    return isMale ? 'male sex' : 'female sex';
  }
  if (rawName === 'age' || rawName.includes('age')) {
    const num = Number(patientData.age ?? parseFloat(valStr));
    if (!Number.isNaN(num)) {
      return `age ${Math.abs(num - Math.round(num)) < 0.05 ? Math.round(num) : num.toFixed(1)}`;
    }
    return `age (${valStr})`;
  }
  if (rawName === 'tlco_pp' || rawName.includes('tlco')) {
    const num = Number(patientData.tlcoPred ?? parseFloat(valStr));
    if (!Number.isNaN(num)) {
      const qual = num < 60 ? 'low ' : num < 80 ? 'reduced ' : 'preserved ';
      return `${qual}TLCO (${num.toFixed(1)}%)`;
    }
    return `TLCO (${valStr})`;
  }
  if (rawName === 'tlc_pp' || rawName.includes('tlc')) {
    const num = Number(patientData.tlcPred ?? parseFloat(valStr));
    if (!Number.isNaN(num)) {
      const qual = num < 80 ? 'reduced ' : num > 120 ? 'elevated ' : '';
      return `${qual}TLC (${num.toFixed(1)}%)`;
    }
    return `TLC (${valStr})`;
  }
  if (rawName === 'fvc_pp' || (rawName.includes('fvc') && !rawName.includes('fev1'))) {
    const num = Number(patientData.fvcPred ?? parseFloat(valStr));
    if (!Number.isNaN(num)) {
      const qual = num < 80 ? 'reduced ' : 'preserved ';
      return `${qual}FVC (${num.toFixed(1)}%)`;
    }
    return `FVC (${valStr})`;
  }
  if (rawName === 'fev1_pp' || (rawName.includes('fev1') && !rawName.includes('fvc'))) {
    const num = Number(patientData.fev1Pred ?? parseFloat(valStr));
    if (!Number.isNaN(num)) return `FEV1 (${num.toFixed(1)}%)`;
    return `FEV1 (${valStr})`;
  }
  if (rawName === 'fev1fvc_abs' || rawName.includes('fev1/fvc')) {
    const num = Number(patientData.fev1Fvc ?? parseFloat(valStr));
    if (!Number.isNaN(num)) {
      const qual = num >= 0.8 ? 'elevated ' : num < 0.7 ? 'low ' : '';
      return `${qual}FEV1/FVC ratio (${num.toFixed(2)})`;
    }
    return `FEV1/FVC ratio (${valStr})`;
  }
  return `${item.feature} (${valStr})`;
}

function joinPhrases(phrases) {
  if (!phrases.length) return '';
  if (phrases.length === 1) return phrases[0];
  if (phrases.length === 2) return `${phrases[0]} and ${phrases[1]}`;
  return `${phrases.slice(0, -1).join(', ')}, and ${phrases[phrases.length - 1]}`;
}

export function buildShapPlainLanguageSummary(featureImportance = [], patientData = {}) {
  const toward = [...featureImportance]
    .filter((f) => Number(f.impact) > 0)
    .sort((a, b) => Number(b.impact) - Number(a.impact));
  const away = [...featureImportance]
    .filter((f) => Number(f.impact) < 0)
    .sort((a, b) => Math.abs(Number(b.impact)) - Math.abs(Number(a.impact)));

  const fmtWithShap = (item) => {
    const v = Number(item.impact);
    const sign = v >= 0 ? '+' : '';
    return `${formatFeaturePhrase(item, patientData)} (${sign}${v.toFixed(2)})`;
  };

  let lead = '';
  if (toward.length > 0) {
    const primary = toward.slice(0, 2);
    const secondary = toward.slice(2, 4);
    lead = `${joinPhrases(primary.map(fmtWithShap))} pushed most toward IPF`;
    if (secondary.length > 0) {
      lead += `, followed by ${joinPhrases(secondary.map(fmtWithShap))}`;
    }
    if (away.length > 0) {
      const awayTop = away.slice(0, 2);
      const maxAway = Math.max(...awayTop.map((x) => Math.abs(Number(x.impact))));
      const verb = maxAway < 0.25 ? 'slightly lowered' : 'lowered';
      lead += `, while ${joinPhrases(awayTop.map(fmtWithShap))} ${verb} the likelihood.`;
    } else {
      lead += ', with no top factors lowering the predicted likelihood.';
    }
  } else if (away.length > 0) {
    lead = `${joinPhrases(away.slice(0, 3).map(fmtWithShap))} pushed most away from IPF, with no top factors increasing the predicted likelihood.`;
  } else {
    lead = 'No individual clinical feature exerted a strong directional shift away from the cohort baseline.';
  }

  lead = lead.charAt(0).toUpperCase() + lead.slice(1);
  const scaleSentence =
    "SHAP values are on the model's log-odds scale (positive values push toward IPF and negative values push away from IPF relative to the training baseline).";

  return `${lead} ${scaleSentence} ${SHAP_MANDATORY_CAVEAT}`;
}
