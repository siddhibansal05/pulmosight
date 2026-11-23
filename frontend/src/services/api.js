/**
 * Centralized API Client (`frontend/src/services/api.js`).
 *
 * Uses `import.meta.env.VITE_API_BASE_URL` (no hard-coded localhost URLs).
 * Supports `operating_mode: 'screening' | 'confirmatory'` on `POST /predict-ipf`.
 */

import { buildShapPlainLanguageSummary } from '../utils/shapSummary';

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function predict(data) {
  const age = Number(data.age);
  const sex = data.sex; // 'M' or 'F'
  const height = Number(data.height); // Explicit meters (1.0 - 2.5 m)
  const weight = Number(data.weight); // Explicit kilograms (25 - 260 kg)
  const fev1Fvc = Number(data.fev1Fvc); // Explicit decimal ratio (0.20 - 1.15)
  const fev1Pred = Number(data.fev1Pred);
  const fvcPred = Number(data.fvcPred);
  const tlcPred = Number(data.tlcPred);
  const tlcoPred = Number(data.tlcoPred);
  const operatingMode = data.operatingMode || 'screening';
  const hasHrct = Boolean(data.hrctFile);

  const bmi = height > 0 ? weight / (height * height) : 25;

  let gapPoints = 0;
  if (sex === 'M') gapPoints += 1;
  if (age >= 61 && age <= 65) gapPoints += 1;
  else if (age > 65) gapPoints += 2;
  if (fvcPred >= 50 && fvcPred <= 75) gapPoints += 1;
  else if (fvcPred < 50) gapPoints += 2;
  if (tlcoPred >= 36 && tlcoPred <= 55) gapPoints += 1;
  else if (tlcoPred < 36) gapPoints += 2;
  const gapStage =
    gapPoints <= 3 ? 'Stage I' : gapPoints <= 5 ? 'Stage II' : 'Stage III';

  try {
    const response = await fetch(`${API_BASE_URL}/predict-ipf`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sex,
        age,
        height,
        weight,
        fev1fvc_abs: fev1Fvc,
        fev1_pp: fev1Pred,
        fvc_pp: fvcPred,
        tlc_pp: tlcPred,
        tlco_pp: tlcoPred,
        operating_mode: operatingMode,
      }),
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${response.status}`);
    }

    const apiRes = await response.json();
    const probability = Number(apiRes.ipf_probability);
    const probabilityPercent = Math.round(probability * 100);
    const riskMap = { low: 'Low', medium: 'Medium', high: 'High' };

    const featureImportance = (apiRes.top_5_shap_features || []).map(
      (item) => ({
        rawFeature: item.feature,
        feature: item.display_name || item.feature,
        shortName: (item.display_name || item.feature).replace(
          ' Predicted',
          ' pred'
        ),
        value: item.value,
        impact: Number(item.shap_value),
        absImpact: Number(item.abs_shap),
        direction: item.direction,
        clinicalNote: `${item.direction} (SHAP = ${
          item.shap_value >= 0 ? `+${item.shap_value}` : item.shap_value
        })`,
      })
    );

    const shapPlainSummary =
      apiRes.shap_plain_summary ||
      buildShapPlainLanguageSummary(featureImportance, data);

    return {
      isOfflineDemo: false,
      offlineBannerMessage: null,
      dataSource: apiRes.data_source || 'real',
      modelVersion: apiRes.model_version || '2.2.0',
      operatingMode: apiRes.operating_mode || operatingMode,
      threshold: apiRes.threshold,
      screeningThreshold: apiRes.screening_threshold,
      confirmatoryThreshold: apiRes.confirmatory_threshold,
      operatingModesSummary: apiRes.operating_modes_summary || null,
      sensitivityDisclosure: apiRes.sensitivity_disclosure || '',
      prediction: apiRes.predicted_label,
      probability,
      probabilityPercent,
      riskLevel: riskMap[apiRes.risk_level] || 'Medium',
      modelUsed: apiRes.model_used,
      timestamp: new Date().toISOString(),
      caseId: `CASE-${Math.floor(100000 + Math.random() * 900000)}`,
      patientSummary: {
        ...data,
        bmi: Number(bmi.toFixed(1)),
        normalizedRatio: Number(fev1Fvc.toFixed(2)),
        gapPoints,
        gapStage,
        hrctPatternLabel: hasHrct
          ? data.hrctFile?.isZipSeries
            ? `Full CT Series Uploaded (${data.hrctFile.name})`
            : 'Preview Image Attached (Upload .zip DICOM series for HRCT/Fusion progression)'
          : 'Not Provided',
      },
      featureImportance,
      shapPlainSummary,
      interpretation: apiRes.interpretation || null,
      interpretationText: apiRes.interpretation_text || '',
      interpretationSource: apiRes.interpretation_source || 'deterministic',
      disclaimer:
        apiRes.disclaimer ||
        'Decision support only, not a clinical diagnosis.',
    };
  } catch (err) {
    const tlcoContrib = Number(((75 - tlcoPred) * 0.005).toFixed(3));
    const fvcContrib = Number(((80 - fvcPred) * 0.004).toFixed(3));
    const tlcContrib = Number(((82 - tlcPred) * 0.003).toFixed(3));
    const ratioContrib = Number(((fev1Fvc - 0.78) * 0.35).toFixed(3));
    const ageContrib = Number(((age - 60) * 0.004).toFixed(3));

    const rawScore =
      0.35 + tlcoContrib + fvcContrib + tlcContrib + ratioContrib + ageContrib;
    const probability = Math.min(0.95, Math.max(0.05, Number(rawScore.toFixed(2))));

    const offlineFeatures = [
      {
        rawFeature: 'tlco_pp',
        feature: 'TLCO % Predicted',
        shortName: 'TLCO % pred',
        value: `${tlcoPred}%`,
        impact: tlcoContrib,
        absImpact: Math.abs(tlcoContrib),
        direction:
          tlcoContrib >= 0 ? 'Increases IPF Risk' : 'Decreases IPF Risk',
        clinicalNote: 'Offline demo approximation (start FastAPI for real SHAP)',
      },
      {
        rawFeature: 'fvc_pp',
        feature: 'FVC % Predicted',
        shortName: 'FVC % pred',
        value: `${fvcPred}%`,
        impact: fvcContrib,
        absImpact: Math.abs(fvcContrib),
        direction:
          fvcContrib >= 0 ? 'Increases IPF Risk' : 'Decreases IPF Risk',
        clinicalNote: 'Offline demo approximation (start FastAPI for real SHAP)',
      },
      {
        rawFeature: 'tlc_pp',
        feature: 'TLC % Predicted',
        shortName: 'TLC % pred',
        value: `${tlcPred}%`,
        impact: tlcContrib,
        absImpact: Math.abs(tlcContrib),
        direction:
          tlcContrib >= 0 ? 'Increases IPF Risk' : 'Decreases IPF Risk',
        clinicalNote: 'Offline demo approximation (start FastAPI for real SHAP)',
      },
      {
        rawFeature: 'age',
        feature: 'Patient Age (yrs)',
        shortName: 'Age',
        value: `${age} yrs`,
        impact: ageContrib,
        absImpact: Math.abs(ageContrib),
        direction:
          ageContrib >= 0 ? 'Increases IPF Risk' : 'Decreases IPF Risk',
        clinicalNote: 'Offline demo approximation (start FastAPI for real SHAP)',
      },
      {
        rawFeature: 'fev1fvc_abs',
        feature: 'FEV1/FVC Ratio',
        shortName: 'FEV1/FVC',
        value: fev1Fvc.toFixed(2),
        impact: ratioContrib,
        absImpact: Math.abs(ratioContrib),
        direction:
          ratioContrib >= 0 ? 'Increases IPF Risk' : 'Decreases IPF Risk',
        clinicalNote: 'Offline demo approximation (start FastAPI for real SHAP)',
      },
    ].sort((a, b) => b.absImpact - a.absImpact);

    return {
      isOfflineDemo: true,
      offlineBannerMessage:
        'Backend offline: demo values, not a real prediction',
      offlineErrorDetail: err?.message || 'Unable to reach FastAPI server',
      dataSource: 'synthetic',
      modelVersion: 'offline-demo',
      operatingMode,
      threshold: 0.5,
      screeningThreshold: 0.5,
      confirmatoryThreshold: 0.5,
      operatingModesSummary: null,
      prediction: probability >= 0.5 ? 'IPF' : 'Non-IPF',
      probability,
      probabilityPercent: Math.round(probability * 100),
      riskLevel:
        probability >= 0.65 ? 'High' : probability >= 0.35 ? 'Medium' : 'Low',
      modelUsed: 'OFFLINE DEMO (Backend Unreachable)',
      timestamp: new Date().toISOString(),
      caseId: 'DEMO-OFFLINE',
      patientSummary: {
        ...data,
        bmi: Number(bmi.toFixed(1)),
        normalizedRatio: Number(fev1Fvc.toFixed(2)),
        gapPoints,
        gapStage,
        hrctPatternLabel: 'Backend Offline',
      },
      featureImportance: offlineFeatures,
      shapPlainSummary: buildShapPlainLanguageSummary(offlineFeatures, data),
      interpretation: null,
      interpretationText:
        'Backend offline: automated interpretation is unavailable until the FastAPI backend is running.\nAutomated interpretation, decision support only. Verify clinically.',
      interpretationSource: 'deterministic',
      disclaimer: 'Decision support only, not a clinical diagnosis.',
    };
  }
}

export async function predictProgressionMultimodal(data) {
  const age = Number(data.age);
  const sex = data.sex || 'M';
  const fvcPred = Number(data.fvcPred);
  const refFvcMl = sex === 'M' ? 4200 : 3100;
  const estFvcMl = Math.min(
    6500,
    Math.max(500, Math.round(refFvcMl * (fvcPred / 100)))
  );

  const formBody = new FormData();
  formBody.append('age', String(age));
  formBody.append('sex', sex);
  formBody.append('fvc', String(estFvcMl));
  formBody.append('percent', String(fvcPred));
  formBody.append('smoking_status', data.smokingStatus || 'Ex-smoker');
  formBody.append('horizon_weeks', '52');

  const rawFile = data.hrctFile?.rawFile;
  const isRealZip =
    rawFile &&
    typeof rawFile.name === 'string' &&
    rawFile.name.toLowerCase().endsWith('.zip');
  if (isRealZip) {
    formBody.append('ct_series_zip', rawFile, rawFile.name);
  }

  try {
    const response = await fetch(`${API_BASE_URL}/predict-progression-multimodal`, {
      method: 'POST',
      body: formBody,
    });
    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      return {
        available: false,
        httpStatus: response.status,
        error: errBody.detail || `HTTP ${response.status}`,
        modality_used: { clinical: false, hrct: false, fusion: false },
        hrct_only: null,
        fusion: null,
      };
    }
    const payload = await response.json();
    return {
      available: true,
      httpStatus: 200,
      error: null,
      ...payload,
    };
  } catch (err) {
    return {
      available: false,
      httpStatus: 0,
      error: err?.message || 'Unable to reach /predict-progression-multimodal',
      modality_used: { clinical: false, hrct: false, fusion: false },
      hrct_only: null,
      fusion: null,
    };
  }
}

export async function fetchModelInfo() {
  const response = await fetch(`${API_BASE_URL}/model-info`);
  if (!response.ok) {
    throw new Error(`Failed to load /model-info (HTTP ${response.status})`);
  }
  return await response.json();
}

export async function fetchEdaInsights() {
  const response = await fetch(`${API_BASE_URL}/eda`);
  if (!response.ok) {
    throw new Error(`Failed to load /eda (HTTP ${response.status})`);
  }
  return await response.json();
}
