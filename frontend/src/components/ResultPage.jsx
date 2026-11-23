import React from 'react';
import {
  FileDown,
  ShieldAlert,
  ArrowLeft,
  CheckCircle2,
  AlertTriangle,
  BarChart3,
  Stethoscope,
  Activity,
  WifiOff,
  Sliders,
  Info,
  Layers,
  ScanLine,
  ShieldCheck,
  ClipboardCheck,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Cell,
} from 'recharts';
import { downloadClinicalReportPdf } from '../utils/pdfGenerator';

function ProbabilityGauge({ probabilityPercent, prediction, riskLevel }) {
  const clamped = Math.min(100, Math.max(0, Number(probabilityPercent) || 0));
  const radius = 84;
  const circumference = Math.PI * radius;
  const strokeDashoffset = circumference - (clamped / 100) * circumference;
  const angleDeg = -90 + (clamped / 100) * 180;

  let arcColor = '#0284c7';
  if (riskLevel === 'High') arcColor = '#e11d48';
  else if (riskLevel === 'Medium') arcColor = '#f59e0b';
  else arcColor = '#10b981';

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-60 h-36 flex items-end justify-center overflow-hidden">
        <svg viewBox="0 0 220 130" className="w-full h-full overflow-visible">
          <path
            d="M 26 110 A 84 84 0 0 1 194 110"
            fill="none"
            stroke="currentColor"
            className="text-slate-200 dark:text-slate-800"
            strokeWidth="16"
            strokeLinecap="round"
          />
          <text x="12" y="124" className="fill-slate-400 text-[10px] font-mono">
            0%
          </text>
          <text x="101" y="14" className="fill-slate-400 text-[10px] font-mono">
            50%
          </text>
          <text x="186" y="124" className="fill-slate-400 text-[10px] font-mono">
            100%
          </text>
          <path
            d="M 26 110 A 84 84 0 0 1 194 110"
            fill="none"
            stroke={arcColor}
            strokeWidth="16"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            style={{ transition: 'stroke-dashoffset 0.6s ease-out, stroke 0.3s ease' }}
          />
          <g transform={`translate(110, 110) rotate(${angleDeg})`}>
            <path d="M -3.5 0 L 0 -70 L 3.5 0 Z" fill={arcColor} />
            <circle cx="0" cy="0" r="7" fill={arcColor} />
            <circle cx="0" cy="0" r="3" fill="#ffffff" />
          </g>
        </svg>
      </div>

      <div className="text-center -mt-1">
        <div className="text-3xl font-extrabold font-mono tracking-tight text-slate-900 dark:text-white">
          {clamped}%
        </div>
        <div className="text-xs font-medium text-slate-500 dark:text-slate-400">
          Calibrated IPF Probability ({prediction})
        </div>
      </div>
    </div>
  );
}

export default function ResultPage({
  result,
  multimodalProgression = null,
  onBackToInput,
  onSwitchOperatingMode,
  darkMode,
}) {
  if (!result) return null;

  const {
    isOfflineDemo,
    offlineBannerMessage,
    dataSource,
    modelVersion,
    operatingMode = 'screening',
    threshold,
    screeningThreshold,
    confirmatoryThreshold,
    operatingModesSummary,
    prediction,
    probabilityPercent,
    riskLevel,
    caseId,
    timestamp,
    modelUsed,
    patientSummary = {},
    featureImportance = [],
    interpretation = null,
    interpretationText = '',
    interpretationSource = 'deterministic',
  } = result;

  const hrctScore = multimodalProgression?.hrct_only || null;
  const fusionScore = multimodalProgression?.fusion || null;
  const cvCiSummary = multimodalProgression?.cv_ci_summary || {};
  const selectedFusionDisplay =
    multimodalProgression?.selected_fusion_display_name ||
    cvCiSummary?.fusion?.method_display_name ||
    'Selected Fusion Method';
  const ctErrorMsg =
    multimodalProgression && !multimodalProgression.available && multimodalProgression.error
      ? multimodalProgression.error
      : null;

  const riskBadgeStyles = {
    Low: {
      bg: 'bg-emerald-50 dark:bg-emerald-950/60',
      border: 'border-emerald-200 dark:border-emerald-800/80',
      text: 'text-emerald-700 dark:text-emerald-300',
      dot: 'bg-emerald-500',
    },
    Medium: {
      bg: 'bg-amber-50 dark:bg-amber-950/60',
      border: 'border-amber-200 dark:border-amber-800/80',
      text: 'text-amber-700 dark:text-amber-300',
      dot: 'bg-amber-500',
    },
    High: {
      bg: 'bg-rose-50 dark:bg-rose-950/60',
      border: 'border-rose-200 dark:border-rose-800/80',
      text: 'text-rose-700 dark:text-rose-300',
      dot: 'bg-rose-500',
    },
  };

  const activeRiskStyle = riskBadgeStyles[riskLevel] || riskBadgeStyles.Medium;

  return (
    <div className="space-y-6">
      {/* CRITICAL SAFETY REQUIREMENT: Red banner when backend is offline */}
      {isOfflineDemo && (
        <div
          role="alert"
          className="rounded-xl bg-rose-600 text-white px-5 py-4 flex items-center justify-between gap-3 shadow-md border border-rose-700"
        >
          <div className="flex items-center gap-3">
            <WifiOff className="w-6 h-6 shrink-0" />
            <div>
              <p className="font-bold text-sm sm:text-base">
                {offlineBannerMessage || 'Backend offline: demo values, not a real prediction'}
              </p>
              <p className="text-xs text-rose-100 mt-0.5">
                Start the FastAPI server (<code className="font-mono">uvicorn backend.api.main:app --port 8000</code>) for real calibrated predictions. PDF export is disabled in offline demo mode.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Top Action Header */}
      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-5 shadow-clinical flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-medical-600 dark:text-sky-400">
            <Stethoscope className="w-4 h-4" />
            <span>Page 2 • Diagnostic Support Result & Explainability</span>
          </div>
          <div className="flex flex-wrap items-center gap-2.5 mt-1">
            <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white">
              Diagnostic Prediction Report
            </h1>
            <span className="px-2.5 py-0.5 rounded-md bg-slate-100 dark:bg-slate-800 text-xs font-mono text-slate-600 dark:text-slate-300">
              {caseId}
            </span>
            <span
              className={`px-2.5 py-0.5 rounded-md text-xs font-mono font-semibold ${
                dataSource === 'real' && !isOfflineDemo
                  ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800'
                  : 'bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300 border border-rose-200 dark:border-rose-800'
              }`}
            >
              data_source: {isOfflineDemo ? 'offline-demo' : dataSource}
            </span>
            {threshold !== undefined && (
              <span className="px-2.5 py-0.5 rounded-md bg-medical-50 dark:bg-sky-950/60 text-xs font-mono text-medical-700 dark:text-sky-300 border border-medical-200 dark:border-sky-800">
                Active Threshold ({operatingMode}): {threshold}
              </span>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={onBackToInput}
            className="inline-flex items-center gap-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 px-4 py-2.5 text-sm font-medium transition"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Edit Patient Input</span>
          </button>

          <button
            type="button"
            disabled={isOfflineDemo}
            onClick={() => downloadClinicalReportPdf(result)}
            title={
              isOfflineDemo
                ? 'PDF export is disabled while the backend is offline'
                : 'Download clinical report as PDF'
            }
            className="inline-flex items-center gap-2 rounded-xl bg-medical-600 hover:bg-medical-700 disabled:bg-slate-300 dark:disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed text-white px-4 py-2.5 text-sm font-semibold shadow-sm transition"
          >
            <FileDown className="w-4 h-4" />
            <span>{isOfflineDemo ? 'PDF Disabled (Offline)' : 'Download Result as PDF'}</span>
          </button>
        </div>
      </div>

      {/* Operating Mode Switcher: Screening Mode (80% OOF Sensitivity) vs Confirmatory Mode (≥92% OOF Specificity) */}
      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-4 shadow-clinical flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div className="flex items-center gap-2.5 text-xs">
          <Sliders className="w-4 h-4 text-medical-600 dark:text-sky-400 shrink-0" />
          <div>
            <span className="font-bold text-slate-900 dark:text-white">
              Clinical Operating Point Mode:
            </span>{' '}
            <span className="text-slate-600 dark:text-slate-400">
              Switch between High-Sensitivity Screening (80% OOF sensitivity) and High-Specificity
              Confirmatory (≥92% OOF specificity) thresholds.
            </span>
          </div>
        </div>

        <div className="inline-flex rounded-xl bg-slate-100 dark:bg-slate-800 p-1 text-xs font-semibold shrink-0">
          <button
            type="button"
            onClick={() => onSwitchOperatingMode && onSwitchOperatingMode('screening')}
            className={`px-3 py-1.5 rounded-lg transition ${
              operatingMode === 'screening'
                ? 'bg-medical-600 text-white shadow-sm'
                : 'text-slate-600 dark:text-slate-300 hover:text-slate-900'
            }`}
          >
            Screening Mode (80% Sens • Thresh {screeningThreshold ?? threshold})
          </button>
          <button
            type="button"
            onClick={() => onSwitchOperatingMode && onSwitchOperatingMode('confirmatory')}
            className={`px-3 py-1.5 rounded-lg transition ${
              operatingMode === 'confirmatory'
                ? 'bg-medical-600 text-white shadow-sm'
                : 'text-slate-600 dark:text-slate-300 hover:text-slate-900'
            }`}
          >
            Confirmatory Mode (92% Spec • Thresh {confirmatoryThreshold ?? threshold})
          </button>
        </div>
      </div>

      {/* Mandatory Clinical Disclaimer Banner */}
      <div
        role="note"
        className="rounded-xl bg-medical-50/90 dark:bg-sky-950/40 border border-medical-200 dark:border-sky-800/70 px-4 py-3.5 flex items-center gap-3 text-sm text-medical-900 dark:text-sky-200 shadow-sm"
      >
        <ShieldAlert className="w-5 h-5 text-medical-600 dark:text-sky-400 shrink-0" />
        <p className="font-semibold">
          This tool supports doctors and does not replace clinical diagnosis. (Decision support only, not a clinical diagnosis.)
        </p>
      </div>

      {/* Requirement C & Item (4): Three-Score Diagnostic & Exploratory Progression Cards */}
      <div className="space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Card 1: Clinical IPF probability (Validated XGBoost) */}
          <div className="rounded-2xl bg-white dark:bg-slate-900 border-2 border-medical-200 dark:border-sky-800/80 p-5 shadow-clinical flex flex-col justify-between space-y-3">
            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <span className="inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-medical-700 dark:text-sky-300">
                  <ShieldCheck className="w-4 h-4" />
                  1. Clinical IPF probability
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800">
                  Validated (XGBoost)
                </span>
              </div>
              <p className="text-[11px] text-slate-600 dark:text-slate-300 font-medium">
                <strong>Predicts:</strong> IPF vs. Non-IPF diagnosis (Zenodo ILD cohort; PFTs + demographics, torch-free).
              </p>
            </div>

            <div className="py-1">
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-extrabold font-mono text-slate-900 dark:text-white">
                  {probabilityPercent}%
                </span>
                <span
                  className={`text-sm font-bold ${
                    prediction === 'IPF'
                      ? 'text-rose-600 dark:text-rose-400'
                      : 'text-emerald-600 dark:text-emerald-400'
                  }`}
                >
                  ({prediction})
                </span>
              </div>
              <p className="text-[11px] font-mono text-slate-500 dark:text-slate-400 mt-1">
                {operatingMode} threshold: {threshold} • {riskLevel} Risk
              </p>
            </div>

            <div className="pt-2 border-t border-slate-100 dark:border-slate-800 text-[11px] text-slate-500 dark:text-slate-400">
              <strong>Diagnostic Metric (Zenodo Test Set):</strong> ROC-AUC{' '}
              <span className="font-mono font-semibold text-slate-700 dark:text-slate-200">
                0.9179 (95% CI: 0.8990–0.9346)
              </span>
              . Separate from the exploratory OSIC FVC progression models in Cards 2–3.
            </div>
          </div>

          {/* Card 2: HRCT progression-risk score (Exploratory, Requires CT series) */}
          <div className="rounded-2xl bg-white dark:bg-slate-900 border border-amber-300/90 dark:border-amber-800/80 p-5 shadow-clinical flex flex-col justify-between space-y-3">
            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <span className="inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-amber-800 dark:text-amber-300">
                  <ScanLine className="w-4 h-4" />
                  2. HRCT progression-risk score
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-900 dark:bg-amber-950/80 dark:text-amber-200 border border-amber-300 dark:border-amber-700">
                  Exploratory, not validated
                </span>
              </div>
              <p className="text-[11px] text-amber-800/90 dark:text-amber-300/90 font-medium">
                <strong>Predicts two OSIC targets (NOT IPF diagnosis):</strong> (a) future FVC at week 52 (regression) and (b) binary rapid-decliner risk (≤−200 mL/yr classifier).
              </p>
            </div>

            <div className="py-1">
              {hrctScore ? (
                <div className="space-y-1.5">
                  <div className="rounded-lg bg-slate-50 dark:bg-slate-800/60 p-2 text-[11px] font-mono">
                    <span className="text-slate-400 block text-[10px] uppercase">Future FVC at Week {hrctScore.horizon_weeks} (Regression)</span>
                    <span className="font-bold text-slate-900 dark:text-white text-sm">
                      {hrctScore.predicted_fvc_at_horizon_ml} mL
                    </span>{' '}
                    <span className="text-slate-500">
                      (Δ {hrctScore.predicted_fvc_change_ml} mL, σ={hrctScore.uncertainty_sigma_ml} mL)
                    </span>
                  </div>
                  <div className="rounded-lg bg-slate-50 dark:bg-slate-800/60 p-2 text-[11px] font-mono">
                    <span className="text-slate-400 block text-[10px] uppercase">Rapid-Decliner Risk (Binary Classifier)</span>
                    <span className="font-bold text-slate-900 dark:text-white text-sm">
                      {Math.round(hrctScore.progression_risk_score * 100)}%
                    </span>{' '}
                    <span className="text-amber-700 dark:text-amber-300 font-sans font-semibold">
                      ({hrctScore.risk_tier})
                    </span>
                  </div>
                </div>
              ) : (
                <div className="rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-dashed border-slate-300 dark:border-slate-700 px-3 py-2.5">
                  <div className="text-sm font-bold text-slate-700 dark:text-slate-200">
                    Requires CT series
                  </div>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                    {ctErrorMsg
                      ? ctErrorMsg
                      : 'Upload a full CT series (.zip of DICOM slices) on Page 1 to compute real HRCT features.'}
                  </p>
                </div>
              )}
            </div>

            <div className="pt-2 border-t border-slate-100 dark:border-slate-800 text-[11px] font-mono text-slate-500 dark:text-slate-400 space-y-1">
              <div>
                Future FVC — CV Laplace LL (95% CI):{' '}
                <strong>{hrctScore?.cv_laplace_ll_95ci || cvCiSummary?.hrct_only?.laplace_ll_95ci || '—'}</strong>
              </div>
              <div>
                Rapid-Decliner Risk — HRCT-only CV AUC (95% CI):{' '}
                <strong>{hrctScore?.cv_auc_95ci || cvCiSummary?.hrct_only?.auc_95ci || '—'}</strong>
              </div>
            </div>
          </div>

          {/* Card 3: Fusion progression-risk score (Exploratory, Selected Method, Requires CT series) */}
          <div className="rounded-2xl bg-white dark:bg-slate-900 border border-amber-300/90 dark:border-amber-800/80 p-5 shadow-clinical flex flex-col justify-between space-y-3">
            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <span className="inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-amber-800 dark:text-amber-300">
                  <Layers className="w-4 h-4" />
                  3. Fusion progression-risk score
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-900 dark:bg-amber-950/80 dark:text-amber-200 border border-amber-300 dark:border-amber-700">
                  Exploratory, not validated
                </span>
              </div>
              <p className="text-[11px] text-amber-800/90 dark:text-amber-300/90 font-medium">
                <strong>Method:</strong> {selectedFusionDisplay}{' '}
                <span className="font-mono px-1.5 py-0.5 rounded bg-amber-100/80 dark:bg-amber-950 text-[10px]">
                  {multimodalProgression?.fusion_vs_clinical_status || cvCiSummary?.fusion?.status || 'nominal_only'}
                </span>{' '}
                • <strong>Predicts two OSIC targets (NOT IPF diagnosis):</strong> (a) future FVC at week 52 (regression) and (b) separate binary rapid-decliner risk classifier.
              </p>
            </div>

            <div className="py-1">
              {fusionScore ? (
                <div className="space-y-1.5">
                  <div className="rounded-lg bg-slate-50 dark:bg-slate-800/60 p-2 text-[11px] font-mono">
                    <span className="text-slate-400 block text-[10px] uppercase">Future FVC at Week {fusionScore.horizon_weeks} (Late Fusion Regression)</span>
                    <span className="font-bold text-slate-900 dark:text-white text-sm">
                      {fusionScore.predicted_fvc_at_horizon_ml} mL
                    </span>{' '}
                    <span className="text-slate-500">
                      (Δ {fusionScore.predicted_fvc_change_ml} mL, σ={fusionScore.uncertainty_sigma_ml} mL)
                    </span>
                  </div>
                  <div className="rounded-lg bg-slate-50 dark:bg-slate-800/60 p-2 text-[11px] font-mono">
                    <span className="text-slate-400 block text-[10px] uppercase">Rapid-Decliner Risk (Separate Binary Classifier)</span>
                    <span className="font-bold text-slate-900 dark:text-white text-sm">
                      {Math.round(fusionScore.progression_risk_score * 100)}%
                    </span>{' '}
                    <span className="text-amber-700 dark:text-amber-300 font-sans font-semibold">
                      ({fusionScore.risk_tier})
                    </span>
                  </div>
                </div>
              ) : (
                <div className="rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-dashed border-slate-300 dark:border-slate-700 px-3 py-2.5">
                  <div className="text-sm font-bold text-slate-700 dark:text-slate-200">
                    Requires CT series
                  </div>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                    {ctErrorMsg
                      ? ctErrorMsg
                      : 'Upload a full CT series (.zip of DICOM slices) on Page 1 to compute multimodal fusion.'}
                  </p>
                </div>
              )}
            </div>

            <div className="pt-2 border-t border-slate-100 dark:border-slate-800 text-[11px] font-mono text-slate-500 dark:text-slate-400 space-y-1">
              <div>
                Future FVC — CV Laplace LL (95% CI):{' '}
                <strong>{fusionScore?.cv_laplace_ll_95ci || cvCiSummary?.fusion?.laplace_ll_95ci || '—'}</strong>
              </div>
              {cvCiSummary?.fusion?.paired_laplace_ll && (
                <div>
                  Paired ΔLL vs. Clinical-only:{' '}
                  <strong>{cvCiSummary.fusion.paired_laplace_ll}</strong>
                </div>
              )}
              <div>
                Rapid-Decliner Risk — Fusion CV AUC (95% CI):{' '}
                <strong>{fusionScore?.cv_auc_95ci || cvCiSummary?.fusion?.auc_95ci || '—'}</strong>
                {cvCiSummary?.clinical_only?.auc_mean && (
                  <span className="block font-sans text-[10px] text-amber-700 dark:text-amber-300">
                    (Lower than Clinical-only Rapid-Decliner AUC {Number(cvCiSummary.clinical_only.auc_mean).toFixed(4)})
                  </span>
                )}
              </div>
              <p className="font-sans text-[10.5px] text-slate-600 dark:text-slate-300 pt-1 leading-snug">
                {multimodalProgression?.selection_rationale ||
                  'Late fusion (mean) has the best mean FVC Laplace LL (nominal only), while its rapid-decliner AUC is lower. Stacking is the only method with a statistically confirmed gain over clinical-only (+0.0104 Laplace LL, p = 0.0006; −1.7 mL MAE, p = 0.0003); across 20 paired comparisons both survive Bonferroni correction (α = 0.0025), but the gain is clinically negligible (under 0.1% of mean FVC). Exploratory.'}
              </p>
            </div>
          </div>
        </div>

        {/* Short explanation of how the three scores relate */}
        <div className="rounded-xl bg-slate-100/80 dark:bg-slate-900/90 border border-slate-200/80 dark:border-slate-800 px-4 py-3 text-xs text-slate-600 dark:text-slate-300 flex items-start gap-2.5">
          <Info className="w-4 h-4 text-medical-600 dark:text-sky-400 shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <span className="font-bold text-slate-800 dark:text-slate-100">
              How these three scores relate:
            </span>{' '}
            {multimodalProgression?.score_relationship_note ||
              'Card 1 (Clinical IPF probability, Zenodo cohort) predicts IPF vs. Non-IPF diagnosis and is clinically validated on held-out test patients. Cards 2 (HRCT) and 3 (Fusion) come from the separate OSIC cohort and predict two distinct exploratory targets — future FVC at week 52 (regression, Laplace LL) and binary rapid-decliner risk (classifier, AUC) — NOT an IPF diagnosis. Late fusion (mean) has the best mean FVC Laplace LL (nominal only), while its rapid-decliner AUC is lower. Stacking is the only method with a statistically confirmed gain over clinical-only (+0.0104 Laplace LL, p = 0.0006; −1.7 mL MAE, p = 0.0003); across 20 paired comparisons both survive Bonferroni correction (α = 0.0025), but the gain is clinically negligible (under 0.1% of mean FVC). Exploratory.'}
          </p>
        </div>
      </div>

      {/* Main Prediction & Gauge Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-6 shadow-clinical flex flex-col justify-between space-y-6">
          <div className="flex items-start justify-between gap-3 border-b border-slate-100 dark:border-slate-800 pb-4">
            <div>
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Primary Classification ({operatingMode.toUpperCase()} MODE)
              </span>
              <div className="flex items-center gap-2.5 mt-1">
                <span
                  className={`text-3xl font-extrabold tracking-tight ${
                    prediction === 'IPF'
                      ? 'text-rose-600 dark:text-rose-400'
                      : 'text-emerald-600 dark:text-emerald-400'
                  }`}
                >
                  {prediction}
                </span>
                {prediction === 'IPF' ? (
                  <AlertTriangle className="w-6 h-6 text-rose-500" />
                ) : (
                  <CheckCircle2 className="w-6 h-6 text-emerald-500" />
                )}
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                {prediction === 'IPF'
                  ? `Probability ≥ ${operatingMode} OOF threshold (${threshold})`
                  : `Probability < ${operatingMode} OOF threshold (${threshold})`}
              </p>
            </div>

            <div className="text-right">
              <span className="block text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
                Risk Level
              </span>
              <span
                className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold border ${activeRiskStyle.bg} ${activeRiskStyle.border} ${activeRiskStyle.text}`}
              >
                <span className={`w-2 h-2 rounded-full ${activeRiskStyle.dot}`} />
                <span>{riskLevel} Risk</span>
              </span>
            </div>
          </div>

          <div className="py-2">
            <ProbabilityGauge
              probabilityPercent={probabilityPercent}
              prediction={prediction}
              riskLevel={riskLevel}
            />
          </div>

          {operatingModesSummary && (
            <div className="grid grid-cols-2 gap-2.5 text-xs">
              <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-200/70 dark:border-slate-700">
                <span className="text-slate-400 block text-[11px]">Screening Mode (80% Sens)</span>
                <span className="font-mono font-bold text-slate-800 dark:text-slate-100">
                  {operatingModesSummary.screening?.predicted_label} (≥
                  {operatingModesSummary.screening?.threshold})
                </span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-200/70 dark:border-slate-700">
                <span className="text-slate-400 block text-[11px]">Confirmatory Mode (92% Spec)</span>
                <span className="font-mono font-bold text-slate-800 dark:text-slate-100">
                  {operatingModesSummary.confirmatory?.predicted_label} (≥
                  {operatingModesSummary.confirmatory?.threshold})
                </span>
              </div>
            </div>
          )}

          <div className="rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/70 dark:border-slate-800 p-4 space-y-2.5">
            <div className="flex items-center justify-between text-xs font-semibold text-slate-700 dark:text-slate-300">
              <span className="inline-flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-medical-600 dark:text-sky-400" />
                Patient Input Summary
              </span>
              <span className="font-mono text-medical-600 dark:text-sky-400">
                GAP Index: {patientSummary.gapStage || 'Stage II'} ({patientSummary.gapPoints ?? 4} pts)
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 pt-1 text-xs">
              <div className="p-2 rounded-lg bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-700/60">
                <span className="text-slate-400 block text-[11px]">Sex / Age</span>
                <span className="font-mono font-semibold text-slate-800 dark:text-slate-200">
                  {patientSummary.sex || 'M'} • {patientSummary.age || '66'}y
                </span>
              </div>
              <div className="p-2 rounded-lg bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-700/60">
                <span className="text-slate-400 block text-[11px]">FVC / TLC</span>
                <span className="font-mono font-semibold text-slate-800 dark:text-slate-200">
                  {patientSummary.fvcPred}% / {patientSummary.tlcPred}%
                </span>
              </div>
              <div className="p-2 rounded-lg bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-700/60">
                <span className="text-slate-400 block text-[11px]">TLCO % Pred</span>
                <span className="font-mono font-semibold text-slate-800 dark:text-slate-200">
                  {patientSummary.tlcoPred}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Card: Explainability (SHAP Top 5 Features Bar Chart) */}
        <div className="lg:col-span-7 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-6 shadow-clinical flex flex-col justify-between">
          <div>
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-100 dark:border-slate-800 pb-4">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-xl bg-medical-50 dark:bg-sky-950/60 text-medical-600 dark:text-sky-400">
                  <BarChart3 className="w-5 h-5" />
                </div>
                <div>
                  <div className="flex items-center gap-1.5">
                    <h2 className="text-base font-bold text-slate-900 dark:text-white">
                      Top Features Driving Prediction
                    </h2>
                    <span className="relative group inline-flex items-center">
                      <button
                        type="button"
                        title="SHAP values show each feature's influence on this score"
                        aria-label="SHAP values show each feature's influence on this score"
                        className="inline-flex items-center justify-center w-5 h-5 rounded-full text-slate-400 hover:text-medical-600 dark:hover:text-sky-400 focus:outline-none focus:ring-2 focus:ring-medical-500/40 transition"
                      >
                        <Info className="w-4 h-4" />
                      </button>
                      <span
                        role="tooltip"
                        className="pointer-events-none opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity duration-150 absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-60 rounded-lg bg-slate-900 dark:bg-slate-800 text-white text-[11px] leading-snug px-2.5 py-2 shadow-lg border border-slate-700 z-20"
                      >
                        SHAP values show each feature&apos;s influence on this score
                      </span>
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Top 5 feature influences on this patient&apos;s score
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3 text-xs font-medium shrink-0">
                <span className="inline-flex items-center gap-1.5 text-rose-600 dark:text-rose-400">
                  <span className="w-2.5 h-2.5 rounded-sm bg-rose-500" />
                  Toward IPF (+)
                </span>
                <span className="inline-flex items-center gap-1.5 text-sky-600 dark:text-sky-400">
                  <span className="w-2.5 h-2.5 rounded-sm bg-sky-500" />
                  Toward Non-IPF (−)
                </span>
              </div>
            </div>

            <div className="h-72 w-full mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={featureImportance}
                  layout="vertical"
                  margin={{ top: 8, right: 28, left: 20, bottom: 8 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    horizontal={true}
                    vertical={true}
                    stroke={darkMode ? '#1e293b' : '#e2e8f0'}
                  />
                  <XAxis
                    type="number"
                    tickFormatter={(v) => (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2))}
                    tick={{ fill: darkMode ? '#94a3b8' : '#475569', fontSize: 11 }}
                  />
                  <YAxis
                    type="category"
                    dataKey="shortName"
                    width={105}
                    tick={{ fill: darkMode ? '#e2e8f0' : '#1e293b', fontSize: 12, fontWeight: 600 }}
                  />
                  <Tooltip
                    cursor={{ fill: darkMode ? 'rgba(30, 41, 59, 0.5)' : 'rgba(241, 245, 249, 0.8)' }}
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null;
                      const item = payload[0].payload;
                      const sign = item.impact >= 0 ? '+' : '';
                      return (
                        <div className="rounded-xl bg-slate-900 text-white p-3 shadow-lg border border-slate-700 text-xs max-w-xs space-y-1">
                          <div className="font-bold text-sky-300">
                            {item.feature}: <span className="font-mono text-white">{item.value}</span>
                          </div>
                          <div className="font-mono font-semibold">
                            SHAP Contribution:{' '}
                            <span className={item.impact >= 0 ? 'text-rose-400' : 'text-sky-400'}>
                              {sign}
                              {item.impact.toFixed(4)} ({item.direction})
                            </span>
                          </div>
                        </div>
                      );
                    }}
                  />
                  <ReferenceLine x={0} stroke={darkMode ? '#64748b' : '#94a3b8'} strokeWidth={1.5} />
                  <Bar dataKey="impact" radius={[4, 4, 4, 4]} barSize={20}>
                    {featureImportance.map((entry, idx) => (
                      <Cell key={`cell-${idx}`} fill={entry.impact >= 0 ? '#e11d48' : '#0284c7'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Five-part Evidence-Grounded "Interpretation of this result" Card */}
            {interpretation ? (
              <div
                role="region"
                aria-label="Interpretation of this result"
                className="mt-4 rounded-xl bg-slate-50/90 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700/80 p-4 space-y-3"
              >
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200/80 dark:border-slate-700/80 pb-2.5">
                  <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-medical-700 dark:text-sky-300">
                    <ClipboardCheck className="w-4 h-4 shrink-0" />
                    <span>Interpretation of this result</span>
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5">
                    {interpretation.physiology?.pattern && (
                      <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700 capitalize">
                        Pattern: {String(interpretation.physiology.pattern).replace('_', '-')}
                      </span>
                    )}
                    {interpretation.model_result?.modes_disagree && (
                      <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-amber-100 dark:bg-amber-950/70 text-amber-800 dark:text-amber-300 border border-amber-300 dark:border-amber-800">
                        Conflicting Modes
                      </span>
                    )}
                    {interpretationSource === 'llm_rephrased' && (
                      <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-sky-100 dark:bg-sky-950/70 text-sky-800 dark:text-sky-300 border border-sky-200 dark:border-sky-800">
                        Validated LLM Wording
                      </span>
                    )}
                  </div>
                </div>

                <div className="space-y-2.5 text-xs leading-relaxed text-slate-800 dark:text-slate-200">
                  <div className="rounded-lg bg-white dark:bg-slate-900/80 border border-slate-200/70 dark:border-slate-700/60 p-2.5">
                    <span className="font-bold text-slate-900 dark:text-white">
                      (a) Physiology:{' '}
                    </span>
                    <span>{interpretation.physiology?.summary}</span>
                  </div>

                  <div className="rounded-lg bg-white dark:bg-slate-900/80 border border-slate-200/70 dark:border-slate-700/60 p-2.5">
                    <span className="font-bold text-slate-900 dark:text-white">
                      (b) Model result:{' '}
                    </span>
                    <span>{interpretation.model_result?.summary}</span>
                  </div>

                  <div className="rounded-lg bg-white dark:bg-slate-900/80 border border-slate-200/70 dark:border-slate-700/60 p-2.5">
                    <span className="font-bold text-slate-900 dark:text-white">
                      (c) What pushed it:{' '}
                    </span>
                    <span>{interpretation.what_pushed_it?.summary}</span>
                  </div>

                  <div className="rounded-lg bg-amber-50/70 dark:bg-amber-950/20 border border-amber-200/80 dark:border-amber-800/60 p-2.5 text-amber-950 dark:text-amber-100">
                    <span className="font-bold">(d) Caution: </span>
                    <span>{interpretation.caution?.summary}</span>
                  </div>

                  <div className="rounded-lg bg-white dark:bg-slate-900/80 border border-slate-200/70 dark:border-slate-700/60 p-2.5 space-y-1">
                    <div className="font-bold text-slate-900 dark:text-white">
                      (e) For clinician consideration{' '}
                      <span className="font-normal text-[11px] text-slate-500 dark:text-slate-400">
                        ({interpretation.clinician_considerations?.guideline_reference})
                      </span>
                      :
                    </div>
                    <ul className="list-disc pl-4 space-y-0.5 text-slate-700 dark:text-slate-300">
                      {(interpretation.clinician_considerations?.items || []).map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>

                  {interpretationSource === 'llm_rephrased' && interpretationText && (
                    <div className="rounded-lg bg-sky-50/70 dark:bg-sky-950/30 border border-sky-200 dark:border-sky-800/70 p-2.5 whitespace-pre-line text-slate-800 dark:text-slate-200">
                      <div className="font-bold text-medical-700 dark:text-sky-300 mb-1">
                        Rephrased Summary (Fact-Validated):
                      </div>
                      {interpretationText}
                    </div>
                  )}
                </div>

                <div className="pt-2 border-t border-slate-200/80 dark:border-slate-700/80 flex items-center gap-1.5 text-[11px] font-semibold text-slate-600 dark:text-slate-300">
                  <ShieldAlert className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400 shrink-0" />
                  <span>
                    {interpretation.notice ||
                      'Automated interpretation, decision support only. Verify clinically.'}
                  </span>
                </div>
              </div>
            ) : (
              interpretationText && (
                <div
                  role="region"
                  aria-label="Interpretation of this result"
                  className="mt-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 p-4 text-xs text-slate-700 dark:text-slate-300 whitespace-pre-line"
                >
                  <div className="font-bold uppercase tracking-wider text-slate-800 dark:text-slate-100 mb-1">
                    Interpretation of this result
                  </div>
                  {interpretationText}
                </div>
              )
            )}
          </div>

          <div className="mt-4 pt-4 border-t border-slate-100 dark:border-slate-800">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2.5">
              Top 5 SHAP Feature Contributions
            </h3>
            <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
              {featureImportance.map((item) => {
                const isPos = item.impact >= 0;
                return (
                  <div
                    key={item.feature}
                    className="flex items-center justify-between gap-2 text-xs p-2 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200/60 dark:border-slate-800"
                  >
                    <div className="min-w-0">
                      <span className="font-semibold text-slate-800 dark:text-slate-200">
                        {item.feature}
                      </span>{' '}
                      <span className="font-mono text-medical-600 dark:text-sky-400 font-medium">
                        ({item.value})
                      </span>
                    </div>
                    <span
                      className={`font-mono font-bold px-2 py-0.5 rounded shrink-0 ${
                        isPos
                          ? 'bg-rose-100/80 text-rose-700 dark:bg-rose-950/80 dark:text-rose-300'
                          : 'bg-sky-100/80 text-sky-700 dark:bg-sky-950/80 dark:text-sky-300'
                      }`}
                    >
                      {isPos ? `+${item.impact.toFixed(4)}` : item.impact.toFixed(4)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-4 shadow-clinical">
        <div className="text-xs text-slate-500 dark:text-slate-400">
          <span className="font-semibold text-slate-700 dark:text-slate-200">Model Engine:</span>{' '}
          {modelUsed} (v{modelVersion}) • Evaluated at{' '}
          {timestamp ? new Date(timestamp).toLocaleTimeString() : 'Current Session'}
        </div>

        <button
          type="button"
          disabled={isOfflineDemo}
          onClick={() => downloadClinicalReportPdf(result)}
          className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl bg-medical-600 hover:bg-medical-700 disabled:bg-slate-300 dark:disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed text-white px-5 py-2.5 text-sm font-semibold shadow-sm transition"
        >
          <FileDown className="w-4 h-4" />
          <span>{isOfflineDemo ? 'PDF Disabled (Backend Offline)' : 'Download Result as PDF'}</span>
        </button>
      </div>
    </div>
  );
}
