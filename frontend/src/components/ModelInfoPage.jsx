import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Award,
  Grid,
  TrendingUp,
  CheckCircle2,
  Info,
  AlertCircle,
  RefreshCw,
  AlertTriangle,
  Sliders,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { fetchModelInfo } from '../services/api';

export default function ModelInfoPage({ darkMode }) {
  const [modelInfo, setModelInfo] = useState(null);
  const [selectedModelName, setSelectedModelName] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadData = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await fetchModelInfo();
      setModelInfo(data);
      setSelectedModelName(data.best_model || data.metrics_table?.[0]?.model || '');
    } catch (err) {
      setError(
        err?.message ||
          'Backend API (/model-info) is unreachable. Start uvicorn backend.api.main:app --port 8000.'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-12 text-center space-y-3">
        <div className="w-8 h-8 border-2 border-medical-600 border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-sm font-medium text-slate-600 dark:text-slate-400">
          Loading real model evaluation metrics from FastAPI (<code className="font-mono">GET /model-info</code>)...
        </p>
      </div>
    );
  }

  if (error || !modelInfo) {
    return (
      <div className="rounded-2xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 p-6 space-y-4">
        <div className="flex items-center gap-3 text-rose-700 dark:text-rose-300">
          <AlertCircle className="w-6 h-6 shrink-0" />
          <div>
            <h2 className="text-base font-bold">
              Unable to load live model metrics from GET /model-info
            </h2>
            <p className="text-xs mt-0.5">{error}</p>
          </div>
        </div>
        <button
          type="button"
          onClick={loadData}
          className="inline-flex items-center gap-2 rounded-xl bg-rose-600 hover:bg-rose-700 text-white px-4 py-2 text-xs font-semibold"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Retry Connection</span>
        </button>
      </div>
    );
  }

  const metricsTable = modelInfo.metrics_table || [];
  const activeModel =
    metricsTable.find((m) => m.model === selectedModelName) || metricsTable[0];
  const cm = activeModel?.confusion_matrix || { tp: 0, fn: 0, fp: 0, tn: 0, total: 0 };
  const rocCurveData = modelInfo.roc_curve || [];
  const perGroupErrors = modelInfo.per_group_error_analysis || [];
  const reliabilityTable = modelInfo.calibration_reliability_table || null;
  const cohortAgeStats = modelInfo.cohort_age_stats || null;
  const interpretationRules = modelInfo.interpretation_rules || null;
  const osicStudy = modelInfo.osic_progression_study || null;
  const osicCompBench = osicStudy?.osic_competition_benchmark || null;
  const fusionComp = osicStudy?.fusion_comparison || null;
  const extractorComp = modelInfo.extractor_comparison || null;
  const operatingModes = modelInfo.operating_modes || {};

  const formatPct = (val) => `${(Number(val) * 100).toFixed(1)}%`;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-5 shadow-clinical flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-medical-600 dark:text-sky-400">
            <Cpu className="w-4 h-4" />
            <span>Page 3 • Real Model Evaluation & Validation Benchmarks (GET /model-info)</span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white mt-1">
            Model Information & Diagnostic Performance
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1 max-w-3xl">
            {modelInfo.selection_criterion}
          </p>
        </div>

        <div className="flex flex-col items-end gap-1.5 text-xs font-mono shrink-0">
          <span className="px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800 font-semibold">
            data_source: {modelInfo.data_source} • torch_at_serving: {String(modelInfo.requires_torch_at_serving)}
          </span>
          <span className="text-slate-500 dark:text-slate-400">
            Screening Thresh (80% OOF Sens): <strong>{modelInfo.screening_threshold}</strong>
          </span>
          <span className="text-slate-500 dark:text-slate-400">
            Confirmatory Thresh (92% OOF Spec): <strong>{modelInfo.confirmatory_threshold}</strong>
          </span>
        </div>
      </div>

      {/* Plain Sensitivity & Per-Group Misclassification Disclosure Banner (Item 6) */}
      {modelInfo.sensitivity_disclosure && (
        <div
          role="note"
          className="rounded-2xl bg-rose-50/80 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-800/70 p-4 flex items-start gap-3 text-xs sm:text-sm text-rose-950 dark:text-rose-100 shadow-clinical"
        >
          <AlertTriangle className="w-5 h-5 text-rose-600 dark:text-rose-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold block mb-0.5">
              Plain Clinical Sensitivity & Misclassification Disclosure
            </span>
            <p className="leading-relaxed">{modelInfo.sensitivity_disclosure}</p>
          </div>
        </div>
      )}

      {/* Classifier Comparison Table (Unified 80% OOF Sensitivity Rule + 95% Bootstrap CIs) */}
      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 shadow-clinical overflow-hidden">
        <div className="p-5 border-b border-slate-100 dark:border-slate-800 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <h2 className="text-base font-bold text-slate-900 dark:text-white">
              Part 1: Zenodo ILD Classifier Comparison (Unified 80% OOF Training Sensitivity Rule)
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              All 4 models (Random Forest, SVM, XGBoost, and PyTorch MLP) use thresholds tuned on
              their calibrated Out-of-Fold training predictions for 80% sensitivity.
            </p>
          </div>
          <span className="inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1 rounded-full bg-medical-50 dark:bg-sky-950/60 text-medical-700 dark:text-sky-300 border border-medical-200 dark:border-sky-800">
            <Award className="w-3.5 h-3.5" />
            Fair Common Threshold Rule • 95% Bootstrap CIs
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-50/80 dark:bg-slate-800/60 border-b border-slate-200/80 dark:border-slate-800 text-[11px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                <th className="py-3.5 px-4">Model</th>
                <th className="py-3.5 px-3 text-right">5-Fold Train CV AUC</th>
                <th className="py-3.5 px-3 text-right">Test ROC-AUC (95% CI)</th>
                <th className="py-3.5 px-3 text-right">PR-AUC</th>
                <th className="py-3.5 px-3 text-right">F1</th>
                <th className="py-3.5 px-3 text-right">Sensitivity (95% CI)</th>
                <th className="py-3.5 px-3 text-right">Specificity (95% CI)</th>
                <th className="py-3.5 px-4 text-right">OOF Thresh (80% Sens)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200/70 dark:divide-slate-800 text-sm">
              {metricsTable.map((m) => {
                const isSelected = m.model === activeModel?.model;
                const aucCi = m.bootstrap_ci_95?.auc_ci_95;
                const sensCi = m.bootstrap_ci_95?.sensitivity_ci_95;
                const specCi = m.bootstrap_ci_95?.specificity_ci_95;
                return (
                  <tr
                    key={m.model}
                    onClick={() => setSelectedModelName(m.model)}
                    className={`cursor-pointer transition ${
                      isSelected
                        ? 'bg-medical-50/70 dark:bg-sky-950/30'
                        : 'hover:bg-slate-50 dark:hover:bg-slate-800/40'
                    }`}
                  >
                    <td className="py-4 px-4">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-slate-900 dark:text-white">
                          {m.model}
                        </span>
                        {m.is_best && (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300 border border-emerald-300/60 dark:border-emerald-800">
                            <CheckCircle2 className="w-3 h-3" />
                            Selected by CV
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-4 px-3 text-right font-mono text-xs font-semibold text-slate-700 dark:text-slate-300">
                      {m.cv_roc_auc_formatted}
                    </td>
                    <td className="py-4 px-3 text-right font-mono font-bold text-medical-600 dark:text-sky-400">
                      {Number(m.auc).toFixed(4)}
                      {aucCi && (
                        <span className="block text-[11px] text-slate-400 font-normal">
                          [{aucCi.lower}–{aucCi.upper}]
                        </span>
                      )}
                    </td>
                    <td className="py-4 px-3 text-right font-mono text-slate-800 dark:text-slate-200">
                      {Number(m.pr_auc).toFixed(4)}
                    </td>
                    <td className="py-4 px-3 text-right font-mono text-slate-800 dark:text-slate-200">
                      {Number(m.f1).toFixed(4)}
                    </td>
                    <td className="py-4 px-3 text-right font-mono text-slate-800 dark:text-slate-200">
                      {formatPct(m.sensitivity)}
                      {sensCi && (
                        <span className="block text-[11px] text-slate-400 font-normal">
                          [{formatPct(sensCi.lower)}–{formatPct(sensCi.upper)}]
                        </span>
                      )}
                    </td>
                    <td className="py-4 px-3 text-right font-mono text-slate-800 dark:text-slate-200">
                      {formatPct(m.specificity)}
                      {specCi && (
                        <span className="block text-[11px] text-slate-400 font-normal">
                          [{formatPct(specCi.lower)}–{formatPct(specCi.upper)}]
                        </span>
                      )}
                    </td>
                    <td className="py-4 px-4 text-right font-mono text-xs font-semibold text-slate-700 dark:text-slate-300">
                      {Number(m.threshold).toFixed(4)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Dual Operating Mode Trade-Off Card: Screening Mode vs Confirmatory Mode */}
      {operatingModes.screening && operatingModes.confirmatory && (
        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-5 shadow-clinical space-y-4">
          <div className="flex items-center gap-2.5 border-b border-slate-100 dark:border-slate-800 pb-3">
            <Sliders className="w-5 h-5 text-medical-600 dark:text-sky-400" />
            <div>
              <h2 className="text-base font-bold text-slate-900 dark:text-white">
                Operating Point Trade-Off ({modelInfo.best_model}): Screening Mode vs. Confirmatory Mode
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {modelInfo.threshold_tuning_method}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            {[
              { key: 'screening', title: 'Screening Mode (Default — Minimizes Missed IPF)', data: operatingModes.screening },
              { key: 'confirmatory', title: 'Confirmatory Mode (High Specificity — Minimizes False Alarms)', data: operatingModes.confirmatory },
            ].map((mode) => (
              <div
                key={mode.key}
                className="rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-700 p-4 space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-slate-900 dark:text-white">{mode.title}</span>
                  <span className="px-2 py-0.5 rounded bg-medical-100 dark:bg-sky-950 text-medical-800 dark:text-sky-300 font-mono font-bold">
                    OOF Threshold = {mode.data.threshold}
                  </span>
                </div>
                <p className="text-slate-500 dark:text-slate-400">{mode.data.description}</p>
                <div className="grid grid-cols-4 gap-2 pt-1 font-mono">
                  <div className="p-2 rounded bg-white dark:bg-slate-900">
                    <span className="text-slate-400 block text-[10px]">Sensitivity</span>
                    <span className="font-bold text-slate-800 dark:text-slate-100">
                      {formatPct(mode.data.sensitivity)}
                    </span>
                  </div>
                  <div className="p-2 rounded bg-white dark:bg-slate-900">
                    <span className="text-slate-400 block text-[10px]">Specificity</span>
                    <span className="font-bold text-slate-800 dark:text-slate-100">
                      {formatPct(mode.data.specificity)}
                    </span>
                  </div>
                  <div className="p-2 rounded bg-white dark:bg-slate-900">
                    <span className="text-slate-400 block text-[10px]">F1 Score</span>
                    <span className="font-bold text-slate-800 dark:text-slate-100">
                      {Number(mode.data.f1).toFixed(3)}
                    </span>
                  </div>
                  <div className="p-2 rounded bg-white dark:bg-slate-900">
                    <span className="text-slate-400 block text-[10px]">TP / FN / FP</span>
                    <span className="font-bold text-slate-800 dark:text-slate-100">
                      {mode.data.confusion_matrix?.tp}/{mode.data.confusion_matrix?.fn}/{mode.data.confusion_matrix?.fp}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Confusion Matrix & ROC Curve Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-6 shadow-clinical flex flex-col justify-between space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-100 dark:border-slate-800 pb-4">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-medical-50 dark:bg-sky-950/60 text-medical-600 dark:text-sky-400">
                <Grid className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-base font-bold text-slate-900 dark:text-white">
                  Untouched Test Confusion Matrix ({activeModel?.model})
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Test split n = {cm.total} ({cm.tp + cm.fn} IPF, {cm.fp + cm.tn} Non-IPF) • Screening Threshold ={' '}
                  {activeModel?.threshold}
                </p>
              </div>
            </div>

            <div className="inline-flex flex-wrap rounded-xl bg-slate-100 dark:bg-slate-800 p-1 text-xs font-semibold">
              {metricsTable.map((m) => (
                <button
                  key={m.model}
                  type="button"
                  onClick={() => setSelectedModelName(m.model)}
                  className={`px-2.5 py-1 rounded-lg transition ${
                    activeModel?.model === m.model
                      ? 'bg-white dark:bg-slate-900 text-medical-600 dark:text-sky-400 shadow-sm'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                  }`}
                >
                  {m.model}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-3">
            <div className="grid grid-cols-12 items-center text-center text-xs font-semibold text-slate-500 dark:text-slate-400">
              <div className="col-span-3" />
              <div className="col-span-9 grid grid-cols-2 gap-3 pb-1">
                <span className="uppercase tracking-wider">Predicted: IPF</span>
                <span className="uppercase tracking-wider">Predicted: Non-IPF</span>
              </div>
            </div>

            <div className="grid grid-cols-12 items-stretch gap-3">
              <div className="col-span-3 flex items-center justify-end pr-2 text-right text-xs font-bold uppercase tracking-wider text-slate-600 dark:text-slate-300">
                Actual: IPF
              </div>
              <div className="col-span-9 grid grid-cols-2 gap-3">
                <div className="rounded-xl bg-medical-600 text-white p-4 text-center shadow-sm">
                  <span className="text-[11px] font-mono uppercase tracking-wider text-medical-100 block">
                    True Positive (TP)
                  </span>
                  <span className="text-2xl font-extrabold font-mono block mt-1">{cm.tp}</span>
                  <span className="text-xs text-medical-100 block mt-0.5">
                    {cm.tp + cm.fn > 0 ? ((cm.tp / (cm.tp + cm.fn)) * 100).toFixed(1) : 0}% sensitivity
                  </span>
                </div>
                <div className="rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800/60 p-4 text-center">
                  <span className="text-[11px] font-mono uppercase tracking-wider text-rose-600 dark:text-rose-400 block">
                    False Negative (FN)
                  </span>
                  <span className="text-2xl font-extrabold font-mono text-rose-700 dark:text-rose-300 block mt-1">
                    {cm.fn}
                  </span>
                  <span className="text-xs text-rose-600/80 dark:text-rose-400/80 block mt-0.5">
                    {cm.tp + cm.fn > 0 ? ((cm.fn / (cm.tp + cm.fn)) * 100).toFixed(1) : 0}% missed IPF
                  </span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-12 items-stretch gap-3">
              <div className="col-span-3 flex items-center justify-end pr-2 text-right text-xs font-bold uppercase tracking-wider text-slate-600 dark:text-slate-300">
                Actual: Non-IPF
              </div>
              <div className="col-span-9 grid grid-cols-2 gap-3">
                <div className="rounded-xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800/60 p-4 text-center">
                  <span className="text-[11px] font-mono uppercase tracking-wider text-amber-700 dark:text-amber-400 block">
                    False Positive (FP)
                  </span>
                  <span className="text-2xl font-extrabold font-mono text-amber-800 dark:text-amber-300 block mt-1">
                    {cm.fp}
                  </span>
                  <span className="text-xs text-amber-700/80 dark:text-amber-400/80 block mt-0.5">
                    {cm.fp + cm.tn > 0 ? ((cm.fp / (cm.fp + cm.tn)) * 100).toFixed(1) : 0}% false alarm
                  </span>
                </div>
                <div className="rounded-xl bg-sky-600 text-white p-4 text-center shadow-sm">
                  <span className="text-[11px] font-mono uppercase tracking-wider text-sky-100 block">
                    True Negative (TN)
                  </span>
                  <span className="text-2xl font-extrabold font-mono block mt-1">{cm.tn}</span>
                  <span className="text-xs text-sky-100 block mt-0.5">
                    {cm.fp + cm.tn > 0 ? ((cm.tn / (cm.fp + cm.tn)) * 100).toFixed(1) : 0}% specificity
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/70 dark:border-slate-800 p-3 text-xs text-slate-600 dark:text-slate-400 flex items-center gap-2">
            <Info className="w-4 h-4 text-medical-600 dark:text-sky-400 shrink-0" />
            <span>
              Imbalance Comparison (5-Fold CV AUC): Class Weights ={' '}
              <strong className="font-mono">
                {modelInfo.imbalance_comparison?.class_weights_formatted}
              </strong>{' '}
              vs. In-Fold SMOTE ={' '}
              <strong className="font-mono">
                {modelInfo.imbalance_comparison?.smote_formatted}
              </strong>
            </span>
          </div>
        </div>

        {/* Right: Real ROC Curve */}
        <div className="lg:col-span-6 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-6 shadow-clinical flex flex-col justify-between space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-4">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-medical-50 dark:bg-sky-950/60 text-medical-600 dark:text-sky-400">
                <TrendingUp className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-base font-bold text-slate-900 dark:text-white">
                  Held-Out Test ROC Curve (Real Zenodo ILD)
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  True Positive Rate (Sensitivity) vs. False Positive Rate (1 − Specificity)
                </p>
              </div>
            </div>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rocCurveData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#1e293b' : '#e2e8f0'} />
                <XAxis
                  dataKey="fpr"
                  type="number"
                  domain={[0, 1]}
                  tickFormatter={(v) => v.toFixed(1)}
                  tick={{ fill: darkMode ? '#94a3b8' : '#475569', fontSize: 11 }}
                />
                <YAxis
                  domain={[0, 1]}
                  tickFormatter={(v) => v.toFixed(1)}
                  tick={{ fill: darkMode ? '#94a3b8' : '#475569', fontSize: 11 }}
                />
                <Tooltip
                  formatter={(value, name) => [Number(value).toFixed(3), name]}
                  labelFormatter={(label) => `FPR: ${Number(label).toFixed(2)}`}
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    borderRadius: '0.75rem',
                    color: '#f8fafc',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '8px' }} />
                <Line type="monotone" dataKey="xgb" name="XGBoost" stroke="#0284c7" strokeWidth={2.8} dot={false} />
                <Line type="monotone" dataKey="rf" name="Random Forest" stroke="#10b981" strokeWidth={2.2} dot={false} />
                <Line type="monotone" dataKey="svm" name="SVM" stroke="#f59e0b" strokeWidth={2.2} dot={false} />
                <Line type="monotone" dataKey="mlp" name="PyTorch MLP" stroke="#8b5cf6" strokeWidth={2.0} dot={false} />
                <Line
                  type="linear"
                  dataKey="baseline"
                  name="Random Chance"
                  stroke="#94a3b8"
                  strokeDasharray="5 5"
                  strokeWidth={1.5}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/70 dark:border-slate-800 p-3 text-xs text-slate-600 dark:text-slate-400">
            <span className="font-semibold text-slate-800 dark:text-slate-200">
              Test-Set Calibration Brier Score:
            </span>{' '}
            Uncalibrated ={' '}
            <strong className="font-mono">
              {modelInfo.calibration_curve?.test_brier_uncalibrated}
            </strong>{' '}
            → Calibrated ={' '}
            <strong className="font-mono">
              {modelInfo.calibration_curve?.test_brier_calibrated}
            </strong>
          </div>
        </div>
      </div>

      {/* Per-Diagnosis-Group Error Analysis Table */}
      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 shadow-clinical overflow-hidden">
        <div className="p-5 border-b border-slate-100 dark:border-slate-800 space-y-1.5">
          <h2 className="text-base font-bold text-slate-900 dark:text-white">
            Per-Diagnosis-Group Error Analysis on Held-Out Test Set ({modelInfo.best_model}, Screening Mode)
          </h2>
          <p className="text-xs text-slate-600 dark:text-slate-400">
            {modelInfo.clinical_error_analysis_note}
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs sm:text-sm">
            <thead>
              <tr className="bg-slate-50/80 dark:bg-slate-800/60 border-b border-slate-200/80 dark:border-slate-800 text-[11px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                <th className="py-3 px-5">Zenodo Group</th>
                <th className="py-3 px-4">Clinical Description</th>
                <th className="py-3 px-4 text-right">Test Cases</th>
                <th className="py-3 px-4 text-right">Predicted IPF</th>
                <th className="py-3 px-4 text-right">Error Type</th>
                <th className="py-3 px-5 text-right">Error Rate (%)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200/70 dark:divide-slate-800">
              {perGroupErrors.map((g) => (
                <tr key={g.group} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                  <td className="py-3 px-5 font-mono font-bold text-slate-900 dark:text-white">
                    {g.group}
                  </td>
                  <td className="py-3 px-4 text-slate-600 dark:text-slate-300">{g.full_name}</td>
                  <td className="py-3 px-4 text-right font-mono">{g.total_test_cases}</td>
                  <td className="py-3 px-4 text-right font-mono">{g.predicted_ipf_count}</td>
                  <td className="py-3 px-4 text-right text-xs text-slate-500">{g.error_type}</td>
                  <td className="py-3 px-5 text-right font-mono font-bold text-rose-600 dark:text-rose-400">
                    {g.error_rate_pct}% ({g.error_count}/{g.total_test_cases})
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Calibration Reliability Table & Automated Result Interpretation Rules */}
      {reliabilityTable && Array.isArray(reliabilityTable.bins) && (
        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 shadow-clinical overflow-hidden">
          <div className="p-5 border-b border-slate-100 dark:border-slate-800 space-y-1.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-base font-bold text-slate-900 dark:text-white">
                Calibrated Probability Reliability Table &amp; Automated Result Interpretation Rules
              </h2>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono font-semibold bg-medical-50 dark:bg-sky-950/70 text-medical-700 dark:text-sky-300 border border-medical-200 dark:border-sky-800">
                Held-out Test n = {reliabilityTable.total_patients} ({reliabilityTable.total_ipf} IPF)
              </span>
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400">
              {reliabilityTable.description} Used directly by the five-part &ldquo;Interpretation of
              this result&rdquo; card on Page 2 and in PDF reports to report empirical IPF rates for
              each patient&apos;s probability bin. Observed calibrated test score range:{' '}
              <span className="font-mono font-semibold">
                {reliabilityTable.observed_prob_min}–{reliabilityTable.observed_prob_max}
              </span>
              .
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs sm:text-sm">
              <thead>
                <tr className="bg-slate-50/80 dark:bg-slate-800/60 border-b border-slate-200/80 dark:border-slate-800 text-[11px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  <th className="py-3 px-5">Calibrated Score Bin</th>
                  <th className="py-3 px-4 text-right">Test Patients (n)</th>
                  <th className="py-3 px-4 text-right">Observed IPF (k)</th>
                  <th className="py-3 px-4 text-right">Empirical IPF Rate</th>
                  <th className="py-3 px-4 text-right">95% Wilson CI</th>
                  <th className="py-3 px-5 text-right">Mean Predicted Prob</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200/70 dark:divide-slate-800">
                {reliabilityTable.bins.map((b) => (
                  <tr key={b.bin_label} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                    <td className="py-2.5 px-5 font-mono font-bold text-slate-900 dark:text-white">
                      {b.pct_range_label}{' '}
                      <span className="text-[11px] font-normal text-slate-400">
                        ({b.bin_label})
                      </span>
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono">{b.n_patients}</td>
                    <td className="py-2.5 px-4 text-right font-mono">{b.n_ipf}</td>
                    <td className="py-2.5 px-4 text-right font-mono font-bold text-medical-700 dark:text-sky-400">
                      {b.empirical_ipf_pct !== null ? `${b.empirical_ipf_pct}%` : '—'}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono text-slate-600 dark:text-slate-300">
                      {b.wilson_ci_95_lower_pct !== null
                        ? `${b.wilson_ci_95_lower_pct}%–${b.wilson_ci_95_upper_pct}%`
                        : '—'}
                    </td>
                    <td className="py-2.5 px-5 text-right font-mono text-slate-600 dark:text-slate-300">
                      {b.mean_predicted_prob !== null ? b.mean_predicted_prob.toFixed(4) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="p-4 bg-slate-50/70 dark:bg-slate-800/40 border-t border-slate-100 dark:border-slate-800 space-y-2 text-xs text-slate-600 dark:text-slate-400">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-700 p-3 space-y-1">
                <span className="font-bold text-slate-800 dark:text-slate-200 block">
                  Configurable PFT Pattern Rules (backend/config/interpretation_rules.json)
                </span>
                <p>
                  Reduced volume cutoff:{' '}
                  <span className="font-mono font-semibold">
                    TLC &lt; {interpretationRules?.physiology?.volume_thresholds?.tlc_reduced_below_pp ?? 80}%
                  </span>{' '}
                  (or FVC &lt; {interpretationRules?.physiology?.volume_thresholds?.fvc_reduced_below_pp ?? 80}% if TLC absent);
                  preserved ratio:{' '}
                  <span className="font-mono font-semibold">
                    FEV1/FVC ≥ {interpretationRules?.physiology?.ratio_thresholds?.fev1fvc_preserved_at_or_above ?? 0.7}
                  </span>
                  ; TLCO severity bands:{' '}
                  <span className="font-mono font-semibold">≥80% / 60–79% / 40–59% / &lt;40%</span>.
                </p>
              </div>
              <div className="rounded-xl bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-700 p-3 space-y-1">
                <span className="font-bold text-slate-800 dark:text-slate-200 block">
                  Cohort Age Reference &amp; Optional LLM Rephrasing
                </span>
                <p>
                  Zenodo cohort median age: IPF{' '}
                  <span className="font-mono font-semibold">
                    {cohortAgeStats?.ipf_median_age ?? 68.9} yrs
                  </span>{' '}
                  vs. Non-IPF{' '}
                  <span className="font-mono font-semibold">
                    {cohortAgeStats?.non_ipf_median_age ?? 48.9} yrs
                  </span>
                  . Optional LLM rephrasing (<code className="font-mono">ENABLE_LLM_INTERPRETATION_REPHRASE</code>) is
                  OFF by default and enforces numeric-conservation and no-diagnosis/treatment validation.
                </p>
              </div>
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              {interpretationRules?.clinical_review_note ||
                'Percent-predicted thresholds are approximations without GLI LLN/z-scores and must be reviewed by a clinician before real clinical use.'}{' '}
              <strong className="text-slate-700 dark:text-slate-200">
                {interpretationRules?.mandatory_notice ||
                  'Automated interpretation, decision support only. Verify clinically.'}
              </strong>
            </p>
          </div>
        </div>
      )}

      {/* Exploratory Study: Part 2 OSIC Progression Table (Both Target Variants + Modality Ablation Status) */}
      {osicStudy && (
        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-amber-300/90 dark:border-amber-800/80 shadow-clinical overflow-hidden space-y-4 pb-4">
          <div className="p-5 bg-amber-50/70 dark:bg-amber-950/30 border-b border-amber-200 dark:border-amber-800/70 space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-100 dark:bg-amber-900/70 text-amber-900 dark:text-amber-200 text-xs font-bold border border-amber-300 dark:border-amber-700">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{osicStudy.exploratory_banner}</span>
              </div>
              <span
                className={`px-3 py-1 rounded-full text-xs font-mono font-bold border ${
                  osicStudy.imaging_model_available
                    ? 'bg-emerald-100 dark:bg-emerald-950/80 text-emerald-800 dark:text-emerald-200 border-emerald-300 dark:border-emerald-800'
                    : 'bg-rose-100 dark:bg-rose-950/80 text-rose-800 dark:text-rose-200 border-rose-300 dark:border-rose-800'
                }`}
              >
                imaging_model_available: {String(osicStudy.imaging_model_available)} (
                {osicStudy.patients_with_dicoms}/{osicStudy.num_patients} CT patients; ≥
                {osicStudy.min_imaging_patients_required} required)
              </span>
            </div>
            <h2 className="text-base font-bold text-slate-900 dark:text-white">
              Part 2 (Exploratory Separate Cohort): OSIC Pulmonary Fibrosis Progression vs. Dummy Baselines
            </h2>
            <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
              {osicStudy.study_note}
            </p>
            {osicStudy.modality_ablation && (
              <p className="text-xs font-mono text-amber-800 dark:text-amber-300">
                3-Way Modality Ablation (Clinical-only vs. Clinical+HU/Vol vs. Clinical+ResNet-18 In-Fold PCA):{' '}
                <strong>
                  {osicStudy.modality_ablation.status === 'completed'
                    ? `completed (${osicStudy.modality_ablation.num_ct_patients ?? osicStudy.patients_with_dicoms} CT patients, PCA=${osicStudy.modality_ablation.pca_components_used}D fitted strictly inside each training fold)`
                    : osicStudy.modality_ablation.status}
                </strong>
              </p>
            )}
          </div>

          {/* Requirement A: OSIC Competition Metric (Modified Laplace LL) 8-Model Internal CV Benchmark */}
          {osicCompBench && osicCompBench.models && (
            <div className="px-5 space-y-2.5">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                <div>
                  <h3 className="text-xs font-bold uppercase tracking-wider text-medical-700 dark:text-sky-300">
                    OSIC Competition Metric Benchmark — Modified Laplace Log-Likelihood (5×5 Nested GroupKFold)
                  </h3>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                    {osicCompBench.evaluation_protocol}. {osicCompBench.metric_definition}. Internal CV on{' '}
                    <code className="font-mono">train.csv</code> (n = {osicCompBench.num_patients} CT patients,{' '}
                    {osicCompBench.num_future_visits} future visits) — <strong>not Kaggle leaderboard scores</strong>.
                  </p>
                </div>
                {osicCompBench.best_internal_model && (
                  <span className="px-2.5 py-1 rounded-full text-[11px] font-mono font-bold bg-sky-100 text-sky-900 dark:bg-sky-950/80 dark:text-sky-200 border border-sky-300 dark:border-sky-800 shrink-0">
                    Best Internal CV: {osicCompBench.best_internal_model}
                  </span>
                )}
              </div>

              <div className="overflow-x-auto rounded-xl border border-slate-200/80 dark:border-slate-800">
                <table className="w-full text-left border-collapse text-xs sm:text-sm">
                  <thead>
                    <tr className="bg-slate-50/80 dark:bg-slate-800/60 border-b border-slate-200/80 dark:border-slate-800 text-[11px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                      <th className="py-2.5 px-4">Model (Internal Nested GroupKFold CV)</th>
                      <th className="py-2.5 px-3 text-right">Modified Laplace LL (95% CI, higher = better)</th>
                      <th className="py-2.5 px-3 text-right">FVC RMSE mL (95% CI)</th>
                      <th className="py-2.5 px-4 text-right">FVC MAE mL (95% CI)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200/70 dark:divide-slate-800">
                    {Object.values(osicCompBench.models).map((m) => {
                      const isBestInternal = m.model === osicCompBench.best_internal_model;
                      const isBase = m.model.includes('Baseline');
                      return (
                        <tr
                          key={`comp-${m.model}`}
                          className={
                            isBestInternal
                              ? 'bg-sky-50/70 dark:bg-sky-950/30 font-semibold'
                              : isBase
                              ? 'bg-slate-100/70 dark:bg-slate-800/70 font-semibold'
                              : 'hover:bg-slate-50 dark:hover:bg-slate-800/40'
                          }
                        >
                          <td className="py-2.5 px-4 font-bold text-slate-900 dark:text-white">
                            {m.model}
                            {isBestInternal && (
                              <span className="ml-2 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300">
                                Highest CV LL
                              </span>
                            )}
                          </td>
                          <td className="py-2.5 px-3 text-right font-mono text-xs font-bold text-medical-700 dark:text-sky-300">
                            {m.laplace_ll_stats?.formatted}
                          </td>
                          <td className="py-2.5 px-3 text-right font-mono text-xs">
                            {m.rmse_stats?.formatted}
                          </td>
                          <td className="py-2.5 px-4 text-right font-mono text-xs">
                            {m.mae_stats?.formatted}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {osicCompBench.image_tabular_design && (
                <p className="text-[11px] text-slate-500 dark:text-slate-400">
                  <strong>Image + Tabular Formulation:</strong> {osicCompBench.image_tabular_design}
                </p>
              )}
              {osicCompBench.efficientnet_b5_note && (
                <p className="text-[11px] text-slate-500 dark:text-slate-400">
                  <strong>EfficientNet-B5 Note:</strong> {osicCompBench.efficientnet_b5_note}
                </p>
              )}
            </div>
          )}

          {/* Requirement B: 6-Method Fusion Comparison Table + Paired Nadeau-Bengio Test */}
          {fusionComp && fusionComp.methods && (
            <div className="px-5 space-y-2.5">
              <div className="rounded-xl bg-amber-50/90 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800/70 p-3.5 text-xs text-amber-950 dark:text-amber-100 space-y-1.5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-bold uppercase tracking-wider text-amber-900 dark:text-amber-200">
                    6-Method Multimodal Fusion Comparison (5×5 Nested GroupKFold — Identical 25 Folds)
                  </span>
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono font-bold bg-white dark:bg-slate-900 text-medical-700 dark:text-sky-300 border border-amber-300 dark:border-amber-700">
                      selected_fusion_method: {fusionComp.selected_fusion_method} ({fusionComp.fusion_vs_clinical_status || 'nominal_only'})
                    </span>
                    {fusionComp.statistically_confirmed_method && (
                      <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono font-bold bg-emerald-100/90 dark:bg-emerald-950/80 text-emerald-900 dark:text-emerald-200 border border-emerald-300 dark:border-emerald-700">
                        statistically_confirmed_method: {fusionComp.statistically_confirmed_method} (clinically negligible)
                      </span>
                    )}
                  </div>
                </div>
                <p className="leading-relaxed font-semibold">{fusionComp.selection_rationale}</p>
                {fusionComp.paired_test_method && (
                  <p className="text-[11px] text-amber-900/90 dark:text-amber-200/90 leading-relaxed">
                    <strong>Paired Statistical Test:</strong> {fusionComp.paired_test_method}
                  </p>
                )}
                {fusionComp.hrct_branch_design && (
                  <p className="text-[11px] text-amber-900/90 dark:text-amber-200/90 leading-relaxed">
                    <strong>HRCT &amp; Winsorization Design:</strong> {fusionComp.hrct_branch_design}
                  </p>
                )}
              </div>

              <div className="overflow-x-auto rounded-xl border border-slate-200/80 dark:border-slate-800">
                <table className="w-full text-left border-collapse text-xs sm:text-sm">
                  <thead>
                    <tr className="bg-slate-50/80 dark:bg-slate-800/60 border-b border-slate-200/80 dark:border-slate-800 text-[11px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                      <th className="py-2.5 px-4">Fusion Method</th>
                      <th className="py-2.5 px-3 text-right">FVC Laplace LL (95% CI)</th>
                      <th className="py-2.5 px-3 text-right">Paired ΔLL vs. Clinical-only (95% CI, p)</th>
                      <th className="py-2.5 px-3 text-right">Paired ΔMAE mL (95% CI, p)</th>
                      <th className="py-2.5 px-3 text-right">Rapid-Decliner AUC (95% CI)</th>
                      <th className="py-2.5 px-4 text-right">Paired Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200/70 dark:divide-slate-800">
                    {Object.values(fusionComp.methods).map((fm) => {
                      const isSelected = fm.key === fusionComp.selected_fusion_method;
                      const isStatConfirmed = fm.key === fusionComp.statistically_confirmed_method;
                      const paired = fm.paired_vs_clinical_only || fusionComp.paired_vs_clinical_only?.[fm.key];
                      return (
                        <tr
                          key={`fusion-${fm.key}`}
                          className={
                            isSelected
                              ? 'bg-amber-50/60 dark:bg-amber-950/20 font-semibold'
                              : isStatConfirmed
                              ? 'bg-emerald-50/50 dark:bg-emerald-950/20 font-semibold'
                              : 'hover:bg-slate-50 dark:hover:bg-slate-800/40'
                          }
                        >
                          <td className="py-2.5 px-4 font-bold text-slate-900 dark:text-white">
                            {fm.method}
                            {isSelected && (
                              <span className="ml-2 inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-100 text-amber-900 dark:bg-amber-950/80 dark:text-amber-200 border border-amber-300 dark:border-amber-700">
                                Best Mean LL (Nominal Only)
                              </span>
                            )}
                            {isStatConfirmed && (
                              <span className="ml-2 inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-100 text-emerald-900 dark:bg-emerald-950/80 dark:text-emerald-200 border border-emerald-300 dark:border-emerald-700">
                                Statistically Confirmed (Negligible)
                              </span>
                            )}
                          </td>
                          <td className="py-2.5 px-3 text-right font-mono text-xs font-bold text-medical-700 dark:text-sky-300">
                            {fm.laplace_ll_stats?.formatted}
                          </td>
                          <td className="py-2.5 px-3 text-right font-mono text-xs">
                            {fm.key === 'clinical_only' ? (
                              <span className="text-slate-400">Reference</span>
                            ) : (
                              paired?.laplace_ll?.formatted || '—'
                            )}
                          </td>
                          <td className="py-2.5 px-3 text-right font-mono text-xs">
                            {fm.key === 'clinical_only' ? (
                              <span className="text-slate-400">Reference</span>
                            ) : (
                              paired?.mae?.formatted || '—'
                            )}
                          </td>
                          <td className="py-2.5 px-3 text-right font-mono text-xs">
                            {fm.auc_stats?.formatted}
                          </td>
                          <td className="py-2.5 px-4 text-right font-mono text-xs">
                            {fm.key === 'clinical_only' ? (
                              <span className="text-slate-400">Reference</span>
                            ) : (
                              <span
                                className={
                                  paired?.status === 'statistically_confirmed'
                                    ? 'text-emerald-600 dark:text-emerald-400 font-bold'
                                    : 'text-amber-700 dark:text-amber-300 font-bold'
                                }
                              >
                                {paired?.status || 'nominal_only'}
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Pretrained CT Feature Extractor Comparison Table (Same 5x5 Folds, Read-Only Benchmark) */}
          {extractorComp && extractorComp.extractors && (
            <div className="px-5 space-y-3">
              <div className="rounded-xl bg-indigo-50/80 dark:bg-indigo-950/30 border border-indigo-200 dark:border-indigo-800/70 p-3.5 text-xs text-indigo-950 dark:text-indigo-100 space-y-1.5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-bold uppercase tracking-wider text-indigo-900 dark:text-indigo-200">
                    OSIC Pretrained CT Feature Extractor Comparison (Identical 5×5 Patient-Grouped Folds)
                  </span>
                  <div className="flex flex-wrap items-center gap-1.5">
                    {extractorComp.protocol?.folds_sha256 && (
                      <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono font-bold bg-white dark:bg-slate-900 text-indigo-700 dark:text-indigo-300 border border-indigo-300 dark:border-indigo-700">
                        folds_sha256: {extractorComp.protocol.folds_sha256.slice(0, 12)}…
                      </span>
                    )}
                    <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono font-bold bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 border border-slate-300 dark:border-slate-700">
                      paired tests: {extractorComp.multiple_testing?.total_paired_tests ?? 0} (Holm-Bonferroni)
                    </span>
                    <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono font-bold bg-white dark:bg-slate-900 text-amber-800 dark:text-amber-200 border border-amber-300 dark:border-amber-700">
                      alt beats R18 / Clin (Holm):{' '}
                      {extractorComp.multiple_testing?.any_new_extractor_beats_resnet18_holm === null
                        ? extractorComp.multiple_testing?.any_new_extractor_beats_resnet18_holm_note || 'not evaluated'
                        : `${extractorComp.multiple_testing?.any_new_extractor_beats_resnet18_holm ? 'Yes' : 'No'} / ${
                            extractorComp.multiple_testing?.any_new_extractor_beats_clinical_only_holm ? 'Yes' : 'No'
                          }`}
                    </span>
                  </div>
                </div>
                <p className="leading-relaxed font-semibold">{extractorComp.overall_verdict}</p>
                <p className="text-[11px] text-indigo-900/85 dark:text-indigo-200/85 leading-relaxed">
                  <strong>Protocol:</strong> Only <code className="font-mono">emb_prefix</code> varies; same{' '}
                  {extractorComp.protocol?.num_patients} patients, in-fold {extractorComp.protocol?.pca_components}D PCA,
                  delta-from-baseline-FVC target winsorized to [500, 6500] mL, Modified Laplace LL, and Nadeau-Bengio
                  paired tests. {extractorComp.deployed_model_policy}
                </p>
              </div>

              <div className="overflow-x-auto rounded-xl border border-slate-200/80 dark:border-slate-800">
                <table className="w-full text-left border-collapse text-xs sm:text-sm">
                  <thead>
                    <tr className="bg-slate-50/80 dark:bg-slate-800/60 border-b border-slate-200/80 dark:border-slate-800 text-[11px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                      <th className="py-2.5 px-4">Extractor / License</th>
                      <th className="py-2.5 px-3">Status</th>
                      <th className="py-2.5 px-3 text-right">Dim / Time / Peak GPU</th>
                      <th className="py-2.5 px-3 text-right">HRCT-Only LL (AUC)</th>
                      <th className="py-2.5 px-3 text-right">Image+Tab LL (RMSE)</th>
                      <th className="py-2.5 px-3 text-right">Best Fusion LL</th>
                      <th className="py-2.5 px-4 text-right">Paired vs. ResNet-18 / Clinical (Holm p)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200/70 dark:divide-slate-800">
                    {Object.values(extractorComp.extractors).map((ext) => {
                      const isDone = ext.status === 'completed';
                      const isPending = String(ext.status || '').startsWith('pending');
                      const bestKey = ext.best_fusion_method_key;
                      const bestFm = bestKey ? ext.fusion_methods?.[bestKey] : null;
                      const r18Stack = ext.paired_vs_resnet18?.late_fusion_mean?.laplace_ll;
                      const clinStack = ext.paired_vs_clinical_only?.stacking?.laplace_ll;
                      const timeStr =
                        ext.extraction_runtime_minutes != null
                          ? `${Number(ext.extraction_runtime_minutes).toFixed(2)}m`
                          : ext.extraction_runtime_seconds != null
                          ? `${Number(ext.extraction_runtime_seconds).toFixed(1)}s`
                          : 'N/A';
                      const gpuStr =
                        ext.peak_gpu_memory_mb != null
                          ? `${Number(ext.peak_gpu_memory_mb).toFixed(0)} MB`
                          : 'N/A';
                      return (
                        <tr
                          key={`ext-${ext.key}`}
                          className={
                            ext.is_reference
                              ? 'bg-sky-50/50 dark:bg-sky-950/20'
                              : 'hover:bg-slate-50 dark:hover:bg-slate-800/40'
                          }
                        >
                          <td className="py-2.5 px-4">
                            <div className="font-bold text-slate-900 dark:text-white">
                              {ext.short_name || ext.display_name}
                              {ext.is_reference && (
                                <span className="ml-1.5 px-1.5 py-0.5 rounded text-[10px] font-semibold bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-200">
                                  Baseline
                                </span>
                              )}
                              {ext.is_control && (
                                <span className="ml-1.5 px-1.5 py-0.5 rounded text-[10px] font-semibold bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300">
                                  Control
                                </span>
                              )}
                            </div>
                            <span className="block text-[10px] text-slate-500 dark:text-slate-400">
                              {ext.license}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 font-mono text-[11px]">
                            <span
                              className={
                                isDone
                                  ? 'px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 dark:bg-emerald-950/70 dark:text-emerald-300 font-semibold'
                                  : isPending
                                  ? 'px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 dark:bg-amber-950/70 dark:text-amber-200 font-semibold'
                                  : 'px-2 py-0.5 rounded-full bg-rose-100 text-rose-800 dark:bg-rose-950/70 dark:text-rose-200 font-semibold'
                              }
                            >
                              {ext.status}
                            </span>
                            {!isDone && ext.status_reason && (
                              <span className="block text-[10px] text-slate-500 dark:text-slate-400 mt-0.5 max-w-[220px] whitespace-normal leading-snug">
                                {ext.status_reason}
                              </span>
                            )}
                          </td>
                          <td className="py-2.5 px-3 text-right font-mono text-xs text-slate-700 dark:text-slate-300">
                            {ext.embedding_size ? `${ext.embedding_size}D` : '—'}
                            <span className="block text-[10px] text-slate-400">
                              {isDone ? `${timeStr} • ${gpuStr}` : '—'}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-right font-mono text-xs">
                            {isDone && ext.hrct_only ? (
                              <>
                                <span className="font-semibold text-slate-900 dark:text-slate-100">
                                  {Number(ext.hrct_only.laplace_ll_stats.mean).toFixed(4)}
                                </span>
                                <span className="block text-[10px] text-slate-400">
                                  AUC {Number(ext.hrct_only.auc_stats.mean).toFixed(4)}
                                </span>
                              </>
                            ) : (
                              <span className="text-slate-400">{ext.status}</span>
                            )}
                          </td>
                          <td className="py-2.5 px-3 text-right font-mono text-xs">
                            {isDone && ext.image_plus_tabular ? (
                              <>
                                <span className="font-semibold text-slate-900 dark:text-slate-100">
                                  {Number(ext.image_plus_tabular.laplace_ll_stats.mean).toFixed(4)}
                                </span>
                                <span className="block text-[10px] text-slate-400">
                                  RMSE {Number(ext.image_plus_tabular.rmse_stats.mean).toFixed(1)} mL
                                </span>
                              </>
                            ) : (
                              <span className="text-slate-400">{ext.status}</span>
                            )}
                          </td>
                          <td className="py-2.5 px-3 text-right font-mono text-xs">
                            {isDone && bestFm ? (
                              <>
                                <span className="font-bold text-medical-700 dark:text-sky-300">
                                  {Number(bestFm.laplace_ll_stats.mean).toFixed(4)}
                                </span>
                                <span className="block text-[10px] text-slate-400">
                                  {ext.best_fusion_method_name}
                                </span>
                              </>
                            ) : (
                              <span className="text-slate-400">{ext.status}</span>
                            )}
                          </td>
                          <td className="py-2.5 px-4 text-right font-mono text-[11px]">
                            {!isDone ? (
                              <span className="text-slate-400">{ext.status}</span>
                            ) : ext.is_reference ? (
                              <span>
                                Ref vs. R18
                                {clinStack && (
                                  <span className="block text-[10px] text-slate-500 dark:text-slate-400">
                                    Stack vs Clin: {clinStack.mean_diff >= 0 ? '+' : ''}
                                    {Number(clinStack.mean_diff).toFixed(4)} (p_raw={Number(clinStack.p_value).toFixed(4)}, p_holm={Number(clinStack.p_value_holm).toFixed(4)})
                                  </span>
                                )}
                              </span>
                            ) : (
                              <span>
                                {r18Stack && (
                                  <span className="block text-slate-800 dark:text-slate-200">
                                    LF vs R18: {r18Stack.mean_diff >= 0 ? '+' : ''}
                                    {Number(r18Stack.mean_diff).toFixed(4)} (p_holm={Number(r18Stack.p_value_holm).toFixed(4)})
                                  </span>
                                )}
                                {clinStack && (
                                  <span className="block text-[10px] text-slate-500 dark:text-slate-400">
                                    Stack vs Clin: {clinStack.mean_diff >= 0 ? '+' : ''}
                                    {Number(clinStack.mean_diff).toFixed(4)} (p_holm={Number(clinStack.p_value_holm).toFixed(4)})
                                  </span>
                                )}
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
                <strong>Limits &amp; Licensing:</strong> All raw embeddings are equalized by in-fold PCA to 15 dimensions;
                3D models require volumetric preprocessing distinct from 2D per-slice models; internal 5×5 CV on 175 OSIC
                patients (not leaderboard scores). MedicalNet &amp; RadImageNet mirrors: MIT; CT-CLIP/CT-RATE: gated,
                CC BY-NC-SA 4.0 non-commercial (weights not redistributed). Full tables &amp; citations in{' '}
                <code className="font-mono">docs/EXTRACTOR_COMPARISON.md</code>.
              </p>
            </div>
          )}

          {/* 3-Way Modality Ablation Comparison Tables (Both Targets, rendered when completed) */}
          {osicStudy.modality_ablation?.status === 'completed' &&
            osicStudy.modality_ablation?.by_target && (
              <div className="px-5 space-y-4">
                <div className="rounded-xl bg-sky-50/80 dark:bg-sky-950/30 border border-sky-200 dark:border-sky-800/70 p-3.5 text-xs text-sky-950 dark:text-sky-100">
                  <span className="font-bold">
                    3-Way Imaging vs. Clinical-Only Ablation (5-Repeat × 5-Fold GroupKFold, In-Fold PCA{' '}
                    {osicStudy.modality_ablation.pca_components_used}D):
                  </span>{' '}
                  {osicStudy.modality_ablation.any_imaging_beats_clinical_only_ci
                    ? 'At least one imaging modality improved over Clinical-only with non-overlapping 95% CIs.'
                    : 'Neither Clinical + HU/Volume nor Clinical + ResNet-18 (In-Fold PCA) beats Clinical-only with non-overlapping 95% CIs; imaging progression remains exploratory.'}
                </div>

                {Object.entries(osicStudy.modality_ablation.by_target).map(([abKey, abObj]) => (
                  <div key={`ablation-${abKey}`} className="space-y-2">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-medical-700 dark:text-sky-300">
                      3-Way Modality Ablation — {abObj.target_label} (n = {abObj.num_patients} CT patients)
                    </h3>
                    <div className="overflow-x-auto rounded-xl border border-slate-200/80 dark:border-slate-800">
                      <table className="w-full text-left border-collapse text-xs sm:text-sm">
                        <thead>
                          <tr className="bg-slate-50/80 dark:bg-slate-800/60 border-b border-slate-200/80 dark:border-slate-800 text-[11px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                            <th className="py-2.5 px-4">Modality / Baseline</th>
                            <th className="py-2.5 px-3 text-right">RMSE (95% CI)</th>
                            <th className="py-2.5 px-3 text-right">MAE (95% CI)</th>
                            <th className="py-2.5 px-3 text-right">Rapid Decliner AUC (95% CI)</th>
                            <th className="py-2.5 px-3 text-right">Beats Baseline (95% CI)?</th>
                            <th className="py-2.5 px-4 text-right">Beats Clinical-Only (95% CI)?</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-200/70 dark:divide-slate-800">
                          {Object.values(abObj.modalities || {}).map((mod) => {
                            const isBase = mod.modality.includes('Baseline');
                            const isClin = mod.modality === 'Clinical-only';
                            const vb = mod.vs_baseline_ci;
                            const vc = mod.vs_clinical_only_ci;
                            const formatVerdict = (v, isBaselineComp = false) => {
                              if (!v) return '—';
                              const beatsReg = Boolean(v.beats_rmse_ci || v.beats_mae_ci);
                              const beatsAuc = Boolean(v.beats_auc_ci);
                              if (isBaselineComp && beatsAuc && !beatsReg) {
                                return 'AUC above 0.5 only; RMSE and MAE no better than baseline';
                              }
                              const wins = [];
                              if (v.beats_rmse_ci) wins.push('RMSE');
                              if (v.beats_mae_ci) wins.push('MAE');
                              if (v.beats_auc_ci) wins.push('AUC');
                              return wins.length > 0 ? `Yes (${wins.join(', ')})` : 'No';
                            };
                            const vbBeatsRegression = Boolean(
                              vb?.beats_rmse_ci || vb?.beats_mae_ci
                            );
                            const vcBeatsAny = Boolean(
                              vc?.beats_rmse_ci || vc?.beats_mae_ci || vc?.beats_auc_ci
                            );
                            return (
                              <tr
                                key={`${abKey}-${mod.modality}`}
                                className={
                                  isBase
                                    ? 'bg-slate-100/70 dark:bg-slate-800/70 font-semibold'
                                    : 'hover:bg-slate-50 dark:hover:bg-slate-800/40'
                                }
                              >
                                <td className="py-2.5 px-4 font-bold text-slate-900 dark:text-white">
                                  {mod.modality}
                                </td>
                                <td className="py-2.5 px-3 text-right font-mono text-xs">
                                  {mod.rmse_stats?.formatted}
                                </td>
                                <td className="py-2.5 px-3 text-right font-mono text-xs">
                                  {mod.mae_stats?.formatted}
                                </td>
                                <td className="py-2.5 px-3 text-right font-mono text-xs">
                                  {mod.auc_stats?.formatted}
                                </td>
                                <td className="py-2.5 px-3 text-right font-mono text-xs">
                                  {isBase ? (
                                    <span className="text-slate-500">Reference</span>
                                  ) : (
                                    <span
                                      className={
                                        vbBeatsRegression
                                          ? 'text-emerald-600 dark:text-emerald-400 font-bold'
                                          : 'text-amber-600 dark:text-amber-400 font-bold'
                                      }
                                    >
                                      {formatVerdict(vb, true)}
                                    </span>
                                  )}
                                </td>
                                <td className="py-2.5 px-4 text-right font-mono text-xs">
                                  {isBase || isClin ? (
                                    <span className="text-slate-500">
                                      {isClin ? 'Reference' : '—'}
                                    </span>
                                  ) : (
                                    <span
                                      className={
                                        vcBeatsAny
                                          ? 'text-emerald-600 dark:text-emerald-400 font-bold'
                                          : 'text-amber-600 dark:text-amber-400 font-bold'
                                      }
                                    >
                                      {formatVerdict(vc, false)}
                                    </span>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ))}
              </div>
            )}

          {Object.entries(osicStudy.target_variants || { robust_theil_sen_slope: { target_label: 'Robust Theil-Sen FVC Slope (mL/week)', models: osicStudy.models } }).map(
            ([variantKey, variantObj]) => (
              <div key={variantKey} className="px-5 space-y-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                  Target Variant: {variantObj.target_label} (n = {variantObj.num_patients ?? osicStudy.num_patients} patients)
                </h3>
                <div className="overflow-x-auto rounded-xl border border-slate-200/80 dark:border-slate-800">
                  <table className="w-full text-left border-collapse text-xs sm:text-sm">
                    <thead>
                      <tr className="bg-slate-50/80 dark:bg-slate-800/60 border-b border-slate-200/80 dark:border-slate-800 text-[11px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                        <th className="py-2.5 px-4">Model / Baseline</th>
                        <th className="py-2.5 px-3 text-right">Nested GroupKFold RMSE (95% CI)</th>
                        <th className="py-2.5 px-3 text-right">Nested GroupKFold MAE (95% CI)</th>
                        <th className="py-2.5 px-3 text-right">Rapid Decliner AUC (95% CI)</th>
                        <th className="py-2.5 px-4 text-right">Beats Baseline?</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200/70 dark:divide-slate-800">
                      {Object.values(variantObj.models || {}).map((m) => {
                        const isBase = m.model.includes('Baseline');
                        return (
                          <tr
                            key={`${variantKey}-${m.model}`}
                            className={
                              isBase
                                ? 'bg-slate-100/70 dark:bg-slate-800/70 font-semibold'
                                : 'hover:bg-slate-50 dark:hover:bg-slate-800/40'
                            }
                          >
                            <td className="py-2.5 px-4 font-bold text-slate-900 dark:text-white">
                              {m.model}
                            </td>
                            <td className="py-2.5 px-3 text-right font-mono text-xs">
                              {m.rmse_stats?.formatted || `${m.cv_rmse_mean} ± ${m.cv_rmse_std}`}
                            </td>
                            <td className="py-2.5 px-3 text-right font-mono text-xs">
                              {m.mae_stats?.formatted || m.cv_mae_mean}
                            </td>
                            <td className="py-2.5 px-3 text-right font-mono text-xs">
                              {m.auc_stats?.formatted || m.cv_auc_mean}
                            </td>
                            <td className="py-2.5 px-4 text-right font-mono text-xs">
                              {isBase ? (
                                <span className="text-slate-500">Reference Baseline</span>
                              ) : (
                                <span
                                  className={
                                    m.clearly_beats_regression_ci && m.beats_classification_baseline
                                      ? 'text-emerald-600 dark:text-emerald-400 font-bold'
                                      : 'text-amber-600 dark:text-amber-400 font-bold'
                                  }
                                >
                                  Reg CI: {m.clearly_beats_regression_ci ? 'Clear' : 'Overlaps'} | Cls:{' '}
                                  {m.beats_classification_baseline ? 'Yes' : 'No'} (Exploratory)
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )
          )}
        </div>
      )}
    </div>
  );
}
