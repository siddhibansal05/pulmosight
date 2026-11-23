import React, { useState, useEffect } from 'react';
import {
  PieChart as PieChartIcon,
  Users,
  HeartPulse,
  BarChart2,
  Calendar,
  Activity,
  AlertCircle,
  RefreshCw,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line,
  Cell,
} from 'recharts';
import { fetchEdaInsights } from '../services/api';

const GROUP_COLORS = {
  IPF: '#e11d48',
  '1_SAR': '#0284c7',
  'o-ILD': '#0ea5e9',
  CTD: '#38bdf8',
  HP: '#10b981',
  'u-ILD': '#64748b',
  'i-NSIP': '#8b5cf6',
};

export default function DataInsightsPage({ darkMode }) {
  const [eda, setEda] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadEda = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await fetchEdaInsights();
      setEda(data);
    } catch (err) {
      setError(
        err?.message ||
          'Backend API (/eda) is unreachable. Start uvicorn backend.api.main:app --port 8000.'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadEda();
  }, []);

  if (loading) {
    return (
      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-12 text-center space-y-3">
        <div className="w-8 h-8 border-2 border-medical-600 border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-sm font-medium text-slate-600 dark:text-slate-400">
          Loading real Zenodo ILD cohort statistics from FastAPI (<code className="font-mono">GET /eda</code>)...
        </p>
      </div>
    );
  }

  if (error || !eda) {
    return (
      <div className="rounded-2xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 p-6 space-y-4">
        <div className="flex items-center gap-3 text-rose-700 dark:text-rose-300">
          <AlertCircle className="w-6 h-6 shrink-0" />
          <div>
            <h2 className="text-base font-bold">Unable to load live EDA data from GET /eda</h2>
            <p className="text-xs mt-0.5">{error}</p>
          </div>
        </div>
        <button
          type="button"
          onClick={loadEda}
          className="inline-flex items-center gap-2 rounded-xl bg-rose-600 hover:bg-rose-700 text-white px-4 py-2 text-xs font-semibold"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Retry Connection</span>
        </button>
      </div>
    );
  }

  const diagnosisData = eda.diagnosis_group_counts || [];
  const ageData = eda.age_distribution || [];
  const survivalData = eda.survival_by_diagnosis || [];

  return (
    <div className="space-y-6">
      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-5 shadow-clinical flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-medical-600 dark:text-sky-400">
            <BarChart2 className="w-4 h-4" />
            <span>Page 4 • Real Zenodo ILD Exploratory Data Analysis (GET /eda)</span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white mt-1">
            Clinical Cohort Data Insights (EDA)
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-1 max-w-3xl">
            Real distributions from <code className="font-mono">database_final_anonymized_for_zenodo.csv</code> (N ={' '}
            {Number(eda.total_rows).toLocaleString()} cases across 1_SAR, o-ILD, IPF, CTD, HP, u-ILD, i-NSIP).
          </p>
        </div>

        <span className="px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800 text-xs font-mono font-semibold shrink-0">
          data_source: {eda.data_source} (MD5: {String(eda.zenodo_md5 || '').slice(0, 8)}…)
        </span>
      </div>

      {/* Cohort KPI Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-4 shadow-clinical flex items-center gap-3.5">
          <div className="p-3 rounded-xl bg-medical-50 dark:bg-sky-950/60 text-medical-600 dark:text-sky-400">
            <Users className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
              Total Zenodo Cohort
            </span>
            <span className="text-2xl font-extrabold font-mono text-slate-900 dark:text-white">
              {Number(eda.total_rows).toLocaleString()}
            </span>
            <span className="text-xs text-slate-500 dark:text-slate-400 block">
              {eda.ipf_rows} IPF ({eda.ipf_prevalence_pct}%) • {eda.non_ipf_rows} Non-IPF
            </span>
          </div>
        </div>

        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-4 shadow-clinical flex items-center gap-3.5">
          <div className="p-3 rounded-xl bg-medical-50 dark:bg-sky-950/60 text-medical-600 dark:text-sky-400">
            <Calendar className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
              Median Age (IPF)
            </span>
            <span className="text-2xl font-extrabold font-mono text-slate-900 dark:text-white">
              {eda.median_age_ipf} yrs
            </span>
            <span className="text-xs text-slate-500 dark:text-slate-400 block">
              vs. {eda.median_age_non_ipf} yrs in Non-IPF
            </span>
          </div>
        </div>

        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-4 shadow-clinical flex items-center gap-3.5">
          <div className="p-3 rounded-xl bg-medical-50 dark:bg-sky-950/60 text-medical-600 dark:text-sky-400">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
              Mean FVC / TLCO (IPF)
            </span>
            <span className="text-2xl font-extrabold font-mono text-slate-900 dark:text-white">
              {eda.mean_fvc_ipf}% / {eda.mean_tlco_ipf}%
            </span>
            <span className="text-xs text-slate-500 dark:text-slate-400 block">
              Baseline % predicted
            </span>
          </div>
        </div>

        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-4 shadow-clinical flex items-center gap-3.5">
          <div className="p-3 rounded-xl bg-rose-50 dark:bg-rose-950/60 text-rose-600 dark:text-rose-400">
            <HeartPulse className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
              Median Follow-Up (IPF)
            </span>
            <span className="text-2xl font-extrabold font-mono text-slate-900 dark:text-white">
              {eda.median_survival_ipf_months} mos
            </span>
            <span className="text-xs text-slate-500 dark:text-slate-400 block">
              From First test to event
            </span>
          </div>
        </div>
      </div>

      {/* Top Row Charts: Diagnosis Group Distribution & Age Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-6 shadow-clinical flex flex-col justify-between space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-4">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-medical-50 dark:bg-sky-950/60 text-medical-600 dark:text-sky-400">
                <PieChartIcon className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-base font-bold text-slate-900 dark:text-white">
                  1. Real Zenodo Diagnosis Group Distribution
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Counts across 1_SAR, o-ILD, IPF, CTD, HP, u-ILD, and i-NSIP
                </p>
              </div>
            </div>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={diagnosisData} margin={{ top: 12, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#1e293b' : '#e2e8f0'} />
                <XAxis
                  dataKey="group"
                  tick={{ fill: darkMode ? '#cbd5e1' : '#334155', fontSize: 12, fontWeight: 600 }}
                />
                <YAxis tick={{ fill: darkMode ? '#94a3b8' : '#475569', fontSize: 11 }} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0].payload;
                    return (
                      <div className="rounded-xl bg-slate-900 text-white p-3 shadow-lg border border-slate-700 text-xs space-y-1">
                        <div className="font-bold text-sky-300">
                          {d.group} — {d.full_name}
                        </div>
                        <div className="font-mono">
                          Patients: <span className="font-bold">{d.count}</span> ({d.percentage}%)
                        </div>
                        <div className="text-slate-300 font-mono text-[11px]">
                          Mean FVC: {d.mean_fvc_pp}% • Mean TLCO: {d.mean_tlco_pp}%
                        </div>
                      </div>
                    );
                  }}
                />
                <Bar dataKey="count" name="Patient Count" radius={[6, 6, 0, 0]} barSize={34}>
                  {diagnosisData.map((entry, index) => (
                    <Cell
                      key={`diag-${index}`}
                      fill={GROUP_COLORS[entry.group] || '#0284c7'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-3 sm:grid-cols-7 gap-1.5 pt-2 border-t border-slate-100 dark:border-slate-800 text-center">
            {diagnosisData.map((d) => (
              <div
                key={d.group}
                className="p-1.5 rounded-lg bg-slate-50 dark:bg-slate-800/50 text-xs"
              >
                <span className="font-mono font-bold text-slate-700 dark:text-slate-300 block truncate">
                  {d.group}
                </span>
                <span className="font-mono text-[11px] text-medical-600 dark:text-sky-400">
                  {d.percentage}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Chart 2: Age Distribution */}
        <div className="lg:col-span-6 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-6 shadow-clinical flex flex-col justify-between space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-4">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-medical-50 dark:bg-sky-950/60 text-medical-600 dark:text-sky-400">
                <Users className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-base font-bold text-slate-900 dark:text-white">
                  2. Baseline Age Distribution by Diagnosis
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Computed as age at event − survival time / 365.25
                </p>
              </div>
            </div>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ageData} margin={{ top: 12, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#1e293b' : '#e2e8f0'} />
                <XAxis
                  dataKey="ageRange"
                  tick={{ fill: darkMode ? '#cbd5e1' : '#334155', fontSize: 12, fontWeight: 500 }}
                />
                <YAxis tick={{ fill: darkMode ? '#94a3b8' : '#475569', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    borderRadius: '0.75rem',
                    color: '#f8fafc',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '6px' }} />
                <Bar dataKey="ipf" name="IPF Cohort" fill="#e11d48" radius={[4, 4, 0, 0]} barSize={22} />
                <Bar dataKey="nonIpf" name="Non-IPF Cohort" fill="#0284c7" radius={[4, 4, 0, 0]} barSize={22} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/70 dark:border-slate-800 p-3 text-xs text-slate-600 dark:text-slate-400">
            <span className="font-semibold text-slate-800 dark:text-slate-200">
              Real Zenodo Age Profile:
            </span>{' '}
            IPF median baseline age is {eda.median_age_ipf} years compared to {eda.median_age_non_ipf}{' '}
            years across Non-IPF ILD groups.
          </div>
        </div>
      </div>

      {/* Bottom Row Chart: Survival by Diagnosis */}
      <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-6 shadow-clinical space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-100 dark:border-slate-800 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-medical-50 dark:bg-sky-950/60 text-medical-600 dark:text-sky-400">
              <HeartPulse className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-900 dark:text-white">
                3. Survival by Diagnosis (0 to 60 Months, Real Zenodo Registry)
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Observed survival percentage from First test across IPF and Non-IPF diagnosis groups
              </p>
            </div>
          </div>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={survivalData} margin={{ top: 12, right: 24, left: 6, bottom: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#1e293b' : '#e2e8f0'} />
              <XAxis
                dataKey="month"
                tickFormatter={(m) => `${m}m`}
                tick={{ fill: darkMode ? '#94a3b8' : '#475569', fontSize: 12 }}
              />
              <YAxis
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
                tick={{ fill: darkMode ? '#94a3b8' : '#475569', fontSize: 12 }}
              />
              <Tooltip
                formatter={(value, name) => [`${Number(value).toFixed(1)}%`, name]}
                labelFormatter={(label) => `Month ${label} Follow-Up`}
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '0.75rem',
                  color: '#f8fafc',
                  fontSize: '12px',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
              <Line
                type="stepAfter"
                dataKey="ctdNsip"
                name="CTD / i-NSIP / 1_SAR"
                stroke="#10b981"
                strokeWidth={2.8}
                dot={{ r: 3 }}
              />
              <Line
                type="stepAfter"
                dataKey="nonIpf"
                name="All Non-IPF Groups Combined"
                stroke="#0284c7"
                strokeWidth={2.6}
                dot={{ r: 3 }}
              />
              <Line
                type="stepAfter"
                dataKey="ipf"
                name="Idiopathic Pulmonary Fibrosis (IPF)"
                stroke="#e11d48"
                strokeWidth={3.2}
                dot={{ r: 3.5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
