import React, { useState, useEffect } from 'react';
import {
  Stethoscope,
  ClipboardList,
  Activity,
  Cpu,
  BarChart2,
  Sun,
  Moon,
  ShieldCheck,
} from 'lucide-react';
import PatientInputPage, { SAMPLE_PATIENT_INPUTS } from './components/PatientInputPage';
import ResultPage from './components/ResultPage';
import ModelInfoPage from './components/ModelInfoPage';
import DataInsightsPage from './components/DataInsightsPage';
import {
  predict,
  predictProgressionMultimodal,
  API_BASE_URL,
} from './services/api';

const NAV_TABS = [
  {
    id: 'input',
    pageNum: 'Page 1',
    label: 'Patient Input',
    icon: ClipboardList,
  },
  {
    id: 'result',
    pageNum: 'Page 2',
    label: 'Diagnostic Result',
    icon: Activity,
  },
  {
    id: 'model',
    pageNum: 'Page 3',
    label: 'Model Info',
    icon: Cpu,
  },
  {
    id: 'eda',
    pageNum: 'Page 4',
    label: 'Data Insights (EDA)',
    icon: BarChart2,
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('input');
  const [darkMode, setDarkMode] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);

  const [formData, setFormData] = useState({
    sex: '',
    age: '',
    height: '',
    weight: '',
    fev1Fvc: '',
    fev1Pred: '',
    fvcPred: '',
    tlcPred: '',
    tlcoPred: '',
    operatingMode: 'screening',
    hrctFile: null,
  });

  const [predictionResult, setPredictionResult] = useState(null);
  const [multimodalProgression, setMultimodalProgression] = useState(null);

  useEffect(() => {
    let mounted = true;
    const defaultData = { ...SAMPLE_PATIENT_INPUTS[0].data, operatingMode: 'screening' };
    Promise.all([
      predict(defaultData),
      predictProgressionMultimodal(defaultData),
    ]).then(([initialRes, initialMulti]) => {
      if (mounted) {
        setPredictionResult(initialRes);
        setMultimodalProgression(initialMulti);
      }
    });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [darkMode]);

  const handlePredictSubmit = async (validData) => {
    setIsPredicting(true);
    try {
      const [res, multiRes] = await Promise.all([
        predict(validData),
        predictProgressionMultimodal(validData),
      ]);
      setPredictionResult(res);
      setMultimodalProgression(multiRes);
      setActiveTab('result');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } finally {
      setIsPredicting(false);
    }
  };

  const handleSwitchOperatingMode = async (newMode) => {
    const baseData =
      formData.age && formData.fvcPred ? formData : SAMPLE_PATIENT_INPUTS[0].data;
    setFormData((prev) => ({ ...prev, operatingMode: newMode }));
    const res = await predict({ ...baseData, operatingMode: newMode });
    setPredictionResult(res);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50/80 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors duration-200">
      <header className="sticky top-0 z-30 bg-white/95 dark:bg-slate-900/95 backdrop-blur-md border-b border-medical-100 dark:border-slate-800 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16 gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-medical-500 to-medical-700 flex items-center justify-center text-white shadow-sm shrink-0">
                <Stethoscope className="w-5 h-5" />
              </div>
              <div className="truncate">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-base sm:text-lg tracking-tight text-slate-900 dark:text-white">
                    PulmoSight <span className="text-medical-600 dark:text-sky-400">CDSS</span>
                  </span>
                  <span className="hidden sm:inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-medical-50 dark:bg-sky-950/80 text-medical-700 dark:text-sky-300 border border-medical-200 dark:border-sky-800">
                    <ShieldCheck className="w-3 h-3" />
                    Real Zenodo ILD Cohort
                  </span>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400 truncate">
                  Idiopathic Pulmonary Fibrosis Clinical Decision Support System
                </p>
              </div>
            </div>

            <nav
              aria-label="Main Navigation"
              className="hidden md:flex items-center gap-1.5 bg-slate-100/90 dark:bg-slate-800/80 p-1.5 rounded-xl border border-slate-200/70 dark:border-slate-700/70"
            >
              {NAV_TABS.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setActiveTab(tab.id)}
                    className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition ${
                      isActive
                        ? 'bg-white dark:bg-slate-900 text-medical-600 dark:text-sky-400 shadow-sm'
                        : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                    }`}
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </nav>

            <div className="flex items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={() => setDarkMode((d) => !d)}
                aria-label={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
                className="inline-flex items-center gap-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 hover:bg-medical-50 dark:bg-slate-800 dark:hover:bg-slate-700 px-3 py-2 text-xs font-semibold text-slate-700 dark:text-slate-200 transition"
              >
                {darkMode ? (
                  <>
                    <Sun className="w-4 h-4 text-amber-400" />
                    <span className="hidden sm:inline">Light Mode</span>
                  </>
                ) : (
                  <>
                    <Moon className="w-4 h-4 text-medical-600" />
                    <span className="hidden sm:inline">Dark Mode</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        <div className="md:hidden border-t border-slate-200/80 dark:border-slate-800 bg-slate-50/90 dark:bg-slate-900 px-2 py-1.5 grid grid-cols-4 gap-1">
          {NAV_TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`flex flex-col items-center justify-center py-1.5 px-1 rounded-lg text-[11px] font-semibold transition ${
                  isActive
                    ? 'bg-medical-600 text-white shadow-sm'
                    : 'text-slate-600 dark:text-slate-400 hover:bg-slate-200/60 dark:hover:bg-slate-800'
                }`}
              >
                <Icon className="w-4 h-4 mb-0.5" />
                <span className="truncate max-w-full">{tab.label}</span>
              </button>
            );
          })}
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {activeTab === 'input' && (
          <PatientInputPage
            formData={formData}
            setFormData={setFormData}
            onSubmitPrediction={handlePredictSubmit}
            isPredicting={isPredicting}
          />
        )}

        {activeTab === 'result' && (
          <ResultPage
            result={predictionResult}
            multimodalProgression={multimodalProgression}
            onBackToInput={() => setActiveTab('input')}
            onSwitchOperatingMode={handleSwitchOperatingMode}
            darkMode={darkMode}
          />
        )}

        {activeTab === 'model' && <ModelInfoPage darkMode={darkMode} />}

        {activeTab === 'eda' && <DataInsightsPage darkMode={darkMode} />}
      </main>

      <footer className="border-t border-slate-200/80 dark:border-slate-800/80 bg-white dark:bg-slate-900 py-4 px-4 sm:px-6 lg:px-8 mt-auto">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-500 dark:text-slate-400">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-700 dark:text-slate-300">
              PulmoSight CDSS v2.2
            </span>
            <span>•</span>
            <span>Decision support only, not a clinical diagnosis.</span>
          </div>
          <div className="font-mono text-[11px] text-slate-400">
            VITE_API_BASE_URL: <code className="text-medical-600 dark:text-sky-400">{API_BASE_URL}</code>
          </div>
        </div>
      </footer>
    </div>
  );
}
