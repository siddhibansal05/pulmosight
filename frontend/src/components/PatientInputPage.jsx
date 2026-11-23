import React, { useState, useRef } from 'react';
import {
  UserRound,
  Activity,
  UploadCloud,
  Sparkles,
  RotateCcw,
  FlaskConical,
  AlertCircle,
  CheckCircle2,
  Info,
} from 'lucide-react';
import HrctPreview from './HrctPreview';

// Real Zenodo sample patient input presets for the "Load sample patient" button on Page 1
export const SAMPLE_PATIENT_INPUTS = [
  {
    id: 'sample-ipf-high',
    label: 'Sample 1: Zenodo IPF Case #3 (66M, Restrictive PFTs)',
    data: {
      sex: 'M',
      age: '66',
      height: '1.71',
      weight: '90.0',
      fev1Fvc: '0.83',
      fev1Pred: '65.6',
      fvcPred: '60.5',
      tlcPred: '64.2',
      tlcoPred: '40.1',
      hrctFile: {
        name: 'thorax_hrct_axial_uip_66M.dcm',
        size: 1498546,
        format: 'DICOM (.dcm)',
        isSample: true,
        pattern: 'uip',
      },
    },
  },
  {
    id: 'sample-non-ipf-low',
    label: 'Sample 2: Zenodo Sarcoidosis Case #16 (51F, Preserved Volumes)',
    data: {
      sex: 'F',
      age: '51',
      height: '1.68',
      weight: '76.0',
      fev1Fvc: '0.72',
      fev1Pred: '71.0',
      fvcPred: '77.9',
      tlcPred: '95.5',
      tlcoPred: '90.8',
      hrctFile: null,
    },
  },
  {
    id: 'sample-ctd-moderate',
    label: 'Sample 3: Zenodo CTD-ILD Case #12 (55M, Mild Diffusion Deficit)',
    data: {
      sex: 'M',
      age: '55',
      height: '1.65',
      weight: '66.0',
      fev1Fvc: '0.72',
      fev1Pred: '92.4',
      fvcPred: '101.9',
      tlcPred: '105.0',
      tlcoPred: '81.6',
      hrctFile: null,
    },
  },
];

const INITIAL_FORM = {
  sex: '',
  age: '',
  height: '',
  weight: '',
  fev1Fvc: '',
  fev1Pred: '',
  fvcPred: '',
  tlcPred: '',
  tlcoPred: '',
  hrctFile: null,
};

export default function PatientInputPage({
  formData,
  setFormData,
  onSubmitPrediction,
  isPredicting,
}) {
  const [errors, setErrors] = useState({});
  const [selectedSampleIdx, setSelectedSampleIdx] = useState(0);
  const [fileError, setFileError] = useState('');
  const fileInputRef = useRef(null);

  // Explicit unit validation matching backend Pydantic schema (Height in meters, FEV1/FVC as decimal ratio)
  const validateField = (name, value) => {
    const trimmed = String(value ?? '').trim();

    switch (name) {
      case 'sex':
        if (!trimmed || (trimmed !== 'M' && trimmed !== 'F')) {
          return 'Please select biological sex (Male or Female).';
        }
        return '';
      case 'age': {
        if (!trimmed) return 'Patient age is required.';
        const num = Number(trimmed);
        if (Number.isNaN(num) || num < 18 || num > 110) {
          return 'Enter a valid adult age between 18 and 110 years.';
        }
        return '';
      }
      case 'height': {
        if (!trimmed) return 'Height in meters is required.';
        const num = Number(trimmed);
        if (Number.isNaN(num) || num < 1.0 || num > 2.5) {
          return 'Enter height strictly in meters between 1.00 m and 2.50 m (e.g., 1.72).';
        }
        return '';
      }
      case 'weight': {
        if (!trimmed) return 'Weight in kilograms is required.';
        const num = Number(trimmed);
        if (Number.isNaN(num) || num < 25 || num > 260) {
          return 'Enter weight in kg between 25 kg and 260 kg.';
        }
        return '';
      }
      case 'fev1Fvc': {
        if (!trimmed) return 'FEV1/FVC decimal ratio is required.';
        const num = Number(trimmed);
        if (Number.isNaN(num) || num < 0.20 || num > 1.15) {
          return 'Enter FEV1/FVC strictly as a decimal ratio between 0.20 and 1.15 (e.g., 0.83).';
        }
        return '';
      }
      case 'fev1Pred': {
        if (!trimmed) return 'FEV1 % predicted is required.';
        const num = Number(trimmed);
        if (Number.isNaN(num) || num < 10 || num > 250) {
          return 'Enter FEV1 % predicted between 10% and 250%.';
        }
        return '';
      }
      case 'fvcPred': {
        if (!trimmed) return 'FVC % predicted is required.';
        const num = Number(trimmed);
        if (Number.isNaN(num) || num < 10 || num > 250) {
          return 'Enter FVC % predicted between 10% and 250%.';
        }
        return '';
      }
      case 'tlcPred': {
        if (!trimmed) return 'TLC % predicted is required.';
        const num = Number(trimmed);
        if (Number.isNaN(num) || num < 10 || num > 250) {
          return 'Enter TLC % predicted between 10% and 250%.';
        }
        return '';
      }
      case 'tlcoPred': {
        if (!trimmed) return 'TLCO % predicted is required.';
        const num = Number(trimmed);
        if (Number.isNaN(num) || num < 10 || num > 250) {
          return 'Enter TLCO % predicted between 10% and 250%.';
        }
        return '';
      }
      default:
        return '';
    }
  };

  const validateAll = (data) => {
    const requiredFields = [
      'sex',
      'age',
      'height',
      'weight',
      'fev1Fvc',
      'fev1Pred',
      'fvcPred',
      'tlcPred',
      'tlcoPred',
    ];
    const nextErrors = {};
    requiredFields.forEach((field) => {
      const msg = validateField(field, data[field]);
      if (msg) nextErrors[field] = msg;
    });
    return nextErrors;
  };

  const handleFieldChange = (name, value) => {
    const updated = { ...formData, [name]: value };
    setFormData(updated);
    const fieldErr = validateField(name, value);
    setErrors((prev) => ({
      ...prev,
      [name]: fieldErr,
    }));
  };

  const handleFieldBlur = (name) => {
    const fieldErr = validateField(name, formData[name]);
    setErrors((prev) => ({
      ...prev,
      [name]: fieldErr,
    }));
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const lower = file.name.toLowerCase();
    const isZip = lower.endsWith('.zip');
    const isDicom = lower.endsWith('.dcm') || lower.endsWith('.dicom');
    const isNifti = lower.endsWith('.nii') || lower.endsWith('.nii.gz');
    const isImage =
      lower.endsWith('.png') ||
      lower.endsWith('.jpg') ||
      lower.endsWith('.jpeg') ||
      file.type === 'image/png' ||
      file.type === 'image/jpeg';

    if (!isZip && !isDicom && !isNifti && !isImage) {
      setFileError(
        'Unsupported file format. Upload a full CT DICOM series (.zip, max 50 MB) for HRCT/Fusion progression scoring, or a preview file (.dcm, .nii, .png, .jpg).'
      );
      return;
    }

    const maxBytes = isZip ? 50 * 1024 * 1024 : 25 * 1024 * 1024;
    if (file.size > maxBytes) {
      setFileError(
        `File exceeds the maximum allowed upload size (${isZip ? '50 MB for CT .zip series' : '25 MB'}).`
      );
      return;
    }

    setFileError('');
    let formatLabel = 'Image (PNG/JPG)';
    if (isZip) formatLabel = 'CT DICOM Series (.zip)';
    if (isDicom) formatLabel = 'DICOM (.dcm)';
    if (isNifti) formatLabel = 'NIfTI (.nii)';

    const previewUrl = isImage ? URL.createObjectURL(file) : null;
    setFormData((prev) => ({
      ...prev,
      hrctFile: {
        name: file.name,
        size: file.size,
        format: formatLabel,
        isZipSeries: isZip,
        previewUrl,
        rawFile: file,
        pattern: lower.includes('normal') || lower.includes('non_ipf') ? 'non_ipf' : 'uip',
      },
    }));
  };

  const handleLoadSample = (indexOverride) => {
    const idx = typeof indexOverride === 'number' ? indexOverride : selectedSampleIdx;
    const sample = SAMPLE_PATIENT_INPUTS[idx] || SAMPLE_PATIENT_INPUTS[0];
    setFormData({ ...sample.data });
    setErrors({});
    setFileError('');
    if (typeof indexOverride !== 'number') {
      setSelectedSampleIdx((prev) => (prev + 1) % SAMPLE_PATIENT_INPUTS.length);
    }
  };

  const handleReset = () => {
    setFormData({ ...INITIAL_FORM });
    setErrors({});
    setFileError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const validationErrors = validateAll(formData);
    setErrors(validationErrors);
    if (Object.keys(validationErrors).length === 0) {
      onSubmitPrediction(formData);
    }
  };

  const heightNum = Number(formData.height);
  const weightNum = Number(formData.weight);
  const bmi =
    heightNum >= 1.0 && heightNum <= 2.5 && weightNum >= 25 && weightNum <= 260
      ? (weightNum / (heightNum * heightNum)).toFixed(1)
      : null;

  const inputBaseClass = (fieldName) =>
    `w-full rounded-lg border px-3.5 py-2.5 text-sm font-medium transition focus:outline-none focus:ring-2 ${
      errors[fieldName]
        ? 'border-rose-400 bg-rose-50/50 text-slate-900 focus:border-rose-500 focus:ring-rose-500/20 dark:border-rose-500/80 dark:bg-rose-950/20 dark:text-slate-100'
        : 'border-slate-200 bg-white text-slate-900 hover:border-medical-300 focus:border-medical-500 focus:ring-medical-500/20 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:hover:border-slate-600'
    }`;

  return (
    <div className="space-y-6">
      <div className="rounded-2xl bg-gradient-to-r from-medical-50 via-white to-sky-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800/90 border border-medical-100 dark:border-slate-800 p-5 shadow-clinical">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div className="space-y-1">
            <div className="inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-medical-600 dark:text-sky-400">
              <FlaskConical className="w-4 h-4" />
              <span>Page 1 • Real Zenodo ILD Clinical Assessment Input</span>
            </div>
            <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white">
              Patient Diagnostic Input Form
            </h1>
            <p className="text-sm text-slate-600 dark:text-slate-400 max-w-2xl">
              Enter patient demographics (Height in meters, Weight in kg) and pulmonary function test
              (PFT) values (FEV1/FVC decimal ratio and % predicted values).
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 shrink-0">
            <button
              type="button"
              onClick={() => handleLoadSample()}
              className="inline-flex items-center gap-2 rounded-xl bg-medical-50 hover:bg-medical-100 dark:bg-sky-950/60 dark:hover:bg-sky-900/70 text-medical-700 dark:text-sky-300 border border-medical-200 dark:border-sky-800/70 px-4 py-2.5 text-sm font-semibold transition shadow-sm"
            >
              <Sparkles className="w-4 h-4 text-medical-600 dark:text-sky-400" />
              <span>Load sample patient</span>
            </button>

            <button
              type="button"
              onClick={handleReset}
              className="inline-flex items-center gap-1.5 rounded-xl bg-white hover:bg-slate-100 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700 px-3.5 py-2.5 text-sm font-medium transition"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset</span>
            </button>
          </div>
        </div>

        <div className="mt-4 pt-3.5 border-t border-medical-100/80 dark:border-slate-800 flex flex-wrap items-center gap-2 text-xs">
          <span className="text-slate-500 dark:text-slate-400 font-medium mr-1">
            Real Zenodo Presets:
          </span>
          {SAMPLE_PATIENT_INPUTS.map((sample, idx) => (
            <button
              key={sample.id}
              type="button"
              onClick={() => {
                setSelectedSampleIdx(idx);
                handleLoadSample(idx);
              }}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white dark:bg-slate-800/90 hover:bg-medical-50 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700 transition"
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  idx === 0 ? 'bg-rose-500' : idx === 1 ? 'bg-emerald-500' : 'bg-amber-500'
                }`}
              />
              <span>{sample.label}</span>
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} noValidate className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-6 shadow-clinical space-y-5">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-4">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-xl bg-medical-50 dark:bg-sky-950/60 text-medical-600 dark:text-sky-400">
                  <UserRound className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-slate-900 dark:text-white">
                    Demographics & Anthropometrics
                  </h2>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Explicit units: Height (m), Weight (kg)
                  </p>
                </div>
              </div>
              {bmi && (
                <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-mono font-semibold bg-medical-50 text-medical-700 dark:bg-sky-950/80 dark:text-sky-300 border border-medical-200 dark:border-sky-800">
                  BMI: {bmi} kg/m²
                </span>
              )}
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wide text-slate-700 dark:text-slate-300 mb-2">
                Biological Sex <span className="text-rose-500">*</span>
              </label>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { value: 'M', label: 'Male (M)' },
                  { value: 'F', label: 'Female (F)' },
                ].map((option) => {
                  const active = formData.sex === option.value;
                  return (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => handleFieldChange('sex', option.value)}
                      className={`flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl border text-sm font-semibold transition ${
                        active
                          ? 'bg-medical-600 text-white border-medical-600 shadow-sm'
                          : errors.sex
                          ? 'border-rose-400 bg-rose-50/40 text-slate-700 dark:border-rose-500/70 dark:bg-rose-950/20 dark:text-slate-200'
                          : 'bg-slate-50 hover:bg-medical-50/50 text-slate-700 border-slate-200 dark:bg-slate-800/60 dark:hover:bg-slate-800 dark:text-slate-300 dark:border-slate-700'
                      }`}
                    >
                      <span>{option.label}</span>
                    </button>
                  );
                })}
              </div>
              {errors.sex && (
                <p className="mt-1.5 flex items-center gap-1.5 text-xs font-medium text-rose-600 dark:text-rose-400">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  <span>{errors.sex}</span>
                </p>
              )}
            </div>

            <div>
              <div className="flex justify-between items-baseline mb-1.5">
                <label
                  htmlFor="input-age"
                  className="text-xs font-semibold uppercase tracking-wide text-slate-700 dark:text-slate-300"
                >
                  Age (years) <span className="text-rose-500">*</span>
                </label>
                <span className="text-[11px] font-mono text-slate-400">Range: 18–110 yrs</span>
              </div>
              <input
                id="input-age"
                type="number"
                min="18"
                max="110"
                step="1"
                placeholder="e.g., 66"
                value={formData.age}
                onChange={(e) => handleFieldChange('age', e.target.value)}
                onBlur={() => handleFieldBlur('age')}
                className={inputBaseClass('age')}
              />
              {errors.age && (
                <p className="mt-1.5 flex items-center gap-1.5 text-xs font-medium text-rose-600 dark:text-rose-400">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  <span>{errors.age}</span>
                </p>
              )}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <div className="flex justify-between items-baseline mb-1.5">
                  <label
                    htmlFor="input-height"
                    className="text-xs font-semibold uppercase tracking-wide text-slate-700 dark:text-slate-300"
                  >
                    Height (m) <span className="text-rose-500">*</span>
                  </label>
                  <span className="text-[11px] font-mono text-slate-400">1.00–2.50 m</span>
                </div>
                <input
                  id="input-height"
                  type="number"
                  min="1.00"
                  max="2.50"
                  step="0.01"
                  placeholder="e.g., 1.71"
                  value={formData.height}
                  onChange={(e) => handleFieldChange('height', e.target.value)}
                  onBlur={() => handleFieldBlur('height')}
                  className={inputBaseClass('height')}
                />
                {errors.height && (
                  <p className="mt-1.5 flex items-center gap-1.5 text-xs font-medium text-rose-600 dark:text-rose-400">
                    <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                    <span>{errors.height}</span>
                  </p>
                )}
              </div>

              <div>
                <div className="flex justify-between items-baseline mb-1.5">
                  <label
                    htmlFor="input-weight"
                    className="text-xs font-semibold uppercase tracking-wide text-slate-700 dark:text-slate-300"
                  >
                    Weight (kg) <span className="text-rose-500">*</span>
                  </label>
                  <span className="text-[11px] font-mono text-slate-400">25–260 kg</span>
                </div>
                <input
                  id="input-weight"
                  type="number"
                  min="25"
                  max="260"
                  step="0.1"
                  placeholder="e.g., 90.0"
                  value={formData.weight}
                  onChange={(e) => handleFieldChange('weight', e.target.value)}
                  onBlur={() => handleFieldBlur('weight')}
                  className={inputBaseClass('weight')}
                />
                {errors.weight && (
                  <p className="mt-1.5 flex items-center gap-1.5 text-xs font-medium text-rose-600 dark:text-rose-400">
                    <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                    <span>{errors.weight}</span>
                  </p>
                )}
              </div>
            </div>

            <div className="rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200/70 dark:border-slate-700/70 p-3.5 text-xs text-slate-600 dark:text-slate-400 flex items-start gap-2.5">
              <Info className="w-4 h-4 text-medical-600 dark:text-sky-400 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-slate-800 dark:text-slate-200">
                  Explicit Unit Contract:
                </span>{' '}
                Height must be entered in meters (e.g., <code className="font-mono">1.72</code>) and
                FEV1/FVC as a decimal ratio (e.g., <code className="font-mono">0.83</code>), matching
                the real Zenodo ILD dataset without heuristic scaling.
              </div>
            </div>
          </div>

          <div className="lg:col-span-7 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-6 shadow-clinical space-y-5">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-4">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-xl bg-medical-50 dark:bg-sky-950/60 text-medical-600 dark:text-sky-400">
                  <Activity className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-slate-900 dark:text-white">
                    Lung Function Values (Spirometry, Volumes & Diffusion)
                  </h2>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Decimal FEV1/FVC ratio (0.20–1.15) & % predicted values (10–250%)
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              <div className="sm:col-span-2">
                <div className="flex justify-between items-baseline mb-1.5">
                  <label
                    htmlFor="input-fev1Fvc"
                    className="text-xs font-semibold uppercase tracking-wide text-slate-700 dark:text-slate-300"
                  >
                    FEV1/FVC Decimal Ratio <span className="text-rose-500">*</span>
                  </label>
                  <span className="text-[11px] font-mono text-slate-400">
                    Decimal Ratio: 0.20–1.15 (e.g., 0.83)
                  </span>
                </div>
                <input
                  id="input-fev1Fvc"
                  type="number"
                  min="0.20"
                  max="1.15"
                  step="0.01"
                  placeholder="e.g., 0.83"
                  value={formData.fev1Fvc}
                  onChange={(e) => handleFieldChange('fev1Fvc', e.target.value)}
                  onBlur={() => handleFieldBlur('fev1Fvc')}
                  className={inputBaseClass('fev1Fvc')}
                />
                {errors.fev1Fvc && (
                  <p className="mt-1.5 flex items-center gap-1.5 text-xs font-medium text-rose-600 dark:text-rose-400">
                    <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                    <span>{errors.fev1Fvc}</span>
                  </p>
                )}
              </div>

              <div>
                <div className="flex justify-between items-baseline mb-1.5">
                  <label
                    htmlFor="input-fev1Pred"
                    className="text-xs font-semibold uppercase tracking-wide text-slate-700 dark:text-slate-300"
                  >
                    FEV1 % Predicted <span className="text-rose-500">*</span>
                  </label>
                  <span className="text-[11px] font-mono text-slate-400">10–250%</span>
                </div>
                <div className="relative">
                  <input
                    id="input-fev1Pred"
                    type="number"
                    step="0.1"
                    placeholder="e.g., 65.6"
                    value={formData.fev1Pred}
                    onChange={(e) => handleFieldChange('fev1Pred', e.target.value)}
                    onBlur={() => handleFieldBlur('fev1Pred')}
                    className={inputBaseClass('fev1Pred')}
                  />
                  <span className="absolute right-3.5 top-2.5 text-xs font-mono text-slate-400 pointer-events-none">
                    %
                  </span>
                </div>
                {errors.fev1Pred && (
                  <p className="mt-1.5 flex items-center gap-1.5 text-xs font-medium text-rose-600 dark:text-rose-400">
                    <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                    <span>{errors.fev1Pred}</span>
                  </p>
                )}
              </div>

              <div>
                <div className="flex justify-between items-baseline mb-1.5">
                  <label
                    htmlFor="input-fvcPred"
                    className="text-xs font-semibold uppercase tracking-wide text-slate-700 dark:text-slate-300"
                  >
                    FVC % Predicted <span className="text-rose-500">*</span>
                  </label>
                  <span className="text-[11px] font-mono text-slate-400">10–250%</span>
                </div>
                <div className="relative">
                  <input
                    id="input-fvcPred"
                    type="number"
                    step="0.1"
                    placeholder="e.g., 60.5"
                    value={formData.fvcPred}
                    onChange={(e) => handleFieldChange('fvcPred', e.target.value)}
                    onBlur={() => handleFieldBlur('fvcPred')}
                    className={inputBaseClass('fvcPred')}
                  />
                  <span className="absolute right-3.5 top-2.5 text-xs font-mono text-slate-400 pointer-events-none">
                    %
                  </span>
                </div>
                {errors.fvcPred && (
                  <p className="mt-1.5 flex items-center gap-1.5 text-xs font-medium text-rose-600 dark:text-rose-400">
                    <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                    <span>{errors.fvcPred}</span>
                  </p>
                )}
              </div>

              <div>
                <div className="flex justify-between items-baseline mb-1.5">
                  <label
                    htmlFor="input-tlcPred"
                    className="text-xs font-semibold uppercase tracking-wide text-slate-700 dark:text-slate-300"
                  >
                    TLC % Predicted <span className="text-rose-500">*</span>
                  </label>
                  <span className="text-[11px] font-mono text-slate-400">10–250%</span>
                </div>
                <div className="relative">
                  <input
                    id="input-tlcPred"
                    type="number"
                    step="0.1"
                    placeholder="e.g., 64.2"
                    value={formData.tlcPred}
                    onChange={(e) => handleFieldChange('tlcPred', e.target.value)}
                    onBlur={() => handleFieldBlur('tlcPred')}
                    className={inputBaseClass('tlcPred')}
                  />
                  <span className="absolute right-3.5 top-2.5 text-xs font-mono text-slate-400 pointer-events-none">
                    %
                  </span>
                </div>
                {errors.tlcPred && (
                  <p className="mt-1.5 flex items-center gap-1.5 text-xs font-medium text-rose-600 dark:text-rose-400">
                    <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                    <span>{errors.tlcPred}</span>
                  </p>
                )}
              </div>

              <div>
                <div className="flex justify-between items-baseline mb-1.5">
                  <label
                    htmlFor="input-tlcoPred"
                    className="text-xs font-semibold uppercase tracking-wide text-slate-700 dark:text-slate-300"
                  >
                    TLCO % Predicted <span className="text-rose-500">*</span>
                  </label>
                  <span className="text-[11px] font-mono text-slate-400">10–250%</span>
                </div>
                <div className="relative">
                  <input
                    id="input-tlcoPred"
                    type="number"
                    step="0.1"
                    placeholder="e.g., 40.1"
                    value={formData.tlcoPred}
                    onChange={(e) => handleFieldChange('tlcoPred', e.target.value)}
                    onBlur={() => handleFieldBlur('tlcoPred')}
                    className={inputBaseClass('tlcoPred')}
                  />
                  <span className="absolute right-3.5 top-2.5 text-xs font-mono text-slate-400 pointer-events-none">
                    %
                  </span>
                </div>
                {errors.tlcoPred && (
                  <p className="mt-1.5 flex items-center gap-1.5 text-xs font-medium text-rose-600 dark:text-rose-400">
                    <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                    <span>{errors.tlcoPred}</span>
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Optional HRCT Scan Upload Card */}
        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-6 shadow-clinical">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 pb-4 border-b border-slate-100 dark:border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-medical-50 dark:bg-sky-950/60 text-medical-600 dark:text-sky-400">
                <UploadCloud className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-base font-bold text-slate-900 dark:text-white">
                    High-Resolution CT (HRCT) Series Upload
                  </h2>
                  <span className="px-2 py-0.5 text-[11px] font-semibold rounded-full bg-amber-100 dark:bg-amber-950/70 text-amber-800 dark:text-amber-300 border border-amber-200 dark:border-amber-800">
                    Optional • Enables Exploratory CT Progression
                  </span>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  The validated Zenodo IPF diagnostic classifier uses clinical PFTs only (torch-free).
                  Upload a full CT DICOM series (<code className="font-mono">.zip</code> of axial{' '}
                  <code className="font-mono">.dcm</code> slices, max 50 MB) to compute the exploratory
                  HRCT and Fusion FVC-decline progression scores on Page 2.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-4">
            <label
              htmlFor="hrct-upload"
              className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-medical-200 dark:border-slate-700 bg-medical-50/40 dark:bg-slate-800/40 hover:bg-medical-50/80 dark:hover:bg-slate-800/70 p-6 text-center cursor-pointer transition"
            >
              <UploadCloud className="w-8 h-8 text-medical-600 dark:text-sky-400 mb-2" />
              <span className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                Click to select a CT DICOM series (.zip) or preview file
              </span>
              <span className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                Full CT series (enables HRCT &amp; Fusion cards):{' '}
                <span className="font-mono font-semibold text-medical-700 dark:text-sky-300">.zip (max 50 MB)</span>{' '}
                • Preview only: <span className="font-mono">.dcm, .nii, .png, .jpg</span>
              </span>
              <input
                ref={fileInputRef}
                id="hrct-upload"
                type="file"
                accept=".zip,.dcm,.dicom,.nii,.nii.gz,.png,.jpg,.jpeg,application/zip,image/png,image/jpeg"
                onChange={handleFileUpload}
                className="sr-only"
              />
            </label>

            {fileError && (
              <p className="mt-2 flex items-center gap-1.5 text-xs font-medium text-rose-600 dark:text-rose-400">
                <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                <span>{fileError}</span>
              </p>
            )}

            <HrctPreview
              hrctFile={formData.hrctFile}
              onRemove={() => {
                setFormData((prev) => ({ ...prev, hrctFile: null }));
                if (fileInputRef.current) fileInputRef.current.value = '';
              }}
            />
          </div>
        </div>

        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200/90 dark:border-slate-800 p-4 sm:px-6 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 shadow-clinical">
          <div className="flex flex-wrap items-center gap-2.5">
            <button
              type="button"
              onClick={() => handleLoadSample()}
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-medical-50 hover:bg-medical-100 dark:bg-slate-800 dark:hover:bg-slate-700 text-medical-700 dark:text-sky-300 border border-medical-200 dark:border-slate-700 px-4 py-2.5 text-sm font-semibold transition"
            >
              <Sparkles className="w-4 h-4" />
              <span>Load sample patient</span>
            </button>

            <button
              type="button"
              onClick={handleReset}
              className="inline-flex items-center justify-center gap-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 px-4 py-2.5 text-sm font-semibold transition"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reset</span>
            </button>
          </div>

          <button
            type="submit"
            disabled={isPredicting}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-medical-600 hover:bg-medical-700 disabled:opacity-60 text-white px-7 py-3 text-sm font-bold shadow-md shadow-medical-600/20 transition"
          >
            {isPredicting ? (
              <>
                <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Running Diagnostic Model...</span>
              </>
            ) : (
              <>
                <CheckCircle2 className="w-4 h-4" />
                <span>Predict</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
