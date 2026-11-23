import React, { useState } from 'react';
import { Layers, Trash2, FileCheck2, ZoomIn, ZoomOut, SlidersHorizontal } from 'lucide-react';

export default function HrctPreview({ hrctFile, onRemove }) {
  const [zoomed, setZoomed] = useState(false);
  const [sliceOffset, setSliceOffset] = useState(0);

  if (!hrctFile) return null;

  const isRasterImage = Boolean(hrctFile.previewUrl);
  const pattern = hrctFile.pattern || 'uip';

  const formatBytes = (bytes) => {
    if (!bytes) return '1.5 MB';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(2)} MB`;
  };

  return (
    <div className="mt-4 rounded-xl border border-medical-200 dark:border-slate-700 bg-slate-900 text-slate-100 overflow-hidden shadow-clinical-md">
      <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2.5 bg-slate-950/90 border-b border-slate-800 text-xs">
        <div className="flex items-center gap-2 min-w-0">
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-medical-500/20 text-medical-300 font-mono font-medium border border-medical-500/30">
            <Layers className="w-3.5 h-3.5" />
            {hrctFile.format || 'HRCT'}
          </span>
          <span className="font-medium text-slate-200 truncate" title={hrctFile.name}>
            {hrctFile.name}
          </span>
          <span className="text-slate-400 font-mono shrink-0">({formatBytes(hrctFile.size)})</span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setZoomed((z) => !z)}
            className="inline-flex items-center gap-1 px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 transition"
          >
            {zoomed ? <ZoomOut className="w-3.5 h-3.5" /> : <ZoomIn className="w-3.5 h-3.5" />}
            <span>{zoomed ? '100%' : '140%'}</span>
          </button>
          {onRemove && (
            <button
              type="button"
              onClick={onRemove}
              className="inline-flex items-center gap-1 px-2 py-1 rounded bg-rose-950/80 hover:bg-rose-900 text-rose-200 border border-rose-800/60 transition"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Remove</span>
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-12 items-center bg-slate-950">
        <div className="sm:col-span-7 relative flex items-center justify-center p-4 min-h-[210px] overflow-hidden bg-black/90 border-b sm:border-b-0 sm:border-r border-slate-800">
          {isRasterImage ? (
            <img
              src={hrctFile.previewUrl}
              alt="Uploaded HRCT Scan Preview"
              className={`max-h-52 w-auto object-contain rounded transition-transform duration-300 ${
                zoomed ? 'scale-125' : 'scale-100'
              }`}
            />
          ) : (
            <svg
              viewBox="0 0 260 210"
              className={`w-full max-w-[250px] h-auto select-none transition-transform duration-300 ${
                zoomed ? 'scale-125' : 'scale-100'
              }`}
              role="img"
              aria-label="Axial HRCT Lung Window Preview"
            >
              <defs>
                <radialGradient id="thoraxWall" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#1e293b" />
                  <stop offset="78%" stopColor="#334155" />
                  <stop offset="94%" stopColor="#475569" />
                  <stop offset="100%" stopColor="#0f172a" />
                </radialGradient>
                <radialGradient id="lungField" cx="50%" cy="50%" r="55%">
                  <stop offset="0%" stopColor="#090d16" />
                  <stop offset="75%" stopColor="#0f172a" />
                  <stop offset="100%" stopColor="#1e293b" />
                </radialGradient>
                <pattern id="honeycomb" width="8" height="8" patternUnits="userSpaceOnUse">
                  <circle cx="4" cy="4" r="2.6" fill="#050811" stroke="#94a3b8" strokeWidth="0.65" strokeOpacity="0.65" />
                </pattern>
              </defs>
              <ellipse cx="130" cy="108" rx="106" ry="82" fill="url(#thoraxWall)" stroke="#64748b" strokeWidth="1.2" />
              <ellipse cx="130" cy="108" rx="96" ry="73" fill="#0f172a" stroke="#cbd5e1" strokeWidth="1.5" strokeDasharray="10 6" strokeOpacity="0.55" />
              <path
                d={`M 112 ${52 + sliceOffset} C 74 48, 44 74, 46 116 C 48 152, 78 166, 110 156 C 118 153, 116 122, 114 98 Z`}
                fill="url(#lungField)"
                stroke="#475569"
                strokeWidth="1"
              />
              <path
                d={`M 148 ${54 + sliceOffset} C 184 50, 214 76, 212 116 C 210 152, 182 166, 150 156 C 142 153, 144 124, 146 98 Z`}
                fill="url(#lungField)"
                stroke="#475569"
                strokeWidth="1"
              />
              <ellipse cx="133" cy="102" rx="22" ry="28" fill="#475569" fillOpacity="0.75" />
              <circle cx="130" cy="158" r="13" fill="#e2e8f0" fillOpacity="0.85" />
              <circle cx="130" cy="158" r="5" fill="#1e293b" />
              <circle cx="123" cy="96" r="4.5" fill="#020617" stroke="#94a3b8" strokeWidth="1" />
              <circle cx="137" cy="96" r="4.5" fill="#020617" stroke="#94a3b8" strokeWidth="1" />
              {pattern !== 'non_ipf' ? (
                <g>
                  <path
                    d="M 48 118 C 50 148, 76 164, 106 155 L 102 142 C 78 148, 60 138, 56 116 Z"
                    fill="url(#honeycomb)"
                    opacity="0.9"
                  />
                  <path
                    d="M 210 118 C 208 148, 184 164, 154 155 L 158 142 C 182 148, 200 138, 204 116 Z"
                    fill="url(#honeycomb)"
                    opacity="0.9"
                  />
                </g>
              ) : (
                <g opacity="0.45">
                  <ellipse cx="82" cy="122" rx="18" ry="16" fill="#94a3b8" />
                  <ellipse cx="178" cy="122" rx="18" ry="16" fill="#94a3b8" />
                </g>
              )}
              <text x="8" y="16" fill="#38bdf8" fontSize="8" fontFamily="monospace">
                AXIAL LUNG WINDOW
              </text>
              <text x="8" y="26" fill="#94a3b8" fontSize="7" fontFamily="monospace">
                W:1500 L:-600 • SL:{24 + sliceOffset}/64
              </text>
            </svg>
          )}
        </div>

        <div className="sm:col-span-5 p-4 space-y-3 text-xs bg-slate-900/95">
          <div className="flex items-center gap-1.5 text-emerald-400 font-medium">
            <FileCheck2 className="w-4 h-4 shrink-0" />
            <span>
              {hrctFile.isZipSeries
                ? 'CT DICOM Series (.zip) Ready for HRCT/Fusion Scoring'
                : 'Preview Attached (Upload .zip DICOM series for CT scoring)'}
            </span>
          </div>
          <div className="space-y-1.5 text-slate-300">
            <div className="flex justify-between border-b border-slate-800 pb-1">
              <span className="text-slate-400">Modality:</span>
              <span className="font-mono text-slate-200">HRCT Chest (Axial)</span>
            </div>
            <div className="flex justify-between border-b border-slate-800 pb-1">
              <span className="text-slate-400">Windowing:</span>
              <span className="font-mono text-slate-200">W:1500 / L:-600 HU</span>
            </div>
            <div className="flex justify-between border-b border-slate-800 pb-1">
              <span className="text-slate-400">Cohort Note:</span>
              <span className="font-medium text-sky-300">Separate OSIC Progression</span>
            </div>
          </div>

          {!isRasterImage && (
            <div className="pt-1">
              <label className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                <span className="inline-flex items-center gap-1">
                  <SlidersHorizontal className="w-3 h-3 text-sky-400" />
                  Axial Slice Scrubber
                </span>
                <span className="font-mono text-sky-300">Slice {24 + sliceOffset}/64</span>
              </label>
              <input
                type="range"
                min="-4"
                max="4"
                step="1"
                value={sliceOffset}
                onChange={(e) => setSliceOffset(Number(e.target.value))}
                className="w-full accent-sky-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
