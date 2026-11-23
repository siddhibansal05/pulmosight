/**
 * Self-contained 2-Page Clinical PDF Report Generator (`frontend/src/utils/pdfGenerator.js`).
 *
 * Page 1: Calibrated IPF Prediction Summary, Patient Demographics & PFTs, Top 5 SHAP Contributions.
 * Page 2: 5-Part "Interpretation of this result" (Physiology, Model result, What pushed it,
 *         Caution, For clinician consideration), Sensitivity Disclosure, & Mandatory Notice.
 *
 * Refuses to export a clinical PDF if `result.isOfflineDemo` is true.
 */

const AUTOMATED_NOTICE =
  'Automated interpretation, decision support only. Verify clinically.';

function normalizeAsciiForPdf(str) {
  return String(str || '')
    .replace(/\u2265/g, '>=')
    .replace(/\u2264/g, '<=')
    .replace(/\u2212|\u2013|\u2014/g, '-')
    .replace(/\u00b1/g, '+/-')
    .replace(/\u2018|\u2019/g, "'")
    .replace(/\u201c|\u201d/g, '"')
    .replace(/\u2022/g, '*')
    .replace(/\u00a0/g, ' ');
}

function escapePdfText(str) {
  return normalizeAsciiForPdf(str)
    .replace(/\\/g, '\\\\')
    .replace(/\(/g, '\\(')
    .replace(/\)/g, '\\)')
    .replace(/[^\x20-\x7E]/g, ' ');
}

function wrapPdfText(text, maxCharsPerLine = 88) {
  const words = normalizeAsciiForPdf(text)
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .filter(Boolean);
  if (words.length === 0) return [];

  const lines = [];
  let currentLine = '';

  const pushWord = (word) => {
    if (word.length <= maxCharsPerLine) {
      if (!currentLine) {
        currentLine = word;
      } else if (currentLine.length + 1 + word.length <= maxCharsPerLine) {
        currentLine = `${currentLine} ${word}`;
      } else {
        lines.push(currentLine);
        currentLine = word;
      }
      return;
    }

    if (currentLine) {
      lines.push(currentLine);
      currentLine = '';
    }
    for (let i = 0; i < word.length; i += maxCharsPerLine) {
      const chunk = word.slice(i, i + maxCharsPerLine);
      if (chunk.length === maxCharsPerLine) {
        lines.push(chunk);
      } else {
        currentLine = chunk;
      }
    }
  };

  words.forEach(pushWord);
  if (currentLine) lines.push(currentLine);
  return lines;
}

export function buildClinicalReportPdfString(result) {
  const {
    caseId: rawCaseId = 'CASE-REPORT',
    timestamp,
    prediction = 'IPF',
    probabilityPercent = 0,
    riskLevel = 'Medium',
    operatingMode = 'screening',
    threshold = 0.1868,
    screeningThreshold = 0.1868,
    confirmatoryThreshold = 0.3429,
    dataSource = 'real',
    modelVersion = '2.2.0',
    modelUsed = 'Calibrated Classifier',
    sensitivityDisclosure = '',
    interpretation = null,
    interpretationText = '',
    interpretationSource = 'deterministic',
    patientSummary = {},
    featureImportance = [],
    disclaimer = 'Decision support only, not a clinical diagnosis.',
  } = result;

  const caseId = String(rawCaseId || 'CASE-REPORT').replace(/^IPF-/i, 'CASE-');
  const noticeText = interpretation?.notice || AUTOMATED_NOTICE;

  const formattedDate = timestamp
    ? new Date(timestamp).toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      })
    : new Date().toLocaleString();

  // ======================== PAGE 1 ========================
  const p1 = [];

  p1.push('0.047 0.557 0.902 rg');
  p1.push('0 765 595.28 77 re f');

  p1.push('1 1 1 rg');
  p1.push('BT /F2 15.5 Tf 36 810 Td (PULMOSIGHT CDSS - IPF DIAGNOSTIC SUPPORT REPORT) Tj ET');
  p1.push(
    `BT /F1 9.2 Tf 36 792 Td (Model: ${escapePdfText(modelUsed)} v${escapePdfText(
      modelVersion
    )}  |  Data Source: ${escapePdfText(dataSource.toUpperCase())}  |  Active Threshold: ${escapePdfText(
      threshold
    )}) Tj ET`
  );
  p1.push(
    `BT /F1 8.8 Tf 36 777 Td (Case ID: ${escapePdfText(caseId)}   |   Generated: ${escapePdfText(
      formattedDate
    )}   |   Page 1 of 2) Tj ET`
  );

  // Prediction Summary Card
  p1.push('0.945 0.969 0.996 rg');
  p1.push('36 665 523 82 re f');
  p1.push('0.729 0.878 0.992 RG 1 w');
  p1.push('36 665 523 82 re S');

  p1.push('0.059 0.090 0.165 rg');
  p1.push('BT /F2 11 Tf 50 728 Td (DIAGNOSTIC PREDICTION SUMMARY) Tj ET');

  if (prediction === 'IPF') {
    p1.push('0.725 0.110 0.110 rg');
  } else {
    p1.push('0.020 0.478 0.333 rg');
  }
  p1.push(`BT /F2 20 Tf 50 698 Td (Prediction: ${escapePdfText(prediction)}) Tj ET`);

  p1.push('0.059 0.090 0.165 rg');
  p1.push(
    `BT /F2 12 Tf 255 704 Td (IPF Probability: ${escapePdfText(probabilityPercent)}%) Tj ET`
  );
  p1.push(
    `BT /F2 12 Tf 415 704 Td (Risk Level: ${escapePdfText(riskLevel.toUpperCase())}) Tj ET`
  );

  p1.push('0.278 0.333 0.412 rg');
  p1.push(
    `BT /F1 9 Tf 50 676 Td (GAP Index Estimate: ${escapePdfText(
      patientSummary.gapStage || 'Stage II'
    )} (${escapePdfText(patientSummary.gapPoints ?? 4)} pts)  |  Screening Thresh: ${escapePdfText(
      screeningThreshold
    )}  |  Confirmatory Thresh: ${escapePdfText(confirmatoryThreshold)}) Tj ET`
  );

  // Section 1: Demographics & PFTs
  p1.push('0.059 0.090 0.165 rg');
  p1.push('BT /F2 11.5 Tf 36 640 Td (1. PATIENT DEMOGRAPHICS & PULMONARY FUNCTION VALUES) Tj ET');
  p1.push('0.82 0.86 0.91 RG 0.75 w 36 633 m 559 633 l S');

  const pftRows = [
    [
      `Biological Sex: ${patientSummary.sex === 'M' ? 'Male (M)' : 'Female (F)'}`,
      `FEV1/FVC Decimal Ratio: ${patientSummary.fev1Fvc ?? '-'}`,
    ],
    [
      `Age: ${patientSummary.age ?? '-'} years`,
      `FEV1 % Predicted: ${patientSummary.fev1Pred ?? '-'}%`,
    ],
    [
      `Height: ${patientSummary.height ?? '-'} m`,
      `FVC % Predicted: ${patientSummary.fvcPred ?? '-'}%`,
    ],
    [
      `Weight: ${patientSummary.weight ?? '-'} kg`,
      `TLC % Predicted: ${patientSummary.tlcPred ?? '-'}%`,
    ],
    [
      `BMI: ${patientSummary.bmi ?? '-'} kg/m2`,
      `TLCO % Predicted: ${patientSummary.tlcoPred ?? '-'}%`,
    ],
  ];

  let yPos = 614;
  pftRows.forEach((row, idx) => {
    if (idx % 2 === 0) {
      p1.push('0.973 0.980 0.988 rg');
      p1.push(`36 ${yPos - 6} 523 20 re f`);
    }
    p1.push('0.12 0.16 0.23 rg');
    p1.push(`BT /F1 10 Tf 46 ${yPos} Td (${escapePdfText(row[0])}) Tj ET`);
    p1.push(`BT /F2 10 Tf 310 ${yPos} Td (${escapePdfText(row[1])}) Tj ET`);
    yPos -= 22;
  });

  const hrctFilename = patientSummary.hrctFile?.name || 'None';
  const hrctLines = wrapPdfText(
    `HRCT uploaded: ${hrctFilename} (not used in this clinical IPF prediction)`,
    88
  );
  const hrctRowHeight = Math.max(20, 8 + hrctLines.length * 13);
  p1.push('0.973 0.980 0.988 rg');
  p1.push(`36 ${yPos - hrctRowHeight + 14} 523 ${hrctRowHeight} re f`);
  p1.push('0.12 0.16 0.23 rg');
  let hrctLineY = yPos;
  hrctLines.forEach((ln) => {
    p1.push(`BT /F1 9.5 Tf 46 ${hrctLineY} Td (${escapePdfText(ln)}) Tj ET`);
    hrctLineY -= 13;
  });
  yPos -= hrctRowHeight + 18;

  // Section 2: Top 5 Feature Contributions
  p1.push('0.059 0.090 0.165 rg');
  p1.push(
    `BT /F2 11.5 Tf 36 ${yPos} Td (2. TOP 5 FEATURE INFLUENCES ON THIS SCORE (SHAP)) Tj ET`
  );
  p1.push(`0.82 0.86 0.91 RG 0.75 w 36 ${yPos - 7} m 559 ${yPos - 7} l S`);
  yPos -= 20;
  p1.push('0.35 0.42 0.50 rg');
  p1.push(
    `BT /F1 8.6 Tf 36 ${yPos} Td (Note: SHAP values show each feature's influence on this score.) Tj ET`
  );

  yPos -= 20;
  p1.push('0.28 0.33 0.41 rg');
  p1.push(`BT /F2 9 Tf 46 ${yPos} Td (FEATURE) Tj ET`);
  p1.push(`BT /F2 9 Tf 195 ${yPos} Td (PATIENT VALUE) Tj ET`);
  p1.push(`BT /F2 9 Tf 295 ${yPos} Td (SHAP IMPACT) Tj ET`);
  p1.push(`BT /F2 9 Tf 390 ${yPos} Td (DIRECTION) Tj ET`);

  yPos -= 18;
  featureImportance.forEach((item, index) => {
    if (index % 2 === 0) {
      p1.push('0.973 0.980 0.988 rg');
      p1.push(`36 ${yPos - 4} 523 18 re f`);
    }
    const sign = item.impact >= 0 ? '+' : '';
    p1.push('0.09 0.13 0.20 rg');
    p1.push(`BT /F2 9.4 Tf 46 ${yPos} Td (${escapePdfText(item.feature)}) Tj ET`);
    p1.push(`BT /F1 9.4 Tf 195 ${yPos} Td (${escapePdfText(item.value)}) Tj ET`);
    p1.push(
      `BT /F2 9.4 Tf 295 ${yPos} Td (${escapePdfText(`${sign}${item.impact.toFixed(3)}`)}) Tj ET`
    );
    p1.push(`BT /F1 9 Tf 390 ${yPos} Td (${escapePdfText(item.direction)}) Tj ET`);
    yPos -= 20;
  });

  // Page 1 footer note directing reader to Page 2
  p1.push('0.996 0.973 0.902 rg');
  p1.push('36 36 523 44 re f');
  p1.push('0.961 0.620 0.043 RG 0.9 w');
  p1.push('36 36 523 44 re S');
  p1.push('0.573 0.251 0.055 rg');
  p1.push(`BT /F2 9.2 Tf 48 63 Td (${escapePdfText(noticeText)}) Tj ET`);
  p1.push(
    'BT /F1 8.5 Tf 48 47 Td (See Page 2 for the complete five-part Interpretation of This Result and per-group error caution.) Tj ET'
  );

  // ======================== PAGE 2 ========================
  const p2 = [];

  p2.push('0.047 0.557 0.902 rg');
  p2.push('0 785 595.28 57 re f');
  p2.push('1 1 1 rg');
  p2.push('BT /F2 14 Tf 36 816 Td (PULMOSIGHT CDSS - INTERPRETATION OF THIS RESULT) Tj ET');
  p2.push(
    `BT /F1 8.8 Tf 36 798 Td (Case ID: ${escapePdfText(caseId)}  |  Model: ${escapePdfText(
      modelUsed
    )}  |  Generated: ${escapePdfText(formattedDate)}  |  Page 2 of 2) Tj ET`
  );

  let y2 = 762;
  p2.push('0.059 0.090 0.165 rg');
  p2.push(
    `BT /F2 11.5 Tf 36 ${y2} Td (3. INTERPRETATION OF THIS RESULT${
      interpretationSource === 'llm_rephrased' ? ' (FACT-VALIDATED REPHRASE)' : ''
    }) Tj ET`
  );
  p2.push(`0.82 0.86 0.91 RG 0.75 w 36 ${y2 - 7} m 559 ${y2 - 7} l S`);
  y2 -= 20;

  const drawSubsectionBox = (title, bodyLines, style = 'normal') => {
    const lineStep = 11.5;
    const boxHeight = 24 + bodyLines.length * lineStep;
    const boxBottom = y2 - boxHeight;

    if (style === 'caution') {
      p2.push('1.000 0.973 0.922 rg');
      p2.push(`36 ${boxBottom} 523 ${boxHeight} re f`);
      p2.push('0.910 0.659 0.255 RG 0.75 w');
      p2.push(`36 ${boxBottom} 523 ${boxHeight} re S`);
      p2.push('0.573 0.251 0.055 rg');
    } else {
      p2.push('0.968 0.980 0.996 rg');
      p2.push(`36 ${boxBottom} 523 ${boxHeight} re f`);
      p2.push('0.780 0.855 0.941 RG 0.7 w');
      p2.push(`36 ${boxBottom} 523 ${boxHeight} re S`);
      p2.push('0.047 0.290 0.431 rg');
    }

    p2.push(`BT /F2 9.4 Tf 46 ${y2 - 14} Td (${escapePdfText(title)}) Tj ET`);
    let lineY = y2 - 27;
    p2.push('0.12 0.16 0.23 rg');
    bodyLines.forEach((ln) => {
      p2.push(`BT /F1 8.8 Tf 46 ${lineY} Td (${escapePdfText(ln)}) Tj ET`);
      lineY -= lineStep;
    });
    y2 = boxBottom - 10;
  };

  if (interpretation) {
    drawSubsectionBox(
      '(a) PHYSIOLOGY',
      wrapPdfText(interpretation.physiology?.summary || 'Physiology summary unavailable.', 94)
    );
    drawSubsectionBox(
      '(b) MODEL RESULT & EMPIRICAL TEST-SET RELIABILITY',
      wrapPdfText(interpretation.model_result?.summary || 'Model result summary unavailable.', 94)
    );
    drawSubsectionBox(
      '(c) WHAT PUSHED THIS SCORE',
      wrapPdfText(interpretation.what_pushed_it?.summary || 'Feature driver summary unavailable.', 94)
    );
    drawSubsectionBox(
      '(d) CAUTION & FIBROSING ILD OVERLAP',
      wrapPdfText(interpretation.caution?.summary || 'Caution summary unavailable.', 94),
      'caution'
    );

    const cc = interpretation.clinician_considerations || {};
    const ccItems = Array.isArray(cc.items) ? cc.items : [];
    const ccLines = [];
    if (cc.guideline_reference) {
      ccLines.push(...wrapPdfText(`Reference: ${cc.guideline_reference}`, 94));
    }
    ccItems.forEach((item) => {
      ccLines.push(...wrapPdfText(`* ${item}`, 92));
    });
    drawSubsectionBox(
      '(e) FOR CLINICIAN CONSIDERATION (GUIDELINE CONTEXT - NOT DIAGNOSIS OR TREATMENT)',
      ccLines.length ? ccLines : wrapPdfText(cc.summary || '', 94)
    );
  } else {
    drawSubsectionBox(
      'INTERPRETATION SUMMARY',
      wrapPdfText(
        interpretationText ||
          'Structured interpretation was unavailable for this report. Automated interpretation, decision support only. Verify clinically.',
        94
      )
    );
  }

  // Bottom anchored boxes on Page 2 (zero overlap)
  const disclaimerFull = `${noticeText} ${disclaimer}`;
  const disclaimerLines = wrapPdfText(disclaimerFull, 82);
  const disclaimerLineStep = 12;
  const disclaimerBoxHeight = 28 + disclaimerLines.length * disclaimerLineStep;
  const disclaimerBoxY = 34;

  if (sensitivityDisclosure) {
    const disclosureLines = wrapPdfText(sensitivityDisclosure, 94);
    const disclosureLineStep = 10.5;
    const disclosureBoxHeight = 24 + disclosureLines.length * disclosureLineStep;
    const disclosureBoxY = disclaimerBoxY + disclaimerBoxHeight + 10;

    if (disclosureBoxY + disclosureBoxHeight < y2) {
      p2.push('0.945 0.969 0.996 rg');
      p2.push(`36 ${disclosureBoxY} 523 ${disclosureBoxHeight} re f`);
      p2.push('0.729 0.878 0.992 RG 0.8 w');
      p2.push(`36 ${disclosureBoxY} 523 ${disclosureBoxHeight} re S`);
      p2.push('0.059 0.090 0.165 rg');
      p2.push(
        `BT /F2 8.8 Tf 46 ${disclosureBoxY + disclosureBoxHeight - 13} Td (MODEL SENSITIVITY & PER-GROUP MISCLASSIFICATION DISCLOSURE (${escapePdfText(
          operatingMode.toUpperCase()
        )} MODE)) Tj ET`
      );
      let dLineY = disclosureBoxY + disclosureBoxHeight - 25;
      disclosureLines.forEach((ln) => {
        p2.push(`BT /F1 8.2 Tf 46 ${dLineY} Td (${escapePdfText(ln)}) Tj ET`);
        dLineY -= disclosureLineStep;
      });
    }
  }

  p2.push('0.996 0.973 0.902 rg');
  p2.push(`36 ${disclaimerBoxY} 523 ${disclaimerBoxHeight} re f`);
  p2.push('0.961 0.620 0.043 RG 1 w');
  p2.push(`36 ${disclaimerBoxY} 523 ${disclaimerBoxHeight} re S`);
  p2.push('0.573 0.251 0.055 rg');
  p2.push(
    `BT /F2 9.8 Tf 46 ${disclaimerBoxY + disclaimerBoxHeight - 16} Td (CLINICAL DECISION SUPPORT NOTICE) Tj ET`
  );
  let discLineY = disclaimerBoxY + disclaimerBoxHeight - 31;
  disclaimerLines.forEach((ln) => {
    p2.push(`BT /F2 9.2 Tf 46 ${discLineY} Td (${escapePdfText(ln)}) Tj ET`);
    discLineY -= disclaimerLineStep;
  });

  const stream1 = p1.join('\n');
  const stream2 = p2.join('\n');

  const objects = [
    '1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n',
    '2 0 obj\n<< /Type /Pages /Kids [3 0 R 7 0 R] /Count 2 >>\nendobj\n',
    '3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595.28 841.89] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>\nendobj\n',
    '4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n',
    '5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n',
    `6 0 obj\n<< /Length ${stream1.length} >>\nstream\n${stream1}\nendstream\nendobj\n`,
    '7 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595.28 841.89] /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 8 0 R >>\nendobj\n',
    `8 0 obj\n<< /Length ${stream2.length} >>\nstream\n${stream2}\nendstream\nendobj\n`,
  ];

  let pdf = '%PDF-1.4\n';
  const offsets = [0];

  objects.forEach((obj) => {
    offsets.push(pdf.length);
    pdf += obj;
  });

  const xrefStart = pdf.length;
  pdf += `xref\n0 ${objects.length + 1}\n`;
  pdf += '0000000000 65535 f \n';
  for (let i = 1; i <= objects.length; i++) {
    const offsetStr = String(offsets[i]).padStart(10, '0');
    pdf += `${offsetStr} 00000 n \n`;
  }

  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF`;
  return pdf;
}

export function downloadClinicalReportPdf(result) {
  if (!result || result.isOfflineDemo) return;

  const caseId = String(result.caseId || 'CASE-REPORT').replace(/^IPF-/i, 'CASE-');
  const pdf = buildClinicalReportPdfString(result);

  const blob = new Blob([pdf], { type: 'application/pdf' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${caseId}-Diagnostic-Report.pdf`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}
