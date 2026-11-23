# OSIC Pretrained CT Feature Extractor Comparison

> **Mandatory Cohort & Exploratory Notice:** *Progression results are exploratory, separate cohort, no IPF labels.*  
> **Strict Evidence Rule:** Every metric below is computed directly from cached feature files in `backend/data/osic/`. Missing feature files are shown as `pending: not evaluated yet`, and unrun extractors are shown as `not run` with their exact reason. No numbers are estimated, copied, or fabricated.  
> **Deployed Model Policy:** Read-only benchmark; never changes the deployed clinical IPF or OSIC progression model automatically. (`deployed_model_changed: false`).

---

## 1. Shared Evaluation Protocol (Identical Across All Extractors)

Only the embedding column prefix (`emb_prefix`) changes across extractors; every other modeling and evaluation step stays identical to `backend/src/osic_benchmarks.py` and `backend/src/imaging_progression.py`:

- **Patients & Visits:** `n = 175` OSIC CT patients, `1359` future visits (`Weeks > Baseline_Weeks`).
- **CV Splits:** `5-Repeat x 5-Fold Nested GroupKFold by Patient (_generate_patient_cv_splits, RANDOM_SEED=42)` (`25` identical outer folds).
- **Cryptographic Fold Identity Hash (`folds_sha256`):** `cd42bc04090b2952d83ed2936753952a6a789323c28058eafbfabbba93272bd4`.
- **Dimensionality Reduction:** In-fold `StandardScaler` + `PCA(n_components=15, random_state=42)` fitted strictly inside training folds.
- **Regularization & Winsorization:** `InFoldWinsorizer(0.01, 0.99) on HU/volume features`; Image+Tabular `Ridge` alpha grid `[100.0, 500.0, 1000.0]`; HRCT-only & Early Fusion `Ridge(alpha=500.0)`; Clinical-only & Stacking `Ridge(alpha=10.0)`; `Delta_FVC = Target_FVC - Baseline_FVC reconstructed and clipped to [500.0, 6500.0] mL`.
- **Metrics & Paired Testing:** `modified_laplace_log_likelihood (sigma >= 70 mL, |error| <= 1000 mL)`, FVC RMSE (mL), FVC MAE (mL), and patient-level rapid-decliner ROC-AUC with 95% CIs; `nadeau_bengio_paired_test (Nadeau & Bengio, Machine Learning, 2003) on identical 25 folds`.
- **Multiple-Comparison Correction:** `Holm-Bonferroni (step-down) with Bonferroni reference` across **`168`** executed paired tests (`72` vs. Current ResNet-18 + `96` vs. Clinical-only; family-wise `α = 0.05`, Bonferroni `α = 0.000298`).
- **Alternative-Extractor Holm Summary Flags:** `any_new_extractor_beats_resnet18_holm`: `False` (`evaluated across 3 completed alternative extractors`); `any_new_extractor_beats_clinical_only_holm`: `False` (`evaluated across 3 completed alternative extractors`).

**Overall Verdict:** Across completed alternative extractors (med3d, radimagenet, ctclip; 168 paired Nadeau-Bengio tests, Holm-Bonferroni α = 0.05), none achieved a statistically confirmed improvement over current ResNet-18. None achieved a Holm-corrected improvement over Clinical-only. Not run: resnet50_imagenet_control (Disabled by user toggle RUN_IMAGENET_RESNET50_CONTROL = False). The deployed model is not changed automatically. Exploratory.

---

## 2. Extractor Inventory, Licenses & Extraction Metadata

| Extractor | Status | Embedding Size | Extraction Time | Peak GPU Memory | Source & License | Redistribution Policy |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Current Baseline: ImageNet 2D ResNet-18** (`resnet_emb_*`) | `completed` | `512D` | `N/A (no JSON)` | `N/A (no JSON)` | torchvision.models.resnet18 (ResNet18_Weights.DEFAULT, ImageNet-1K) (**BSD-3-Clause (torchvision)**) | Standard torchvision pretrained weights; derived features cached locally. |
| **MedicalNet (Med3D) 3D ResNet-18** (`med3d_emb_*`) | `completed` | `512D` | `45.30 min` | `860.2 MB` | Hugging Face mirror TencentMedicalNet/MedicalNet-Resnet18 (resnet_18_23dataset.pth) (**MIT License**) | MIT-licensed mirror of Tencent/MedicalNet; only derived numeric embeddings cached. |
| **RadImageNet 2D ResNet-50** (`radimagenet_emb_*`) | `completed` | `2048D` | `39.32 min` | `763.7 MB` | Hugging Face mirror Lab-Rasool/RadImageNet (ResNet50.pt, official BMEII-AI/RadImageNet PyTorch weights) (**MIT License**) | MIT-licensed RadImageNet mirror; strict 318-key check enforced during extraction. |
| **CT-CLIP 3D CT-ViT Image Encoder** (`ctclip_emb_*`) | `completed` | `512D` | `8.24 min` | `2711.5 MB` | Gated Hugging Face dataset ibrahimhamamci/CT-RATE (models/CT-CLIP-Related/CT-CLIP_v2.pt) & GitHub ibrahimethemhamamci/CT-CLIP (**CC BY-NC-SA 4.0 (Gated, Non-Commercial — do NOT redistribute weights)**) | Gated, CC BY-NC-SA 4.0 non-commercial license; model weights are NEVER bundled or redistributed. |
| **Optional Control: ImageNet 2D ResNet-50** (`resnet50_imagenet_control_emb_*`) | `not run` (Disabled by user toggle RUN_IMAGENET_RESNET50_CONTROL = False) | `2048D` | — | — | torchvision.models.resnet50 (ResNet50_Weights.DEFAULT, ImageNet-1K) (**BSD-3-Clause (torchvision)**) | Optional architectural control isolating ResNet-50 capacity vs. RadImageNet pretraining. |

---

## 3. Extractor Performance Summary (5×5 Patient-Grouped Nested CV, Identical Folds)

### 3A. HRCT-Only Branch & Image + Tabular Benchmark

| Extractor | Status | Folds Hash Match | HRCT-Only Laplace LL (95% CI) | HRCT-Only Rapid-Decliner AUC (95% CI) | Image+Tabular Laplace LL (95% CI) | Image+Tabular RMSE mL (95% CI) | Image+Tabular MAE mL (95% CI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 (Current)** | `completed` | `PASS (cd42bc0409)` | `-6.8245 ± 0.0528 (95% CI: -6.8452–-6.8038)` | `0.4807 ± 0.0332 (95% CI: 0.4677–0.4938)` | `-6.8100 ± 0.0840 (95% CI: -6.8429–-6.7771)` | `230.786 ± 12.291 (95% CI: 225.968–235.604)` | `168.717 ± 12.802 (95% CI: 163.699–173.735)` |
| **Med3D (3D ResNet-18)** | `completed` | `PASS (cd42bc0409)` | `-6.9036 ± 0.1008 (95% CI: -6.9432–-6.8641)` | `0.5354 ± 0.0836 (95% CI: 0.5026–0.5682)` | `-6.8748 ± 0.0967 (95% CI: -6.9127–-6.8369)` | `244.736 ± 14.265 (95% CI: 239.144–250.328)` | `179.501 ± 15.844 (95% CI: 173.290–185.712)` |
| **RadImageNet (ResNet-50)** | `completed` | `PASS (cd42bc0409)` | `-6.8578 ± 0.0397 (95% CI: -6.8734–-6.8422)` | `0.4748 ± 0.0443 (95% CI: 0.4574–0.4921)` | `-6.8660 ± 0.0632 (95% CI: -6.8908–-6.8412)` | `240.261 ± 7.318 (95% CI: 237.392–243.129)` | `177.455 ± 10.572 (95% CI: 173.311–181.599)` |
| **CT-CLIP (3D CT-ViT)** | `completed` | `PASS (cd42bc0409)` | `-6.8383 ± 0.0587 (95% CI: -6.8613–-6.8153)` | `0.5630 ± 0.0604 (95% CI: 0.5393–0.5867)` | `-6.8344 ± 0.0917 (95% CI: -6.8704–-6.7985)` | `236.804 ± 11.255 (95% CI: 232.392–241.216)` | `172.426 ± 14.404 (95% CI: 166.779–178.072)` |
| **ResNet-50 Control (ImageNet)** | `not run` | — | `not run` | `not run` | `not run` | `not run` | `not run` |

### 3B. Full 6-Method Fusion Comparison Per Extractor

#### Current Baseline: ImageNet 2D ResNet-18 (`status: completed`)
- **Embedding Prefix / Size:** `resnet_emb_*` (`512D` -> in-fold PCA `15D`) | **Folds SHA-256:** `cd42bc04090b2952d83ed2936753952a6a789323c28058eafbfabbba93272bd4` (identical to canonical: `True`)

| Method | FVC Laplace LL (95% CI) | FVC RMSE mL (95% CI) | FVC MAE mL (95% CI) | Rapid-Decliner AUC (95% CI) |
| :--- | :---: | :---: | :---: | :---: |
| **Clinical-only** | `-6.8076 ± 0.0877 (95% CI: -6.8420–-6.7732)` | `232.596 ± 8.996 (95% CI: 229.070–236.123)` | `168.656 ± 12.728 (95% CI: 163.667–173.646)` | `0.6343 ± 0.0518 (95% CI: 0.6141–0.6546)` |
| **HRCT-only (HU/Volume + In-Fold PCA 15D)** | `-6.8245 ± 0.0528 (95% CI: -6.8452–-6.8038)` | `235.792 ± 11.963 (95% CI: 231.102–240.481)` | `171.448 ± 8.299 (95% CI: 168.195–174.701)` | `0.4807 ± 0.0332 (95% CI: 0.4677–0.4938)` |
| **Early fusion** | `-6.8115 ± 0.0834 (95% CI: -6.8442–-6.7788)` | `231.173 ± 11.792 (95% CI: 226.550–235.795)` | `168.968 ± 12.675 (95% CI: 164.000–173.937)` | `0.5539 ± 0.0679 (95% CI: 0.5273–0.5806)` |
| **Late fusion (mean)** | `-6.7928 ± 0.0716 (95% CI: -6.8209–-6.7647)` | `230.735 ± 9.523 (95% CI: 227.002–234.468)` | `166.333 ± 10.247 (95% CI: 162.316–170.350)` | `0.5636 ± 0.0666 (95% CI: 0.5375–0.5897)` |
| **Weighted late fusion (in-fold tuned)** | `-6.7988 ± 0.0808 (95% CI: -6.8305–-6.7671)` | `232.122 ± 9.457 (95% CI: 228.415–235.829)` | `167.265 ± 11.530 (95% CI: 162.745–171.785)` | `0.6088 ± 0.0544 (95% CI: 0.5875–0.6302)` |
| **Stacking (OOF meta-learner)** | `-6.7971 ± 0.0924 (95% CI: -6.8334–-6.7609)` | `231.116 ± 10.470 (95% CI: 227.012–235.221)` | `166.911 ± 13.460 (95% CI: 161.635–172.187)` | `0.5727 ± 0.0562 (95% CI: 0.5507–0.5947)` |

#### MedicalNet (Med3D) 3D ResNet-18 (`status: completed`)
- **Embedding Prefix / Size:** `med3d_emb_*` (`512D` -> in-fold PCA `15D`) | **Folds SHA-256:** `cd42bc04090b2952d83ed2936753952a6a789323c28058eafbfabbba93272bd4` (identical to canonical: `True`)

| Method | FVC Laplace LL (95% CI) | FVC RMSE mL (95% CI) | FVC MAE mL (95% CI) | Rapid-Decliner AUC (95% CI) |
| :--- | :---: | :---: | :---: | :---: |
| **Clinical-only** | `-6.8076 ± 0.0877 (95% CI: -6.8420–-6.7732)` | `232.596 ± 8.996 (95% CI: 229.070–236.123)` | `168.656 ± 12.728 (95% CI: 163.667–173.646)` | `0.6343 ± 0.0518 (95% CI: 0.6141–0.6546)` |
| **HRCT-only (HU/Volume + In-Fold PCA 15D)** | `-6.9036 ± 0.1008 (95% CI: -6.9432–-6.8641)` | `252.655 ± 17.065 (95% CI: 245.965–259.344)` | `185.044 ± 16.925 (95% CI: 178.410–191.679)` | `0.5354 ± 0.0836 (95% CI: 0.5026–0.5682)` |
| **Early fusion** | `-6.8862 ± 0.1016 (95% CI: -6.9260–-6.8464)` | `246.562 ± 16.645 (95% CI: 240.037–253.087)` | `181.557 ± 17.193 (95% CI: 174.817–188.296)` | `0.5823 ± 0.0470 (95% CI: 0.5639–0.6007)` |
| **Late fusion (mean)** | `-6.8373 ± 0.0906 (95% CI: -6.8728–-6.8018)` | `237.996 ± 11.132 (95% CI: 233.632–242.359)` | `173.540 ± 13.583 (95% CI: 168.215–178.864)` | `0.6157 ± 0.0808 (95% CI: 0.5840–0.6474)` |
| **Weighted late fusion (in-fold tuned)** | `-6.8109 ± 0.0885 (95% CI: -6.8456–-6.7762)` | `233.475 ± 9.314 (95% CI: 229.824–237.126)` | `169.258 ± 12.823 (95% CI: 164.231–174.284)` | `0.6527 ± 0.0715 (95% CI: 0.6247–0.6807)` |
| **Stacking (OOF meta-learner)** | `-6.8086 ± 0.0857 (95% CI: -6.8422–-6.7750)` | `233.160 ± 8.951 (95% CI: 229.651–236.669)` | `168.883 ± 12.377 (95% CI: 164.031–173.734)` | `0.4796 ± 0.1228 (95% CI: 0.4315–0.5278)` |

#### RadImageNet 2D ResNet-50 (`status: completed`)
- **Embedding Prefix / Size:** `radimagenet_emb_*` (`2048D` -> in-fold PCA `15D`) | **Folds SHA-256:** `cd42bc04090b2952d83ed2936753952a6a789323c28058eafbfabbba93272bd4` (identical to canonical: `True`)

| Method | FVC Laplace LL (95% CI) | FVC RMSE mL (95% CI) | FVC MAE mL (95% CI) | Rapid-Decliner AUC (95% CI) |
| :--- | :---: | :---: | :---: | :---: |
| **Clinical-only** | `-6.8076 ± 0.0877 (95% CI: -6.8420–-6.7732)` | `232.596 ± 8.996 (95% CI: 229.070–236.123)` | `168.656 ± 12.728 (95% CI: 163.667–173.646)` | `0.6343 ± 0.0518 (95% CI: 0.6141–0.6546)` |
| **HRCT-only (HU/Volume + In-Fold PCA 15D)** | `-6.8578 ± 0.0397 (95% CI: -6.8734–-6.8422)` | `241.205 ± 10.659 (95% CI: 237.027–245.384)` | `175.926 ± 7.461 (95% CI: 173.001–178.851)` | `0.4748 ± 0.0443 (95% CI: 0.4574–0.4921)` |
| **Early fusion** | `-6.8607 ± 0.0666 (95% CI: -6.8868–-6.8346)` | `238.973 ± 7.589 (95% CI: 235.998–241.948)` | `176.297 ± 11.334 (95% CI: 171.854–180.740)` | `0.5549 ± 0.0678 (95% CI: 0.5283–0.5814)` |
| **Late fusion (mean)** | `-6.8152 ± 0.0578 (95% CI: -6.8379–-6.7926)` | `233.225 ± 6.230 (95% CI: 230.783–235.667)` | `169.471 ± 8.466 (95% CI: 166.153–172.790)` | `0.5578 ± 0.0706 (95% CI: 0.5302–0.5855)` |
| **Weighted late fusion (in-fold tuned)** | `-6.8087 ± 0.0844 (95% CI: -6.8418–-6.7756)` | `232.537 ± 8.490 (95% CI: 229.209–235.865)` | `168.866 ± 12.087 (95% CI: 164.128–173.605)` | `0.6074 ± 0.0520 (95% CI: 0.5870–0.6277)` |
| **Stacking (OOF meta-learner)** | `-6.8090 ± 0.0810 (95% CI: -6.8407–-6.7773)` | `232.848 ± 8.303 (95% CI: 229.593–236.102)` | `168.946 ± 11.511 (95% CI: 164.434–173.458)` | `0.5715 ± 0.0632 (95% CI: 0.5467–0.5963)` |

#### CT-CLIP 3D CT-ViT Image Encoder (`status: completed`)
- **Embedding Prefix / Size:** `ctclip_emb_*` (`512D` -> in-fold PCA `15D`) | **Folds SHA-256:** `cd42bc04090b2952d83ed2936753952a6a789323c28058eafbfabbba93272bd4` (identical to canonical: `True`)

| Method | FVC Laplace LL (95% CI) | FVC RMSE mL (95% CI) | FVC MAE mL (95% CI) | Rapid-Decliner AUC (95% CI) |
| :--- | :---: | :---: | :---: | :---: |
| **Clinical-only** | `-6.8076 ± 0.0877 (95% CI: -6.8420–-6.7732)` | `232.596 ± 8.996 (95% CI: 229.070–236.123)` | `168.656 ± 12.728 (95% CI: 163.667–173.646)` | `0.6343 ± 0.0518 (95% CI: 0.6141–0.6546)` |
| **HRCT-only (HU/Volume + In-Fold PCA 15D)** | `-6.8383 ± 0.0587 (95% CI: -6.8613–-6.8153)` | `238.755 ± 8.125 (95% CI: 235.570–241.940)` | `173.495 ± 9.293 (95% CI: 169.852–177.138)` | `0.5630 ± 0.0604 (95% CI: 0.5393–0.5867)` |
| **Early fusion** | `-6.8320 ± 0.0935 (95% CI: -6.8687–-6.7954)` | `235.976 ± 11.993 (95% CI: 231.275–240.677)` | `171.981 ± 14.770 (95% CI: 166.192–177.771)` | `0.6145 ± 0.0552 (95% CI: 0.5928–0.6361)` |
| **Late fusion (mean)** | `-6.8010 ± 0.0757 (95% CI: -6.8307–-6.7714)` | `232.398 ± 8.299 (95% CI: 229.144–235.651)` | `167.489 ± 10.992 (95% CI: 163.180–171.798)` | `0.6156 ± 0.0700 (95% CI: 0.5881–0.6430)` |
| **Weighted late fusion (in-fold tuned)** | `-6.7998 ± 0.0831 (95% CI: -6.8324–-6.7672)` | `231.855 ± 8.445 (95% CI: 228.544–235.165)` | `167.363 ± 11.870 (95% CI: 162.710–172.016)` | `0.6216 ± 0.0693 (95% CI: 0.5944–0.6487)` |
| **Stacking (OOF meta-learner)** | `-6.7998 ± 0.0923 (95% CI: -6.8359–-6.7636)` | `231.875 ± 9.470 (95% CI: 228.163–235.587)` | `167.324 ± 13.397 (95% CI: 162.072–172.575)` | `0.5638 ± 0.0864 (95% CI: 0.5300–0.5977)` |

#### Optional Control: ImageNet 2D ResNet-50 (`status: not run`)
- **Status:** `not run` — Disabled by user toggle RUN_IMAGENET_RESNET50_CONTROL = False

---

## 4. Paired Nadeau-Bengio Statistical Comparisons (`168` Total Tests, Holm-Bonferroni (step-down) with Bonferroni reference)

### 4A. Primary Question — New Extractors vs. Current ResNet-18 Pipeline (Same 25 Folds)

| Extractor | Configuration | Metric | Paired Δ vs. ResNet-18 (95% CI) | Raw p | Holm p | Bonferroni p | Better (Holm α=0.05)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Med3D (3D ResNet-18)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `laplace_ll` | `-0.0791 [-0.1362, -0.0221]` | `0.0086` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `rmse` | `+16.8632 [+3.4627, +30.2637]` | `0.0158` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `mae` | `+13.5962 [+3.3979, +23.7945]` | `0.0111` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `auc` | `+0.0547 [-0.0126, +0.1219]` | `0.1067` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Image + Tabular (Benchmark, Ridge GridSearch) | `laplace_ll` | `-0.0648 [-0.1013, -0.0282]` | `0.0012` | `0.1944` | `0.2080` | `No` |
| **Med3D (3D ResNet-18)** | Image + Tabular (Benchmark, Ridge GridSearch) | `rmse` | `+13.9496 [+6.8888, +21.0104]` | `0.0004` | `0.0702` | `0.0728` | `No` |
| **Med3D (3D ResNet-18)** | Image + Tabular (Benchmark, Ridge GridSearch) | `mae` | `+10.7842 [+4.1498, +17.4185]` | `0.0026` | `0.4029` | `0.4425` | `No` |
| **Med3D (3D ResNet-18)** | Image + Tabular (Benchmark, Ridge GridSearch) | `auc` | `+0.0283 [-0.0423, +0.0989]` | `0.4156` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Early fusion | `laplace_ll` | `-0.0747 [-0.1281, -0.0214]` | `0.0080` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Early fusion | `rmse` | `+15.3892 [+5.1309, +25.6475]` | `0.0049` | `0.7399` | `0.8286` | `No` |
| **Med3D (3D ResNet-18)** | Early fusion | `mae` | `+12.5884 [+3.0893, +22.0875]` | `0.0115` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Early fusion | `auc` | `+0.0283 [-0.0423, +0.0989]` | `0.4156` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Late fusion (mean) | `laplace_ll` | `-0.0445 [-0.0716, -0.0174]` | `0.0024` | `0.3794` | `0.4112` | `No` |
| **Med3D (3D ResNet-18)** | Late fusion (mean) | `rmse` | `+7.2609 [+3.2620, +11.2598]` | `0.0010` | `0.1572` | `0.1671` | `No` |
| **Med3D (3D ResNet-18)** | Late fusion (mean) | `mae` | `+7.2068 [+2.7077, +11.7059]` | `0.0030` | `0.4482` | `0.4986` | `No` |
| **Med3D (3D ResNet-18)** | Late fusion (mean) | `auc` | `+0.0522 [-0.0236, +0.1279]` | `0.1683` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Weighted late fusion (in-fold tuned) | `laplace_ll` | `-0.0121 [-0.0248, +0.0007]` | `0.0624` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Weighted late fusion (in-fold tuned) | `rmse` | `+1.3531 [-0.7520, +3.4583]` | `0.1971` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Weighted late fusion (in-fold tuned) | `mae` | `+1.9927 [-0.1002, +4.0856]` | `0.0611` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Weighted late fusion (in-fold tuned) | `auc` | `+0.0439 [-0.0491, +0.1369]` | `0.3398` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Stacking (OOF meta-learner) | `laplace_ll` | `-0.0115 [-0.0297, +0.0067]` | `0.2058` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Stacking (OOF meta-learner) | `rmse` | `+2.0434 [-0.4772, +4.5639]` | `0.1073` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Stacking (OOF meta-learner) | `mae` | `+1.9714 [-1.0128, +4.9557]` | `0.1854` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Stacking (OOF meta-learner) | `auc` | `-0.0931 [-0.1745, -0.0117]` | `0.0267` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `laplace_ll` | `-0.0333 [-0.1088, +0.0422]` | `0.3721` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `rmse` | `+5.4139 [-10.1949, +21.0226]` | `0.4810` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `mae` | `+4.4775 [-9.5015, +18.4566]` | `0.5149` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `auc` | `-0.0060 [-0.0534, +0.0414]` | `0.7976` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Image + Tabular (Benchmark, Ridge GridSearch) | `laplace_ll` | `-0.0560 [-0.1143, +0.0023]` | `0.0589` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Image + Tabular (Benchmark, Ridge GridSearch) | `rmse` | `+9.4747 [-2.7120, +21.6613]` | `0.1217` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Image + Tabular (Benchmark, Ridge GridSearch) | `mae` | `+8.7375 [-0.8312, +18.3062]` | `0.0716` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Image + Tabular (Benchmark, Ridge GridSearch) | `auc` | `+0.0009 [-0.0511, +0.0529]` | `0.9717` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Early fusion | `laplace_ll` | `-0.0492 [-0.1117, +0.0133]` | `0.1171` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Early fusion | `rmse` | `+7.8004 [-4.1682, +19.7690]` | `0.1912` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Early fusion | `mae` | `+7.3287 [-3.3782, +18.0356]` | `0.1706` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Early fusion | `auc` | `+0.0009 [-0.0511, +0.0529]` | `0.9717` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Late fusion (mean) | `laplace_ll` | `-0.0224 [-0.0553, +0.0105]` | `0.1723` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Late fusion (mean) | `rmse` | `+2.4902 [-3.7009, +8.6813]` | `0.4146` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Late fusion (mean) | `mae` | `+3.1383 [-2.4673, +8.7439]` | `0.2593` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Late fusion (mean) | `auc` | `-0.0057 [-0.0405, +0.0291]` | `0.7375` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Weighted late fusion (in-fold tuned) | `laplace_ll` | `-0.0099 [-0.0191, -0.0007]` | `0.0363` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Weighted late fusion (in-fold tuned) | `rmse` | `+0.4153 [-2.9150, +3.7456]` | `0.7991` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Weighted late fusion (in-fold tuned) | `mae` | `+1.6017 [+0.0545, +3.1488]` | `0.0430` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Weighted late fusion (in-fold tuned) | `auc` | `-0.0015 [-0.0859, +0.0829]` | `0.9718` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Stacking (OOF meta-learner) | `laplace_ll` | `-0.0119 [-0.0260, +0.0023]` | `0.0968` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Stacking (OOF meta-learner) | `rmse` | `+1.7311 [-0.9105, +4.3728]` | `0.1888` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Stacking (OOF meta-learner) | `mae` | `+2.0348 [-0.2852, +4.3547]` | `0.0828` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Stacking (OOF meta-learner) | `auc` | `-0.0012 [-0.0235, +0.0211]` | `0.9122` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `laplace_ll` | `-0.0138 [-0.0401, +0.0126]` | `0.2917` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `rmse` | `+2.9637 [-7.2964, +13.2238]` | `0.5566` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `mae` | `+2.0466 [-3.2811, +7.3744]` | `0.4356` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `auc` | `+0.0823 [+0.0049, +0.1597]` | `0.0380` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Image + Tabular (Benchmark, Ridge GridSearch) | `laplace_ll` | `-0.0244 [-0.0517, +0.0028]` | `0.0767` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Image + Tabular (Benchmark, Ridge GridSearch) | `rmse` | `+6.0180 [-1.9997, +14.0356]` | `0.1344` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Image + Tabular (Benchmark, Ridge GridSearch) | `mae` | `+3.7085 [-1.3275, +8.7444]` | `0.1416` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Image + Tabular (Benchmark, Ridge GridSearch) | `auc` | `+0.0605 [-0.0036, +0.1247]` | `0.0632` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Early fusion | `laplace_ll` | `-0.0206 [-0.0498, +0.0087]` | `0.1593` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Early fusion | `rmse` | `+4.8034 [-3.2540, +12.8607]` | `0.2305` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Early fusion | `mae` | `+3.0132 [-2.1910, +8.2174]` | `0.2438` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Early fusion | `auc` | `+0.0605 [-0.0036, +0.1247]` | `0.0632` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Late fusion (mean) | `laplace_ll` | `-0.0082 [-0.0173, +0.0008]` | `0.0733` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Late fusion (mean) | `rmse` | `+1.6628 [-2.8652, +6.1908]` | `0.4559` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Late fusion (mean) | `mae` | `+1.1559 [-0.7471, +3.0589]` | `0.2220` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Late fusion (mean) | `auc` | `+0.0520 [-0.0147, +0.1187]` | `0.1207` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Weighted late fusion (in-fold tuned) | `laplace_ll` | `-0.0010 [-0.0074, +0.0055]` | `0.7610` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Weighted late fusion (in-fold tuned) | `rmse` | `-0.2672 [-3.9077, +3.3733]` | `0.8809` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Weighted late fusion (in-fold tuned) | `mae` | `+0.0981 [-1.1713, +1.3675]` | `0.8746` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Weighted late fusion (in-fold tuned) | `auc` | `+0.0128 [-0.0187, +0.0442]` | `0.4110` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Stacking (OOF meta-learner) | `laplace_ll` | `-0.0026 [-0.0080, +0.0028]` | `0.3270` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Stacking (OOF meta-learner) | `rmse` | `+0.7584 [-2.2601, +3.7769]` | `0.6088` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Stacking (OOF meta-learner) | `mae` | `+0.4128 [-0.7032, +1.5289]` | `0.4527` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Stacking (OOF meta-learner) | `auc` | `-0.0089 [-0.0616, +0.0438]` | `0.7302` | `1.0000` | `1.0000` | `No` |

### 4B. Secondary Question — Completed Extractors vs. Clinical-Only (Same 25 Folds)

| Extractor | Configuration | Metric | Paired Δ vs. Clinical-Only (95% CI) | Raw p | Holm p | Bonferroni p | Better (Holm α=0.05)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 (Current)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `laplace_ll` | `-0.0169 [-0.0726, +0.0388]` | `0.5364` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `rmse` | `+3.1953 [-5.3867, +11.7772]` | `0.4497` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `mae` | `+2.7917 [-5.2557, +10.8390]` | `0.4809` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `auc` | `-0.1536 [-0.2040, -0.1032]` | `0.0000` | `0.0003` | `0.0003` | `No` |
| **ResNet-18 (Current)** | Image + Tabular (Benchmark, Ridge GridSearch) | `laplace_ll` | `-0.0024 [-0.0316, +0.0268]` | `0.8660` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | Image + Tabular (Benchmark, Ridge GridSearch) | `rmse` | `-1.8103 [-8.3664, +4.7457]` | `0.5740` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | Image + Tabular (Benchmark, Ridge GridSearch) | `mae` | `+0.0606 [-4.6599, +4.7811]` | `0.9791` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | Image + Tabular (Benchmark, Ridge GridSearch) | `auc` | `-0.0804 [-0.1127, -0.0481]` | `0.0000` | `0.0048` | `0.0048` | `No` |
| **ResNet-18 (Current)** | Early fusion | `laplace_ll` | `-0.0039 [-0.0337, +0.0260]` | `0.7915` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | Early fusion | `rmse` | `-1.4237 [-7.6135, +4.7661]` | `0.6393` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | Early fusion | `mae` | `+0.3116 [-4.4831, +5.1064]` | `0.8944` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | Early fusion | `auc` | `-0.0804 [-0.1127, -0.0481]` | `0.0000` | `0.0048` | `0.0048` | `No` |
| **ResNet-18 (Current)** | Late fusion (mean) | `laplace_ll` | `+0.0148 [-0.0072, +0.0367]` | `0.1778` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | Late fusion (mean) | `rmse` | `-1.8616 [-6.1849, +2.4618]` | `0.3830` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | Late fusion (mean) | `mae` | `-2.3235 [-5.7036, +1.0565]` | `0.1688` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | Late fusion (mean) | `auc` | `-0.0708 [-0.1191, -0.0225]` | `0.0058` | `0.8636` | `0.9803` | `No` |
| **ResNet-18 (Current)** | Weighted late fusion (in-fold tuned) | `laplace_ll` | `+0.0088 [-0.0023, +0.0198]` | `0.1153` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | Weighted late fusion (in-fold tuned) | `rmse` | `-0.4746 [-3.8246, +2.8754]` | `0.7725` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | Weighted late fusion (in-fold tuned) | `mae` | `-1.3917 [-3.2449, +0.4615]` | `0.1342` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | Weighted late fusion (in-fold tuned) | `auc` | `-0.0255 [-0.0815, +0.0305]` | `0.3563` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | Stacking (OOF meta-learner) | `laplace_ll` | `+0.0104 [+0.0049, +0.0159]` | `0.0006` | `0.1024` | `0.1082` | `No` |
| **ResNet-18 (Current)** | Stacking (OOF meta-learner) | `rmse` | `-1.4799 [-3.4216, +0.4619]` | `0.1288` | `1.0000` | `1.0000` | `No` |
| **ResNet-18 (Current)** | Stacking (OOF meta-learner) | `mae` | `-1.7454 [-2.5898, -0.9010]` | `0.0003` | `0.0438` | `0.0451` | `Yes` |
| **ResNet-18 (Current)** | Stacking (OOF meta-learner) | `auc` | `-0.0616 [-0.1577, +0.0345]` | `0.1982` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `laplace_ll` | `-0.0961 [-0.1609, -0.0312]` | `0.0054` | `0.8080` | `0.9110` | `No` |
| **Med3D (3D ResNet-18)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `rmse` | `+20.0584 [+7.6524, +32.4644]` | `0.0028` | `0.4183` | `0.4623` | `No` |
| **Med3D (3D ResNet-18)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `mae` | `+16.3879 [+4.9260, +27.8497]` | `0.0070` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `auc` | `-0.0990 [-0.1982, +0.0003]` | `0.0505` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Image + Tabular (Benchmark, Ridge GridSearch) | `laplace_ll` | `-0.0672 [-0.1206, -0.0138]` | `0.0158` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Image + Tabular (Benchmark, Ridge GridSearch) | `rmse` | `+12.1393 [+1.6302, +22.6483]` | `0.0254` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Image + Tabular (Benchmark, Ridge GridSearch) | `mae` | `+10.8448 [+1.4612, +20.2284]` | `0.0253` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Image + Tabular (Benchmark, Ridge GridSearch) | `auc` | `-0.0521 [-0.0952, -0.0089]` | `0.0201` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Early fusion | `laplace_ll` | `-0.0786 [-0.1454, -0.0118]` | `0.0229` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Early fusion | `rmse` | `+13.9655 [+0.3338, +27.5971]` | `0.0451` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Early fusion | `mae` | `+12.9000 [+1.1352, +24.6649]` | `0.0330` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Early fusion | `auc` | `-0.0521 [-0.0952, -0.0089]` | `0.0201` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Late fusion (mean) | `laplace_ll` | `-0.0297 [-0.0582, -0.0012]` | `0.0415` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Late fusion (mean) | `rmse` | `+5.3993 [+0.3992, +10.3994]` | `0.0355` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Late fusion (mean) | `mae` | `+4.8833 [+0.1256, +9.6409]` | `0.0447` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Late fusion (mean) | `auc` | `-0.0186 [-0.0809, +0.0437]` | `0.5428` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Weighted late fusion (in-fold tuned) | `laplace_ll` | `-0.0033 [-0.0180, +0.0114]` | `0.6486` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Weighted late fusion (in-fold tuned) | `rmse` | `+0.8786 [-1.4112, +3.1683]` | `0.4362` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Weighted late fusion (in-fold tuned) | `mae` | `+0.6010 [-1.8548, +3.0569]` | `0.6181` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Weighted late fusion (in-fold tuned) | `auc` | `+0.0184 [-0.0187, +0.0554]` | `0.3163` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Stacking (OOF meta-learner) | `laplace_ll` | `-0.0010 [-0.0164, +0.0143]` | `0.8915` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Stacking (OOF meta-learner) | `rmse` | `+0.5635 [-2.0647, +3.1916]` | `0.6621` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Stacking (OOF meta-learner) | `mae` | `+0.2261 [-2.3528, +2.8049]` | `0.8579` | `1.0000` | `1.0000` | `No` |
| **Med3D (3D ResNet-18)** | Stacking (OOF meta-learner) | `auc` | `-0.1547 [-0.3276, +0.0182]` | `0.0771` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `laplace_ll` | `-0.0502 [-0.1458, +0.0454]` | `0.2893` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `rmse` | `+8.6091 [-7.3313, +24.5495]` | `0.2760` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `mae` | `+7.2692 [-8.1218, +22.6602]` | `0.3394` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `auc` | `-0.1596 [-0.2024, -0.1167]` | `0.0000` | `0.0000` | `0.0000` | `No` |
| **RadImageNet (ResNet-50)** | Image + Tabular (Benchmark, Ridge GridSearch) | `laplace_ll` | `-0.0584 [-0.1103, -0.0066]` | `0.0288` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Image + Tabular (Benchmark, Ridge GridSearch) | `rmse` | `+7.6643 [-0.7112, +16.0399]` | `0.0711` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Image + Tabular (Benchmark, Ridge GridSearch) | `mae` | `+8.7981 [+0.3726, +17.2236]` | `0.0414` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Image + Tabular (Benchmark, Ridge GridSearch) | `auc` | `-0.0795 [-0.1208, -0.0382]` | `0.0006` | `0.0917` | `0.0957` | `No` |
| **RadImageNet (ResNet-50)** | Early fusion | `laplace_ll` | `-0.0531 [-0.1074, +0.0013]` | `0.0551` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Early fusion | `rmse` | `+6.3767 [-2.2726, +15.0260]` | `0.1412` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Early fusion | `mae` | `+7.6404 [-1.5570, +16.8377]` | `0.0993` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Early fusion | `auc` | `-0.0795 [-0.1208, -0.0382]` | `0.0006` | `0.0917` | `0.0957` | `No` |
| **RadImageNet (ResNet-50)** | Late fusion (mean) | `laplace_ll` | `-0.0076 [-0.0540, +0.0387]` | `0.7364` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Late fusion (mean) | `rmse` | `+0.6286 [-6.1032, +7.3604]` | `0.8488` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Late fusion (mean) | `mae` | `+0.8148 [-6.5600, +8.1895]` | `0.8216` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Late fusion (mean) | `auc` | `-0.0765 [-0.1234, -0.0296]` | `0.0026` | `0.3948` | `0.4306` | `No` |
| **RadImageNet (ResNet-50)** | Weighted late fusion (in-fold tuned) | `laplace_ll` | `-0.0011 [-0.0074, +0.0051]` | `0.7101` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Weighted late fusion (in-fold tuned) | `rmse` | `-0.0593 [-0.6264, +0.5077]` | `0.8309` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Weighted late fusion (in-fold tuned) | `mae` | `+0.2099 [-0.8664, +1.2863]` | `0.6908` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Weighted late fusion (in-fold tuned) | `auc` | `-0.0270 [-0.0752, +0.0212]` | `0.2595` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Stacking (OOF meta-learner) | `laplace_ll` | `-0.0014 [-0.0102, +0.0074]` | `0.7428` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Stacking (OOF meta-learner) | `rmse` | `+0.2512 [-1.2204, +1.7229]` | `0.7277` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Stacking (OOF meta-learner) | `mae` | `+0.2894 [-1.2276, +1.8064]` | `0.6973` | `1.0000` | `1.0000` | `No` |
| **RadImageNet (ResNet-50)** | Stacking (OOF meta-learner) | `auc` | `-0.0628 [-0.1742, +0.0485]` | `0.2558` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `laplace_ll` | `-0.0307 [-0.0785, +0.0171]` | `0.1977` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `rmse` | `+6.1590 [+4.0562, +8.2617]` | `0.0000` | `0.0005` | `0.0005` | `No` |
| **CT-CLIP (3D CT-ViT)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `mae` | `+4.8383 [-1.9601, +11.6367]` | `0.1549` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | HRCT-only (HU/Volume + In-Fold PCA 15D) | `auc` | `-0.0713 [-0.1326, -0.0100]` | `0.0244` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Image + Tabular (Benchmark, Ridge GridSearch) | `laplace_ll` | `-0.0268 [-0.0425, -0.0112]` | `0.0017` | `0.2628` | `0.2830` | `No` |
| **CT-CLIP (3D CT-ViT)** | Image + Tabular (Benchmark, Ridge GridSearch) | `rmse` | `+4.2076 [-0.6635, +9.0788]` | `0.0873` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Image + Tabular (Benchmark, Ridge GridSearch) | `mae` | `+3.7691 [+0.4173, +7.1208]` | `0.0291` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Image + Tabular (Benchmark, Ridge GridSearch) | `auc` | `-0.0199 [-0.0870, +0.0473]` | `0.5473` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Early fusion | `laplace_ll` | `-0.0244 [-0.0421, -0.0068]` | `0.0087` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Early fusion | `rmse` | `+3.3796 [-2.5734, +9.3326]` | `0.2528` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Early fusion | `mae` | `+3.3248 [-0.3534, +7.0030]` | `0.0744` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Early fusion | `auc` | `-0.0199 [-0.0870, +0.0473]` | `0.5473` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Late fusion (mean) | `laplace_ll` | `+0.0065 [-0.0120, +0.0251]` | `0.4732` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Late fusion (mean) | `rmse` | `-0.1988 [-1.4627, +1.0651]` | `0.7483` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Late fusion (mean) | `mae` | `-1.1676 [-4.1176, +1.7824]` | `0.4220` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Late fusion (mean) | `auc` | `-0.0188 [-0.0858, +0.0483]` | `0.5689` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Weighted late fusion (in-fold tuned) | `laplace_ll` | `+0.0078 [-0.0009, +0.0165]` | `0.0768` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Weighted late fusion (in-fold tuned) | `rmse` | `-0.7418 [-1.6218, +0.1383]` | `0.0947` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Weighted late fusion (in-fold tuned) | `mae` | `-1.2936 [-2.9017, +0.3145]` | `0.1099` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Weighted late fusion (in-fold tuned) | `auc` | `-0.0128 [-0.0983, +0.0728]` | `0.7609` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Stacking (OOF meta-learner) | `laplace_ll` | `+0.0078 [+0.0007, +0.0149]` | `0.0327` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Stacking (OOF meta-learner) | `rmse` | `-0.7215 [-2.8772, +1.4342]` | `0.4963` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Stacking (OOF meta-learner) | `mae` | `-1.3326 [-2.7257, +0.0606]` | `0.0600` | `1.0000` | `1.0000` | `No` |
| **CT-CLIP (3D CT-ViT)** | Stacking (OOF meta-learner) | `auc` | `-0.0705 [-0.1847, +0.0436]` | `0.2144` | `1.0000` | `1.0000` | `No` |

---

## 5. Methodological Limitations

- Dimensionality equalization via PCA: All raw embedding vectors (512-D for ResNet-18, Med3D, and CT-CLIP; 2,048-D for RadImageNet and ImageNet ResNet-50) are compressed strictly inside each training fold to 15 PCA components (PCA(n_components=15, random_state=42)), which prevents overfitting on n = 175 patients but may discard fine-grained variance in higher-dimensional embeddings.
- Modality-specific preprocessing differences: 3D volumetric backbones (Med3D resampled lung-masked volumes with z-scoring; CT-CLIP full-chest [-1000, 1000] HU volumes resampled to (1.5, 0.75, 0.75) mm and cropped/padded to 240x480x480) require different spatial and intensity preprocessing than 2D slice-wise models (lung-masked [-1000, 0] HU slices resized to 224x224 and mean-pooled).
- Internal cross-validation only (not Kaggle leaderboard scores): All metrics come from 5-Repeat x 5-Fold patient-grouped internal cross-validation on n = 175 OSIC train.csv patients (1,359 future visits) and are not comparable to hidden Kaggle OSIC leaderboard scores.
- Exploratory separate cohort without IPF labels: OSIC is an independent pulmonary fibrosis cohort with no IPF vs. Non-IPF diagnostic labels. Progression results are strictly exploratory and never alter the deployed Zenodo IPF diagnostic classifier or the deployed OSIC serving pipeline automatically.
- Non-commercial weight restriction for CT-CLIP / CT-RATE: CT-CLIP and CT-RATE weights are gated under CC BY-NC-SA 4.0 (non-commercial) and must never be redistributed inside handover packages.

---

## 6. Licenses & Literature Citations

- **MedicalNet (Med3D) Mirror (`TencentMedicalNet/MedicalNet-Resnet18`):** MIT License.
- **RadImageNet Mirror (`Lab-Rasool/RadImageNet`, official `BMEII-AI/RadImageNet` weights):** MIT License.
- **CT-CLIP & CT-RATE (`ibrahimhamamci/CT-RATE`, `ibrahimethemhamamci/CT-CLIP`):** Gated access, **CC BY-NC-SA 4.0 (Non-Commercial)** — weights must **not** be redistributed.
- **ImageNet ResNet-18 / ResNet-50 (`torchvision`):** BSD-3-Clause.

### References
1. Chen, S., Ma, K., & Zheng, Y. *Med3D: Transfer Learning for 3D Medical Image Analysis*. arXiv:1904.00625 (2019). https://arxiv.org/abs/1904.00625
2. Mei, X., Liu, Z., Robson, P. M., et al. *RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning*. Radiology: Artificial Intelligence, 4(5), e210315 (2022). https://doi.org/10.1148/ryai.210315
3. Hamamci, I. E., Er, S., Almas, F., et al. *Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography (CT-RATE / CT-CLIP)*. arXiv:2403.17834 (2024). https://arxiv.org/abs/2403.17834
4. Nadeau, C., & Bengio, Y. *Inference for the Generalization Error*. Machine Learning, 52(3), 239-281 (2003). https://doi.org/10.1023/A:1024068626366
