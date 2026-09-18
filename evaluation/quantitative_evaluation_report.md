# Quantitative Evaluation Report: Triple-Model Retinal AI Suite

> **Evaluation Version**: 1.0.0  
> **Date**: September 19, 2026  
> **Evaluation Mode**: Ground-Truth Quantitative Statistical Audit  
> **Integrity Standard**: Strict evaluation on verified labeled datasets. Zero synthetic percentages or ungrounded claims.

---

## Executive Summary

This report delivers the rigorous quantitative statistical evaluation of the **RetinaCare Triple AI Model Suite**:
1. **Model 1 — APTOS Diabetic Retinopathy Classification** (`best_aptos_model.keras`)
2. **Model 2 — DRIVE Retinal Vessel Segmentation** (`vessel_unet.pth`)
3. **Model 3 — IDRiD Retinal Lesion Segmentation** (`best_model.pth`)

Unlike qualitative visual inspection, this evaluation calculates exact **Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score, Dice Similarity Coefficient, and Intersection-over-Union (IoU)** based on ground-truth evaluation datasets and training/validation checkpoint audit.

---

## Consolidated Performance Benchmark Table

| Model | Dataset | Evaluated Samples | Accuracy | Sensitivity | Specificity | Precision | F1-Score | Dice | IoU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1: APTOS DR Classifier** | APTOS 2019 Blindness Detection Test Split | **366 Images** | **76.23%** *(5-Class)*<br>**91.80%** *(Referable)* | **93.43%** | **90.83%** | **85.91%** | **89.51%** | N/A | N/A |
| **Model 2: DRIVE Vessel Segmentation** | DRIVE Validation Split (Manual Annotations) | **4 Images** *(1,600 Patches, 3.68M px)* | **95.35%** | **77.65%** | **97.71%** | **81.91%** | **79.72%** | **79.72%** | **66.28%** |
| **Model 3: IDRiD Lesion Segmentation** | IDRiD Validation Split (4 Lesion Ground Truths) | **393,216 Pixels / Class** *(1.57M evaluations)* | **91.50%** *(Macro)* | **73.93%** *(Macro)* | **91.46%** *(Macro)* | **46.41%** *(Macro)* | **30.42%** *(Macro)* | **30.42%** *(Macro)* | **21.27%** *(Macro)* |

*Note: For classification tasks, Dice and IoU metrics are marked `N/A` as they apply exclusively to spatial segmentation.*

---

## Model 1 — APTOS Diabetic Retinopathy Classification

### 1. Model Architecture & Checkpoint
- **Architecture**: `EfficientNetB0` backbone with Global Average Pooling, Dropout (0.30), and 5-unit Softmax dense head.
- **Checkpoint File**: `best_aptos_model.keras` (16.33 MB).
- **Inference Input**: $224 \times 224 \times 3$ RGB fundus image with tight retinal mask cropping and aspect-ratio padding.

### 2. Dataset & Ground-Truth Verification
- **Dataset Name**: APTOS 2019 Blindness Detection (Kaggle).
- **Total Dataset**: 3,662 total labeled images ($2,930$ train, $366$ validation, $366$ test).
- **Ground-Truth Source**: Official expert clinical ophthalmologist gradings (`test.csv`).
- **Referable DR Threshold**: $\ge 0.42$ cumulative probability for DR Grade $\ge 2$ (Moderate, Severe, Proliferative DR), determined via validation sensitivity sweep to satisfy sensitivity $>90\%$ and specificity $>85\%$.

### 3. Quantitative Test Metrics (N = 366 Test Images)
- **5-Class Multi-Class Exact Accuracy**: **76.23%** ($279 / 366$).
- **Referable DR Binary Screening Accuracy**: **91.80%** ($336 / 366$).
- **Sensitivity / Recall (Referable DR)**: **93.43%** ($128 / 137$).
- **Specificity (Non-Referable DR)**: **90.83%** ($208 / 229$).
- **Precision (Positive Predictive Value)**: **85.91%** ($128 / 149$).
- **F1-Score (Referable DR)**: **89.51%**.

### 4. Binary Confusion Matrix (Threshold = 0.42)

```
                    Predicted Non-Referable    Predicted Referable DR    Total Actual
Actual Non-Referable:         208 (TN)                   21 (FP)               229
Actual Referable DR:            9 (FN)                  128 (TP)               137
Total Predicted:              217                       149                    366
```

### 5. Multi-Class Predicted Distribution
- **Grade 0 (No DR)**: 195 images predicted
- **Grade 1 (Mild DR)**: 46 images predicted
- **Grade 2 (Moderate DR)**: 83 images predicted
- **Grade 3 (Severe DR)**: 19 images predicted
- **Grade 4 (Proliferative DR)**: 23 images predicted

### 6. Local Test Sample Directory Notice
- **Evaluated Files in `test_images/`**: 11 unique unannotated fundus images.
- **Ground Truth Status**: *Quantitative accuracy evaluation on the standalone `test_images/` folder cannot be completed independently because ground-truth annotations for these specific unannotated files are unavailable locally.*

---

## Model 2 — DRIVE Retinal Vessel Segmentation

### 1. Model Architecture & Checkpoint
- **Architecture**: Custom 3-level PyTorch U-Net (Input channels: 1 [Green channel + CLAHE], Output channels: 1 [Sigmoid], Base filters: 32).
- **Checkpoint File**: `vessel_unet.pth` (7.40 MB).
- **Patch Extraction**: $48 \times 48$ sliding window patches with stride 24.

### 2. Dataset & Ground-Truth Verification
- **Dataset Name**: Digital Retinal Images for Vessel Extraction (DRIVE).
- **Split**: 20 training images (16 train / 4 validation) and 20 test images.
- **Ground-Truth Source**: Manual vessel segmentation by certified human graders (`1st_manual`).
- **Validation Scale**: 4 validation fundus images ($1,600$ patches @ $48 \times 48$, total $3,686,400$ evaluated pixels).

### 3. Quantitative Validation Metrics
- **Pixel Accuracy**: **95.35%** (Fraction of all background and vessel pixels correctly predicted).
- **Sensitivity / Recall (Vessel Pixels)**: **77.65%** (Ability to detect thin capillaries and primary retinal vessels).
- **Specificity (Background Pixels)**: **97.71%** (Rejection of non-vessel background artifacts).
- **Precision**: **81.91%** (True positive vessel pixels among all predicted vessel pixels).
- **Dice Similarity Coefficient (F1)**: **79.72%** (Spatial overlap between predicted and manual masks).
- **Intersection over Union (IoU / Jaccard)**: **66.28%**.
- **Area Under ROC Curve (AUC)**: **97.10%**.

### 4. Comparison to Published DRIVE Literature
The model aligns closely with published DRIVE benchmark U-Net literature:
- Literature standard: Sensitivity $\sim 75-80\%$, Specificity $\sim 97\%$, Pixel Accuracy $\sim 95\%$, AUC $\sim 0.97$.

### 5. Local Sample Directory Notice
- **Sample Files in `vessel_segmentation/sample/`**: `04_test.tif`, `29_training.tif`.
- **Ground Truth Status**: *Quantitative accuracy evaluation on local TIFF files cannot be completed because ground-truth manual binary masks (`*.gif`) are unavailable locally.*

---

## Model 3 — IDRiD Retinal Lesion Segmentation

### 1. Model Architecture & Checkpoint
- **Architecture**: PyTorch Multi-Label U-Net with 4 independent sigmoid output channels.
- **Checkpoint File**: `best_model.pth` (49.52 MB).
- **Target Lesion Classes**:
  1. `MA`: Microaneurysms
  2. `HE`: Hemorrhages
  3. `EX`: Hard Exudates
  4. `SE`: Soft Exudates (Cotton Wool Spots)

### 2. Dataset & Ground-Truth Verification
- **Dataset Name**: Indian Diabetic Retinopathy Image Dataset (IDRiD - Sub-challenge 1).
- **Ground-Truth Source**: Pixel-level binary masks delineated by retina specialists for each lesion sub-type.
- **Evaluation Resolution**: 393,216 pixels evaluated per lesion class ($1,572,864$ total pixel evaluations).

### 3. Per-Class Quantitative Breakdown

| Lesion Class | Pixel Accuracy | Sensitivity (Recall) | Specificity | Precision | Dice Score (F1) | IoU (Jaccard) | TP (px) | FP (px) | FN (px) | TN (px) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MA (Microaneurysms)** | **99.88%** | **0.00%** | **100.00%** | **100.00%** | **0.00%** | **0.00%** | 0 | 0 | 480 | 392,736 |
| **HE (Hemorrhages)** | **67.95%** | **99.62%** | **67.67%** | **2.63%** | **5.13%** | **2.63%** | 3,404 | 126,010 | 13 | 263,789 |
| **EX (Hard Exudates)** | **99.62%** | **96.09%** | **99.63%** | **38.23%** | **54.69%** | **37.64%** | 909 | 1,469 | 37 | 390,801 |
| **SE (Soft Exudates)** | **98.56%** | **100.00%** | **98.54%** | **44.80%** | **61.88%** | **44.80%** | 4,601 | 5,669 | 0 | 382,946 |

### 4. Aggregate Multi-Lesion Metrics

| Metric Type | Pixel Accuracy | Sensitivity (Recall) | Specificity | Precision | Dice Score (F1) | IoU (Jaccard) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Macro-Average** *(Unweighted class mean)* | **91.50%** | **73.93%** | **91.46%** | **46.41%** | **30.42%** | **21.27%** |
| **Micro-Average** *(Global pixel aggregation)* | **91.50%** | **94.39%** | **91.48%** | **6.27%** | **11.77%** | **6.25%** |

### 5. Clinical & Technical Lesion Findings
1. **Hard Exudates (EX) & Soft Exudates (SE)**: Good segmentation performance with Dice scores of **54.69%** and **61.88%** and sensitivities $\ge 96\%$.
2. **Hemorrhages (HE)**: High sensitivity (**99.62%**) but exhibits high false positive area (precision **2.63%**), causing over-segmentation.
3. **Microaneurysms (MA)**: Under-segmented due to extreme class imbalance (only 480 true positive pixels out of 393,216, or 0.12% area), requiring focal loss or patch-based fine-tuning.

---

## Limitations & Integrity Disclaimers

1. **Dataset Representation**: Ground-truth metrics reflect the benchmark test/validation splits of the respective open-access datasets (APTOS 2019, DRIVE, IDRiD). Performance on field cameras with variable optical artifacts may vary.
2. **Local Sample Annotations**: The local sample images in `test_images/` serve as integration test vectors and visual verification targets; they lack accompanying ground-truth pixel masks on disk.
3. **Clinical Intended Use**: This system is designed as an automated triage and decision-support tool. All classifications and segmentation overlays must be reviewed by certified ophthalmological professionals prior to clinical decision-making.

---

## Generated Artifacts in `evaluation_outputs/`

- `aptos_confusion_matrix.png`: Annotated 2x2 confusion matrix for Referable DR classification.
- `drive_metrics_barchart.png`: Comprehensive bar chart of DRIVE segmentation metrics.
- `idrid_per_class_metrics.png`: Grouped bar chart comparing Dice, IoU, Sensitivity, and Precision across all 4 lesion classes.
- `idrid_confusion_matrices.png`: 4-panel pixel-level confusion matrix heatmaps.
- `model_comparison_benchmark.png`: Comparative overview chart across all 3 AI models.
- `model_metrics.json`: Machine-readable JSON repository of all evaluated numbers.
