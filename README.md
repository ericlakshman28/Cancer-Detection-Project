# 🩺 Explainable AI for Breast Cancer Detection Using Mammograms

An explainable, multimodal deep learning pipeline for breast cancer classification on the **CBIS-DDSM** dataset — combining a ResNet50 image classifier with structured clinical data, and explaining every prediction with **Grad-CAM** and **SHAP**.

> ECS7036P Individual Project — Applications track
> **Author:** Eric Kamalendran

---

## Table of Contents

- [Overview](#overview)
- [Why explainability matters here](#why-explainability-matters-here)
- [Pipeline](#pipeline)
- [Results](#results)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Dataset](#dataset)
- [Tech stack](#tech-stack)
- [Limitations & future work](#limitations--future-work)
- [References](#references)
- [License](#license)

---

## Overview

Breast cancer is one of the leading causes of cancer-related deaths among women worldwide, and early, accurate diagnosis significantly improves survival. Deep learning models can classify mammograms with high accuracy, but they typically operate as "black boxes" — offering little insight into *why* a prediction was made, which limits clinical trust and adoption.

This project builds:

1. A **baseline** image-only classifier (ResNet50, transfer learning) for benign vs. malignant classification.
2. A **multimodal** extension that fuses mammogram image features with structured clinical data (breast density, mass shape/margins, assessment) via feature-level fusion.
3. An **explainability layer** — Grad-CAM highlights the image regions driving each prediction, and SHAP attributes the multimodal model's output to individual clinical features.

The goal is a reproducible, interpretable pipeline that could plausibly sit inside a **clinical decision-support tool**, not just a leaderboard score.

## Why explainability matters here

A model that says "malignant" with no explanation is of limited use to a radiologist who has to justify a diagnosis. This project treats interpretability as a first-class requirement, not an afterthought:

- **Grad-CAM** shows *where* in the mammogram the model is looking.
- **SHAP** shows *which clinical variables* (e.g. breast density, mass margins) pushed the prediction toward malignant vs. benign.
- **Recall is explicitly prioritised** in evaluation, since missing a true malignant case (a false negative) is far costlier in a screening context than a false alarm.

## Pipeline

```
Dataset → Data Pre-processing → Feature Extraction → Multimodal Feature Fusion
        → Deep Learning Classifier → Explainable AI → Model Evaluation
        → Clinical Decision Support System → Decision Support for Radiologist
```

**Stage by stage:**

| Stage | What happens |
|---|---|
| **Data loading** | Merge CBIS-DDSM mass/calc CSVs, clean pathology labels to benign/malignant |
| **Patient-level split** | Train/val/test split by `patient_id` to prevent data leakage |
| **Image path resolution** | Map CSV `.dcm` paths to the actual `.jpg` files via SeriesUID matching |
| **Preprocessing** | Artefact removal, CLAHE contrast enhancement, resize to 224×224, cached to disk |
| **Augmentation** *(train only)* | Rotation, flip, zoom/crop, brightness jitter |
| **Baseline model** | ResNet50 (ImageNet-pretrained), fine-tuned, image-only |
| **Multimodal model** | ResNet50 embedding + MLP-encoded clinical features, fused and classified |
| **Training** | Adam optimiser, early stopping on validation AUC, mixed precision |
| **Evaluation** | Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, confusion matrix, ROC curve |
| **Explainability** | Grad-CAM (image regions) + SHAP (clinical feature attribution) |

## Results

> _Fill in after running the pipeline on your machine/Colab — metrics depend on your train/val/test split and hardware._

| Metric | Baseline (image-only) | Multimodal (image + clinical) |
|---|---|---|
| Accuracy | — | — |
| Precision | — | — |
| Recall | — | — |
| Specificity | — | — |
| F1 | — | — |
| ROC-AUC | — | — |

Confusion matrices, ROC curves, Grad-CAM overlays, and SHAP summary plots are saved to `outputs/figures/` after running the notebook/scripts.

## Project structure

```
.
├── breast_cancer_xai.ipynb     # End-to-end Colab-ready notebook (recommended entry point)
├── config.py                   # Central config: paths, hyperparameters
├── requirements.txt
├── src/
│   ├── data_preprocessing.py   # Artefact removal, CLAHE, resize, augmentation
│   ├── dataset.py              # PyTorch Datasets, patient-level split, path resolution
│   ├── models.py                # ResNet50 baseline + multimodal fusion model
│   ├── train.py                 # Training loop (baseline & multimodal)
│   ├── evaluate.py              # Metrics, confusion matrix, ROC curve
│   ├── gradcam.py               # Grad-CAM implementation
│   ├── shap_explain.py          # SHAP explainability for clinical features
│   └── utils.py                 # Seeding, checkpointing, early stopping
└── outputs/
    ├── checkpoints/             # Saved model weights
    └── figures/                 # Confusion matrices, ROC curves, Grad-CAM, SHAP plots
```

## Getting started

### Option A — Notebook (recommended, Colab-friendly)

1. Open `breast_cancer_xai.ipynb` in Google Colab.
2. Set the runtime to GPU (`Runtime → Change runtime type → GPU`).
3. Run the cells top to bottom. The dataset-location cell auto-detects (or downloads via `kagglehub`) the CBIS-DDSM files — see [Dataset](#dataset) below.

### Option B — Scripts

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt

# 1. Train the image-only ResNet50 baseline
python -m src.train --mode baseline --epochs 20

# 2. Train the multimodal (image + clinical) model
python -m src.train --mode multimodal --epochs 20

# 3. Evaluate a trained checkpoint
python -m src.evaluate --checkpoint outputs/checkpoints/baseline_best.pt --mode baseline

# 4. Generate Grad-CAM visualisations
python -m src.gradcam --checkpoint outputs/checkpoints/baseline_best.pt --num_samples 8

# 5. Generate SHAP explanations for the multimodal model's clinical branch
python -m src.shap_explain --checkpoint outputs/checkpoints/multimodal_best.pt
```

## Dataset

This project uses the **Curated Breast Imaging Subset of DDSM (CBIS-DDSM)**, via its preprocessed JPEG export on Kaggle:

🔗 https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

The dataset is **not included in this repository** (several GB). Download it and point `DATA_ROOT` in `config.py` (or the notebook's config cell) at the extracted folder. The notebook includes an auto-detection/auto-download step using `kagglehub`.

## Tech stack

- **PyTorch** / **Torchvision** — ResNet50 transfer learning
- **OpenCV** — CLAHE, artefact removal
- **scikit-learn** — metrics, one-hot encoding, patient-level splitting
- **SHAP** — clinical feature attribution
- **Matplotlib / Seaborn** — evaluation plots

## Limitations & future work

- Relies on a single public dataset (CBIS-DDSM), which may limit generalisability to other imaging equipment/populations.
- Future work: evaluate on additional datasets, explore Vision Transformer / EfficientNetV2 backbones, and extend fusion beyond simple feature concatenation (e.g. attention-based fusion).

## References

1. Liao, L., & Aagaard, E. M. (2024). *An open codebase for enhancing transparency in deep learning-based breast cancer diagnosis utilizing CBIS-DDSM data.* Scientific Reports, 14(1). https://doi.org/10.1038/s41598-024-78648-0
2. Ghasemi, A., Hashtarkhani, S., Schwartz, D. L., & Shaban-Nejad, A. (2024). *Explainable artificial intelligence in breast cancer detection and risk prediction: A systematic scoping review.* Cancer Innovation, 3(5). https://doi.org/10.1002/cai2.136
3. Nakach, F.-Z., Idri, A., & Goceri, E. (2024). *A comprehensive investigation of multimodal deep learning fusion strategies for breast cancer classification.* Artificial Intelligence Review, 57(12). https://doi.org/10.1007/s10462-024-10984-z
4. Murty, P. S. R. C., et al. (2024). *Integrative hybrid deep learning for enhanced breast cancer diagnosis: leveraging the Wisconsin Breast Cancer Database and the CBIS-DDSM dataset.* Scientific Reports, 14(1). https://doi.org/10.1038/s41598-024-74305-8
5. Muhammad, D., & Bendechache, M. (2024). *Unveiling the black box: a systematic review of explainable artificial intelligence in medical image analysis.* Computational and Structural Biotechnology Journal, 24, 542–560. https://doi.org/10.1016/j.csbj.2024.08.005
6. CBIS-DDSM: Breast Cancer Image Dataset. Kaggle. https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

## License

This project is for academic coursework (ECS7036P). Add a license here if you intend to open-source it (e.g. MIT).
