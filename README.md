[README (3).md](https://github.com/user-attachments/files/31144202/README.3.md)
# Explainable AI for Breast Cancer Detection Using Mammograms

An explainable, multimodal deep learning pipeline for breast cancer classification on the **CBIS-DDSM** dataset — comparing multiple backbone architectures, fusing image features with structured clinical data, and explaining every prediction with **Grad-CAM, LIME, and SHAP**.

> ECS7036P Individual Project — Applications track

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Results](#results)
- [Explainability](#explainability)
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

1. A **baseline** image-only classifier, comparing **ResNet50, EfficientNetV2-S, and ViT-B/16** as candidate backbones under an equal training budget.
2. A **multimodal** extension that fuses mammogram image features with structured clinical data (breast density, mass shape/margins, assessment) via feature-level fusion.
3. A **layered explainability suite**: Grad-CAM and LIME for the image pathway, a quantitative faithfulness check on Grad-CAM's explanations, and SHAP for the clinical-feature pathway.
4. A **full evaluation suite**: not just accuracy, but ROC/PR curves, calibration, threshold sensitivity, and training-curve diagnostics.

## Pipeline

```
Dataset → Data Pre-processing → Feature Extraction → Multimodal Feature Fusion
        → Deep Learning Classifier → Explainable AI → Model Evaluation
        → Clinical Decision Support System → Decision Support for Radiologist
```

| Stage | What happens |
|---|---|
| **Data loading** | Merge CBIS-DDSM mass/calc CSVs, clean pathology labels to benign/malignant |
| **Patient-level split** | Train/val/test split by `patient_id` to prevent data leakage |
| **Image path resolution** | Map CSV `.dcm` paths to the actual `.jpg` files via SeriesUID matching |
| **Preprocessing** | Artefact removal, CLAHE contrast enhancement, resize to 224×224, cached to disk |
| **Augmentation** *(train only)* | Rotation, flip, zoom/crop, brightness jitter |
| **Baseline models** | ResNet50 / EfficientNetV2-S / ViT-B/16 (ImageNet-pretrained), image-only |
| **Multimodal model** | ResNet50 embedding + MLP-encoded clinical features, fused and classified |
| **Training** | Adam optimiser, early stopping on validation AUC, mixed precision, per-epoch history tracked |
| **Evaluation** | Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, confusion matrix, ROC/PR curves, calibration, threshold sensitivity |
| **Explainability** | Grad-CAM + LIME (image), faithfulness metrics (validates Grad-CAM), SHAP (clinical) |

## Results

Test-set performance from a full executed run (70/15/15 patient-level split, 20-epoch budget):

| Metric | Baseline (ResNet50) | Multimodal (image + clinical) |
|---|---|---|
| Accuracy | 66.0% | **82.3%** |
| Precision | 52.2% | **72.4%** |
| Recall | 55.3% | **81.2%** |
| Specificity | 71.9% | **82.9%** |
| F1 | 53.7% | **76.6%** |
| ROC-AUC | 0.708 | **0.911** |

**Takeaway**: fusing clinical data with image features produced a large, consistent improvement across every metric — most notably a jump to 0.911 ROC-AUC and 81.2% Recall, showing the multimodal model both discriminates and catches malignant cases substantially better than the image-only baseline.

**Architecture comparison** (equal 20-epoch budget for all three, `compare_architectures.py`):

| Backbone | Recall | ROC-AUC |
|---|---|---|
| ResNet50 | 0.553 | 0.708 |
| EfficientNetV2-S | 0.548 | **0.736** |
| ViT-B/16 | **0.604** | 0.696 |

Under a shorter (10-epoch) budget in an earlier run, ViT-B/16's Recall was only 0.228 — the equal-budget result shows this was a training-time artifact, not a fundamental architectural mismatch: ViT needs substantially more fine-tuning time than the CNN-based backbones to adapt from ImageNet pretraining to this small, domain-shifted medical dataset.

**Faithfulness** (`faithfulness.py`): Mean Deletion AUC 0.504 (lower = more faithful), Mean Insertion AUC 0.728 (higher = more faithful) — a meaningful gap in the right direction, indicating Grad-CAM's highlighted regions genuinely drive predictions rather than just looking plausible.

**Caveat — run-to-run variance**: baseline accuracy has varied noticeably across runs with the same seed and hyperparameters (66–72% observed), plausibly due to overfitting sensitivity (large train/val AUC gap by epoch 20) combined with which epoch gets checkpointed as "best." The multimodal model has been comparatively more stable. This is a single-split result — patient-level k-fold cross-validation would give a materially more robust estimate and is a natural next step.

## Explainability

**Image pathway** — *what regions drove the prediction?*
- **Grad-CAM**, applied to both the baseline and the multimodal model's image pathway (via `MultimodalImageWrapper`).
- **LIME**, a perturbation-based method — segments the image, hides combinations of segments, fits a local surrogate. Agreement with Grad-CAM on the same region is stronger evidence than either method alone.
- **Deletion/insertion faithfulness metrics** — quantitatively tests Grad-CAM's explanations rather than just displaying them.

**Clinical-feature pathway** — *which clinical variables drove the prediction?*
- **SHAP** (`KernelExplainer`), with the image branch's embedding computed once and cached to avoid re-running the full CNN forward pass per perturbation (which previously caused OOM/crashes). Reports both a per-sample beeswarm plot and an aggregate global-importance bar plot.

**Putting it together** — `clinical_decision_support.py` assembles a case's prediction, Grad-CAM heatmap, and active clinical findings into a single combined view, answering the project's proposal pipeline (`... -> Model Evaluation -> Clinical Decision Support System -> Decision Support for Radiologist`) by reusing the pieces above rather than introducing a new model. This is a demonstrative assembly step, not a deployed clinical tool.

## Project structure

```
.
├── breast_cancer_xai.ipynb            # End-to-end Colab/Kaggle-ready notebook (recommended entry point)
├── config.py                          # Central config: paths, hyperparameters
├── requirements.txt
├── src/
│   ├── data_preprocessing.py          # Artefact removal, CLAHE, resize, augmentation
│   ├── dataset.py                     # PyTorch Datasets, patient-level split, path resolution, caching
│   ├── models.py                       # ResNet50/EfficientNetV2/ViT baselines + multimodal fusion + checkpoint loader
│   ├── train.py                        # Training loop w/ per-epoch history + shared data-prep
│   ├── compare_architectures.py        # ResNet50 vs. EfficientNetV2-S vs. ViT-B/16 comparison + training-curve plot
│   ├── compare_baseline_multimodal.py  # Baseline vs. multimodal comparison table + bar chart
│   ├── evaluate.py                     # Metrics, confusion matrix, ROC/PR curves, calibration, threshold sweep
│   ├── gradcam.py                      # Grad-CAM (baseline + multimodal via wrapper)
│   ├── lime_explain.py                 # LIME image explanations
│   ├── faithfulness.py                 # Deletion/insertion faithfulness metrics for Grad-CAM
│   ├── shap_explain.py                 # SHAP explainability (embedding-cached, crash-safe)
│   ├── clinical_decision_support.py    # Combines prediction + Grad-CAM + clinical findings into one view
│   └── utils.py                        # Seeding, checkpointing, early stopping
└── outputs/
    ├── checkpoints/                    # Saved model weights (not committed — see note below)
    └── figures/                        # All plots: confusion matrices, ROC/PR curves, calibration,
                                         # training curves, Grad-CAM/LIME/SHAP/faithfulness, comparison bar charts
```

### A note on trained-weight checkpoints

`.pt` checkpoints saved by `train.py` include both the model weights and the optimizer state (for resuming training), which can push file size well past GitHub's 100MB limit. If you want to publish trained weights:

```python
import torch
torch.save(model.state_dict(), "baseline_weights_only.pt")  # weights only, much smaller
```

Then load with `models.load_trained_model(path, mode, clinical_input_dim)`, or `model.load_state_dict(torch.load(path))` directly. For large files, prefer [Git LFS](https://git-lfs.com/) or attaching them as a GitHub Release asset rather than committing directly.

## Getting started

### Option A — Notebook (recommended, Colab/Kaggle-friendly)

1. Open `25733547 (5).ipynb` in Google Colab or Kaggle Notebooks.
2. Set the runtime to GPU.
3. Run the cells top to bottom. The dataset-location cell auto-detects (or downloads via `kagglehub`) the CBIS-DDSM files — see [Dataset](#dataset) below.

### Option B — Scripts

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt

export BC_DATA_ROOT=/path/to/cbis-ddsm   # folder containing csv/ and jpeg/

# 1. Train the ResNet50 baseline and the multimodal model
python -m src.train --mode baseline --epochs 20
python -m src.train --mode multimodal --epochs 20

# 2. Evaluate each individually (metrics + ROC/PR/calibration/threshold plots)
python -m src.evaluate --checkpoint outputs/checkpoints/baseline_best.pt --mode baseline
python -m src.evaluate --checkpoint outputs/checkpoints/multimodal_best.pt --mode multimodal

# 3. Compare baseline vs. multimodal side by side (table + bar chart)
python -m src.compare_baseline_multimodal \
    --baseline_checkpoint outputs/checkpoints/baseline_best.pt \
    --multimodal_checkpoint outputs/checkpoints/multimodal_best.pt

# 4. Compare backbone architectures (reuses the baseline checkpoint above if present)
python -m src.compare_architectures --epochs 20

# 5. Explainability
python -m src.gradcam --checkpoint outputs/checkpoints/baseline_best.pt --mode baseline --num_samples 8
python -m src.gradcam --checkpoint outputs/checkpoints/multimodal_best.pt --mode multimodal --num_samples 8
python -m src.lime_explain --checkpoint outputs/checkpoints/baseline_best.pt --num_samples 4
python -m src.faithfulness --checkpoint outputs/checkpoints/baseline_best.pt --num_samples 5
python -m src.shap_explain --checkpoint outputs/checkpoints/multimodal_best.pt

# 6. Clinical decision support view (prediction + Grad-CAM + clinical findings, combined)
python -m src.clinical_decision_support --checkpoint outputs/checkpoints/multimodal_best.pt --num_samples 3
```

## Dataset

This project uses the **Curated Breast Imaging Subset of DDSM (CBIS-DDSM)**, via its preprocessed JPEG export on Kaggle:

🔗 https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

The dataset is **not included in this repository** (several GB). Download it and set `BC_DATA_ROOT` as an environment variable (or edit `config.py`) to point at the extracted folder containing `csv/` and `jpeg/`.

**Note on image paths**: the CBIS-DDSM CSVs reference original `.dcm` file paths, but the Kaggle JPEG export renames the actual files inside each folder. `dataset.py`'s `load_and_resolve()` handles this automatically by matching on the DICOM SeriesInstanceUID folder name rather than trusting the filename.

## Tech stack

- **PyTorch** / **Torchvision** — ResNet50, EfficientNetV2, ViT transfer learning
- **OpenCV** — CLAHE, artefact removal
- **scikit-learn** — metrics, calibration, one-hot encoding, patient-level splitting
- **SHAP**, **LIME**, **scikit-image** — explainability
- **Matplotlib / Seaborn** — evaluation and explanation plots

## Limitations & future work

- Relies on a single public dataset (CBIS-DDSM), which may limit generalisability to other imaging equipment/populations.
- Observed run-to-run variance in baseline performance — patient-level k-fold cross-validation would give a more robust performance estimate than a single split.
- Threshold tuning (choosing a cutoff targeting a specific recall, e.g. ≥95%) is a natural next step given the strong underlying ROC-AUC.
- Future work: extend fusion beyond simple feature concatenation (e.g. attention-based fusion), and explore counterfactual or case-based explanations as a further clinically-oriented explainability layer.

## References

[1] Cancer Research UK. (2022). Breast cancer statistics. In Cancer Research UK. Cancer Research UK. https://www.cancerresearchuk.org/health-professional/cancer-statistics/statistics-by-cancer-type/breast-cancer

[2] Murty, P. S. R. C., Anuradha, C., Naidu, P. A., Mandru, D., Ashok, M., Atheeswaran, A., Rajeswaran, N., Saravanan, V. (2024). Integrative hybrid deep learning for enhanced breast cancer diagnosis: leveraging theWisconsin Breast Cancer Database and the CBIS-DDSM dataset. Scientific Reports, 14(1). https://doi.org/10.1038/s41598-024-74305-8

[3] Fatima-Zahrae Nakach, Idri, A., & Evgin Goceri. (2024). A comprehensive investigation of multimodal deep learning fusion strategies for breast cancer classification. Artificial Intelligence Review, 57(12). https://doi.org/10.1007/s10462-024-10984-z

[4] Ghasemi, A., Hashtarkhani, S., Schwartz, D. L., & Shaban‐Nejad, A. (2024). Explainable artificial intelligence in breast cancer detection and risk prediction: A systematic scoping review. Cancer Innovation, 3(5). https://doi.org/10.1002/cai2.136

[5] CBIS-DDSM: Breast Cancer Image Dataset. (n.d.). In www.kaggle.com. Retrieved August 17, 2026, from https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

[6] Ahmed, S., Naira Elazab, El-Gayar, M. M., Elmogy, M., & Fouda, Y. M. (2025). Multi-Scale Vision Transformer with Optimized Feature Fusion for Mammographic Breast Cancer Classification. Diagnostics, 15(11), 1361–1361. https://doi.org/10.3390/diagnostics15111361

[7] Al-Hejri, A. M., Sable, A. H., Al-Tam, R. M., Al-antari, M. A., Alshamrani, S. S., Alshmrany, K. M., & Alatebi, W. (2025). A hybrid explainable federated-based vision transformer framework for breast cancer prediction via risk factors. Scientific Reports, 15(1). https://doi.org/10.1038/s41598-025-96527-0

[8]Ali, A., Alghamdi, M., Marzuki, S., Tengku Din, T. A., Yamin, M. S., Alrashidi, M., Alkhazi, I., & Ahmed, N. (2025). Exploring AI Approaches for Breast Cancer Detection and Diagnosis: A Review Article. Breast Cancer: Targets and Therapy, Volume 17, 927–947. https://doi.org/10.2147/bctt.s550307

[9]Ahn, J. S., Shin, S., Yang, S.-A., Park, E., Kim, K. H., Cho, S. I., Ock, C. Y., & Kim, S. (2023). Artificial Intelligence in Breast Cancer Diagnosis and Personalized Medicine. Journal of Breast Cancer, 26(5). https://doi.org/10.4048/jbc.2023.26.e45

[10]Zhang, Y., Liu, Y.-L., Nie, K., Zhou, J., Chen, Z., Chen, J.-H., Wang, X., Kim, B., Parajuli, R., Mehta, R. S., Wang, M., & Su, M.-Y. (2023). Deep Learning-based Automatic Diagnosis of Breast Cancer on MRI Using Mask R-CNN for Detection Followed by ResNet50 for Classification. Academic Radiology, 30 Suppl 2(Suppl 2), S161–S171. https://doi.org/10.1016/j.acra.2022.12.038

[11] Sossavi, E., Roy, C., & Molière, S. (2026). Artificial intelligence in breast cancer screening: A systematic review and meta-analysis of integration strategies. European Journal of Radiology Open, 16, 100727. https://doi.org/10.1016/j.ejro.2026.100727

[12] Sandeep Saharan, Wani, N. A., Chatterji, S., Kumar, N., & Abdullah Mohammed Almuhaideb. (2025). A Deep Learning and Explainable Artificial Intelligence based Scheme for Breast Cancer Detection. Scientific Reports, 15(1). https://doi.org/10.1038/s41598-024-80535-7

[13]Hernström, V., Viktoria Josefsson, Sartor, H., Schmidt, D., Larsson, A.-M., Solveig Hofvind, Andersson, I., Rosso, A., Hagberg, O., & Lång, K. (2025). Screening performance and characteristics of breast cancer detected in the Mammography Screening with Artificial Intelligence trial (MASAI): a randomised, controlled, parallel-group, non-inferiority, single-blinded, screening accuracy study. The Lancet Digital Health, 7(3). https://doi.org/10.1016/s2589-7500(24)00267-x

[14] Lauritzen, A. D., von Euler-Chelpin, M. C., Lynge, E., Vejborg, I., Nielsen, M., Karssemeijer, N., & Lillholm, M. (2023). Assessing Breast Cancer Risk by Combining AI for Lesion Detection and Mammographic Texture. Radiology, 308(2). https://doi.org/10.1148/radiol.230227

[15] Añez, D., Conti, G., Uriarte, J. J., Serrano-Olmedo, J.-J., Martínez-Murillo, R., & Casanova-Carvajal, O. (2025). Artificial Intelligence Pipeline for Mammography-Based Breast Cancer Detection: An Integrated Systematic Review and Large-Scale Experimental Validation. Medicina, 61(12), 2237. https://doi.org/10.3390/medicina61122237





