"""
Evaluation on the held-out test set: Accuracy, Precision, Recall,
Specificity, F1, ROC-AUC, confusion matrix, and ROC curve.
Matches Sections 9 and 11 of breast_cancer_xai.ipynb exactly.

Usage:
    python -m src.evaluate --checkpoint outputs/checkpoints/baseline_best.pt --mode baseline
    python -m src.evaluate --checkpoint outputs/checkpoints/multimodal_best.pt --mode multimodal
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import (
    load_and_resolve, patient_level_split, fit_clinical_encoder,
    build_image_cache, MammogramDataset, MultimodalMammogramDataset,
)
from src.data_preprocessing import get_eval_transforms
from src.models import build_model
from src.utils import get_device, load_checkpoint


@torch.no_grad()
def collect_predictions(model, loader, device, mode):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        if mode == "baseline":
            images, labels = batch
            logits = model(images.to(device))
        else:
            images, clinical, labels = batch
            logits = model(images.to(device), clinical.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_probs)


def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probs),
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
    }


def plot_confusion_matrix(cm, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_probs, auc, out_path, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["baseline", "multimodal"], required=True)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    device = get_device()
    df = load_and_resolve(config.COL_IMAGE_PATH)
    df = build_image_cache(df)
    train_df, _, test_df = patient_level_split(df)

    if args.mode == "baseline":
        test_ds = MammogramDataset(test_df, transform=get_eval_transforms())
        model = build_model("baseline")
    else:
        encoder, clinical_cols = fit_clinical_encoder(train_df)
        test_ds = MultimodalMammogramDataset(test_df, encoder, clinical_cols, transform=get_eval_transforms())
        model = build_model("multimodal", clinical_input_dim=test_ds.clinical_feature_dim)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = model.to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    y_true, y_probs = collect_predictions(model, test_loader, device, args.mode)
    metrics = compute_metrics(y_true, y_probs)

    print("\n=== Test set performance ===")
    for k in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]:
        marker = "  <-- emphasised (minimise false negatives)" if k == "recall" else ""
        print(f"{k:>12}: {metrics[k]:.4f}{marker}")

    cm_path = os.path.join(config.FIGURES_DIR, f"{args.mode}_confusion_matrix.png")
    roc_path = os.path.join(config.FIGURES_DIR, f"{args.mode}_roc_curve.png")
    plot_confusion_matrix(metrics["confusion_matrix"], cm_path)
    plot_roc_curve(y_true, y_probs, metrics["roc_auc"], roc_path)
    print(f"\nSaved confusion matrix to {cm_path}")
    print(f"Saved ROC curve to {roc_path}")


if __name__ == "__main__":
    main()
