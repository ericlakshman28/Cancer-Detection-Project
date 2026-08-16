"""
Evaluation on the held-out test set: Accuracy, Precision, Recall,
Specificity, F1, ROC-AUC, confusion matrix, ROC curve, Precision-Recall
curve, metric-vs-threshold plot, and calibration curve. Matches Sections
9 and 11 of breast_cancer_xai.ipynb exactly.

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
    precision_recall_curve, average_precision_score,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.train import prepare_data
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


def plot_pr_curve(y_true, y_probs, out_path, title="Precision-Recall Curve"):
    """Under class imbalance, ROC-AUC can look better than the model's
    real-world usefulness suggests — the PR curve is the more honest
    picture for a screening task where the positive class is rarer."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return ap


def plot_metric_vs_threshold(y_true, y_probs, out_path, title="Metric vs. Decision Threshold"):
    """Shows how Accuracy/Precision/Recall/F1 trade off as the decision
    threshold moves, instead of only reporting the (somewhat arbitrary)
    default of 0.5."""
    thresholds = np.linspace(0.01, 0.99, 50)
    accs, precs, recs, f1s = [], [], [], []
    for t in thresholds:
        m = compute_metrics(y_true, y_probs, threshold=t)
        accs.append(m["accuracy"]); precs.append(m["precision"])
        recs.append(m["recall"]); f1s.append(m["f1"])
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, accs, label="Accuracy")
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, recs, label="Recall")
    plt.plot(thresholds, f1s, label="F1")
    plt.xlabel("Threshold"); plt.ylabel("Metric value"); plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_calibration_curve(y_true, y_probs, out_path, n_bins=10, title="Calibration Curve"):
    """Shows whether a predicted '80% malignant' actually corresponds to
    malignancy ~80% of the time — directly relevant to clinical trust,
    separately from raw discriminative performance."""
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["baseline", "multimodal"], required=True)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    device = get_device()
    _, _, _, _, _, test_ds, _, _ = prepare_data(args.mode)

    if args.mode == "baseline":
        model = build_model("baseline")
    else:
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
    pr_path = os.path.join(config.FIGURES_DIR, f"{args.mode}_pr_curve.png")
    threshold_path = os.path.join(config.FIGURES_DIR, f"{args.mode}_metric_vs_threshold.png")
    calibration_path = os.path.join(config.FIGURES_DIR, f"{args.mode}_calibration_curve.png")

    plot_confusion_matrix(metrics["confusion_matrix"], cm_path)
    plot_roc_curve(y_true, y_probs, metrics["roc_auc"], roc_path)
    ap = plot_pr_curve(y_true, y_probs, pr_path)
    plot_metric_vs_threshold(y_true, y_probs, threshold_path)
    plot_calibration_curve(y_true, y_probs, calibration_path)

    print(f"\nAverage Precision: {ap:.4f}")
    print(f"Saved confusion matrix to {cm_path}")
    print(f"Saved ROC curve to {roc_path}")
    print(f"Saved PR curve to {pr_path}")
    print(f"Saved metric-vs-threshold plot to {threshold_path}")
    print(f"Saved calibration curve to {calibration_path}")


if __name__ == "__main__":
    main()
