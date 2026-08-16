"""
Loads both trained baseline and multimodal checkpoints, evaluates both,
and produces the side-by-side comparison table + bar chart matching
Section 11 of breast_cancer_xai.ipynb.

Usage:
    python -m src.compare_baseline_multimodal \\
        --baseline_checkpoint outputs/checkpoints/baseline_best.pt \\
        --multimodal_checkpoint outputs/checkpoints/multimodal_best.pt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.train import prepare_data
from src.evaluate import collect_predictions, compute_metrics
from src.models import build_model
from src.utils import get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--multimodal_checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    device = get_device()

    # Baseline
    _, _, _, _, _, test_ds_baseline, _, _ = prepare_data("baseline")
    baseline_model = build_model("baseline").to(device)
    load_checkpoint(args.baseline_checkpoint, baseline_model, map_location=device)
    test_loader_baseline = DataLoader(test_ds_baseline, batch_size=args.batch_size, shuffle=False)
    y_true, y_probs = collect_predictions(baseline_model, test_loader_baseline, device, "baseline")
    metrics = compute_metrics(y_true, y_probs)

    # Multimodal
    _, _, _, _, _, test_ds_mm, _, _ = prepare_data("multimodal")
    multimodal_model = build_model("multimodal", clinical_input_dim=test_ds_mm.clinical_feature_dim).to(device)
    load_checkpoint(args.multimodal_checkpoint, multimodal_model, map_location=device)
    test_loader_mm = DataLoader(test_ds_mm, batch_size=args.batch_size, shuffle=False)
    y_true_mm, y_probs_mm = collect_predictions(multimodal_model, test_loader_mm, device, "multimodal")
    metrics_mm = compute_metrics(y_true_mm, y_probs_mm)

    metric_keys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    comparison = pd.DataFrame({
        "Baseline (image-only)": {k: metrics[k] for k in metric_keys},
        "Multimodal (image+clinical)": {k: metrics_mm[k] for k in metric_keys},
    }).T

    print("\n=== Baseline vs. Multimodal ===")
    print(comparison.to_string())

    csv_path = os.path.join(config.OUTPUT_DIR, "baseline_vs_multimodal.csv")
    comparison.to_csv(csv_path)
    print(f"\nSaved comparison table to {csv_path}")

    # Bar chart
    x = np.arange(len(metric_keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, [metrics[k] for k in metric_keys], width, label="Baseline")
    ax.bar(x + width/2, [metrics_mm[k] for k in metric_keys], width, label="Multimodal")
    ax.set_xticks(x); ax.set_xticklabels(metric_keys, rotation=30)
    ax.set_ylabel("Score"); ax.set_title("Baseline vs. Multimodal — All Metrics")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    bar_path = os.path.join(config.FIGURES_DIR, "baseline_vs_multimodal_bar.png")
    plt.savefig(bar_path, dpi=150)
    print(f"Saved comparison bar chart to {bar_path}")


if __name__ == "__main__":
    main()
