"""
Compares ResNet50, EfficientNetV2-S, and ViT-B/16 as image-only baselines,
as named in the proposal. Matches Section 12 of breast_cancer_xai.ipynb,
including the training-curve comparison plot and comparison bar chart.

By default, auto-detects and reuses an existing ResNet50 checkpoint at
outputs/checkpoints/baseline_best.pt (e.g. from `python -m src.train
--mode baseline`) instead of retraining it — retraining ResNet50 from
scratch under a possibly-different epoch budget than your main baseline
would silently produce a *weaker* ResNet50 result than what you already
have, making the comparison unfair to it. Pass --force_retrain_resnet50
to override this. Note: if you reuse an existing checkpoint, no training
history is available for it, so it's omitted from the combined AUC curve
plot (but still included in the bar chart and comparison table).

Usage:
    python -m src.compare_architectures --epochs 20
    python -m src.compare_architectures --epochs 20 --resnet50_checkpoint outputs/checkpoints/baseline_best.pt
    python -m src.compare_architectures --epochs 20 --force_retrain_resnet50
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
from src.train import prepare_data, train_model
from src.evaluate import collect_predictions, compute_metrics
from src.models import build_image_backbone, build_model
from src.utils import set_seed, get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.COMPARISON_EPOCHS,
                         help="Epoch budget for ALL three architectures — keep this equal "
                              "across architectures for a fair comparison.")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--resnet50_checkpoint", default=None,
        help="Path to an already-trained ResNet50 baseline checkpoint. If omitted, "
             "auto-detects outputs/checkpoints/baseline_best.pt if present.",
    )
    parser.add_argument(
        "--force_retrain_resnet50", action="store_true",
        help="Retrain ResNet50 from scratch even if an existing checkpoint is found.",
    )
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = get_device()
    print(f"Using device: {device}")

    _, _, _, train_ds, val_ds, test_ds, _, _ = prepare_data("baseline")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    results = {}
    histories = {}

    # ResNet50: reuse an existing checkpoint whenever possible — see
    # module docstring for why retraining-by-default would be unfair.
    resnet_checkpoint = args.resnet50_checkpoint
    if resnet_checkpoint is None and not args.force_retrain_resnet50:
        default_ckpt = os.path.join(config.CHECKPOINT_DIR, "baseline_best.pt")
        if os.path.exists(default_ckpt):
            resnet_checkpoint = default_ckpt
            print(f"Auto-detected existing ResNet50 checkpoint at {default_ckpt} — "
                  f"reusing it instead of retraining. Pass --force_retrain_resnet50 "
                  f"to override.")

    if resnet_checkpoint and not args.force_retrain_resnet50:
        print("\n=== Loading existing ResNet50 checkpoint ===")
        resnet_model = build_model("baseline").to(device)
        load_checkpoint(resnet_checkpoint, resnet_model, map_location=device)
        # No history available for a reused checkpoint — omitted from the
        # combined training-curve plot below, but still in the bar chart.
    else:
        reason = "forced via --force_retrain_resnet50" if args.force_retrain_resnet50 else "no existing checkpoint found"
        print(f"\n=== Training resnet50 from scratch ({reason}) ===")
        resnet_model, _, resnet_history = train_model(
            "baseline", train_loader, val_loader, device,
            epochs=args.epochs, model_fn=lambda: build_image_backbone("resnet50"),
            ckpt_name="resnet50",
        )
        histories["resnet50"] = resnet_history
    y_true, y_probs = collect_predictions(resnet_model, test_loader, device, "baseline")
    results["resnet50"] = compute_metrics(y_true, y_probs)

    for backbone_name in ["efficientnet_v2_s", "vit_b_16"]:
        print(f"\n=== Training {backbone_name} ===")
        model, _, history = train_model(
            "baseline", train_loader, val_loader, device,
            epochs=args.epochs,
            model_fn=lambda name=backbone_name: build_image_backbone(name),
            ckpt_name=backbone_name,
        )
        y_true, y_probs = collect_predictions(model, test_loader, device, "baseline")
        results[backbone_name] = compute_metrics(y_true, y_probs)
        histories[backbone_name] = history

    metric_keys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    comparison = pd.DataFrame({
        name: {k: m[k] for k in metric_keys}
        for name, m in results.items()
    }).T

    print("\n=== Architecture comparison ===")
    print(comparison.to_string())

    csv_path = os.path.join(config.OUTPUT_DIR, "architecture_comparison.csv")
    comparison.to_csv(csv_path)
    print(f"\nSaved comparison table to {csv_path}")

    # Bar chart: all architectures across all six metrics
    arch_names = list(results.keys())
    x = np.arange(len(metric_keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(arch_names):
        ax.bar(x + (i - 1) * width, [results[name][k] for k in metric_keys], width, label=name)
    ax.set_xticks(x); ax.set_xticklabels(metric_keys, rotation=30)
    ax.set_ylabel("Score"); ax.set_title("Architecture Comparison — All Metrics")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    bar_path = os.path.join(config.FIGURES_DIR, "architecture_comparison_bar.png")
    plt.savefig(bar_path, dpi=150)
    print(f"Saved comparison bar chart to {bar_path}")

    # Combined validation AUC over epochs (only for architectures that
    # were actually trained this run, i.e. have a history)
    if histories:
        plt.figure(figsize=(8, 5))
        for name, history in histories.items():
            epochs_range = range(1, len(history["val_auc"]) + 1)
            plt.plot(epochs_range, history["val_auc"], marker="o", label=name)
        plt.xlabel("Epoch"); plt.ylabel("Validation ROC-AUC")
        plt.title("Architecture Comparison — Validation AUC over Training")
        plt.legend()
        plt.tight_layout()
        curve_path = os.path.join(config.FIGURES_DIR, "architecture_comparison_training_curves.png")
        plt.savefig(curve_path, dpi=150)
        print(f"Saved training-curve comparison to {curve_path}")


if __name__ == "__main__":
    main()
