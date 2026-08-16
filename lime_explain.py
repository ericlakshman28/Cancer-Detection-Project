"""
LIME explanations for the image-only baseline — a complementary,
perturbation-based explanation to Grad-CAM's gradient-based approach.
Matches Section 14 of breast_cancer_xai.ipynb.

Because the method is fundamentally different from Grad-CAM (segment +
perturb + fit a local surrogate, vs. backprop gradients), agreement
between the two on the same region is a stronger trust signal than
either method alone.

Usage:
    python -m src.lime_explain --checkpoint outputs/checkpoints/baseline_best.pt --num_samples 4
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.train import prepare_data
from src.data_preprocessing import preprocess_mammogram, get_eval_transforms
from src.models import build_model
from src.utils import get_device, load_checkpoint


def build_lime_predict_fn(model, device):
    def predict(images_batch: np.ndarray) -> np.ndarray:
        """images_batch: (N, H, W, 3) uint8 array of LIME's perturbed
        segments. Applies the same eval transform used at training time."""
        batch_tensors = [get_eval_transforms()(img.astype(np.uint8)) for img in images_batch]
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
    return predict


def main():
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--num_lime_samples", type=int, default=200,
                         help="Perturbations per explained image — cost scales linearly with this.")
    args = parser.parse_args()

    device = get_device()
    _, _, test_df, _, _, test_ds, _, _ = prepare_data("baseline")

    model = build_model("baseline").to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    predict_fn = build_lime_predict_fn(model, device)
    explainer = lime_image.LimeImageExplainer()
    class_names = {0: "Benign", 1: "Malignant"}

    n = min(args.num_samples, len(test_df))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i in range(n):
        row = test_df.iloc[i]
        raw_image = preprocess_mammogram(row["resolved_image_path"])  # HWC uint8

        explanation = explainer.explain_instance(
            raw_image, predict_fn, top_labels=1, hide_color=0, num_samples=args.num_lime_samples
        )
        pred_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            pred_label, positive_only=True, num_features=8, hide_rest=False
        )
        axes[i].imshow(mark_boundaries(temp / 255.0, mask))
        axes[i].set_title(f"LIME: predicted {class_names[pred_label]}", fontsize=9)
        axes[i].axis("off")

    plt.tight_layout()
    out_path = os.path.join(config.FIGURES_DIR, "lime_samples.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved LIME visualisations to {out_path}")


if __name__ == "__main__":
    main()
