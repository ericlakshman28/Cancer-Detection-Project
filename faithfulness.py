"""
Deletion/insertion faithfulness metrics for Grad-CAM: quantitatively tests
whether the regions Grad-CAM highlights actually drive the prediction,
rather than just looking plausible. Matches Section 15 of
breast_cancer_xai.ipynb.

- Deletion: progressively blank the highest-ranked pixels, track how fast
  the predicted-class probability drops. Lower AUC = more faithful.
- Insertion: start blank, progressively reveal the highest-ranked pixels,
  track how fast probability recovers. Higher AUC = more faithful.

Usage:
    python -m src.faithfulness --checkpoint outputs/checkpoints/baseline_best.pt --num_samples 5
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
from src.models import build_model
from src.gradcam import GradCAM
from src.utils import get_device, load_checkpoint


def deletion_insertion_curves(model, image_tensor, cam, class_idx, device, steps=20):
    """image_tensor: normalised (C,H,W) tensor as fed to the model.
    cam: Grad-CAM heatmap (H,W), already resized to match image_tensor.
    Returns (deletion_probs, insertion_probs, deletion_auc, insertion_auc).
    """
    H, W = cam.shape
    order = np.argsort(-cam.flatten())  # most important pixel first
    n_pixels = H * W
    step_size = max(1, n_pixels // steps)
    baseline_value = image_tensor.mean().item()  # simple mean-fill baseline

    deletion_probs, insertion_probs = [], []
    with torch.no_grad():
        for step in range(steps + 1):
            k = min(step * step_size, n_pixels)
            idx = order[:k]
            ys, xs = np.unravel_index(idx, (H, W))

            deleted = image_tensor.clone()
            deleted[:, ys, xs] = baseline_value
            probs = torch.softmax(model(deleted.unsqueeze(0).to(device)), dim=1)
            deletion_probs.append(probs[0, class_idx].item())

            inserted = torch.full_like(image_tensor, baseline_value)
            inserted[:, ys, xs] = image_tensor[:, ys, xs]
            probs = torch.softmax(model(inserted.unsqueeze(0).to(device)), dim=1)
            insertion_probs.append(probs[0, class_idx].item())

    # Mean of the sampled curve approximates the area under it — simple
    # and avoids numpy-version differences in trapezoidal integration.
    deletion_auc = float(np.mean(deletion_probs))
    insertion_auc = float(np.mean(insertion_probs))
    return deletion_probs, insertion_probs, deletion_auc, insertion_auc


def evaluate_gradcam_faithfulness(model, dataset, gradcam_obj, device, n_samples=5, steps=20):
    del_aucs, ins_aucs = [], []
    for i in range(min(n_samples, len(dataset))):
        image, label = dataset[i]
        input_tensor = image.unsqueeze(0).to(device)
        cam, pred_class = gradcam_obj.generate(input_tensor)
        _, _, del_auc, ins_auc = deletion_insertion_curves(model, image.to(device), cam, pred_class, device, steps=steps)
        del_aucs.append(del_auc)
        ins_aucs.append(ins_auc)
    print(f"Mean Deletion AUC over {len(del_aucs)} samples (lower = more faithful): {np.mean(del_aucs):.4f}")
    print(f"Mean Insertion AUC over {len(ins_aucs)} samples (higher = more faithful): {np.mean(ins_aucs):.4f}")
    return del_aucs, ins_aucs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    device = get_device()
    _, _, _, _, _, test_ds, _, _ = prepare_data("baseline")

    model = build_model("baseline").to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    gradcam = GradCAM(model, model.backbone.layer4[-1])

    # Illustrate curves for one sample, then report the aggregate over several
    image0, label0 = test_ds[0]
    cam0, pred_class0 = gradcam.generate(image0.unsqueeze(0).to(device))
    deletion_probs, insertion_probs, del_auc0, ins_auc0 = deletion_insertion_curves(
        model, image0.to(device), cam0, pred_class0, device, steps=args.steps
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(np.linspace(0, 1, len(deletion_probs)), deletion_probs)
    axes[0].set_title(f"Deletion curve (AUC={del_auc0:.3f})")
    axes[0].set_xlabel("Fraction of pixels removed"); axes[0].set_ylabel("Predicted-class probability")
    axes[1].plot(np.linspace(0, 1, len(insertion_probs)), insertion_probs)
    axes[1].set_title(f"Insertion curve (AUC={ins_auc0:.3f})")
    axes[1].set_xlabel("Fraction of pixels inserted"); axes[1].set_ylabel("Predicted-class probability")
    plt.tight_layout()
    out_path = os.path.join(config.FIGURES_DIR, "gradcam_faithfulness.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved faithfulness curves to {out_path}")

    evaluate_gradcam_faithfulness(model, test_ds, gradcam, device, n_samples=args.num_samples, steps=args.steps)


if __name__ == "__main__":
    main()
