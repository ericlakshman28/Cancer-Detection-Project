"""
Grad-CAM for ResNet50: highlights the mammogram regions that most
influenced the model's prediction. Matches Section 12 of
breast_cancer_xai.ipynb exactly.

Usage:
    python -m src.gradcam --checkpoint outputs/checkpoints/baseline_best.pt --num_samples 8
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import load_and_resolve, patient_level_split, build_image_cache, MammogramDataset
from src.data_preprocessing import get_eval_transforms
from src.models import build_model
from src.utils import get_device, load_checkpoint


class GradCAM:
    """Hooks the last convolutional block of a ResNet backbone and computes
    class-discriminative localisation maps."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def overlay_heatmap(image_tensor, cam, alpha=0.4):
    mean = np.array(config.IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(config.IMAGENET_STD).reshape(3, 1, 1)
    img = image_tensor.cpu().numpy() * std + mean
    img = np.clip(img.transpose(1, 2, 0), 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = (1 - alpha) * img + alpha * heatmap
    return np.clip(overlay, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    df = load_and_resolve(config.COL_IMAGE_PATH)
    df = build_image_cache(df)
    _, _, test_df = patient_level_split(df)
    test_ds = MammogramDataset(test_df, transform=get_eval_transforms())

    model = build_model("baseline").to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    gradcam = GradCAM(model, model.backbone.layer4[-1])

    n = min(args.num_samples, len(test_ds))
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    axes = axes.flatten()

    class_names = {0: "Benign", 1: "Malignant"}
    for i in range(n):
        image, label = test_ds[i]
        input_tensor = image.unsqueeze(0).to(device)
        cam, pred_class = gradcam.generate(input_tensor)
        overlay = overlay_heatmap(image, cam)

        axes[i].imshow(overlay)
        axes[i].set_title(
            f"True: {class_names[int(label)]} | Pred: {class_names[pred_class]}", fontsize=9,
        )
        axes[i].axis("off")

    plt.tight_layout()
    out_path = os.path.join(config.FIGURES_DIR, "gradcam_samples.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved Grad-CAM visualisations to {out_path}")


if __name__ == "__main__":
    main()
