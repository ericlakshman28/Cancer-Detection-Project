"""
Clinical Decision Support View: assembles a case's prediction, Grad-CAM
heatmap, and active clinical findings into a single combined view.
Matches Section 17 of breast_cancer_xai.ipynb.

This is a demonstrative assembly step, not a new model or method — it
answers the proposal's pipeline diagram (Dataset -> ... -> Explainable AI
-> Model Evaluation -> Clinical Decision Support System -> Decision
Support for Radiologist) by reusing the multimodal model, the Grad-CAM
wrapper, and the clinical feature names already built by the other
scripts in this package, and presenting their outputs together the way a
radiologist would actually want to see them.

Usage:
    python -m src.clinical_decision_support --checkpoint outputs/checkpoints/multimodal_best.pt --num_samples 3
"""
import argparse
import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.train import prepare_data
from src.models import build_model
from src.gradcam import GradCAM, MultimodalImageWrapper, overlay_heatmap
from src.utils import get_device, load_checkpoint

CLASS_NAMES = {0: "Benign", 1: "Malignant"}


def clinical_decision_support_view(index, model, wrapper, gradcam, test_ds, feature_names, device,
                                    out_path=None, show=False):
    """Combines the multimodal model's prediction, its Grad-CAM heatmap,
    and this case's active clinical findings into one view.

    Returns a dict summarising the prediction, confidence, true label, and
    the clinical features that were "on" for this case — suitable for
    logging or further use, independent of the plot itself.
    """
    image, clinical, label = test_ds[index]
    input_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor, clinical.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0]
    pred_class = int(probs.argmax())
    confidence = float(probs[pred_class])

    wrapper.set_clinical(clinical.unsqueeze(0).to(device))
    cam, _ = gradcam.generate(input_tensor, class_idx=pred_class)
    overlay = overlay_heatmap(image, cam)

    clinical_np = clinical.numpy()
    active_features = [feature_names[i] for i in range(len(clinical_np)) if clinical_np[i] > 0.5]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(overlay)
    axes[0].set_title(
        f"Prediction: {CLASS_NAMES[pred_class]} ({confidence:.1%} confidence)\n"
        f"True label: {CLASS_NAMES[int(label)]}"
    )
    axes[0].axis("off")

    axes[1].axis("off")
    axes[1].set_title("Clinical Decision Support Summary")
    summary_lines = "Active clinical findings:\n\n" + "\n".join(f"\u2022 {f}" for f in active_features)
    axes[1].text(0.0, 0.95, summary_lines, fontsize=10, va="top", wrap=True, transform=axes[1].transAxes)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return {
        "index": index,
        "predicted_class": CLASS_NAMES[pred_class],
        "confidence": confidence,
        "true_label": CLASS_NAMES[int(label)],
        "active_clinical_features": active_features,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    device = get_device()
    _, _, _, _, _, test_ds, encoder, clinical_cols = prepare_data("multimodal")
    feature_names = encoder.get_feature_names_out(clinical_cols).tolist()

    model = build_model("multimodal", clinical_input_dim=test_ds.clinical_feature_dim).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    wrapper = MultimodalImageWrapper(model)
    target_layer = model.image_encoder[7][-1]  # last block of layer4
    gradcam = GradCAM(wrapper, target_layer)

    n = min(args.num_samples, len(test_ds))
    for i in range(n):
        out_path = os.path.join(config.FIGURES_DIR, f"clinical_decision_support_case{i}.png")
        result = clinical_decision_support_view(i, model, wrapper, gradcam, test_ds, feature_names, device,
                                                  out_path=out_path)
        print(result)
        print(f"Saved view to {out_path}")


if __name__ == "__main__":
    main()
