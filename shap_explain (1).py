"""
SHAP explainability for the multimodal model's clinical branch. Matches
Section 13 of breast_cancer_xai.ipynb exactly, including the
embedding-caching fix (avoids re-running the full ResNet50 forward pass
per SHAP perturbation, which previously caused OOM/crashes).

Usage:
    python -m src.shap_explain --checkpoint outputs/checkpoints/multimodal_best.pt
"""
import argparse
import os
import sys

import numpy as np
import shap
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import (
    load_and_resolve, patient_level_split, fit_clinical_encoder,
    build_image_cache, MultimodalMammogramDataset,
)
from src.data_preprocessing import get_eval_transforms
from src.models import build_model
from src.utils import get_device, load_checkpoint


@torch.no_grad()
def build_prediction_fn(model, fixed_image, device):
    """Runs ResNet50 on the fixed image exactly once, then returns a
    lightweight function that reuses that cached embedding for every SHAP
    perturbation — avoiding thousands of repeated CNN forward passes."""
    model.eval()
    img_tensor = fixed_image.unsqueeze(0).to(device)
    image_embedding = torch.flatten(model.image_encoder(img_tensor), 1)

    def predict(clinical_batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            clinical_tensor = torch.tensor(clinical_batch, dtype=torch.float32).to(device)
            n = clinical_tensor.shape[0]
            img_feats = image_embedding.repeat(n, 1)
            clin_feats = model.clinical_encoder(clinical_tensor)
            fused = torch.cat([img_feats, clin_feats], dim=1)
            logits = model.classifier(fused)
            return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    return predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_background", type=int, default=30)
    parser.add_argument("--num_explain", type=int, default=10)
    parser.add_argument("--nsamples", type=int, default=200)
    args = parser.parse_args()

    device = get_device()
    df = load_and_resolve(config.COL_IMAGE_PATH)
    df = build_image_cache(df)
    train_df, _, test_df = patient_level_split(df)
    encoder, clinical_cols = fit_clinical_encoder(train_df)

    test_ds = MultimodalMammogramDataset(test_df, encoder, clinical_cols, transform=get_eval_transforms())

    model = build_model("multimodal", clinical_input_dim=test_ds.clinical_feature_dim)
    model = model.to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    feature_names = encoder.get_feature_names_out(clinical_cols).tolist()

    n_bg = min(args.num_background, len(test_ds))
    background = test_ds.clinical_features[:n_bg]

    fixed_image, _, _ = test_ds[0]
    predict_fn = build_prediction_fn(model, fixed_image, device)

    explainer = shap.KernelExplainer(predict_fn, background)

    n_explain = min(args.num_explain, len(test_ds))
    to_explain = test_ds.clinical_features[:n_explain]
    shap_values = explainer.shap_values(to_explain, nsamples=args.nsamples)

    plt.figure()
    shap.summary_plot(shap_values, to_explain, feature_names=feature_names, show=False)
    out_path = os.path.join(config.FIGURES_DIR, "shap_clinical_summary.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved SHAP summary plot to {out_path}")


if __name__ == "__main__":
    main()
