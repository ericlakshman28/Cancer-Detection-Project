"""
Model architecture definitions, matching Section 6 of
breast_cancer_xai.ipynb exactly.

A .pt checkpoint saved by the training scripts only contains trained
WEIGHTS (a state_dict) — it does not contain the model architecture. This
file is required alongside any checkpoint to reload a trained model.
"""
import os
import sys
import torch
import torch.nn as nn
from torchvision import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ResNet50Baseline(nn.Module):
    """Image-only baseline: ResNet50 with the final layer replaced for
    binary classification (benign=0, malignant=1)."""

    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class ClinicalEncoder(nn.Module):
    """Small MLP encoding structured clinical features (breast density,
    mass shape, mass margins, assessment) into a dense embedding."""

    def __init__(self, input_dim, hidden_dim=config.CLINICAL_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultimodalFusionModel(nn.Module):
    """Feature-level fusion of ResNet50 image embeddings and clinical
    embeddings, followed by a classification head.

    `image_encoder` is the ResNet50 backbone with its final fc layer
    stripped: nn.Sequential(conv1, bn1, relu, maxpool, layer1, layer2,
    layer3, layer4, avgpool). index [7] is layer4 — used by Grad-CAM to
    hook the last conv block for this model too (see gradcam.py).

    IMPORTANT: `clinical_input_dim` must match the number of columns
    produced by the fitted OneHotEncoder on your clinical columns — check
    `train_ds_mm.clinical_feature_dim` if unsure.
    """

    def __init__(self, clinical_input_dim, num_classes=config.NUM_CLASSES, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.image_feature_dim = backbone.fc.in_features
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.clinical_encoder = ClinicalEncoder(clinical_input_dim)
        fusion_input_dim = self.image_feature_dim + config.CLINICAL_HIDDEN_DIM
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, config.FUSION_HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(config.FUSION_HIDDEN_DIM, num_classes),
        )

    def forward(self, image, clinical):
        img_feats = torch.flatten(self.image_encoder(image), 1)
        clin_feats = self.clinical_encoder(clinical)
        fused = torch.cat([img_feats, clin_feats], dim=1)
        return self.classifier(fused)


def build_model(mode: str, clinical_input_dim: int = None):
    """Factory: mode is 'baseline' or 'multimodal'."""
    if mode == "baseline":
        return ResNet50Baseline()
    elif mode == "multimodal":
        if clinical_input_dim is None:
            raise ValueError("clinical_input_dim is required for multimodal model")
        return MultimodalFusionModel(clinical_input_dim)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_image_backbone(name: str, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
    """Factory for comparing multiple pretrained backbones as image-only
    classifiers (see compare_architectures.py / notebook Section 12).
    Supports the three architectures named in the proposal: ResNet50,
    EfficientNetV2, and a Vision Transformer.
    """
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_v2_s(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
    elif name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.vit_b_16(weights=weights)
        net.heads.head = nn.Linear(net.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return net


def load_trained_model(checkpoint_path: str, mode: str, clinical_input_dim: int = None, device=None):
    """Convenience loader: builds the right architecture and loads a
    checkpoint's weights into it, ready for inference.

    Example:
        model = load_trained_model("outputs/checkpoints/baseline_best.pt", mode="baseline")
        model.eval()
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(mode, clinical_input_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model
