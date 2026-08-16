"""
Image preprocessing for mammograms: artefact removal, CLAHE contrast
enhancement, resize, and augmentation. Mirrors Section 3 of
breast_cancer_xai.ipynb exactly.
"""
import cv2
import numpy as np
from torchvision import transforms

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def remove_artefacts(image: np.ndarray) -> np.ndarray:
    """Keeps only the largest connected bright region (the breast tissue),
    dropping scan labels / borders. Expects a grayscale uint8 image."""
    _, thresh = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_and(image, image, mask=mask)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID_SIZE
    )
    return clahe.apply(image)


def preprocess_mammogram(image_path: str) -> np.ndarray:
    """Full pipeline for a single mammogram: artefact removal -> CLAHE ->
    resize -> grayscale-to-RGB (ResNet50 expects 3 channels)."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = remove_artefacts(image)
    image = apply_clahe(image)
    image = cv2.resize(
        image, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA
    )
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def get_train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomResizedCrop(
                config.IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )
