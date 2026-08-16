"""
Training script for the ResNet50 baseline and the multimodal fusion model.
Matches Sections 7, 8, and 10 of breast_cancer_xai.ipynb, including
per-epoch training-curve tracking.

Usage:
    python -m src.train --mode baseline --epochs 20
    python -m src.train --mode multimodal --epochs 20
"""
import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import (
    load_and_resolve,
    patient_level_split,
    fit_clinical_encoder,
    build_image_cache,
    MammogramDataset,
    MultimodalMammogramDataset,
)
from src.data_preprocessing import get_train_transforms, get_eval_transforms
from src.models import build_model
from src.utils import set_seed, get_device, save_checkpoint, EarlyStopping


def run_epoch(model, loader, criterion, optimizer, device, mode, train: bool, scaler=None):
    model.train() if train else model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    use_amp = scaler is not None
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if train else "eval", leave=False):
            if mode == "baseline":
                images, labels = batch
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                images, clinical, labels = batch
                images = images.to(device, non_blocking=True)
                clinical = clinical.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images, clinical)
                    loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return avg_loss, auc


def train_model(mode, train_loader, val_loader, device, epochs=config.NUM_EPOCHS, lr=config.LEARNING_RATE,
                 clinical_input_dim=None, model_fn=None, ckpt_name=None):
    """model_fn (optional): a zero-arg callable returning a fresh model —
    used by compare_architectures.py to train alternative backbones
    without touching the default baseline/multimodal construction.
    ckpt_name (optional): override the checkpoint filename, so alternative
    backbones don't overwrite baseline_best.pt / multimodal_best.pt.

    Returns (model, best_ckpt_path, history) — history is a dict of
    per-epoch train/val loss and AUC, for plotting training curves.
    """
    model = (model_fn() if model_fn else build_model(mode, clinical_input_dim)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    best_ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name or mode}_best.pt")

    # Mixed precision: Tensor-Core GPUs run fp16 matmuls roughly 2x faster
    # than fp32, at essentially no accuracy cost for this task.
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    history = {"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_auc = run_epoch(model, train_loader, criterion, optimizer, device, mode, train=True, scaler=scaler)
        val_loss, val_auc = run_epoch(model, val_loader, criterion, optimizer, device, mode, train=False, scaler=scaler)
        scheduler.step(val_auc)

        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} train_auc={train_auc:.4f} "
            f"| val_loss={val_loss:.4f} val_auc={val_auc:.4f} | {time.time()-t0:.1f}s"
        )

        if early_stopping.step(val_auc):
            save_checkpoint(model, optimizer, epoch, val_auc, best_ckpt_path)
            print(f"  -> saved new best checkpoint (val_auc={val_auc:.4f})")
        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    return model, best_ckpt_path, history


def plot_training_curves(history, title_prefix="", save_path=None):
    """Plots train vs. val loss and ROC-AUC over epochs — shows whether a
    model is still improving, overfitting, or underfitting when early
    stopping kicked in."""
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_range, history["train_loss"], label="Train")
    axes[0].plot(epochs_range, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title_prefix} Loss"); axes[0].legend()

    axes[1].plot(epochs_range, history["train_auc"], label="Train")
    axes[1].plot(epochs_range, history["val_auc"], label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title(f"{title_prefix} ROC-AUC"); axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curves to {save_path}")
    plt.close(fig)


def prepare_data(mode):
    """Shared data-prep used by train.py, evaluate.py, compare_architectures.py, etc."""
    df = load_and_resolve(config.COL_IMAGE_PATH)
    df = build_image_cache(df)
    train_df, val_df, test_df = patient_level_split(df)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}  patients split, no overlap")

    if mode == "baseline":
        train_ds = MammogramDataset(train_df, transform=get_train_transforms())
        val_ds = MammogramDataset(val_df, transform=get_eval_transforms())
        test_ds = MammogramDataset(test_df, transform=get_eval_transforms())
        encoder, clinical_cols = None, None
    else:
        encoder, clinical_cols = fit_clinical_encoder(train_df)
        train_ds = MultimodalMammogramDataset(train_df, encoder, clinical_cols, transform=get_train_transforms())
        val_ds = MultimodalMammogramDataset(val_df, encoder, clinical_cols, transform=get_eval_transforms())
        test_ds = MultimodalMammogramDataset(test_df, encoder, clinical_cols, transform=get_eval_transforms())

    return train_df, val_df, test_df, train_ds, val_ds, test_ds, encoder, clinical_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "multimodal"], required=True)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = get_device()
    print(f"Using device: {device}")

    _, _, _, train_ds, val_ds, _, _, _ = prepare_data(args.mode)

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

    clinical_input_dim = train_ds.clinical_feature_dim if args.mode == "multimodal" else None
    model, ckpt_path, history = train_model(
        args.mode, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, clinical_input_dim=clinical_input_dim,
    )
    print(f"Training complete. Best checkpoint: {ckpt_path}")

    curve_path = os.path.join(config.FIGURES_DIR, f"{args.mode}_training_curves.png")
    plot_training_curves(history, title_prefix=args.mode.capitalize(), save_path=curve_path)


if __name__ == "__main__":
    main()
