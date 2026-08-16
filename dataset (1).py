"""
PyTorch Dataset classes for CBIS-DDSM, matching Sections 4, 4b, 4c, and 5
of breast_cancer_xai.ipynb exactly (full mammograms, simple largest-file
path disambiguation, threaded image caching).
"""
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
import torch
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_preprocessing import preprocess_mammogram

LABEL_MAP = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}


def load_metadata() -> pd.DataFrame:
    """Loads and concatenates the mass/calc train+test CSVs into one
    DataFrame with a clean binary `label` column."""
    frames = []
    for path in config.CSV_FILES:
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
        else:
            print(f"[warn] metadata file not found, skipping: {path}")

    if not frames:
        raise FileNotFoundError(
            "No CBIS-DDSM CSV files found. Set BC_DATA_ROOT or edit config.py."
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[config.COL_PATHOLOGY, config.COL_IMAGE_PATH])
    df["label"] = df[config.COL_PATHOLOGY].str.upper().map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def patient_level_split(df: pd.DataFrame):
    """Splits by unique patient_id so no patient's images appear in more
    than one split (prevents data leakage)."""
    patients = df[config.COL_PATIENT_ID].unique()
    train_p, temp_p = train_test_split(
        patients, train_size=config.TRAIN_FRAC, random_state=config.RANDOM_SEED
    )
    relative_val = config.VAL_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    val_p, test_p = train_test_split(
        temp_p, train_size=relative_val, random_state=config.RANDOM_SEED
    )
    train_df = df[df[config.COL_PATIENT_ID].isin(train_p)].reset_index(drop=True)
    val_df = df[df[config.COL_PATIENT_ID].isin(val_p)].reset_index(drop=True)
    test_df = df[df[config.COL_PATIENT_ID].isin(test_p)].reset_index(drop=True)
    return train_df, val_df, test_df


def fit_clinical_encoder(train_df: pd.DataFrame):
    """Fits a one-hot encoder on clinical columns using only the training
    split (avoids leaking test-set category information)."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cols = [c for c in config.CLINICAL_FEATURE_COLUMNS if c in train_df.columns]
    encoder.fit(train_df[cols].astype(str))
    return encoder, cols


def build_uid_to_jpeg_map(jpeg_root: str) -> dict:
    """One-time scan of jpeg/: maps each DICOM series-UID folder name to
    the jpg file(s) inside it. The CBIS-DDSM CSVs store the *original
    DICOM* file path, but the Kaggle JPEG export renames files inside each
    folder — the folder name itself is preserved, so we resolve by that."""
    uid_map = {}
    for entry in os.scandir(jpeg_root):
        if entry.is_dir():
            jpgs = sorted(glob.glob(os.path.join(entry.path, "*.jpg")))
            if jpgs:
                uid_map[entry.name] = jpgs
    return uid_map


def resolve_image_paths(df: pd.DataFrame, uid_map: dict, path_column: str):
    """Resolves each row's original .dcm path to the matching real .jpg
    path via the SeriesUID folder (2nd-to-last path component)."""
    resolved = []
    for raw_path in df[path_column]:
        series_uid = str(raw_path).strip().replace("\\", "/").split("/")[-2]
        candidates = uid_map.get(series_uid)
        if not candidates:
            resolved.append(None)
        elif len(candidates) == 1:
            resolved.append(candidates[0])
        else:
            # Multiple files in the folder (e.g. cropped image + mask) —
            # the full-resolution mammogram is almost always the largest.
            resolved.append(max(candidates, key=os.path.getsize))
    return resolved


def load_and_resolve(path_column: str = config.COL_IMAGE_PATH) -> pd.DataFrame:
    """Loads metadata and resolves image paths against the actual jpeg/
    folder contents, dropping any rows that couldn't be matched."""
    df = load_metadata()
    uid_map = build_uid_to_jpeg_map(config.JPEG_ROOT)
    df["resolved_image_path"] = resolve_image_paths(df, uid_map, path_column)

    n_before = len(df)
    unmatched = df["resolved_image_path"].isna().sum()
    df = df.dropna(subset=["resolved_image_path"]).reset_index(drop=True)
    print(
        f"Resolved {len(df)}/{n_before} images "
        f"({unmatched} dropped — no matching jpg found under jpeg/)"
    )
    return df


def build_image_cache(df: pd.DataFrame, cache_dir: str = config.CACHE_DIR, num_workers=None) -> pd.DataFrame:
    """Preprocesses every image once (artefact removal, CLAHE, resize) and
    caches the result as a .npy file. OpenCV releases the GIL during its
    C++ calls, so a thread pool gives real parallelism here."""
    os.makedirs(cache_dir, exist_ok=True)
    num_workers = num_workers or min(16, (os.cpu_count() or 4) * 4)

    def _process(item):
        idx, src_path = item
        cache_path = os.path.join(cache_dir, f"{idx}.npy")
        if not os.path.exists(cache_path):
            img = preprocess_mammogram(src_path)
            np.save(cache_path, img)
        return idx, cache_path

    items = list(zip(df.index, df["resolved_image_path"]))
    results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process, item) for item in items]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Caching preprocessed images"):
            idx, cache_path = f.result()
            results[idx] = cache_path

    df = df.copy()
    df["cache_path"] = df.index.map(results)
    return df


class MammogramDataset(Dataset):
    """Image-only dataset for the ResNet50 baseline. Reads pre-processed
    images from the cache built by build_image_cache()."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.load(row["cache_path"])
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row["label"], dtype=torch.long)
        return image, label


class MultimodalMammogramDataset(Dataset):
    """Image + one-hot-encoded clinical feature dataset for the fusion model."""

    def __init__(self, df, encoder, clinical_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.encoder = encoder
        self.clinical_cols = clinical_cols
        self.transform = transform
        self.clinical_features = encoder.transform(
            self.df[clinical_cols].astype(str)
        ).astype(np.float32)

    def __len__(self):
        return len(self.df)

    @property
    def clinical_feature_dim(self) -> int:
        return self.clinical_features.shape[1]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.load(row["cache_path"])
        if self.transform:
            image = self.transform(image)
        clinical = torch.tensor(self.clinical_features[idx], dtype=torch.float32)
        label = torch.tensor(row["label"], dtype=torch.long)
        return image, clinical, label
