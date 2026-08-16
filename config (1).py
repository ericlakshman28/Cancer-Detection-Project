"""
Central configuration for the Breast Cancer XAI pipeline.
Mirrors breast_cancer_xai.ipynb's config cell exactly. Set BC_DATA_ROOT as
an environment variable, or edit DATA_ROOT below, to point at your local
copy of CBIS-DDSM (downloaded from Kaggle).
"""
import os

# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
DATA_ROOT = os.environ.get("BC_DATA_ROOT", "./data/cbis-ddsm")

CSV_FILES = [
    os.path.join(DATA_ROOT, "csv", "mass_case_description_train_set.csv"),
    os.path.join(DATA_ROOT, "csv", "mass_case_description_test_set.csv"),
    os.path.join(DATA_ROOT, "csv", "calc_case_description_train_set.csv"),
    os.path.join(DATA_ROOT, "csv", "calc_case_description_test_set.csv"),
]

JPEG_ROOT = os.path.join(DATA_ROOT, "jpeg")

# ---- CBIS-DDSM column names ----
COL_PATIENT_ID = "patient_id"
COL_PATHOLOGY = "pathology"                 # BENIGN / MALIGNANT / BENIGN_WITHOUT_CALLBACK
COL_IMAGE_PATH = "image file path"          # full mammogram — used for training
COL_CROPPED_IMAGE_PATH = "cropped image file path"   # tight lesion crop (available if you want to switch to it)
COL_BREAST_DENSITY = "breast_density"
COL_MASS_SHAPE = "mass shape"
COL_MASS_MARGINS = "mass margins"
COL_ASSESSMENT = "assessment"
CLINICAL_FEATURE_COLUMNS = [COL_BREAST_DENSITY, COL_MASS_SHAPE, COL_MASS_MARGINS, COL_ASSESSMENT]

# ------------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------------
IMAGE_SIZE = 224
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ------------------------------------------------------------------
# SPLITS
# ------------------------------------------------------------------
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42

# ------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------
BATCH_SIZE = 64
NUM_WORKERS = min(4, os.cpu_count() or 2)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
NUM_CLASSES = 2

# Epoch budget for the multi-backbone architecture comparison
# (Section 12 / compare_architectures.py). Kept equal to NUM_EPOCHS by
# default for a fair comparison — architectures like ViT typically need
# more fine-tuning time than CNNs to adapt from ImageNet to a small,
# visually different medical dataset, so a shorter budget systematically
# disadvantages them rather than reflecting their true ceiling.
COMPARISON_EPOCHS = NUM_EPOCHS

CLINICAL_HIDDEN_DIM = 64
FUSION_HIDDEN_DIM = 256

# ------------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------------
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
CACHE_DIR = "./image_cache"

for d in (OUTPUT_DIR, CHECKPOINT_DIR, FIGURES_DIR):
    os.makedirs(d, exist_ok=True)
