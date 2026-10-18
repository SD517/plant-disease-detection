"""Central configuration for the Plant Leaf Disease Detection project."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"          # put the downloaded PlantVillage folder here
SPLIT_DIR = DATA_DIR / "split"      # created by `python main.py prepare`
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# ---- Data -------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15   # stratified, group-aware

# ---- Models (transfer learning from ImageNet) -------------------------
MODEL_NAMES = ["mobilenetv2", "resnet50"]
DROPOUT = 0.3
USE_CLASS_WEIGHTS = True            # PlantVillage is imbalanced

# Phase 1: backbone frozen, train only the new classification head
HEAD_EPOCHS = 8
HEAD_LR = 1e-3
# Phase 2: unfreeze the top layers of the backbone and fine-tune gently
FINE_TUNE_EPOCHS = 12
FINE_TUNE_LR = 1e-5
UNFREEZE_LAYERS = {"mobilenetv2": 40, "resnet50": 30}
PATIENCE = 4                        # early-stopping patience (val_loss)

# ---- Web app ----------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.60         # below this the UI asks for a better photo
MAX_UPLOAD_MB = 8
