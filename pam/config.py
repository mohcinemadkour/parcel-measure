"""Shared constants for the Property Area Measurement (PAM) prototype."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CANONICAL_GSD = 0.15          # metres per pixel the model works at
MIN_GSD, MAX_GSD = 0.05, 0.60  # accepted input range (FR-02)
SQFT_PER_M2 = 10.7639
FT_PER_M = 3.28084

CLASSES = ["other", "lawn", "roof", "hardscape", "tree"]
CLASS_COLORS = {          # RGBA overlay colours
    "other": (0, 0, 0, 0),
    "lawn": (76, 175, 80, 150),
    "roof": (229, 57, 53, 150),
    "hardscape": (158, 158, 158, 150),
    "tree": (27, 94, 32, 150),
}
N_CLASSES = len(CLASSES)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TILE = 512       # inference window (px at canonical GSD)
OVERLAP = 64

MODEL_DIR = Path(os.environ.get("PAM_MODEL_DIR", ROOT / "models"))
_FP32, _INT8W = MODEL_DIR / "pam_unet.onnx", MODEL_DIR / "pam_unet_int8w.onnx"
# fp32 model if present, otherwise the int8-weight model shipped in the repo
# (same predictions within 0.1 point of test accuracy; 24.6 MB instead of 97.7 MB)
MODEL_ONNX = Path(os.environ["PAM_MODEL_FILE"]) if os.environ.get("PAM_MODEL_FILE") else (_FP32 if _FP32.exists() else _INT8W)
MODEL_CARD = MODEL_DIR / "model_card.json"

# Review-flag thresholds (spec section 7)
MIN_CONFIDENCE = 0.80
MAX_CANOPY_SHARE = 0.35
MAX_RELIABLE_GSD = 0.30
MIN_REGION_M2 = 1.0
HOLE_FILL_M2 = 0.5
SIMPLIFY_M = 0.3
DEFAULT_PITCH = 6.0  # rise per 12 of run
