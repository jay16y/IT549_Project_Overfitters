# app/config.py — Configuration for Pill Recognition Backend

import os

# Paths — all relative to pill_backend/ folder
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH      = os.path.join(BASE_DIR, "models", "best_model.pth")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "models", "pill_index.faiss")
METADATA_PATH   = os.path.join(BASE_DIR, "models", "index_metadata.csv")
REFERENCE_CSV   = os.path.join(BASE_DIR, "models", "reference_mapping.csv")

# Model settings
BACKBONE_NAME   = "dinov2_vitb14"
EMBEDDING_DIM   = 512
LORA_RANK       = 32
LORA_ALPHA      = 64
SUB_CENTERS     = 3
NUM_CLASSES     = 2047
IMG_SIZE        = 518

# API settings
TOP_K           = 5
MAX_FILE_SIZE   = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES   = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
