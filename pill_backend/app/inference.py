# app/inference.py — Inference Engine
# Loads model + FAISS index once at startup, then handles queries

import torch
import numpy as np
import pandas as pd
import faiss
from PIL import Image
from torchvision import transforms
from io import BytesIO

from app.config import (
    MODEL_PATH, FAISS_INDEX_PATH, METADATA_PATH,
    REFERENCE_CSV, BACKBONE_NAME, EMBEDDING_DIM,
    LORA_RANK, LORA_ALPHA, SUB_CENTERS, NUM_CLASSES,
    IMG_SIZE, TOP_K,
)
from app.model import PillModelV3


class PillRecognitionEngine:
    """
    Singleton engine that loads model + FAISS index once,
    then processes pill image queries.
    """

    def __init__(self):
        self.model = None
        self.index = None
        self.metadata = None
        self.device = None
        self.transform = None
        self.loaded = False

    def load(self):
        """Load model, FAISS index, and metadata. Called once at startup."""
        if self.loaded:
            return

        print("Loading Pill Recognition Engine...")

        # Device — use CPU for deployment (GPU optional)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            self.device = torch.device("cpu")
            print("  Device: CPU")

        # Load model
        print(f"  Loading model from {MODEL_PATH}...")
        self.model = PillModelV3(
            num_classes=NUM_CLASSES,
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            embedding_dim=EMBEDDING_DIM,
            sub_centers=SUB_CENTERS,
            unfreeze_blocks=8,
        )

        checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"  Model loaded (epoch: {checkpoint.get('epoch', 'N/A')})")

        # Load FAISS index
        print(f"  Loading FAISS index from {FAISS_INDEX_PATH}...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"  FAISS index: {self.index.ntotal} vectors, dim={self.index.d}")

        # Load metadata
        print(f"  Loading metadata from {METADATA_PATH}...")
        self.metadata = pd.read_csv(METADATA_PATH)
        print(f"  Metadata: {len(self.metadata)} entries")

        # Load reference CSV for extra pill info (shape, color, imprint)
        # Load pill info from enriched index_metadata
        try:
            meta_df = pd.read_csv(METADATA_PATH)
            self.pill_info = {}
            for _, row in meta_df.drop_duplicates("pill_id").iterrows():
                self.pill_info[int(row["pill_id"])] = {
                    "ndc"      : str(row.get("ndc_clean", "")),
                    "drug_name": str(row.get("drug_name", "Unknown")),
                    "shape"    : str(row.get("shape", "")),
                    "colors"     : str(row.get("colors", "")),
                    "imprint"  : str(row.get("imprint", "")),
                    "size_mm"  : str(row.get("size_mm", "")),
                }
            print(f"  Pill info: {len(self.pill_info)} pills")
        except Exception as e:
            print(f"  Warning: Could not load reference CSV: {e}")
            self.pill_info = {}

        # Image transform (same as eval transform during training)
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.loaded = True
        print("Engine loaded successfully!\n")

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Convert raw image bytes to model-ready tensor."""
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(img)
        return tensor.unsqueeze(0)  # add batch dim

    @torch.no_grad()
    def get_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extract embedding from preprocessed image tensor."""
        image_tensor = image_tensor.to(self.device)
        embedding = self.model(image_tensor)
        return embedding.cpu().numpy().astype("float32")

    def search(self, embedding: np.ndarray, top_k: int = TOP_K) -> list:
        """Search FAISS index and return top-K results with metadata."""
        # Normalize query
        faiss.normalize_L2(embedding)

        # Search
        distances, indices = self.index.search(embedding, top_k * 3)

        # Deduplicate by pill_id (return unique pills only)
        results = []
        seen_pills = set()

        for i in range(distances.shape[1]):
            idx = indices[0, i]
            if idx < 0 or idx >= len(self.metadata):
                continue

            row = self.metadata.iloc[idx]
            pill_id = int(row["pill_id"])

            if pill_id in seen_pills:
                continue
            seen_pills.add(pill_id)

            # Get pill info
            info = self.pill_info.get(pill_id, {})

            result = {
                "rank": len(results) + 1,
                "pill_id": pill_id,
                "drug_name": info.get("drug_name", str(row.get("drug_name", "Unknown"))),
                "similarity": round(float(distances[0, i]) * 100, 2),
                "ndc": info.get("ndc", ""),
                "shape": info.get("shape", ""),
                "colors": info.get("colors", ""),
                "imprint": info.get("imprint", ""),
                "size_mm": info.get("size_mm", ""),
            }
            results.append(result)

            if len(results) >= top_k:
                break

        return results

    def predict(self, image_bytes: bytes, top_k: int = TOP_K) -> list:
        """Full pipeline: image bytes → top-K pill matches."""
        if not self.loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")

        # Preprocess
        tensor = self.preprocess(image_bytes)

        # Get embedding
        embedding = self.get_embedding(tensor)

        # Search
        results = self.search(embedding, top_k)

        return results


# Global singleton
engine = PillRecognitionEngine()
