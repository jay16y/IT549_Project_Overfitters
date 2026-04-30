# pill_model_v2.py — SOTA Pill Recognition Model
#
# Architecture:
#   DINOv2 ViT-B/14 (frozen + LoRA adapters)
#   → Multi-scale pooling (CLS + GeM patches)
#   → 512-d embedding
#   → Sub-center ArcFace (K=3 sub-centers per class)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════
# LoRA — Low-Rank Adaptation
# Trains only ~1% of parameters but matches full fine-tuning
# ══════════════════════════════════════════════════════
class LoRALinear(nn.Module):
    """
    LoRA layer: replaces a Linear layer with a frozen base + low-rank update.
    out = W_base @ x + (B @ A) @ x * (alpha / r)

    Only A and B are trainable, drastically reducing trainable params.
    """

    def __init__(self, base_layer, rank=8, alpha=16):
        super().__init__()
        self.base_layer = base_layer
        # Freeze base
        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # B starts at 0 → no change at init

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


def apply_lora_to_dinov2(model, rank=8, alpha=16, target_modules=("qkv", "proj")):
    """
    Apply LoRA to DINOv2's attention layers.
    Only the QKV projections and output projections get LoRA adapters.
    """
    replaced = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue

        # Find parent module
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        # Replace with LoRA
        new_module = LoRALinear(module, rank=rank, alpha=alpha)
        setattr(parent, parts[-1], new_module)
        replaced += 1

    print(f"  Replaced {replaced} linear layers with LoRA adapters (rank={rank})")
    return model


# ══════════════════════════════════════════════════════
# GeM Pooling — Generalized Mean Pooling
# Better than avg/max pooling for fine-grained retrieval
# ══════════════════════════════════════════════════════
class GeMPool(nn.Module):
    """
    GeM pooling: f(x) = (mean(x^p))^(1/p)
    p=1 → average pool, p=∞ → max pool
    Learnable p adapts to data.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (B, N, D) where N is num_patches
        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = x.mean(dim=1)  # average over patches
        x = x.pow(1.0 / self.p)
        return x


# ══════════════════════════════════════════════════════
# Sub-center ArcFace — handles intra-class variance
# Each pill class has K sub-centers to model different angles/lighting
# ══════════════════════════════════════════════════════
class SubCenterArcFace(nn.Module):
    """
    Sub-center ArcFace loss.
    Each class has K sub-centers — sample is matched to closest sub-center.
    This handles intra-class variance much better than vanilla ArcFace.

    Reference: Deng et al. "Sub-center ArcFace" (ECCV 2020)
    """

    def __init__(self, embedding_dim, num_classes, K=3, margin=0.3, scale=30.0):
        super().__init__()
        self.K = K
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Weights: (num_classes * K, embedding_dim)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * K, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity to ALL sub-centers: (B, num_classes * K)
        cosine = F.linear(embeddings, weights)

        # Reshape to (B, num_classes, K) and take MAX across sub-centers
        cosine = cosine.view(-1, self.num_classes, self.K)
        cosine, _ = cosine.max(dim=2)  # (B, num_classes)

        # Apply margin
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


# ══════════════════════════════════════════════════════
# MAIN MODEL
# ══════════════════════════════════════════════════════
class PillRecognitionModel(nn.Module):
    """
    SOTA Pill Recognition Model.

    Pipeline:
      Image (224×224)
        → DINOv2 ViT-B/14 (with LoRA adapters)
        → CLS token + GeM-pooled patch tokens
        → Concatenate (768 + 768 = 1536)
        → BatchNorm → Linear → 512-d embedding
        → Sub-center ArcFace (during training only)
    """

    def __init__(
        self,
        num_classes,
        embedding_dim=512,
        backbone_name="dinov2_vitb14",
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
        sub_centers=3,
        arcface_margin=0.3,
        arcface_scale=30.0,
    ):
        super().__init__()

        # ── Load DINOv2 backbone ──────────────────────
        print(f"Loading {backbone_name}...")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            backbone_name,
            pretrained=True,
        )

        # Get feature dim (768 for ViT-B/14)
        self.feature_dim = self.backbone.embed_dim
        print(f"  Backbone feature dim: {self.feature_dim}")

        # ── Apply LoRA OR full freeze ─────────────────
        if use_lora:
            # Freeze entire backbone first
            for p in self.backbone.parameters():
                p.requires_grad = False

            # Add LoRA adapters
            self.backbone = apply_lora_to_dinov2(
                self.backbone, rank=lora_rank, alpha=lora_alpha
            )
        else:
            # Full fine-tuning (more compute, more risk)
            for p in self.backbone.parameters():
                p.requires_grad = True

        # ── GeM pooling for patch tokens ──────────────
        self.gem_pool = GeMPool(p=3.0)

        # ── Embedding head ────────────────────────────
        # Input: CLS (768) + GeM(patches) (768) = 1536
        self.embedding_head = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # ── Sub-center ArcFace head ───────────────────
        self.arcface = SubCenterArcFace(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            K=sub_centers,
            margin=arcface_margin,
            scale=arcface_scale,
        )

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

    def get_embedding(self, x):
        """Extract L2-normalized embedding for FAISS retrieval."""
        # DINOv2 forward returns dict with x_norm_clstoken and x_norm_patchtokens
        out = self.backbone.forward_features(x)
        cls_token = out["x_norm_clstoken"]              # (B, 768)
        patch_tokens = out["x_norm_patchtokens"]        # (B, N, 768)

        # GeM pool over patch tokens
        gem_features = self.gem_pool(patch_tokens)      # (B, 768)

        # Concatenate CLS + GeM-pooled patches
        combined = torch.cat([cls_token, gem_features], dim=1)  # (B, 1536)

        # Project to embedding
        embedding = self.embedding_head(combined)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def forward(self, x, labels=None):
        """
        Training: pass labels → returns ArcFace logits + embedding
        Inference: no labels → returns embedding only
        """
        embedding = self.get_embedding(x)

        if labels is not None:
            logits = self.arcface(embedding, labels)
            return logits, embedding
        else:
            return embedding


def build_pill_model(num_classes, **kwargs):
    """Factory function."""
    model = PillRecognitionModel(num_classes=num_classes, **kwargs)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,} ({100*trainable/total:.2f}%)")

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = build_pill_model(
        num_classes=2047,
        embedding_dim=512,
        use_lora=True,
        lora_rank=8,
        sub_centers=3,
    )
    model = model.to(device)

    # Test forward pass
    x = torch.randn(4, 3, 224, 224).to(device)
    labels = torch.tensor([0, 1, 2, 3]).to(device)

    print("\nTesting training forward...")
    logits, emb = model(x, labels)
    print(f"  Logits shape    : {logits.shape}")     # (4, 2047)
    print(f"  Embedding shape : {emb.shape}")        # (4, 512)

    print("\nTesting inference forward...")
    model.eval()
    with torch.no_grad():
        emb = model(x)
    print(f"  Embedding shape : {emb.shape}")        # (4, 512)
    print(f"  Embedding norm  : {torch.norm(emb, dim=1)}")  # ~1.0
