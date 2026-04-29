# app/model.py — DINOv2 + LoRA + Sub-center ArcFace (inference only)
# This is the exact model architecture from train_v3.py / resume_training.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=32, alpha=64):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        in_f = base_layer.in_features
        out_f = base_layer.out_features
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


class GeMPool(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)


class SubCenterArcFace(nn.Module):
    def __init__(self, emb_dim, num_classes, K=3, margin=0.3, scale=30.0):
        super().__init__()
        self.K = K
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * K, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, emb, labels):
        emb = F.normalize(emb, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        cos = F.linear(emb, w).view(-1, self.num_classes, self.K).max(dim=2)[0]
        sin = torch.sqrt(torch.clamp(1.0 - cos * cos, 0, 1))
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        one_hot = torch.zeros_like(cos).scatter_(1, labels.unsqueeze(1), 1)
        return ((one_hot * phi) + ((1.0 - one_hot) * cos)) * self.scale


class PillModelV3(nn.Module):
    """Exact same architecture as training — needed to load weights."""

    def __init__(self, num_classes, lora_rank=32, lora_alpha=64,
                 embedding_dim=512, sub_centers=3,
                 arcface_margin=0.3, arcface_scale=30.0,
                 unfreeze_blocks=8):
        super().__init__()

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
        )
        self.feature_dim = self.backbone.embed_dim

        # Freeze all
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze last N blocks (must match training)
        total_blocks = len(self.backbone.blocks)
        for i in range(total_blocks - unfreeze_blocks, total_blocks):
            for p in self.backbone.blocks[i].parameters():
                p.requires_grad = True

        # Apply LoRA
        for name, module in self.backbone.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(t in name for t in ("qkv", "proj")):
                continue
            parts = name.split(".")
            parent = self.backbone
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1],
                    LoRALinear(module, rank=lora_rank, alpha=lora_alpha))

        self.gem_pool = GeMPool(p=3.0)

        self.embedding_head = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        self.arcface = SubCenterArcFace(
            embedding_dim, num_classes, sub_centers,
            arcface_margin, arcface_scale
        )

    def get_embedding(self, x):
        out = self.backbone.forward_features(x)
        cls_tok = out["x_norm_clstoken"]
        patch_tok = out["x_norm_patchtokens"]
        gem = self.gem_pool(patch_tok)
        combined = torch.cat([cls_tok, gem], dim=1)
        emb = self.embedding_head(combined)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x, labels=None):
        emb = self.get_embedding(x)
        if labels is not None:
            return self.arcface(emb, labels), emb
        return emb
