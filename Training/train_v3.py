# train_v3.py — IMPROVED Training Pipeline
#
# Key improvements over v2:
#   1. Image-based split (all pills in train) — already done by rebuild_and_evaluate.py
#   2. Higher LoRA rank (32) — more learning capacity
#   3. Partial backbone unfreezing (last 4 transformer blocks)
#   4. DINOv2 native resolution (518px) — RTX 6000 has enough VRAM
#   5. Enriched FAISS evaluation (train consumer + augmented refs)
#   6. Longer training with patience=15
#   7. Proper config saving for reproducibility

import os
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from collections import defaultdict
from PIL import Image
import faiss

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ══════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════
class Config:
    # Paths
    consumer_csv  = "data/consumer_mapping.csv"
    reference_csv = "data/reference_mapping.csv"
    save_dir      = "checkpoints_v3"

    # Model
    backbone_name    = "dinov2_vitb14"
    embedding_dim    = 512
    use_lora         = True
    lora_rank        = 32               # ← INCREASED from 16
    lora_alpha       = 64               # ← INCREASED (2x rank)
    sub_centers      = 3
    arcface_margin   = 0.3
    arcface_scale    = 30.0
    unfreeze_blocks  = 4                # ← NEW: unfreeze last 4 transformer blocks

    # Training
    epochs           = 60               # ← INCREASED from 40
    batch_size       = 48               # ← adjusted for 518px images
    img_size         = 518              # ← DINOv2 native resolution!
    grad_accum_steps = 2                # effective batch = 96

    # Learning rates
    lr_backbone      = 5e-6             # very low for unfrozen blocks
    lr_lora          = 5e-5
    lr_head          = 5e-4
    weight_decay     = 1e-4

    # Scheduler
    T_0              = 15
    T_mult           = 1
    eta_min          = 1e-7

    # Warmup
    warmup_epochs    = 5                # ← INCREASED from 2

    # Augmentation
    cutmix_prob      = 0.3
    mixup_prob        = 0.2
    cutmix_alpha     = 1.0
    mixup_alpha      = 0.2

    # Regularization
    label_smoothing  = 0.1
    grad_clip_norm   = 5.0

    # SWA
    use_swa          = True
    swa_start_epoch  = 45
    swa_lr           = 1e-5

    # Early stopping
    patience         = 15               # ← INCREASED from 10

    # Hardware
    num_workers      = 8
    mixed_precision  = True

    # Evaluation
    eval_every       = 2                # evaluate every 2 epochs (faster)
    top_k            = [1, 3, 5, 10]

    # Enriched evaluation
    num_ref_aug      = 5                # augmented versions per reference


# ══════════════════════════════════════════════════════
# TRANSFORMS
# ══════════════════════════════════════════════════════
def get_train_transform(img_size):
    return transforms.Compose([
        transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])


def get_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_aug_transform(img_size):
    return transforms.Compose([
        transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ══════════════════════════════════════════════════════
# CUTMIX & MIXUP
# ══════════════════════════════════════════════════════
def cutmix(images, labels, alpha=1.0):
    batch_size = images.size(0)
    indices = torch.randperm(batch_size, device=images.device)
    lam = np.random.beta(alpha, alpha)
    H, W = images.shape[2], images.shape[3]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
    lam = 1 - ((y2 - y1) * (x2 - x1) / (H * W))
    return images, labels, labels[indices], lam


def mixup(images, labels, alpha=0.2):
    indices = torch.randperm(images.size(0), device=images.device)
    lam = max(np.random.beta(alpha, alpha), 1 - np.random.beta(alpha, alpha))
    mixed = lam * images + (1 - lam) * images[indices]
    return mixed, labels, labels[indices], lam


# ══════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════
class PillDataset(Dataset):
    def __init__(self, csv_path, split, img_size):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = get_train_transform(img_size) if split == "train" else get_eval_transform(img_size)
        self.class_counts = self.df["pill_id"].value_counts().to_dict()

    def get_sampler_weights(self):
        return torch.DoubleTensor([1.0 / self.class_counts[l] for l in self.df["pill_id"]])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["img_path"]).convert("RGB")
        except Exception:
            return self.__getitem__(random.randint(0, len(self.df) - 1))
        return self.transform(img), int(row["pill_id"])


class SimpleDataset(Dataset):
    def __init__(self, paths, ids, img_size, augment=False, num_aug=1):
        self.paths = paths
        self.ids = ids
        self.num_aug = num_aug
        self.transform = get_aug_transform(img_size) if augment else get_eval_transform(img_size)

    def __len__(self):
        return len(self.paths) * self.num_aug

    def __getitem__(self, idx):
        i = idx // self.num_aug
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.transform(img), self.ids[i]


# ══════════════════════════════════════════════════════
# MODEL (same as v2 but with backbone unfreezing)
# ══════════════════════════════════════════════════════
import math

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        in_f, out_f = base_layer.in_features, base_layer.out_features
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.base_layer(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


class GeMPool(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)
        return x


class SubCenterArcFace(nn.Module):
    def __init__(self, emb_dim, num_classes, K=3, margin=0.3, scale=30.0):
        super().__init__()
        self.K, self.num_classes = K, num_classes
        self.margin, self.scale = margin, scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * K, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)
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
    def __init__(self, num_classes, cfg):
        super().__init__()

        # Load DINOv2
        print(f"  Loading {cfg.backbone_name}...")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", cfg.backbone_name, pretrained=True
        )
        self.feature_dim = self.backbone.embed_dim

        # Freeze entire backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze last N transformer blocks
        if cfg.unfreeze_blocks > 0:
            total_blocks = len(self.backbone.blocks)
            for i in range(total_blocks - cfg.unfreeze_blocks, total_blocks):
                for p in self.backbone.blocks[i].parameters():
                    p.requires_grad = True
            print(f"  Unfroze last {cfg.unfreeze_blocks}/{total_blocks} transformer blocks")

        # Apply LoRA to attention layers
        if cfg.use_lora:
            replaced = 0
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
                        LoRALinear(module, rank=cfg.lora_rank, alpha=cfg.lora_alpha))
                replaced += 1
            print(f"  Applied LoRA (rank={cfg.lora_rank}) to {replaced} layers")

        # GeM pooling
        self.gem_pool = GeMPool(p=3.0)

        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, cfg.embedding_dim),
            nn.BatchNorm1d(cfg.embedding_dim),
        )

        # ArcFace
        self.arcface = SubCenterArcFace(
            cfg.embedding_dim, num_classes, cfg.sub_centers,
            cfg.arcface_margin, cfg.arcface_scale
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


# ══════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════
@torch.no_grad()
def extract_emb(model, loader, device):
    model.eval()
    embs, lbls = [], []
    for imgs, ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        embs.append(model(imgs).cpu().numpy())
        lbls.append(np.array(ids))
    return np.vstack(embs).astype("float32"), np.concatenate(lbls)


def evaluate_enriched(model, device, cfg, consumer_df):
    """Evaluate using enriched FAISS index (refs + augmented + train consumer)."""
    model.eval()

    # Build enriched index
    ref_df = pd.read_csv(cfg.reference_csv)

    # A. Clean reference
    ds_ref = SimpleDataset(ref_df["img_path"].tolist(), ref_df["pill_id"].tolist(), cfg.img_size)
    ld_ref = DataLoader(ds_ref, batch_size=cfg.batch_size*2, num_workers=cfg.num_workers, pin_memory=True)
    emb_ref, lbl_ref = extract_emb(model, ld_ref, device)

    # B. Augmented reference
    ds_aug = SimpleDataset(ref_df["img_path"].tolist(), ref_df["pill_id"].tolist(),
                           cfg.img_size, augment=True, num_aug=cfg.num_ref_aug)
    ld_aug = DataLoader(ds_aug, batch_size=cfg.batch_size*2, num_workers=cfg.num_workers, pin_memory=True)
    emb_aug, lbl_aug = extract_emb(model, ld_aug, device)

    # C. Train consumer images
    train_df = consumer_df[consumer_df["split"] == "train"]
    ds_train = SimpleDataset(train_df["img_path"].tolist(), train_df["pill_id"].tolist(), cfg.img_size)
    ld_train = DataLoader(ds_train, batch_size=cfg.batch_size*2, num_workers=cfg.num_workers, pin_memory=True)
    emb_train, lbl_train = extract_emb(model, ld_train, device)

    # Combine
    all_emb = np.vstack([emb_ref, emb_aug, emb_train])
    all_lbl = np.concatenate([lbl_ref, lbl_aug, lbl_train])
    faiss.normalize_L2(all_emb)

    index = faiss.IndexFlatIP(all_emb.shape[1])
    index.add(all_emb)

    # Query: val set
    val_df = consumer_df[consumer_df["split"] == "val"]
    ds_val = SimpleDataset(val_df["img_path"].tolist(), val_df["pill_id"].tolist(), cfg.img_size)
    ld_val = DataLoader(ds_val, batch_size=cfg.batch_size*2, num_workers=cfg.num_workers, pin_memory=True)
    qry_emb, qry_lbl = extract_emb(model, ld_val, device)
    faiss.normalize_L2(qry_emb)

    _, indices = index.search(qry_emb, 10)

    results = {}
    for k in cfg.top_k:
        correct = sum(qry_lbl[i] in all_lbl[indices[i, :k]] for i in range(len(qry_lbl)))
        results[f"top{k}"] = 100 * correct / len(qry_lbl)

    return results, index, all_emb, all_lbl


def train():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Auto-tune for GPU
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem >= 40:
            cfg.batch_size = 48
            cfg.img_size = 518
        elif gpu_mem >= 20:
            cfg.batch_size = 24
            cfg.img_size = 518
        elif gpu_mem >= 10:
            cfg.batch_size = 16
            cfg.img_size = 224
            cfg.grad_accum_steps = 4
        else:
            cfg.batch_size = 8
            cfg.img_size = 224
            cfg.grad_accum_steps = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("PILL RECOGNITION V3 — IMPROVED TRAINING")
    print("=" * 70)
    print(f"  Device          : {device}")
    if torch.cuda.is_available():
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM            : {gpu_mem:.1f} GB")
    print(f"  Image size      : {cfg.img_size}")
    print(f"  Batch size      : {cfg.batch_size} (effective: {cfg.batch_size * cfg.grad_accum_steps})")
    print(f"  LoRA rank       : {cfg.lora_rank}")
    print(f"  Unfreeze blocks : {cfg.unfreeze_blocks}")

    # Load data
    print(f"\n📦 Loading data...")
    consumer_df = pd.read_csv(cfg.consumer_csv)
    num_classes = int(consumer_df["pill_id"].max()) + 1
    print(f"  Total classes   : {num_classes}")

    train_set = PillDataset(cfg.consumer_csv, "train", cfg.img_size)
    sampler = WeightedRandomSampler(train_set.get_sampler_weights(),
                                     len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size,
                               sampler=sampler, num_workers=cfg.num_workers,
                               pin_memory=True, drop_last=True,
                               persistent_workers=(cfg.num_workers > 0))
    print(f"  Train           : {len(train_set)} images, {len(train_loader)} batches")

    # Build model
    print(f"\n🏗️  Building model...")
    model = PillModelV3(num_classes, cfg).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total_p:,}")
    print(f"  Trainable params: {train_p:,} ({100*train_p/total_p:.2f}%)")

    # Optimizer with 3 param groups
    backbone_params, lora_params, head_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(p)
        elif "backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.lr_backbone, "name": "backbone"})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": cfg.lr_lora, "name": "lora"})
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.lr_head, "name": "head"})

    optimizer = optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.eta_min)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = GradScaler("cuda") if cfg.mixed_precision and device.type == "cuda" else None

    # SWA
    swa_model = AveragedModel(model) if cfg.use_swa else None
    swa_scheduler = SWALR(optimizer, swa_lr=cfg.swa_lr) if cfg.use_swa else None

    print(f"\n  Param groups:")
    for g in param_groups:
        n = sum(p.numel() for p in g["params"])
        print(f"    {g['name']:>10}: {n:>10,} params, lr={g['lr']}")

    # Training loop
    best_top1 = 0
    best_epoch = 0
    patience_count = 0
    history = []

    print(f"\n🚀 Starting training for {cfg.epochs} epochs...")
    print("=" * 70)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, correct, total = 0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # CutMix / MixUp
            use_cut = np.random.rand() < cfg.cutmix_prob
            use_mix = (not use_cut) and np.random.rand() < cfg.mixup_prob

            if use_cut:
                images, la, lb, lam = cutmix(images, labels, cfg.cutmix_alpha)
            elif use_mix:
                images, la, lb, lam = mixup(images, labels, cfg.mixup_alpha)

            with autocast(device_type="cuda", enabled=cfg.mixed_precision):
                if use_cut or use_mix:
                    log_a, _ = model(images, la)
                    log_b, _ = model(images, lb)
                    loss = lam * criterion(log_a, la) + (1 - lam) * criterion(log_b, lb)
                    logits, used_labels = log_a, la
                else:
                    logits, _ = model(images, labels)
                    loss = criterion(logits, labels)
                    used_labels = labels
                loss = loss / cfg.grad_accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * images.size(0) * cfg.grad_accum_steps
            with torch.no_grad():
                correct += (logits.argmax(1) == used_labels).sum().item()
                total += images.size(0)

            if (batch_idx + 1) % 25 == 0:
                print(f"    Batch {batch_idx+1}/{len(train_loader)} — "
                      f"loss: {total_loss/total:.4f}, acc: {100*correct/total:.2f}%")

        # Scheduler
        if cfg.use_swa and epoch >= cfg.swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        elapsed = time.time() - t0
        avg_loss = total_loss / total
        acc = 100 * correct / total
        print(f"\nEpoch {epoch}/{cfg.epochs} — loss: {avg_loss:.4f}, "
              f"acc: {acc:.2f}%, time: {elapsed:.0f}s")

        # Evaluate
        if epoch % cfg.eval_every == 0:
            print("  📊 Evaluating with enriched index...")
            results, _, _, _ = evaluate_enriched(model, device, cfg, consumer_df)

            for k, v in sorted(results.items()):
                print(f"    {k:<6}: {v:.2f}%")

            history.append({"epoch": epoch, "loss": avg_loss, "acc": acc, **results})
            pd.DataFrame(history).to_csv(os.path.join(cfg.save_dir, "history.csv"), index=False)

            top1 = results["top1"]
            if top1 > best_top1 + 0.001:
                best_top1 = top1
                best_epoch = epoch
                patience_count = 0

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_top1": best_top1,
                    "results": results,
                    "config": {k: v for k, v in vars(cfg).items()
                               if not k.startswith("_")},
                }, os.path.join(cfg.save_dir, "best_model.pth"))
                print(f"  ⭐ New best! top1={best_top1:.2f}%")
            else:
                patience_count += 1
                print(f"  No improvement ({patience_count}/{cfg.patience})")

            if patience_count >= cfg.patience:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break

        print("-" * 70)

    # ── SWA Final ─────────────────────────────────────
    if cfg.use_swa and swa_model and epoch >= cfg.swa_start_epoch:
        print("\n📊 SWA: updating batch norm...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_results, _, _, _ = evaluate_enriched(swa_model, device, cfg, consumer_df)
        print("  SWA results:")
        for k, v in sorted(swa_results.items()):
            print(f"    {k:<6}: {v:.2f}%")

        if swa_results.get("top1", 0) > best_top1:
            torch.save({
                "epoch": "swa",
                "model_state_dict": swa_model.state_dict(),
                "best_top1": swa_results["top1"],
                "results": swa_results,
                "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
            }, os.path.join(cfg.save_dir, "best_model_swa.pth"))
            print(f"  ⭐ SWA is better! Saved.")

    # ── Final Test ────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 FINAL TEST EVALUATION")
    print("=" * 70)

    ck = torch.load(os.path.join(cfg.save_dir, "best_model.pth"),
                    map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])

    # Build enriched index for test
    model.eval()
    ref_df = pd.read_csv(cfg.reference_csv)

    ds_ref = SimpleDataset(ref_df["img_path"].tolist(), ref_df["pill_id"].tolist(), cfg.img_size)
    ld_ref = DataLoader(ds_ref, batch_size=cfg.batch_size*2, num_workers=cfg.num_workers, pin_memory=True)
    emb_ref, lbl_ref = extract_emb(model, ld_ref, device)

    ds_aug = SimpleDataset(ref_df["img_path"].tolist(), ref_df["pill_id"].tolist(),
                           cfg.img_size, augment=True, num_aug=cfg.num_ref_aug)
    ld_aug = DataLoader(ds_aug, batch_size=cfg.batch_size*2, num_workers=cfg.num_workers, pin_memory=True)
    emb_aug, lbl_aug = extract_emb(model, ld_aug, device)

    train_df = consumer_df[consumer_df["split"] == "train"]
    ds_tr = SimpleDataset(train_df["img_path"].tolist(), train_df["pill_id"].tolist(), cfg.img_size)
    ld_tr = DataLoader(ds_tr, batch_size=cfg.batch_size*2, num_workers=cfg.num_workers, pin_memory=True)
    emb_tr, lbl_tr = extract_emb(model, ld_tr, device)

    all_emb = np.vstack([emb_ref, emb_aug, emb_tr])
    all_lbl = np.concatenate([lbl_ref, lbl_aug, lbl_tr])
    faiss.normalize_L2(all_emb)
    index = faiss.IndexFlatIP(all_emb.shape[1])
    index.add(all_emb)

    # Test queries
    test_df = consumer_df[consumer_df["split"] == "test"]
    ds_test = SimpleDataset(test_df["img_path"].tolist(), test_df["pill_id"].tolist(), cfg.img_size)
    ld_test = DataLoader(ds_test, batch_size=cfg.batch_size*2, num_workers=cfg.num_workers, pin_memory=True)
    qry_emb, qry_lbl = extract_emb(model, ld_test, device)
    faiss.normalize_L2(qry_emb)
    _, idxs = index.search(qry_emb, 10)

    print(f"\n  {'Metric':<15} {'Accuracy':>10}")
    print(f"  {'-'*27}")
    for k in cfg.top_k:
        correct = sum(qry_lbl[i] in all_lbl[idxs[i, :k]] for i in range(len(qry_lbl)))
        acc = 100 * correct / len(qry_lbl)
        print(f"  top{k:<11} {acc:>9.2f}%")

    # Save final index
    faiss_dir = "faiss_index_v3"
    os.makedirs(faiss_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(faiss_dir, "pill_index.faiss"))

    meta_df = pd.DataFrame({"index_position": range(len(all_lbl)), "pill_id": all_lbl})
    names = dict(zip(consumer_df["pill_id"], consumer_df["drug_name"]))
    ref_names = dict(zip(ref_df["pill_id"], ref_df["drug_name"]))
    names.update(ref_names)
    meta_df["drug_name"] = meta_df["pill_id"].map(names)
    meta_df.to_csv(os.path.join(faiss_dir, "index_metadata.csv"), index=False)

    print(f"\n✅ Training complete!")
    print(f"   Best model: {cfg.save_dir}/best_model.pth")
    print(f"   FAISS index: {faiss_dir}/pill_index.faiss ({index.ntotal} vectors)")


if __name__ == "__main__":
    train()