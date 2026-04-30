# resume_training.py — Continue training from best_model.pth
# Loads checkpoint, continues with lower LR for 30 more epochs

import os, time, random, math
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
from PIL import Image
import faiss

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ══════════════════════════════════════════════════════
# CONFIG — lower LR for resume
# ══════════════════════════════════════════════════════
CONSUMER_CSV   = "data/consumer_mapping.csv"
REFERENCE_CSV  = "data/reference_mapping.csv"
CHECKPOINT     = "checkpoints_v3/best_model.pth"
SAVE_DIR       = "checkpoints_v3_resumed"
IMG_SIZE       = 518
BATCH_SIZE     = 16
GRAD_ACCUM     = 6
NUM_WORKERS    = 8
EXTRA_EPOCHS   = 40           # train 40 more epochs
PATIENCE       = 15

# Lower LRs for fine-tuning on top of existing weights
LR_BACKBONE    = 1e-6         # very low — backbone mostly frozen
LR_LORA        = 1e-5         # lower than original 5e-5
LR_HEAD        = 1e-4         # lower than original 5e-4
WEIGHT_DECAY   = 1e-4

# Unfreeze ALL backbone blocks this time
UNFREEZE_BLOCKS = 8          # was 4 before → now full backbone

CUTMIX_PROB    = 0.3
MIXUP_PROB     = 0.2
LABEL_SMOOTH   = 0.1
GRAD_CLIP      = 5.0
NUM_REF_AUG    = 10           # more augmentation than before (was 5)

USE_SWA        = True
SWA_START      = 25           # start SWA after 25 of the 40 extra epochs
SWA_LR         = 1e-6

# ══════════════════════════════════════════════════════
# TRANSFORMS
# ══════════════════════════════════════════════════════
def get_train_transform(img_size):
    return transforms.Compose([
        transforms.Resize((int(img_size*1.1), int(img_size*1.1))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.85,1.15)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1,1.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02,0.2)),
    ])

def get_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def get_aug_transform(img_size):
    return transforms.Compose([
        transforms.Resize((int(img_size*1.1), int(img_size*1.1))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1,1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

# ══════════════════════════════════════════════════════
# CUTMIX & MIXUP
# ══════════════════════════════════════════════════════
def cutmix(images, labels, alpha=1.0):
    B = images.size(0)
    idx = torch.randperm(B, device=images.device)
    lam = np.random.beta(alpha, alpha)
    H, W = images.shape[2], images.shape[3]
    cut_h = int(H * np.sqrt(1-lam))
    cut_w = int(W * np.sqrt(1-lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1,y2 = np.clip(cy-cut_h//2,0,H), np.clip(cy+cut_h//2,0,H)
    x1,x2 = np.clip(cx-cut_w//2,0,W), np.clip(cx+cut_w//2,0,W)
    images[:,:,y1:y2,x1:x2] = images[idx,:,y1:y2,x1:x2]
    lam = 1 - ((y2-y1)*(x2-x1)/(H*W))
    return images, labels, labels[idx], lam

def mixup(images, labels, alpha=0.2):
    idx = torch.randperm(images.size(0), device=images.device)
    lam = max(np.random.beta(alpha,alpha), 1-np.random.beta(alpha,alpha))
    return lam*images+(1-lam)*images[idx], labels, labels[idx], lam

# ══════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════
class PillDataset(Dataset):
    def __init__(self, csv_path, split, img_size):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"]==split].reset_index(drop=True)
        self.transform = get_train_transform(img_size) if split=="train" else get_eval_transform(img_size)
        self.class_counts = self.df["pill_id"].value_counts().to_dict()

    def get_sampler_weights(self):
        return torch.DoubleTensor([1.0/self.class_counts[l] for l in self.df["pill_id"]])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try: img = Image.open(row["img_path"]).convert("RGB")
        except: return self.__getitem__(random.randint(0, len(self.df)-1))
        return self.transform(img), int(row["pill_id"])


class SimpleDataset(Dataset):
    def __init__(self, paths, ids, transform):
        self.paths, self.ids, self.transform = paths, ids, transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try: img = Image.open(self.paths[idx]).convert("RGB")
        except: img = Image.new("RGB", (224,224))
        return self.transform(img), self.ids[idx]

# ══════════════════════════════════════════════════════
# MODEL (identical to train_v3.py)
# ══════════════════════════════════════════════════════
class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=32, alpha=64):
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
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0/self.p)


class SubCenterArcFace(nn.Module):
    def __init__(self, emb_dim, num_classes, K=3, margin=0.3, scale=30.0):
        super().__init__()
        self.K, self.num_classes = K, num_classes
        self.margin, self.scale = margin, scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes*K, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, emb, labels):
        emb = F.normalize(emb, p=2, dim=1)
        w   = F.normalize(self.weight, p=2, dim=1)
        cos = F.linear(emb, w).view(-1, self.num_classes, self.K).max(dim=2)[0]
        sin = torch.sqrt(torch.clamp(1.0 - cos*cos, 0, 1))
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        one_hot = torch.zeros_like(cos).scatter_(1, labels.unsqueeze(1), 1)
        return ((one_hot * phi) + ((1.0 - one_hot) * cos)) * self.scale


class PillModelV3(nn.Module):
    def __init__(self, num_classes, lora_rank=32, lora_alpha=64,
                 embedding_dim=512, sub_centers=3,
                 arcface_margin=0.3, arcface_scale=30.0,
                 unfreeze_blocks=12):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
        )
        self.feature_dim = self.backbone.embed_dim

        # Freeze all first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze last N blocks
        total_blocks = len(self.backbone.blocks)
        for i in range(total_blocks - unfreeze_blocks, total_blocks):
            for p in self.backbone.blocks[i].parameters():
                p.requires_grad = True
        print(f"  Unfroze {unfreeze_blocks}/{total_blocks} transformer blocks")

        # Apply LoRA
        replaced = 0
        for name, module in self.backbone.named_modules():
            if not isinstance(module, nn.Linear): continue
            if not any(t in name for t in ("qkv","proj")): continue
            parts = name.split(".")
            parent = self.backbone
            for p in parts[:-1]: parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(module, lora_rank, lora_alpha))
            replaced += 1
        print(f"  Applied LoRA rank={lora_rank} to {replaced} layers")

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
            embedding_dim, num_classes, sub_centers, arcface_margin, arcface_scale
        )

    def get_embedding(self, x):
        out = self.backbone.forward_features(x)
        cls = out["x_norm_clstoken"]
        pat = out["x_norm_patchtokens"]
        gem = self.gem_pool(pat)
        emb = self.embedding_head(torch.cat([cls, gem], dim=1))
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x, labels=None):
        emb = self.get_embedding(x)
        if labels is not None:
            return self.arcface(emb, labels), emb
        return emb


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════
@torch.no_grad()
def extract_emb(model, loader, device):
    model.eval()
    embs, lbls = [], []
    for imgs, ids in loader:
        embs.append(model(imgs.to(device, non_blocking=True)).cpu().numpy())
        lbls.append(np.array(ids))
    return np.vstack(embs).astype("float32"), np.concatenate(lbls)


def build_enriched_index(model, device, consumer_df, ref_df,
                         img_size, batch_size, num_workers, num_ref_aug):
    """Build FAISS index: ref(clean) + ref(aug x N) + train consumer."""
    eval_tf = get_eval_transform(img_size)
    aug_tf  = get_aug_transform(img_size)
    all_emb, all_lbl = [], []

    # Ref clean
    ds = SimpleDataset(ref_df["img_path"].tolist(), ref_df["pill_id"].tolist(), eval_tf)
    ld = DataLoader(ds, batch_size=batch_size*2, num_workers=num_workers, pin_memory=True)
    e, l = extract_emb(model, ld, device)
    all_emb.append(e); all_lbl.append(l)

    # Ref augmented
    for _ in range(num_ref_aug):
        ds = SimpleDataset(ref_df["img_path"].tolist(), ref_df["pill_id"].tolist(), aug_tf)
        ld = DataLoader(ds, batch_size=batch_size*2, num_workers=num_workers, pin_memory=True)
        e, l = extract_emb(model, ld, device)
        all_emb.append(e); all_lbl.append(l)

    # Train consumer
    tr = consumer_df[consumer_df["split"] == "train"]
    ds = SimpleDataset(tr["img_path"].tolist(), tr["pill_id"].tolist(), eval_tf)
    ld = DataLoader(ds, batch_size=batch_size*2, num_workers=num_workers, pin_memory=True)
    e, l = extract_emb(model, ld, device)
    all_emb.append(e); all_lbl.append(l)

    all_emb = np.vstack(all_emb)
    all_lbl = np.concatenate(all_lbl)
    faiss.normalize_L2(all_emb)
    index = faiss.IndexFlatIP(all_emb.shape[1])
    index.add(all_emb)
    return index, all_emb, all_lbl


def evaluate(model, device, consumer_df, ref_df, split,
             img_size, batch_size, num_workers, num_ref_aug, top_k=[1,3,5,10]):
    index, _, all_lbl = build_enriched_index(
        model, device, consumer_df, ref_df, img_size, batch_size, num_workers, num_ref_aug
    )
    qry_df = consumer_df[consumer_df["split"] == split]
    eval_tf = get_eval_transform(img_size)
    ds = SimpleDataset(qry_df["img_path"].tolist(), qry_df["pill_id"].tolist(), eval_tf)
    ld = DataLoader(ds, batch_size=batch_size*2, num_workers=num_workers, pin_memory=True)
    qry_emb, qry_lbl = extract_emb(model, ld, device)
    faiss.normalize_L2(qry_emb)
    _, indices = index.search(qry_emb, max(top_k))
    results = {}
    for k in top_k:
        correct = sum(qry_lbl[i] in all_lbl[indices[i,:k]] for i in range(len(qry_lbl)))
        results[f"top{k}"] = 100 * correct / len(qry_lbl)
    return results


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Load data
    consumer_df = pd.read_csv(CONSUMER_CSV)
    ref_df      = pd.read_csv(REFERENCE_CSV)
    num_classes = int(consumer_df["pill_id"].max()) + 1
    print(f"\nClasses : {num_classes}")

    # Train loader
    train_set = PillDataset(CONSUMER_CSV, "train", IMG_SIZE)
    sampler   = WeightedRandomSampler(train_set.get_sampler_weights(),
                                       len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               drop_last=True, persistent_workers=True)
    print(f"Train   : {len(train_set)} images, {len(train_loader)} batches")

    # Build model
    print(f"\nBuilding model...")
    model = PillModelV3(
        num_classes     = num_classes,
        lora_rank       = 32,
        lora_alpha      = 64,
        embedding_dim   = 512,
        sub_centers     = 3,
        unfreeze_blocks = UNFREEZE_BLOCKS,
    ).to(device)

    # Load checkpoint weights
    print(f"Loading checkpoint: {CHECKPOINT}")
    ck = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    prev_top1 = ck.get("best_top1", 0)
    print(f"Previous best top-1: {prev_top1:.2f}%")

    total_p   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,}/{total_p:,} ({100*trainable/total_p:.2f}%)")

    # Optimizer — 3 groups with lower LRs
    backbone_p, lora_p, head_p = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if "lora_" in name:    lora_p.append(p)
        elif "backbone" in name: backbone_p.append(p)
        else:                  head_p.append(p)

    param_groups = []
    if backbone_p: param_groups.append({"params": backbone_p, "lr": LR_BACKBONE, "name": "backbone"})
    if lora_p:     param_groups.append({"params": lora_p,     "lr": LR_LORA,     "name": "lora"})
    if head_p:     param_groups.append({"params": head_p,     "lr": LR_HEAD,     "name": "head"})

    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=1e-8)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler    = GradScaler("cuda") if device.type == "cuda" else None

    # SWA
    swa_model     = AveragedModel(model) if USE_SWA else None
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR) if USE_SWA else None

    print(f"\nParam groups:")
    for g in param_groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:>10}: {n:>10,} params  lr={g['lr']}")

    print(f"\n🚀 Resuming training for {EXTRA_EPOCHS} more epochs...")
    print(f"   Previous best : {prev_top1:.2f}%")
    print(f"   Unfreeze      : {UNFREEZE_BLOCKS} blocks (was 4)")
    print(f"   LR backbone   : {LR_BACKBONE} (was 5e-6)")
    print(f"   LR LoRA       : {LR_LORA} (was 5e-5)")
    print(f"   LR head       : {LR_HEAD} (was 5e-4)")
    print(f"   Ref aug       : x{NUM_REF_AUG} (was x5)")
    print("=" * 70)

    best_top1     = prev_top1
    patience_count = 0
    history       = []

    for epoch in range(1, EXTRA_EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss, correct, total = 0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            use_cut = np.random.rand() < CUTMIX_PROB
            use_mix = (not use_cut) and np.random.rand() < MIXUP_PROB

            if use_cut:
                images, la, lb, lam = cutmix(images, labels)
            elif use_mix:
                images, la, lb, lam = mixup(images, labels)

            with autocast(device_type="cuda", enabled=(scaler is not None)):
                if use_cut or use_mix:
                    log_a, _ = model(images, la)
                    log_b, _ = model(images, lb)
                    loss = lam * criterion(log_a, la) + (1-lam) * criterion(log_b, lb)
                    logits, used_labels = log_a, la
                else:
                    logits, _ = model(images, labels)
                    loss = criterion(logits, labels)
                    used_labels = labels
                loss = loss / GRAD_ACCUM

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * images.size(0) * GRAD_ACCUM
            with torch.no_grad():
                correct += (logits.argmax(1) == used_labels).sum().item()
                total   += images.size(0)

            if (batch_idx + 1) % 25 == 0:
                print(f"    Batch {batch_idx+1}/{len(train_loader)} — "
                      f"loss: {total_loss/total:.4f}, acc: {100*correct/total:.2f}%")

        # Scheduler step
        if USE_SWA and epoch >= SWA_START:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        elapsed  = time.time() - t0
        avg_loss = total_loss / total
        acc      = 100 * correct / total
        print(f"\nEpoch {epoch}/{EXTRA_EPOCHS} — "
              f"loss: {avg_loss:.4f}, acc: {acc:.2f}%, time: {elapsed:.0f}s")

        # Evaluate every 2 epochs
        if epoch % 2 == 0:
            print("  📊 Evaluating...")
            results = evaluate(
                model, device, consumer_df, ref_df, "val",
                IMG_SIZE, BATCH_SIZE, NUM_WORKERS, NUM_REF_AUG
            )
            for k, v in sorted(results.items()):
                print(f"    {k:<6}: {v:.2f}%")

            history.append({"epoch": epoch, "loss": avg_loss, "acc": acc, **results})
            pd.DataFrame(history).to_csv(os.path.join(SAVE_DIR, "history.csv"), index=False)

            top1 = results["top1"]
            if top1 > best_top1 + 0.001:
                best_top1 = top1
                patience_count = 0
                torch.save({
                    "epoch"            : f"resumed_{epoch}",
                    "model_state_dict" : model.state_dict(),
                    "best_top1"        : best_top1,
                    "results"          : results,
                }, os.path.join(SAVE_DIR, "best_model.pth"))
                print(f"  ⭐ New best! top1={best_top1:.2f}%")
            else:
                patience_count += 1
                print(f"  No improvement ({patience_count}/{PATIENCE})")

            if patience_count >= PATIENCE:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break

        print("-" * 70)

    # SWA final
    if USE_SWA and swa_model and epoch >= SWA_START:
        print("\n📊 SWA: updating batch norm...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_results = evaluate(
            swa_model, device, consumer_df, ref_df, "val",
            IMG_SIZE, BATCH_SIZE, NUM_WORKERS, NUM_REF_AUG
        )
        print("  SWA val results:")
        for k, v in sorted(swa_results.items()):
            print(f"    {k:<6}: {v:.2f}%")
        if swa_results.get("top1", 0) > best_top1:
            torch.save({
                "epoch"           : "swa",
                "model_state_dict": swa_model.state_dict(),
                "best_top1"       : swa_results["top1"],
                "results"         : swa_results,
            }, os.path.join(SAVE_DIR, "best_model_swa.pth"))
            print("  ⭐ SWA is better! Saved.")

    # Final test evaluation
    print("\n" + "=" * 70)
    print("📊 FINAL TEST EVALUATION")
    print("=" * 70)

    best_ck = torch.load(os.path.join(SAVE_DIR, "best_model.pth"),
                         map_location=device, weights_only=False)
    model.load_state_dict(best_ck["model_state_dict"])

    test_results = evaluate(
        model, device, consumer_df, ref_df, "test",
        IMG_SIZE, BATCH_SIZE, NUM_WORKERS, NUM_REF_AUG
    )

    print(f"\n  {'Metric':<10} {'Accuracy':>10}")
    print(f"  {'-'*22}")
    for k, v in sorted(test_results.items()):
        print(f"  {k:<10} {v:>9.2f}%")

    # Save final FAISS index
    faiss_dir = "faiss_index_final"
    os.makedirs(faiss_dir, exist_ok=True)
    final_index, _, final_lbl = build_enriched_index(
        model, device, consumer_df, ref_df,
        IMG_SIZE, BATCH_SIZE, NUM_WORKERS, NUM_REF_AUG
    )
    faiss.write_index(final_index, os.path.join(faiss_dir, "pill_index.faiss"))

    meta = pd.DataFrame({"index_position": range(len(final_lbl)), "pill_id": final_lbl})
    names = {**dict(zip(consumer_df["pill_id"], consumer_df["drug_name"])),
             **dict(zip(ref_df["pill_id"], ref_df["drug_name"]))}
    meta["drug_name"] = meta["pill_id"].map(names)
    meta.to_csv(os.path.join(faiss_dir, "index_metadata.csv"), index=False)

    print(f"\n✅ Done!")
    print(f"   Best model : {SAVE_DIR}/best_model.pth")
    print(f"   FAISS index: {faiss_dir}/pill_index.faiss ({final_index.ntotal} vectors)")
    print(f"   Previous best : {prev_top1:.2f}%")
    print(f"   New best      : {best_top1:.2f}%")
    print(f"   Test top-1    : {test_results['top1']:.2f}%")
    print(f"   Test top-5    : {test_results['top5']:.2f}%")


if __name__ == "__main__":
    main()