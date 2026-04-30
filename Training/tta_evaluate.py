# tta_evaluate.py — Test Time Augmentation evaluation
# Run on existing best_model.pth, no retraining needed

import os
import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random, math
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
np.random.seed(42)

CONSUMER_CSV  = "data/consumer_mapping.csv"
REFERENCE_CSV = "data/reference_mapping.csv"
CHECKPOINT    = "checkpoints_v3/best_model.pth"
IMG_SIZE      = 518
BATCH_SIZE    = 32
NUM_WORKERS   = 8
N_TTA         = 8  # augmented versions per query image

def get_tta_transforms(img_size):
    """8 different augmented views of the same image."""
    return [
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((int(img_size*1.1), int(img_size*1.1))),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(90,90)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(180,180)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(270,270)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((int(img_size*1.15), int(img_size*1.15))),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ]),
    ]

class TTADataset(Dataset):
    def __init__(self, paths, ids, transforms_list):
        self.paths = paths
        self.ids = ids
        self.transforms_list = transforms_list

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        # Return all augmented versions
        views = torch.stack([t(img) for t in self.transforms_list])
        return views, self.ids[idx]

class SimpleDataset(Dataset):
    def __init__(self, paths, ids, img_size):
        self.paths, self.ids = paths, ids
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try: img = Image.open(self.paths[idx]).convert("RGB")
        except: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        return self.transform(img), self.ids[idx]

# Inline model definition (same as train_v3.py)
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
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0/self.p)

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
        sin = torch.sqrt(torch.clamp(1.0 - cos*cos, 0, 1))
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        one_hot = torch.zeros_like(cos).scatter_(1, labels.unsqueeze(1), 1)
        return ((one_hot * phi) + ((1.0 - one_hot) * cos)) * self.scale

class PillModelV3(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", cfg["backbone_name"], pretrained=True
        )
        self.feature_dim = self.backbone.embed_dim
        for p in self.backbone.parameters():
            p.requires_grad = False
        if cfg.get("use_lora", True):
            replaced = 0
            for name, module in self.backbone.named_modules():
                if not isinstance(module, nn.Linear): continue
                if not any(t in name for t in ("qkv", "proj")): continue
                parts = name.split(".")
                parent = self.backbone
                for p in parts[:-1]: parent = getattr(parent, p)
                setattr(parent, parts[-1],
                        LoRALinear(module, rank=cfg.get("lora_rank",32),
                                   alpha=cfg.get("lora_alpha",64)))
                replaced += 1
        self.gem_pool = GeMPool(p=3.0)
        self.embedding_head = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, cfg.get("embedding_dim", 512)),
            nn.BatchNorm1d(cfg.get("embedding_dim", 512)),
        )
        self.arcface = SubCenterArcFace(
            cfg.get("embedding_dim", 512), num_classes,
            cfg.get("sub_centers", 3), cfg.get("arcface_margin", 0.3),
            cfg.get("arcface_scale", 30.0)
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ck = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    cfg.setdefault("backbone_name", "dinov2_vitb14")
    cfg.setdefault("embedding_dim", 512)
    cfg.setdefault("lora_rank", 32)
    cfg.setdefault("lora_alpha", 64)
    cfg.setdefault("sub_centers", 3)
    cfg.setdefault("use_lora", True)

    consumer_df = pd.read_csv(CONSUMER_CSV)
    num_classes  = int(consumer_df["pill_id"].max()) + 1

    print(f"Loading model (best val top-1: {ck.get('best_top1','N/A'):.2f}%)...")
    model = PillModelV3(num_classes, cfg)
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ── Build enriched FAISS index ────────────────────
    print("\nBuilding enriched FAISS index...")
    ref_df    = pd.read_csv(REFERENCE_CSV)
    train_df  = consumer_df[consumer_df["split"] == "train"]

    all_emb, all_lbl = [], []

    def extract_simple(paths, ids):
        ds = SimpleDataset(paths, ids, IMG_SIZE)
        ld = DataLoader(ds, batch_size=BATCH_SIZE*2, num_workers=NUM_WORKERS, pin_memory=True)
        embs, lbls = [], []
        with torch.no_grad():
            for imgs, l in ld:
                embs.append(model(imgs.to(device)).cpu().numpy())
                lbls.append(np.array(l))
        return np.vstack(embs).astype("float32"), np.concatenate(lbls)

    # Reference clean
    e, l = extract_simple(ref_df["img_path"].tolist(), ref_df["pill_id"].tolist())
    all_emb.append(e); all_lbl.append(l)
    print(f"  Reference (clean)     : {len(e)}")

    # Reference augmented x10
    aug_tf = transforms.Compose([
        transforms.Resize((int(IMG_SIZE*1.1), int(IMG_SIZE*1.1))),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.3,0.3,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    for aug_i in range(10):
        aug_paths = ref_df["img_path"].tolist()
        aug_ids   = ref_df["pill_id"].tolist()
        ds_aug = SimpleDataset(aug_paths, aug_ids, IMG_SIZE)
        ds_aug.transform = aug_tf
        ld_aug = DataLoader(ds_aug, batch_size=BATCH_SIZE*2, num_workers=NUM_WORKERS, pin_memory=True)
        embs, lbls = [], []
        with torch.no_grad():
            for imgs, l in ld_aug:
                embs.append(model(imgs.to(device)).cpu().numpy())
                lbls.append(np.array(l))
        all_emb.append(np.vstack(embs).astype("float32"))
        all_lbl.append(np.concatenate(lbls))
    print(f"  Reference (aug x10)   : {len(ref_df)*10}")

    # Train consumer
    e, l = extract_simple(train_df["img_path"].tolist(), train_df["pill_id"].tolist())
    all_emb.append(e); all_lbl.append(l)
    print(f"  Train consumer        : {len(e)}")

    all_emb = np.vstack(all_emb)
    all_lbl = np.concatenate(all_lbl)
    faiss.normalize_L2(all_emb)
    index = faiss.IndexFlatIP(all_emb.shape[1])
    index.add(all_emb)
    print(f"  Total index size      : {index.ntotal}")

    # ── TTA Evaluation ────────────────────────────────
    print("\nRunning TTA evaluation on TEST set...")
    test_df = consumer_df[consumer_df["split"] == "test"]
    tta_transforms = get_tta_transforms(IMG_SIZE)

    tta_dataset = TTADataset(
        test_df["img_path"].tolist(),
        test_df["pill_id"].tolist(),
        tta_transforms,
    )
    tta_loader = DataLoader(tta_dataset, batch_size=8,
                            num_workers=NUM_WORKERS, pin_memory=True)

    all_qry_emb = []
    all_qry_lbl = []

    with torch.no_grad():
        for batch_idx, (views, labels) in enumerate(tta_loader):
            # views shape: (B, N_TTA, C, H, W)
            B, N, C, H, W = views.shape
            views = views.view(B * N, C, H, W).to(device)
            embs = model(views)                    # (B*N, 512)
            embs = embs.view(B, N, -1).mean(dim=1) # average over TTA views
            embs = F.normalize(embs, p=2, dim=1)  # renormalize

            all_qry_emb.append(embs.cpu().numpy())
            all_qry_lbl.append(np.array(labels))

            if (batch_idx+1) % 50 == 0:
                print(f"  Processed {(batch_idx+1)*8}/{len(test_df)} queries")

    qry_emb = np.vstack(all_qry_emb).astype("float32")
    qry_lbl = np.concatenate(all_qry_lbl)
    faiss.normalize_L2(qry_emb)

    _, indices = index.search(qry_emb, 10)

    # Results
    print("\n" + "="*50)
    print("RESULTS WITH TTA")
    print("="*50)
    print(f"\n  {'Metric':<15} {'Accuracy':>10}")
    print(f"  {'-'*27}")
    for k in [1, 3, 5, 10]:
        correct = sum(qry_lbl[i] in all_lbl[indices[i,:k]] for i in range(len(qry_lbl)))
        acc = 100 * correct / len(qry_lbl)
        print(f"  top{k:<11} {acc:>9.2f}%")

    print(f"\n  Query images  : {len(qry_lbl)}")
    print(f"  Index size    : {index.ntotal}")
    print(f"  TTA views     : {N_TTA}")

if __name__ == "__main__":
    main()