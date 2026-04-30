import os, torch, numpy as np, pandas as pd, faiss
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from resume_training import PillModelV3

# Config - MUST match V3 training
IMG_SIZE       = 518
BATCH_SIZE     = 64
NUM_WORKERS    = 4
NUM_AUG        = 10
CHECKPOINT     = "checkpoints_v3_resumed/best_model.pth"
CONSUMER_CSV   = "data/consumer_mapping.csv"
REFERENCE_CSV  = "data/reference_mapping.csv"
OUTPUT_DIR     = "faiss_index_v3_correct"

# Load V3 model
device = torch.device("cuda")
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)

model = PillModelV3(
    num_classes=2047,
    lora_rank=32, lora_alpha=64,
    embedding_dim=512, sub_centers=3,
    unfreeze_blocks=8,
)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()
print(f"Model loaded! Epoch: {checkpoint.get('epoch')}")

# Datasets
class SimpleDataset(Dataset):
    def __init__(self, paths, ids):
        self.paths = paths
        self.ids = ids
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        return self.transform(img), self.ids[idx]

class AugDataset(Dataset):
    def __init__(self, paths, ids):
        self.paths = paths
        self.ids = ids
        self.transform = transforms.Compose([
            transforms.Resize((int(IMG_SIZE*1.15), int(IMG_SIZE*1.15))),
            transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
            transforms.GaussianBlur(3, (0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.paths) * NUM_AUG
    def __getitem__(self, idx):
        i = idx // NUM_AUG
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        return self.transform(img), self.ids[i]

@torch.no_grad()
def extract(model, loader, device):
    embs, lbls = [], []
    for imgs, ids in loader:
        imgs = imgs.to(device)
        e = model(imgs)
        embs.append(e.cpu().numpy())
        lbls.append(np.array(ids))
    return np.vstack(embs).astype("float32"), np.concatenate(lbls)

# Load data
ref = pd.read_csv(REFERENCE_CSV)
consumer = pd.read_csv(CONSUMER_CSV)
train = consumer[consumer["split"]=="train"]

all_emb, all_lbl = [], []

# Part A: Reference clean
print("A. Reference clean...")
ds = SimpleDataset(ref["img_path"].tolist(), ref["pill_id"].tolist())
ld = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
e, l = extract(model, ld, device)
all_emb.append(e); all_lbl.append(l)
print(f"   {len(e)} embeddings")

# Part B: Reference augmented
print(f"B. Reference augmented x{NUM_AUG}...")
ds = AugDataset(ref["img_path"].tolist(), ref["pill_id"].tolist())
ld = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
e, l = extract(model, ld, device)
all_emb.append(e); all_lbl.append(l)
print(f"   {len(e)} embeddings")

# Part C: Train consumer
print("C. Train consumer...")
ds = SimpleDataset(train["img_path"].tolist(), train["pill_id"].tolist())
ld = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
e, l = extract(model, ld, device)
all_emb.append(e); all_lbl.append(l)
print(f"   {len(e)} embeddings")

# Combine
emb = np.vstack(all_emb)
lbl = np.concatenate(all_lbl)
print(f"\nTotal: {len(emb)} embeddings")

# Normalize + build index
faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(512)
index.add(emb)

# Save index
os.makedirs(OUTPUT_DIR, exist_ok=True)
faiss.write_index(index, f"{OUTPUT_DIR}/pill_index.faiss")

# Build metadata
pill_names = consumer.drop_duplicates("pill_id")[["pill_id","drug_name"]]

meta = pd.DataFrame({"index_position": range(len(lbl)), "pill_id": lbl})
meta = meta.merge(pill_names, on="pill_id", how="left")

meta.to_csv(f"{OUTPUT_DIR}/index_metadata.csv", index=False)

# Sanity check
print("\nSanity check - searching first 5 images...")
test = emb[:5]
d, idx = index.search(test, 5)
for i in range(5):
    q = lbl[i]
    r = lbl[idx[i,0]]
    ok = "YES" if q == r else "NO"
    print(f"  Query pill_id={q}, Top1 pill_id={r} → {ok} (sim={d[i,0]:.3f})")

print(f"\nDone! Saved to {OUTPUT_DIR}/")
