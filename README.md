# IT549_Project_Overfitters

# 💊 PillScan — AI-Powered Pill Recognition System

> **IT549 Deep Learning Project — Team Overfitters**

A full-stack pill recognition system that identifies pills from photos using state-of-the-art deep learning. Upload a pill image and get instant identification with shape, color, imprint, and size details.

🔗 **Live Demo:** [https://Pjshana-Pill-recognition.hf.space](https://Pjshana-Pill-recognition.hf.space)  
📦 **Dataset:** [NLM C3PI Dataset]

---

## 🏆 Results

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | **73.90%** |
| Top-3 Accuracy | **83.66%** |
| Top-5 Accuracy | **86.86%** |
| Top-10 Accuracy | **91.53%** |

---

## 🧠 Model Architecture

```
Input Image (518×518)
        ↓
DINOv2 ViT-B/14 (pretrained, frozen)
+ LoRA Adapters (rank=32, alpha=64) on QKV+proj layers
        ↓
CLS Token (768-d) + GeM-Pooled Patch Tokens (768-d)
        ↓
Concatenate → 1536-d
        ↓
Linear(1536→1024) → BatchNorm → GELU → Dropout(0.3)
        ↓
Linear(1024→512) → BatchNorm
        ↓
512-d L2-normalized Embedding
        ↓
Sub-center ArcFace (K=3, margin=0.3, scale=30.0)
```

**Key Design Choices:**
- **DINOv2 ViT-B/14** — Best vision transformer for fine-grained visual similarity
- **LoRA Fine-tuning** — Only 6.61% of parameters trainable (~6.1M / 92.7M), prevents overfitting
- **Sub-center ArcFace** — Handles intra-class variance (same pill, different lighting/angles)
- **GeM Pooling** — Better than average/max pooling for retrieval tasks
- **FAISS IndexFlatIP** — Exact cosine similarity search over 33,681 embeddings

---

## 📦 Dataset

**Source:** NLM C3PI (Computational Photography Project for Pill Identification)

| Split | Images | Unique Pills |
|-------|--------|-------------|
| Train | 8,601 | 2,047 |
| Val | 2,172 | 2,047 |
| Test | 3,127 | 2,047 |
| Reference | 2,280 | 2,047 |
| **Total** | **16,180** | **2,047** |

**FAISS Index composition:**
- Reference images (clean): 2,280
- Reference images (augmented ×10): 22,800
- Train consumer images: 8,601
- **Total: 33,681 vectors**

---

## 🗂️ Project Structure

```
IT549_Project_Overfitters/
│
├── training/                      # ML training scripts
│   ├── train_v3.py                # Main V3 training script
│   ├── resume_training.py         # PillModelV3 architecture
│   ├── pill_model_v2.py           # Base components (LoRA, GeM, ArcFace)
│   ├── pill_dataset_v2.py         # Dataset loader & augmentations
│   ├── rebuild_index_v3.py        # FAISS index builder (V3)
│   ├── tta_evaluate.py            # Test-time augmentation evaluation
│   └── pill_metadata.csv          # Shape/color/imprint metadata
│
├── pill_backend/                  # FastAPI backend
│   ├── app/
│   │   ├── config.py              # Configuration
│   │   ├── model.py               # PillModelV3 (deployment version)
│   │   ├── inference.py           # Recognition engine
│   │   └── main.py                # API endpoints
│   ├── models/                    # Model files (not in repo, too large)
│   │   ├── best_model.pth         # ~350MB — download from HF Space
│   │   ├── pill_index.faiss       # ~500MB — download from HF Space
│   │   ├── index_metadata.csv     # Pill metadata
│   │   └── reference_mapping.csv  # Reference image mapping
│   ├── run.py                     # Server entry point
│   ├── requirements.txt
|   ├── test_api.py
|   |__ enrich_metadata
│
├── pill_frontend/                 # React frontend
│   ├── src/
│   │   ├── App.jsx
|   |   ├── index.css
|   |   ├── main.jsx
│   │   └── components/
│   │       ├── Header.jsx
│   │       ├── UploadZone.jsx     # Drag & drop + camera capture
│   │       ├── LoadingState.jsx   # Scanning animation
│   │       ├── Results.jsx        # Top-5 result cards
│   │       └── Footer.jsx
│   ├── package.json
│   └── vite.config.js
|   |__ index.html
|   |__ tailwind.config.js
|   |__ postcss.config.js
|   |__ package-lock.js
│
├── Dockerfile                     # HuggingFace Spaces deployment
└── README.md
```

---

## 🚀 How It Works

1. User uploads a pill photo via drag & drop or camera
2. FastAPI backend receives the image
3. DINOv2 + LoRA model extracts a 512-d embedding
4. FAISS performs cosine similarity search over 33,681 reference embeddings
5. Top-5 unique pill matches returned with similarity scores
6. Frontend displays results with drug name, shape, color, imprint, size, and reference image

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | PyTorch + DINOv2 ViT-B/14 |
| Fine-tuning | LoRA (rank=32) |
| Loss | Sub-center ArcFace (K=3) |
| Vector Search | FAISS IndexFlatIP |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Tailwind CSS + Vite |
| Deployment | HuggingFace Spaces (Docker) + Vercel |
| Reference Images | HuggingFace Datasets |

---

## ⚙️ Local Setup

### Backend
```bash
cd pill_backend
pip install -r requirements.txt

# Download model files from HF Space and place in models/
# best_model.pth, pill_index.faiss, index_metadata.csv, reference_mapping.csv

python run.py
# API running at http://localhost:8000
```

### Frontend
```bash
cd pill_frontend
npm install
npm run dev
# UI running at http://localhost:3000
```

### API Endpoints
```
GET  /health    → Model status
GET  /stats     → Model info
POST /predict   → Upload image → top-5 matches
```

---

## 🏋️ Training

```bash
cd training

# Train V3 model from scratch
python train_v3.py

# Resume training from checkpoint
python resume_training.py

# Build FAISS index after training
python rebuild_index_v3.py

# Evaluate with test-time augmentation
python tta_evaluate.py
```

**Training config:**
- Epochs: 60 (+ 40 resumed)
- Batch size: 64
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Scheduler: CosineAnnealingLR


---

## ⚠️ Limitations

- Dataset contains **US FDA pills only** — Indian pills (e.g., Paracetamol, Dolo 650) are not recognized
- Real-world performance may vary due to lighting, camera angle, and background
- This tool is **not a substitute** for professional medical advice

---

## 👥 Team Overfitters

IT549 — Deep Learning  
Dhirubhai Ambani Institute of Information and Communication Technology (DA-IICT)

---

## 📄 References

- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Sub-center ArcFace (ECCV 2020)](https://arxiv.org/abs/2004.01569)
- [NLM C3PI Dataset](https://www.nlm.nih.gov/)
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
