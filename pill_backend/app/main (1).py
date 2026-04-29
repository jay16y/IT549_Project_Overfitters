# app/main.py — FastAPI Pill Recognition API

import time
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.inference import engine
from app.config import MAX_FILE_SIZE, ALLOWED_TYPES, TOP_K

# ══════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════
app = FastAPI(
    title="Pill Recognition API",
    description="Upload a pill image → get top-5 matching drug identifications",
    version="1.0.0",
)

# CORS — allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════
# STARTUP — load model once
# ══════════════════════════════════════════════════════
@app.on_event("startup")
async def startup():
    """Load model and FAISS index when server starts."""
    engine.load()


# ══════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════
@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "running",
        "model_loaded": engine.loaded,
        "index_size": engine.index.ntotal if engine.index else 0,
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if engine.loaded else "loading",
        "model_loaded": engine.loaded,
        "device": str(engine.device) if engine.device else "not set",
        "index_vectors": engine.index.ntotal if engine.index else 0,
        "pills_in_database": len(engine.pill_info) if engine.pill_info else 0,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(default=TOP_K, ge=1, le=20),
):
    """
    Upload a pill image and get top-K matching pills.

    - **file**: Image file (JPEG, PNG, WebP)
    - **top_k**: Number of results to return (default: 5)

    Returns list of matching pills with drug name, similarity score,
    shape, color, imprint, and NDC code.
    """
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. "
                   f"Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    # Read file
    image_bytes = await file.read()

    # Validate file size
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {len(image_bytes)/1024/1024:.1f} MB. "
                   f"Max: {MAX_FILE_SIZE/1024/1024:.0f} MB",
        )

    # Run prediction
    start_time = time.time()

    try:
        results = engine.predict(image_bytes, top_k=top_k)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

    elapsed = time.time() - start_time

    return {
        "success": True,
        "inference_time_ms": round(elapsed * 1000, 1),
        "num_results": len(results),
        "results": results,
    }


@app.get("/stats")
async def stats():
    """Get system statistics."""
    if not engine.loaded:
        return {"status": "not loaded"}

    return {
        "model": "DINOv2 ViT-B/14 + LoRA + Sub-center ArcFace",
        "embedding_dim": 512,
        "index_vectors": engine.index.ntotal,
        "pills_in_database": len(engine.pill_info),
        "device": str(engine.device),
        "top_k_default": TOP_K,
    }
