# =============================================================================
# DRScan AI — Production Dockerfile (Cloud Run Optimized)
# Multi-stage build for minimal image size + fast cold starts
# =============================================================================

# ---- Stage 1: Builder (install dependencies) ----
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for OpenCV, TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: Production ----
FROM python:3.11-slim AS production

# Labels
LABEL maintainer="DRScan AI Team"
LABEL description="DRScan AI — Multi-Modality Medical Image Analysis Platform"

# System runtime deps (OpenCV needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# ── Pre-cache BiomedCLIP model during build (avoids cold-start download) ──
# This bakes the ~700MB HuggingFace model into the image
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', \
    cache_dir='/app/model_cache/biomedclip')" || echo "[WARN] BiomedCLIP pre-cache skipped (network unavailable)"

# ── Copy application code ──
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# ── Download retinal model from GCS at startup (see entrypoint.sh) ──
COPY deploy/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ── Environment Variables ──
# Cloud Run sets PORT automatically; default 8080
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HuggingFace cache → use pre-cached model
ENV HF_HOME=/app/model_cache/biomedclip
ENV TRANSFORMERS_CACHE=/app/model_cache/biomedclip

# Retinal model path (downloaded at startup from GCS)
ENV RETINAL_MODEL_PATH=/app/models/best_model.h5

# GCS bucket for the retinal model (set during deployment)
ENV GCS_MODEL_BUCKET=""
ENV GCS_MODEL_PATH="models/best_model.h5"

# TensorFlow optimizations
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=1
ENV OMP_NUM_THREADS=4

# ── Health check ──
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# ── Expose port ──
EXPOSE ${PORT}

# ── Start via entrypoint (downloads model, then launches uvicorn) ──
ENTRYPOINT ["/app/entrypoint.sh"]
