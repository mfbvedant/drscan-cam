# =============================================================================
# DRScan AI — Lightweight Dockerfile (Fast Build)
# Skips model pre-caching — models download at first request
# =============================================================================

FROM python:3.10-slim

LABEL maintainer="DRScan AI Team"

# System deps for OpenCV + TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY deploy/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Environment
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=1
ENV OMP_NUM_THREADS=4
ENV RETINAL_MODEL_PATH=/app/models/best_model.h5
ENV GCS_MODEL_BUCKET=""
ENV GCS_MODEL_PATH="models/best_model.h5"

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
