#!/bin/bash
# =============================================================================
# DRScan AI — Container Entrypoint
# Downloads retinal model from GCS (if not cached), then starts the server.
# =============================================================================
set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║       DRScan AI — Starting Up                       ║"
echo "╚══════════════════════════════════════════════════════╝"

# ── 1. Download retinal model from GCS if not present ──
MODEL_DIR="/app/models"
MODEL_FILE="${MODEL_DIR}/best_model.h5"

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_FILE}" ]; then
    echo "[✓] Retinal model already cached: ${MODEL_FILE}"
else
    if [ -n "${GCS_MODEL_BUCKET}" ]; then
        echo "[↓] Downloading retinal model from gs://${GCS_MODEL_BUCKET}/${GCS_MODEL_PATH}..."
        gsutil -q cp "gs://${GCS_MODEL_BUCKET}/${GCS_MODEL_PATH}" "${MODEL_FILE}" 2>/dev/null || {
            # Fallback: try gcloud storage
            gcloud storage cp "gs://${GCS_MODEL_BUCKET}/${GCS_MODEL_PATH}" "${MODEL_FILE}" 2>/dev/null || {
                echo "[!] Could not download retinal model from GCS"
                echo "[!] Retinal modality will use demo/mock mode"
                export RETINAL_MODEL_PATH=""
            }
        }
        if [ -f "${MODEL_FILE}" ]; then
            SIZE=$(du -sh "${MODEL_FILE}" | cut -f1)
            echo "[✓] Retinal model downloaded: ${SIZE}"
        fi
    else
        echo "[!] GCS_MODEL_BUCKET not set — retinal model unavailable"
        echo "[!] Set GCS_MODEL_BUCKET env var to enable retinal inference"
        export RETINAL_MODEL_PATH=""
    fi
fi

# ── 2. Performance tuning ──
# Set workers based on available CPUs (Cloud Run provides $CPU)
WORKERS=${WORKERS:-1}
echo "[*] Workers: ${WORKERS}"
echo "[*] Port: ${PORT:-8080}"

# ── 3. Start Uvicorn ──
echo "[*] Launching DRScan AI server..."
exec python -m uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8080}" \
    --workers "${WORKERS}" \
    --timeout-keep-alive 120 \
    --log-level info
