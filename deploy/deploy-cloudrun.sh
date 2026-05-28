#!/bin/bash
# =============================================================================
# DRScan AI — Cloud Run Deployment Script
# Deploys the app to Google Cloud Run with optimized settings
# =============================================================================
set -euo pipefail

# ── Configuration ──
PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"
REGION="${GCP_REGION:-asia-south1}"
SERVICE_NAME="drscan-ai"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
GCS_BUCKET="${GCS_MODEL_BUCKET:-${PROJECT_ID}-drscan-models}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║     DRScan AI — Cloud Run Deployment                ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Project:  ${PROJECT_ID}"
echo "║  Region:   ${REGION}"
echo "║  Service:  ${SERVICE_NAME}"
echo "║  Image:    ${IMAGE_NAME}"
echo "║  Bucket:   ${GCS_BUCKET}"
echo "╚══════════════════════════════════════════════════════╝"

# ── Step 1: Enable required APIs ──
echo ""
echo "═══ Step 1: Enabling GCP APIs ═══"
gcloud services enable \
    run.googleapis.com \
    containerregistry.googleapis.com \
    cloudbuild.googleapis.com \
    storage.googleapis.com \
    --project="${PROJECT_ID}"

# ── Step 2: Create GCS bucket & upload retinal model ──
echo ""
echo "═══ Step 2: Setting up model storage ═══"
gsutil mb -p "${PROJECT_ID}" -l "${REGION}" "gs://${GCS_BUCKET}" 2>/dev/null || echo "[*] Bucket already exists"

if [ -f "../../best_model.h5" ]; then
    echo "[↑] Uploading retinal model to GCS (4.4 GB — this may take a few minutes)..."
    gsutil -o "GSUtil:parallel_composite_upload_threshold=150M" \
        cp "../../best_model.h5" "gs://${GCS_BUCKET}/models/best_model.h5"
    echo "[✓] Model uploaded"
elif [ -f "$HOME/Downloads/best_model.h5" ]; then
    echo "[↑] Uploading retinal model from Downloads..."
    gsutil -o "GSUtil:parallel_composite_upload_threshold=150M" \
        cp "$HOME/Downloads/best_model.h5" "gs://${GCS_BUCKET}/models/best_model.h5"
    echo "[✓] Model uploaded"
else
    echo "[!] best_model.h5 not found — upload manually:"
    echo "    gsutil cp /path/to/best_model.h5 gs://${GCS_BUCKET}/models/best_model.h5"
fi

# ── Step 3: Build container image with Cloud Build ──
echo ""
echo "═══ Step 3: Building container image ═══"
cd "$(dirname "$0")/.."
gcloud builds submit \
    --tag "${IMAGE_NAME}" \
    --project="${PROJECT_ID}" \
    --timeout=1800 \
    --machine-type=e2-highcpu-8

# ── Step 4: Deploy to Cloud Run ──
echo ""
echo "═══ Step 4: Deploying to Cloud Run ═══"
gcloud run deploy "${SERVICE_NAME}" \
    --image="${IMAGE_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --platform=managed \
    --allow-unauthenticated \
    --port=8080 \
    --memory=4Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=2 \
    --timeout=300 \
    --concurrency=4 \
    --cpu-boost \
    --set-env-vars="\
GCS_MODEL_BUCKET=${GCS_BUCKET},\
GCS_MODEL_PATH=models/best_model.h5,\
TF_CPP_MIN_LOG_LEVEL=2,\
TF_ENABLE_ONEDNN_OPTS=1,\
OMP_NUM_THREADS=4,\
WORKERS=1"

# ── Step 5: Get URL ──
echo ""
echo "═══ Deployment Complete! ═══"
URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --format="value(status.url)")

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  DRScan AI is LIVE!                                 ║"
echo "║                                                     ║"
echo "║  URL: ${URL}"
echo "║                                                     ║"
echo "║  Landing:  ${URL}/"
echo "║  Scanner:  ${URL}/scan"
echo "║  API Docs: ${URL}/api/docs"
echo "║  Health:   ${URL}/api/health"
echo "╚══════════════════════════════════════════════════════╝"
