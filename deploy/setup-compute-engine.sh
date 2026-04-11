#!/bin/bash
# =============================================================================
# DRScan AI — Compute Engine GPU VM Setup
# Creates a GPU-enabled VM for high-performance inference
# Use this when you need faster inference (14s → ~1s with GPU)
# =============================================================================
set -euo pipefail

# ── Configuration ──
PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"
ZONE="${GCP_ZONE:-asia-south1-a}"
INSTANCE_NAME="drscan-ai-gpu"
MACHINE_TYPE="n1-standard-4"       # 4 vCPUs, 15 GB RAM
GPU_TYPE="nvidia-tesla-t4"          # Cost-effective GPU
GPU_COUNT=1
GCS_BUCKET="${GCS_MODEL_BUCKET:-${PROJECT_ID}-drscan-models}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║   DRScan AI — Compute Engine GPU VM Setup           ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Instance:  ${INSTANCE_NAME}"
echo "║  Zone:      ${ZONE}"
echo "║  Machine:   ${MACHINE_TYPE} + ${GPU_TYPE}"
echo "║  Project:   ${PROJECT_ID}"
echo "╚══════════════════════════════════════════════════════╝"

# ── Step 1: Enable Compute Engine API ──
gcloud services enable compute.googleapis.com --project="${PROJECT_ID}"

# ── Step 2: Create the VM with GPU ──
echo ""
echo "═══ Creating GPU VM ═══"
gcloud compute instances create "${INSTANCE_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --image-family=tf-latest-gpu \
    --image-project=deeplearning-platform-release \
    --scopes=cloud-platform \
    --tags=http-server,https-server \
    --metadata=startup-script='#!/bin/bash
set -e

# ── Install project dependencies ──
echo "[1/5] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq python3-pip libgl1-mesa-glx git

# ── Clone/copy app ──
echo "[2/5] Setting up application..."
cd /opt
if [ ! -d "drscan-ai" ]; then
    mkdir -p drscan-ai
fi
cd drscan-ai

# ── Install Python dependencies ──
echo "[3/5] Installing Python packages..."
pip3 install fastapi uvicorn[standard] python-multipart \
    tensorflow opencv-python-headless Pillow numpy matplotlib \
    open_clip_torch transformers huggingface_hub aiofiles

# ── Download retinal model from GCS ──
echo "[4/5] Downloading retinal model..."
mkdir -p /opt/drscan-ai/models
gsutil -q cp "gs://'"${GCS_BUCKET}"'/models/best_model.h5" /opt/drscan-ai/models/best_model.h5 || \
    echo "[WARN] Could not download model from GCS"

# ── Set environment ──
export RETINAL_MODEL_PATH=/opt/drscan-ai/models/best_model.h5
export PORT=8080
export TF_CPP_MIN_LOG_LEVEL=2

echo "[5/5] Server ready to start!"
echo "To start: cd /opt/drscan-ai && python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8080"
'

# ── Step 3: Open firewall for HTTP ──
echo ""
echo "═══ Configuring firewall ═══"
gcloud compute firewall-rules create allow-drscan-http \
    --project="${PROJECT_ID}" \
    --allow=tcp:8080 \
    --target-tags=http-server \
    --description="Allow DRScan AI HTTP traffic" 2>/dev/null || echo "[*] Firewall rule already exists"

# ── Step 4: Wait for VM and get IP ──
echo ""
echo "═══ Waiting for VM to be ready ═══"
sleep 15

EXTERNAL_IP=$(gcloud compute instances describe "${INSTANCE_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  GPU VM Created!                                    ║"
echo "║                                                     ║"
echo "║  External IP: ${EXTERNAL_IP}"
echo "║  SSH:  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"
echo "║                                                     ║"
echo "║  After SSH, upload your code and run:               ║"
echo "║  cd /opt/drscan-ai                                  ║"
echo "║  # Copy your code here, then:                       ║"
echo "║  python3 -m uvicorn backend.main:app \\              ║"
echo "║      --host 0.0.0.0 --port 8080                     ║"
echo "║                                                     ║"
echo "║  App: http://${EXTERNAL_IP}:8080                    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "To upload code:"
echo "  gcloud compute scp --recurse ./ ${INSTANCE_NAME}:/opt/drscan-ai/ --zone=${ZONE}"
