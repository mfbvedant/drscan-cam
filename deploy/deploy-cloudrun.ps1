# =============================================================================
# DRScan AI — Cloud Run Deployment (Windows PowerShell)
# Run this from the project root: .\deploy\deploy-cloudrun.ps1
# =============================================================================

param(
    [string]$ProjectId = "",
    [string]$Region = "asia-south1",
    [string]$ServiceName = "drscan-ai"
)

# Prompt for project ID if not provided
if (-not $ProjectId) {
    $ProjectId = Read-Host "Enter your GCP Project ID"
}

$ImageName = "gcr.io/$ProjectId/$ServiceName"
$GcsBucket = "$ProjectId-drscan-models"

Write-Host ""
Write-Host "========================================="  -ForegroundColor Cyan
Write-Host "  DRScan AI - Cloud Run Deployment"  -ForegroundColor Cyan
Write-Host "========================================="  -ForegroundColor Cyan
Write-Host "  Project:  $ProjectId"
Write-Host "  Region:   $Region"
Write-Host "  Service:  $ServiceName"
Write-Host "  Image:    $ImageName"
Write-Host "  Bucket:   $GcsBucket"
Write-Host "========================================="  -ForegroundColor Cyan

# Step 1: Enable APIs
Write-Host ""
Write-Host "[1/5] Enabling GCP APIs..."  -ForegroundColor Yellow
gcloud services enable run.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com storage.googleapis.com --project=$ProjectId

# Step 2: Upload retinal model to GCS
Write-Host ""
Write-Host "[2/5] Setting up model storage..."  -ForegroundColor Yellow

# Create bucket
try {
    gsutil mb -p $ProjectId -l $Region "gs://$GcsBucket" 2>$null
} catch {
    Write-Host "  Bucket already exists"  -ForegroundColor Gray
}

$ModelPath = Join-Path $HOME "Downloads\best_model.h5"
if (Test-Path $ModelPath) {
    Write-Host "  Uploading retinal model to GCS... this may take a few minutes"  -ForegroundColor Cyan
    gsutil -o "GSUtil:parallel_composite_upload_threshold=150M" cp $ModelPath "gs://$GcsBucket/models/best_model.h5"
    Write-Host "  Model uploaded!"  -ForegroundColor Green
} else {
    Write-Host "  WARNING: best_model.h5 not found at $ModelPath"  -ForegroundColor Red
    Write-Host "  Upload manually with: gsutil cp YOUR_PATH gs://$GcsBucket/models/best_model.h5"
}

# Step 3: Build with Cloud Build
Write-Host ""
Write-Host "[3/5] Building container image with Cloud Build..."  -ForegroundColor Yellow
Push-Location (Split-Path $PSScriptRoot)
gcloud builds submit --tag "$ImageName" --project=$ProjectId --timeout=1800 --machine-type=e2-highcpu-8

# Step 4: Deploy to Cloud Run
Write-Host ""
Write-Host "[4/5] Deploying to Cloud Run..."  -ForegroundColor Yellow
$EnvVars = "GCS_MODEL_BUCKET=$GcsBucket,GCS_MODEL_PATH=models/best_model.h5,TF_CPP_MIN_LOG_LEVEL=2,TF_ENABLE_ONEDNN_OPTS=1,OMP_NUM_THREADS=4,WORKERS=1"

gcloud run deploy $ServiceName --image="$ImageName" --project=$ProjectId --region=$Region --platform=managed --allow-unauthenticated --port=8080 --memory=16Gi --cpu=4 --min-instances=1 --max-instances=2 --timeout=300 --concurrency=4 --cpu-boost --set-env-vars=$EnvVars

# Step 5: Get URL
Write-Host ""
Write-Host "[5/5] Getting deployment URL..."  -ForegroundColor Yellow
$Url = gcloud run services describe $ServiceName --project=$ProjectId --region=$Region --format="value(status.url)"

Pop-Location

Write-Host ""
Write-Host "========================================="  -ForegroundColor Green
Write-Host "  DRScan AI is LIVE!"  -ForegroundColor Green
Write-Host "========================================="  -ForegroundColor Green
Write-Host "  URL: $Url"  -ForegroundColor Cyan
Write-Host ""
Write-Host "  Landing:  $Url/"
Write-Host "  Scanner:  $Url/scan"
Write-Host "  API Docs: $Url/api/docs"
Write-Host "  Health:   $Url/api/health"
Write-Host "========================================="  -ForegroundColor Green
