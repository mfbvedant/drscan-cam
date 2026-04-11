"""
DRScan Cam — Health Check Router
"""

from fastapi import APIRouter
from backend.config import APP_VERSION, INFERENCE_MODE, Modality
from backend.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint — returns system status."""
    return HealthResponse(
        status="healthy",
        version=APP_VERSION,
        inference_mode=str(INFERENCE_MODE),
        modalities=[m.value for m in Modality],
    )
