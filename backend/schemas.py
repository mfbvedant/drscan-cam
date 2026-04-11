"""
DRScan Cam — Pydantic Schemas
Request / response models for the prediction API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    """Single class prediction with confidence."""
    class_name: str = Field(..., description="Predicted condition name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0–1)")


class PredictionResponse(BaseModel):
    """Full prediction response from /api/predict."""
    # Primary diagnosis
    diagnosis: str = Field(..., description="Primary diagnosis label")
    severity: int = Field(..., ge=0, le=4, description="Severity grade (0–4)")
    severity_label: str = Field(..., description="Human-readable severity label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Top prediction confidence")

    # Detailed predictions
    top_predictions: List[PredictionResult] = Field(
        default_factory=list,
        description="Top N class predictions with confidences",
    )

    # Explainability
    heatmap_base64: Optional[str] = Field(
        None, description="Grad-CAM heatmap overlay as base64 JPEG"
    )
    original_base64: Optional[str] = Field(
        None, description="Preprocessed original image as base64 JPEG"
    )

    # Clinical context
    modality: str = Field(..., description="Imaging modality used")
    recommendation: str = Field(..., description="Clinical recommendation text")
    regions_of_concern: List[str] = Field(
        default_factory=list,
        description="Detected regions of concern",
    )

    # Metadata
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    model_version: str = Field(default="demo-v1", description="Model version used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    inference_mode: str
    modalities: List[str]


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
