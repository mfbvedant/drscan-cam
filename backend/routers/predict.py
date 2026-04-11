"""
DRScan Cam — Prediction Router
Main endpoint for medical image analysis.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.config import (
    Modality,
    MAX_FILE_SIZE,
    SEVERITY_LEVELS,
    MODALITY_CLASSES,
    MODALITY_INFO,
)
from backend.schemas import PredictionResponse, ErrorResponse
from backend.services.preprocessing import (
    load_image_from_bytes,
    preprocess_for_model,
    validate_image,
    check_image_quality,
    image_to_base64,
)
from backend.services.gradcam import generate_heatmap, heatmap_to_base64
from backend.services.inference import engine
from backend.services.report import generate_report

router = APIRouter(tags=["predict"])


@router.post(
    "/api/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict(
    file: UploadFile = File(..., description="Medical image file"),
    modality: str = Form(..., description="Imaging modality: chest_xray, brain_mri, or lung_ct"),
):
    """
    Analyze a medical image and return diagnosis, confidence, heatmap, and recommendations.
    """
    # --- Validate modality ---
    valid_modalities = [m.value for m in Modality]
    if modality not in valid_modalities:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid modality '{modality}'. Must be one of: {valid_modalities}",
        )

    # --- Read file ---
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB.",
        )

    # --- Validate image ---
    validation_error = validate_image(file_bytes, file.filename or "image.jpg")
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)

    try:
        # --- Load & preprocess ---
        image = load_image_from_bytes(file_bytes, file.filename or "image.jpg")
        preprocessed, resized = preprocess_for_model(image)

        # --- Quality check (non-blocking) ---
        quality = check_image_quality(resized)

        # --- Run inference ---
        result = engine.predict(resized, preprocessed, modality)

        # --- Generate heatmap ---
        heatmap_overlay, regions = generate_heatmap(resized, preprocessed)

        # --- Generate report ---
        report = generate_report(
            modality=modality,
            diagnosis=result["diagnosis"],
            severity=result["severity"],
            confidence=result["confidence"],
            regions=regions,
            top_predictions=result["top_predictions"],
        )

        # --- Build response ---
        severity_info = SEVERITY_LEVELS.get(result["severity"], SEVERITY_LEVELS[0])

        return PredictionResponse(
            diagnosis=result["diagnosis"],
            severity=result["severity"],
            severity_label=severity_info["label"],
            confidence=result["confidence"],
            top_predictions=result["top_predictions"],
            heatmap_base64=heatmap_to_base64(heatmap_overlay),
            original_base64=image_to_base64(resized),
            modality=modality,
            recommendation=report["recommendation"],
            regions_of_concern=regions,
            inference_time_ms=result["inference_time_ms"],
            model_version=result["model_version"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )


@router.get("/api/modalities")
async def list_modalities():
    """List all supported imaging modalities with metadata."""
    return {
        modality.value: {
            "name": MODALITY_INFO[modality]["name"],
            "icon": MODALITY_INFO[modality]["icon"],
            "description": MODALITY_INFO[modality]["description"],
            "classes": MODALITY_CLASSES[modality],
        }
        for modality in Modality
    }
