"""
DRScan Cam — Configuration
Defines modalities, class labels, severity levels, and app settings.
"""

import os
from enum import Enum
from typing import Dict, List


class Modality(str, Enum):
    """Supported medical imaging modalities."""
    CHEST_XRAY = "chest_xray"
    BRAIN_MRI = "brain_mri"
    LUNG_CT = "lung_ct"
    RETINAL_FUNDUS = "retinal_fundus"


class InferenceMode(str, Enum):
    """Model inference modes."""
    DEMO = "demo"          # Image-analysis based (works immediately)
    CUSTOM = "custom"      # User-provided trained model


# ---------------------------------------------------------------------------
# Per-modality class labels
# ---------------------------------------------------------------------------
MODALITY_CLASSES: Dict[str, List[str]] = {
    Modality.CHEST_XRAY: [
        "Normal",
        "Bacterial Pneumonia",
        "Viral Pneumonia",
        "Pleural Effusion",
        "Cardiomegaly",
        "Pneumothorax",
        "Atelectasis",
        "Lung Opacity",
    ],
    Modality.BRAIN_MRI: [
        "Normal",
        "Glioma",
        "Meningioma",
        "Pituitary Tumor",
        "Metastatic Lesion",
        "Multiple Sclerosis",
    ],
    Modality.LUNG_CT: [
        "Normal",
        "Lung Nodule",
        "Ground Glass Opacity",
        "Consolidation",
        "Emphysema",
        "Pulmonary Fibrosis",
    ],
    Modality.RETINAL_FUNDUS: [
        "No DR",
        "Mild DR",
        "Moderate DR",
        "Severe DR",
        "Proliferative DR",
    ],
}

# ---------------------------------------------------------------------------
# Severity scale (0–4)
# ---------------------------------------------------------------------------
SEVERITY_LEVELS = {
    0: {
        "label": "Normal",
        "color": "#10b981",
        "badge": "success",
        "recommendation": (
            "No significant abnormalities detected. "
            "Routine follow-up as clinically indicated."
        ),
    },
    1: {
        "label": "Mild",
        "color": "#22d3ee",
        "badge": "info",
        "recommendation": (
            "Minor findings noted. Clinical correlation recommended. "
            "Consider follow-up imaging in 6–12 months."
        ),
    },
    2: {
        "label": "Moderate",
        "color": "#f59e0b",
        "badge": "warning",
        "recommendation": (
            "Moderate findings detected. Follow-up imaging recommended "
            "within 3–6 months. Specialist consultation advised."
        ),
    },
    3: {
        "label": "Severe",
        "color": "#f97316",
        "badge": "danger",
        "recommendation": (
            "Significant abnormalities detected. Urgent specialist "
            "consultation recommended. Additional imaging may be required."
        ),
    },
    4: {
        "label": "Critical",
        "color": "#ef4444",
        "badge": "critical",
        "recommendation": (
            "Critical findings identified. Immediate specialist referral "
            "required. Consider emergent intervention."
        ),
    },
}

# ---------------------------------------------------------------------------
# Retinal DR-specific severity recommendations & urgency levels
# ---------------------------------------------------------------------------
RETINAL_RECOMMENDATIONS = {
    0: "No diabetic retinopathy detected. Routine screening recommended in 12 months.",
    1: "Mild non-proliferative DR detected. Follow-up screening recommended in 6–9 months.",
    2: "Moderate non-proliferative DR detected. Referral to ophthalmologist recommended within 3 months.",
    3: "Severe non-proliferative DR detected. Urgent referral to ophthalmologist within 1 month.",
    4: "Proliferative diabetic retinopathy detected. Immediate referral to retina specialist required.",
}

RETINAL_URGENCY_LEVELS = {
    0: "routine",
    1: "low",
    2: "moderate",
    3: "high",
    4: "critical",
}

RETINAL_SEVERITY_COLORS = {
    0: "#22c55e",
    1: "#84cc16",
    2: "#eab308",
    3: "#f97316",
    4: "#ef4444",
}

# ---------------------------------------------------------------------------
# Modality display metadata
# ---------------------------------------------------------------------------
MODALITY_INFO = {
    Modality.CHEST_XRAY: {
        "name": "Chest X-Ray",
        "icon": "lungs",
        "description": (
            "Analyze chest radiographs for pneumonia, effusions, "
            "cardiomegaly, and other thoracic conditions."
        ),
    },
    Modality.BRAIN_MRI: {
        "name": "Brain MRI",
        "icon": "brain",
        "description": (
            "Detect brain tumors, lesions, and structural abnormalities "
            "in MRI scans."
        ),
    },
    Modality.LUNG_CT: {
        "name": "Lung CT",
        "icon": "microscope",
        "description": (
            "Identify lung nodules, ground glass opacities, and other "
            "pulmonary pathologies in CT images."
        ),
    },
    Modality.RETINAL_FUNDUS: {
        "name": "Retinal Fundus",
        "icon": "visibility",
        "description": (
            "Screen for diabetic retinopathy severity (Grade 0–4) "
            "from retinal fundus photographs with Grad-CAM explainability."
        ),
    },
}

# ---------------------------------------------------------------------------
# App Settings
# ---------------------------------------------------------------------------
APP_NAME = "DRScan AI"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "AI-Powered Medical Image Analysis & Diabetic Retinopathy Screening Platform"

INFERENCE_MODE = os.getenv("INFERENCE_MODE", InferenceMode.DEMO)
MODEL_PATH = os.getenv("MODEL_PATH", "backend/models/")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
RETINAL_IMAGE_SIZE = (384, 384)
RETINAL_MODEL_PATH = os.getenv("RETINAL_MODEL_PATH", r"C:\Users\VEDAN\Downloads\best_model.h5")
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".dcm", ".dicom"}
