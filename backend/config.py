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
}

# ---------------------------------------------------------------------------
# App Settings
# ---------------------------------------------------------------------------
APP_NAME = "DRScan Cam"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AI-Powered Medical Image Analysis Platform"

INFERENCE_MODE = os.getenv("INFERENCE_MODE", InferenceMode.DEMO)
MODEL_PATH = os.getenv("MODEL_PATH", "backend/models/")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".dcm", ".dicom"}
