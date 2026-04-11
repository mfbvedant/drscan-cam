# 🏥 DRScan Cam — AI-Powered Medical Image Analysis Platform

> Upload a chest X-ray, brain MRI, or lung CT → Get instant AI-powered diagnosis, confidence scores, Grad-CAM heatmaps, and clinical recommendations.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Run the server
python -m uvicorn backend.main:app --reload --port 8080

# Open in browser
# http://localhost:8080
```

---

## 🏗️ Architecture

```
Doctor uploads X-ray / MRI / CT
        ↓
  Cloud Run (API Layer)
  ┌─────────────────────────┐
  │  FastAPI Backend         │
  │  • Image validation      │
  │  • DICOM parsing         │
  │  • Preprocessing         │
  │  • Routes to model       │
  └────────┬────────────────┘
           ↓
  GKE (Model Inference)
  ┌─────────────────────────┐
  │  Inference Engine        │
  │  • Multi-modality models │
  │  • Grad-CAM heatmaps    │
  │  • Clinical reporting    │
  └────────┬────────────────┘
           ↓
  Dashboard renders:
  • Diagnosis + Severity Grade
  • Confidence Score (radial gauge)
  • Grad-CAM Attention Heatmap
  • Clinical Recommendation
  • Downloadable Report
```

---

## 📁 Project Structure

```
drscan-cam/
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Modalities, classes, severity levels
│   ├── schemas.py               # Pydantic request/response models
│   ├── routers/
│   │   ├── predict.py           # POST /api/predict
│   │   └── health.py            # GET /api/health
│   ├── services/
│   │   ├── inference.py         # Pluggable model inference engine
│   │   ├── preprocessing.py     # Image loading, DICOM, quality checks
│   │   ├── gradcam.py           # Grad-CAM heatmap generation
│   │   └── report.py            # Clinical report builder
│   ├── models/                  # Model weights (gitignored)
│   └── requirements.txt
│
├── frontend/
│   ├── index.html               # Dashboard SPA
│   ├── css/styles.css           # Premium dark-mode design system
│   ├── js/app.js                # Upload, API, result rendering
│   └── assets/                  # Sample images, favicon
│
└── .gitignore
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict` | Analyze medical image |
| `GET` | `/api/health` | System health check |
| `GET` | `/api/modalities` | List supported modalities |
| `GET` | `/api/docs` | Interactive API docs (Swagger) |

### POST /api/predict

```bash
curl -X POST http://localhost:8080/api/predict \
  -F "file=@chest_xray.png" \
  -F "modality=chest_xray"
```

**Response:**
```json
{
  "diagnosis": "Bacterial Pneumonia",
  "severity": 3,
  "severity_label": "Severe",
  "confidence": 0.92,
  "top_predictions": [
    {"class_name": "Bacterial Pneumonia", "confidence": 0.92},
    {"class_name": "Viral Pneumonia", "confidence": 0.04}
  ],
  "heatmap_base64": "/9j/4AAQ...",
  "recommendation": "Significant abnormalities detected...",
  "regions_of_concern": ["high-intensity region in inferior right area"],
  "inference_time_ms": 45.2
}
```

---

## 🩻 Supported Modalities

| Modality | Classes |
|----------|---------|
| **Chest X-Ray** | Normal, Bacterial Pneumonia, Viral Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax, Atelectasis, Lung Opacity |
| **Brain MRI** | Normal, Glioma, Meningioma, Pituitary Tumor, Metastatic Lesion, Multiple Sclerosis |
| **Lung CT** | Normal, Lung Nodule, Ground Glass Opacity, Consolidation, Emphysema, Pulmonary Fibrosis |

---

## 🔧 Tech Stack

- **Backend:** FastAPI, Python 3.13, OpenCV, NumPy, Pillow
- **Frontend:** HTML5, CSS3 (Glassmorphism), Vanilla JS
- **Medical:** DICOM (pydicom), Grad-CAM heatmaps
- **Deployment:** GKE + Cloud Run (GCP)

---

## 📄 License

Built for hackathon demonstration purposes. Not for clinical use without proper validation and regulatory approval.
