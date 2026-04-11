"""
DRScan Cam — Inference Engine
Pluggable model inference supporting:
  - biomedclip: Zero-shot classification via BiomedCLIP (default when available)
  - demo: Feature-based analysis fallback
  - custom: User-provided model

On startup, attempts to load BiomedCLIP. Falls back to demo mode if unavailable.
"""

import time
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
import cv2

from backend.config import (
    Modality,
    InferenceMode,
    MODALITY_CLASSES,
    SEVERITY_LEVELS,
)
from backend.services.retinal import retinal_engine


class InferenceEngine:
    """
    Pluggable inference engine.
    Auto-selects BiomedCLIP if available, otherwise falls back to demo mode.
    """

    def __init__(self):
        self.mode = InferenceMode.DEMO
        self.model = None
        self._model_version = "demo-v1"
        self._biomedclip = None

    def initialize(self):
        """Try to load BiomedCLIP. Falls back to demo mode if unavailable."""
        try:
            from backend.services.biomedclip import biomedclip
            biomedclip.load()
            self._biomedclip = biomedclip
            self.mode = "biomedclip"
            self._model_version = "BiomedCLIP-v1"
            print("[Inference] BiomedCLIP loaded successfully - using AI mode")
        except Exception as e:
            print(f"[Inference] BiomedCLIP not available ({e}), using demo mode")
            self.mode = InferenceMode.DEMO
            self._model_version = "demo-v1"

        # Initialize retinal engine
        retinal_engine.initialize()
        print(f"[Inference] Retinal engine: {retinal_engine.mode}")

    def load_custom_model(self, model_path: str, framework: str = "tensorflow"):
        """Load a custom trained model."""
        if framework == "tensorflow":
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
        elif framework == "pytorch":
            import torch
            self.model = torch.load(model_path, map_location="cpu")
            self.model.eval()
        self.mode = InferenceMode.CUSTOM
        self._model_version = f"custom-{framework}"

    def predict(
        self,
        image: Image.Image,
        preprocessed: np.ndarray,
        modality: str,
    ) -> Dict:
        """
        Run inference on a medical image.

        Returns dict with:
          - diagnosis: str
          - severity: int (0-4)
          - confidence: float
          - top_predictions: list of {class_name, confidence}
          - inference_time_ms: float
          - model_version: str
        """
        # BiomedCLIP mode (real AI)
        if self.mode == "biomedclip" and self._biomedclip is not None:
            return self._biomedclip.predict(image, modality)

        # Custom model mode
        if self.mode == InferenceMode.CUSTOM and self.model is not None:
            result = self._custom_predict(preprocessed, modality)
            return result

        # Demo mode (fallback)
        start = time.perf_counter()
        result = self._demo_predict(image, modality)
        elapsed_ms = (time.perf_counter() - start) * 1000
        result["inference_time_ms"] = round(elapsed_ms, 1)
        result["model_version"] = self._model_version
        return result

    def get_attention_map(self, image: Image.Image) -> Optional[np.ndarray]:
        """Get attention map from model if available."""
        if self._biomedclip is not None:
            return self._biomedclip.get_attention_map(image)
        return None

    # ------------------------------------------------------------------
    # Demo mode — image feature analysis (fallback when no model loaded)
    # ------------------------------------------------------------------
    def _demo_predict(self, image: Image.Image, modality: str) -> Dict:
        """Analyze image features to produce plausible, varying predictions."""
        classes = MODALITY_CLASSES.get(modality, MODALITY_CLASSES[Modality.CHEST_XRAY])
        features = self._extract_features(image)
        raw_scores = self._feature_to_scores(features, classes, modality)
        probs = self._softmax(raw_scores, temperature=1.5)

        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)

        top_predictions = [
            {"class_name": classes[i], "confidence": round(float(p), 4)}
            for i, p in indexed
        ]

        top_idx, top_conf = indexed[0]
        diagnosis = classes[top_idx]
        severity = self._confidence_to_severity(top_conf, top_idx)

        return {
            "diagnosis": diagnosis,
            "severity": severity,
            "confidence": round(float(top_conf), 4),
            "top_predictions": top_predictions,
        }

    def _extract_features(self, image: Image.Image) -> Dict[str, float]:
        """Extract visual features from the image."""
        gray = np.array(image.convert("L"), dtype=np.float32)
        rgb = np.array(image.convert("RGB"), dtype=np.float32)

        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 128.0

        hist, _ = np.histogram(gray.ravel(), bins=64, range=(0, 255))
        hist = hist.astype(np.float32) / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        edge_density = np.mean(edges > 0)

        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        texture = np.var(laplacian) / 10000.0

        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        if right_half.shape[1] < left_half.shape[1]:
            right_half = np.pad(right_half, ((0, 0), (0, 1)), mode='edge')
        asymmetry = np.mean(np.abs(left_half - np.fliplr(right_half))) / 255.0

        cy, cx = h // 2, w // 2
        r = min(h, w) // 4
        yy, xx = np.ogrid[:h, :w]
        center_mask = ((yy - cy)**2 + (xx - cx)**2) <= r**2
        center_intensity = np.mean(gray[center_mask]) / 255.0
        peripheral_intensity = np.mean(gray[~center_mask]) / 255.0
        center_bias = center_intensity - peripheral_intensity

        return {
            "brightness": brightness,
            "contrast": contrast,
            "entropy": entropy / 6.0,
            "edge_density": edge_density,
            "texture": min(texture, 1.0),
            "asymmetry": asymmetry,
            "center_bias": center_bias + 0.5,
        }

    def _feature_to_scores(
        self, features: Dict[str, float], classes: List[str], modality: str
    ) -> np.ndarray:
        """Map image features to class scores."""
        n = len(classes)
        scores = np.zeros(n, dtype=np.float32)

        b = features["brightness"]
        c = features["contrast"]
        ed = features["edge_density"]
        t = features["texture"]
        a = features["asymmetry"]
        cb = features["center_bias"]

        feature_hash = hash(
            (round(b, 2), round(c, 2), round(features["entropy"], 2), round(ed, 3))
        ) % 1000 / 1000.0

        scores[0] = (b * 0.4 + (1 - a) * 0.3 + (1 - ed) * 0.3) * 2.5

        for i in range(1, n):
            phase = (i * 2.3 + feature_hash * 6.28)
            w_bright = 0.5 + 0.5 * np.sin(phase)
            w_edge = 0.5 + 0.5 * np.cos(phase * 0.7)
            w_asym = 0.5 + 0.5 * np.sin(phase * 1.3)
            w_text = 0.5 + 0.5 * np.cos(phase * 0.5)

            scores[i] = (
                w_bright * (1 - b) * 0.5
                + w_edge * ed * 0.8
                + w_asym * a * 0.6
                + w_text * t * 0.4
                + (1 - scores[0] / 2.5) * 0.5
                + cb * 0.2
            )

        for i in range(n):
            phase_i = (i * 1.7 + feature_hash * 3.14)
            scores[i] += 0.15 * np.sin(phase_i)

        return scores

    def _softmax(self, scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Temperature-scaled softmax."""
        scores = scores / temperature
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()

    def _confidence_to_severity(self, confidence: float, class_idx: int) -> int:
        """Map prediction to severity grade."""
        if class_idx == 0:
            return 0
        if confidence > 0.85:
            return 4
        elif confidence > 0.70:
            return 3
        elif confidence > 0.50:
            return 2
        elif confidence > 0.30:
            return 1
        else:
            return 0

    # ------------------------------------------------------------------
    # Custom model inference
    # ------------------------------------------------------------------
    def _custom_predict(self, preprocessed: np.ndarray, modality: str) -> Dict:
        """Run inference using the loaded custom model."""
        classes = MODALITY_CLASSES.get(modality, MODALITY_CLASSES[Modality.CHEST_XRAY])
        start = time.perf_counter()

        if self.model is None:
            raise RuntimeError("No custom model loaded.")

        try:
            input_data = np.expand_dims(preprocessed, axis=0)
            predictions = self.model.predict(input_data, verbose=0)
            probs = predictions[0]
        except Exception:
            import torch
            tensor = torch.from_numpy(preprocessed).unsqueeze(0).permute(0, 3, 1, 2).float()
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1).numpy()[0]

        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)

        top_predictions = [
            {"class_name": classes[min(i, len(classes)-1)], "confidence": round(float(p), 4)}
            for i, p in indexed
        ]

        top_idx, top_conf = indexed[0]
        diagnosis = classes[min(top_idx, len(classes)-1)]
        severity = self._confidence_to_severity(float(top_conf), top_idx)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "diagnosis": diagnosis,
            "severity": severity,
            "confidence": round(float(top_conf), 4),
            "top_predictions": top_predictions,
            "inference_time_ms": round(elapsed_ms, 1),
            "model_version": self._model_version,
        }


# Global singleton
engine = InferenceEngine()
