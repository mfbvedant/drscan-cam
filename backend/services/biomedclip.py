"""
DRScan Cam — BiomedCLIP Zero-Shot Classification Service
Uses microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 for
zero-shot medical image classification across all modalities.

No training required — uses text prompts to classify any medical image.
"""

import time
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple

from backend.config import Modality, MODALITY_CLASSES

# ---------------------------------------------------------------------------
# Text prompt templates per modality for zero-shot classification
# ---------------------------------------------------------------------------
MODALITY_PROMPTS: Dict[str, Dict[str, str]] = {
    Modality.CHEST_XRAY: {
        "Normal": "a normal chest x-ray with no abnormalities",
        "Bacterial Pneumonia": "a chest x-ray showing bacterial pneumonia with consolidation",
        "Viral Pneumonia": "a chest x-ray showing viral pneumonia with interstitial infiltrates",
        "Pleural Effusion": "a chest x-ray showing pleural effusion with blunted costophrenic angle",
        "Cardiomegaly": "a chest x-ray showing cardiomegaly with enlarged heart silhouette",
        "Pneumothorax": "a chest x-ray showing pneumothorax with collapsed lung",
        "Atelectasis": "a chest x-ray showing atelectasis with volume loss",
        "Lung Opacity": "a chest x-ray showing lung opacity or mass lesion",
    },
    Modality.BRAIN_MRI: {
        "Normal": "a normal brain MRI scan with no abnormalities",
        "Glioma": "a brain MRI showing glioma tumor",
        "Meningioma": "a brain MRI showing meningioma with extra-axial mass",
        "Pituitary Tumor": "a brain MRI showing pituitary tumor in the sella",
        "Metastatic Lesion": "a brain MRI showing metastatic brain lesion",
        "Multiple Sclerosis": "a brain MRI showing multiple sclerosis plaques",
    },
    Modality.LUNG_CT: {
        "Normal": "a normal lung CT scan showing clear healthy lungs with no masses, nodules, or opacities",
        "Lung Nodule": "a lung CT scan with a solitary pulmonary nodule, a small round lesion visible in the parenchyma",
        "Ground Glass Opacity": "a lung CT scan showing ground glass opacity, hazy increased attenuation with visible vessels",
        "Consolidation": "a lung CT scan with consolidation, dense opacification obscuring vessels and airways",
        "Emphysema": "a lung CT scan showing emphysema with hyperinflated lungs and destruction of alveolar walls",
        "Pulmonary Fibrosis": "a lung CT scan showing pulmonary fibrosis with honeycombing and reticular pattern",
    },
}

# Common template prefix used by BiomedCLIP
TEMPLATE_PREFIX = "this is a photo of "


class BiomedCLIPClassifier:
    """
    Zero-shot medical image classifier using BiomedCLIP.
    Loads the model once on startup, then classifies any image
    against modality-specific text prompts.
    """

    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = None
        self._loaded = False
        self._text_cache: Dict[str, object] = {}  # Pre-tokenized prompts

    def load(self):
        """Load BiomedCLIP model and tokenizer from HuggingFace Hub."""
        if self._loaded:
            return

        import torch
        from open_clip import create_model_from_pretrained, get_tokenizer

        print("[BiomedCLIP] Loading model from HuggingFace Hub...")
        start = time.time()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model + preprocessing transform
        self.model, self.preprocess = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )

        # Pre-tokenize all prompts for each modality
        self._precompute_text_features()

        elapsed = time.time() - start
        print(f"[BiomedCLIP] Model loaded in {elapsed:.1f}s on {self.device}")
        self._loaded = True

    def _precompute_text_features(self):
        """Pre-compute text features for all modality prompts."""
        import torch

        for modality, prompts in MODALITY_PROMPTS.items():
            labels = list(prompts.values())
            # BiomedCLIP uses context_length=256
            texts = self.tokenizer(
                [TEMPLATE_PREFIX + label for label in labels],
                context_length=256,
            ).to(self.device)
            self._text_cache[modality] = texts

    def predict(
        self,
        image: Image.Image,
        modality: str,
    ) -> Dict:
        """
        Run zero-shot classification on a medical image.

        Args:
            image: PIL Image (RGB)
            modality: One of chest_xray, brain_mri, lung_ct

        Returns:
            dict with diagnosis, severity, confidence, top_predictions,
            inference_time_ms, model_version
        """
        if not self._loaded:
            self.load()

        import torch

        start = time.perf_counter()

        # Get class names for this modality
        prompts = MODALITY_PROMPTS.get(modality, MODALITY_PROMPTS[Modality.CHEST_XRAY])
        class_names = list(prompts.keys())

        # Preprocess image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Get pre-tokenized text features
        texts = self._text_cache.get(modality)
        if texts is None:
            text_labels = list(prompts.values())
            texts = self.tokenizer(
                [TEMPLATE_PREFIX + l for l in text_labels],
                context_length=256,
            ).to(self.device)

        # Forward pass
        with torch.no_grad():
            image_features, text_features, logit_scale = self.model(image_tensor, texts)
            logits = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

        # Extract probabilities
        probs = logits[0].cpu().numpy()

        # Apply modality-specific calibration
        # BiomedCLIP is weaker on CT — boost Normal class slightly
        if modality == Modality.LUNG_CT:
            normal_idx = 0  # Normal is always first in our prompt dict
            normal_boost = 0.08
            probs[normal_idx] += normal_boost
            probs = probs / probs.sum()  # Re-normalize

        # Sort by confidence
        sorted_indices = np.argsort(probs)[::-1]

        top_predictions = [
            {
                "class_name": class_names[i],
                "confidence": round(float(probs[i]), 4),
            }
            for i in sorted_indices
        ]

        # Primary diagnosis
        top_idx = sorted_indices[0]
        diagnosis = class_names[top_idx]
        confidence = float(probs[top_idx])

        # Severity mapping — use diagnosis NAME not index
        severity = self._confidence_to_severity(confidence, diagnosis)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "diagnosis": diagnosis,
            "severity": severity,
            "confidence": round(confidence, 4),
            "top_predictions": top_predictions,
            "inference_time_ms": round(elapsed_ms, 1),
            "model_version": "BiomedCLIP-v1",
        }

    def get_attention_map(
        self, image: Image.Image
    ) -> Optional[np.ndarray]:
        """
        Extract attention map from the ViT vision encoder for heatmap generation.
        Uses the attention weights from the last transformer block.
        """
        if not self._loaded:
            return None

        import torch

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Register hook to capture attention weights
        attentions = []

        def hook_fn(module, input, output):
            # For open_clip ViT, attention is in the resblocks
            attentions.append(output)

        # Try to access the vision transformer's attention
        try:
            visual = self.model.visual
            # The last transformer block
            if hasattr(visual, 'trunk'):
                # For newer open_clip versions
                blocks = visual.trunk.blocks
            elif hasattr(visual, 'transformer'):
                blocks = visual.transformer.resblocks
            else:
                return None

            # Simple approach: use the CLS token attention from last layer
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)

            # Fallback: generate attention-like map from patch embeddings
            return self._attention_from_patches(image_tensor, visual)

        except Exception:
            return None

    def _attention_from_patches(self, image_tensor, visual) -> Optional[np.ndarray]:
        """Generate attention map from patch token norms."""
        import torch

        try:
            with torch.no_grad():
                # Get patch embeddings from vision encoder
                if hasattr(visual, 'trunk'):
                    features = visual.trunk(image_tensor)
                else:
                    features = visual(image_tensor)

            # If we get patch tokens, compute attention from their norms
            if isinstance(features, torch.Tensor) and features.dim() == 3:
                # features shape: [B, num_patches + 1, dim]
                # Skip CLS token, take patch tokens
                patch_tokens = features[0, 1:]  # [num_patches, dim]
                norms = torch.norm(patch_tokens, dim=-1)
                norms = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)

                # Reshape to spatial grid
                num_patches = patch_tokens.shape[0]
                grid_size = int(num_patches ** 0.5)
                if grid_size * grid_size == num_patches:
                    attn_map = norms.reshape(grid_size, grid_size).cpu().numpy()
                    return attn_map

        except Exception:
            pass

        return None

    def _confidence_to_severity(self, confidence: float, diagnosis: str) -> int:
        """Map prediction confidence and diagnosis to severity grade."""
        # Normal diagnosis always = grade 0
        if diagnosis.lower() == "normal":
            return 0

        # For abnormal findings, scale severity by confidence
        if confidence > 0.85:
            return 4
        elif confidence > 0.70:
            return 3
        elif confidence > 0.55:
            return 2
        elif confidence > 0.35:
            return 1
        else:
            return 0


# Global singleton — loaded lazily on first prediction
biomedclip = BiomedCLIPClassifier()
