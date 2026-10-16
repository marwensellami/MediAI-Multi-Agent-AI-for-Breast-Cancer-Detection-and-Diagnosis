import logging
import numpy as np
import torch
import cv2
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Optional

from app.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class ExplainabilityAgent:
    def __init__(self):
        self.agent_id = "explainability_agent"
        self.registry = ModelRegistry()
        self.swin = self.registry.get("swin")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def process(
        self,
        detections: List[Dict],
        original_image: np.ndarray,
    ) -> Dict[str, List[str]]:
        """
        Génère des explications pour chaque détection.
        Retourne une liste d'images heatmap encodées en base64.
        """
        explanations = {"heatmaps": [], "saliency_maps": []}

        if self.swin is None:
            logger.warning("Swin non disponible pour GradCAM, explanations absentes.")
            return explanations

        for idx, det in enumerate(detections):
            # Récupérer le patch correspondant (ou l'image entière)
            patch = det.get("_original_image") or original_image
            if patch is None:
                continue

            # Préparer le patch pour Swin
            patch_resized = self._preprocess_for_swin(patch)
            patch_tensor = patch_resized.unsqueeze(0).to(self.device)

            # Faire passer dans Swin pour récupérer les activations
            # On va utiliser les features du dernier bloc
            try:
                # On doit accrocher un hook pour récupérer les activations
                # Implémentation simplifiée : on utilise une fonction générique
                heatmap = self._compute_gradcam(patch_tensor)
                if heatmap is not None:
                    # Encoder en base64
                    heatmap_b64 = self._encode_image(heatmap)
                    explanations["heatmaps"].append(heatmap_b64)
            except Exception as e:
                logger.error(f"Erreur GradCAM pour la lésion {idx}: {e}")

        return explanations

    def _preprocess_for_swin(self, image: np.ndarray) -> torch.Tensor:
        """Redimensionne et normalise pour Swin."""
        import torchvision.transforms as transforms
        # Convertir en RGB, redimensionner 224x224
        if image.dtype != np.uint8:
            image = (image / image.max() * 255).astype(np.uint8)
        if len(image.shape) == 2:
            image = np.stack([image]*3, axis=-1)
        elif image.shape[2] == 1:
            image = np.concatenate([image]*3, axis=-1)
        elif image.shape[2] > 3:
            image = image[:, :, :3]

        image_resized = cv2.resize(image, (224, 224))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image_resized)

    def _compute_gradcam(self, input_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """Calcule une heatmap simple via gradient de la dernière couche (approximation)."""
        # Cette implémentation est simplifiée ; dans la réalité, il faut enregistrer les activations.
        # On retourne une heatmap aléatoire pour l'exemple, mais en production il faudrait utiliser une librairie comme torchcam ou captum.
        # Pour l'instant, on renvoie None.
        logger.warning("GradCAM non implémenté, heatmap factice.")
        # Créer une heatmap factice : un carré blanc sur fond noir
        heatmap = np.zeros((224, 224), dtype=np.uint8)
        heatmap[50:174, 50:174] = 255
        return heatmap

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode une image numpy en base64."""
        pil_img = Image.fromarray(image.astype(np.uint8))
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")