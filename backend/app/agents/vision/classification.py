# backend/app/agents/vision/classification.py
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from typing import List, Dict, Any

from app.ml.model_registry import ModelRegistry
from app.ml.loaders.vision_loader import VisionLoader
from app.ml.models import SwinClassifier

logger = logging.getLogger(__name__)

# Statistiques ImageNet pour la normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Mapping des classes pour Swin fine‑tuné (3 classes)
CLASS_MAPPING = {
    0: "Normal",
    1: "Bénin",
    2: "Malin",
}

class ClassificationAgent:
    """
    Agent de classification des lésions mammaires.
    Utilise Swin Transformer fine‑tuné (3 classes : Normal, Bénin, Malin).
    """

    def __init__(self):
        self.agent_id = "classification_agent"
        self.registry = ModelRegistry()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = (224, 224)

        # Charger le modèle Swin fine‑tuné (3 classes)
        self.model = self._load_model()

        # Transformations pour le prétraitement des patches
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        logger.info("✅ Classification Agent initialisé avec Swin fine‑tuné (3 classes).")

    def _load_model(self) -> nn.Module:
        """
        Charge le modèle Swin fine‑tuné (3 classes) depuis le cache ou le registre.
        Si le fine‑tuné n'est pas disponible, charge un backbone Swin générique + une tête aléatoire (3 classes).
        """
        # 1. Essayer de charger le modèle fine‑tuné
        try:
            model = VisionLoader.load_swin_finetuned()
            if model is not None:
                logger.info("✅ Swin fine‑tuné (3 classes) chargé avec succès.")
                return model.to(self.device).eval()
        except Exception as e:
            logger.warning(f"⚠️ Échec du chargement du modèle fine‑tuné : {e}")

        # 2. Fallback : charger le backbone générique + tête aléatoire (3 classes)
        logger.warning("⚠️ Utilisation du backbone Swin générique avec tête aléatoire (3 classes).")
        backbone = VisionLoader.load_swin()
        model = SwinClassifier(backbone, num_classes=3)
        # Initialisation aléatoire de la tête
        model.fc = nn.Linear(model.num_features, 3)
        model.to(self.device)
        model.eval()
        logger.info("✅ Swin générique + tête aléatoire (3 classes) chargé.")
        return model

    async def process(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Pour chaque détection, extrait le patch, le classifie et met à jour les champs :
        - malignancy_score : probabilité de la classe "Malin" (index 2)
        - malignancy_label : "Normal", "Bénin" ou "Malin"
        - feature_vector : vecteur de caractéristiques (embedding)
        """
        if not detections:
            return detections

        with torch.no_grad():
            for det in detections:
                # 1. Récupérer l'image originale et le masque/bbox
                image = det.get("_original_image")
                if image is None:
                    logger.warning("⚠️ Pas d'image originale dans la détection, impossible de classifier.")
                    self._set_default_classification(det)
                    continue

                mask = det.get("mask")
                bbox = det.get("bbox")

                # 2. Extraire le patch à partir du masque ou de la bbox
                patch = self._extract_patch(image, mask, bbox)
                if patch is None:
                    logger.warning("⚠️ Échec de l'extraction du patch.")
                    self._set_default_classification(det)
                    continue

                # 3. Prétraiter le patch pour Swin
                input_tensor = self._preprocess_patch(patch)
                input_tensor = input_tensor.unsqueeze(0).to(self.device)

                # 4. Inférence
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                prob_normal, prob_benign, prob_malignant = probs[0].tolist()

                # 5. Classe prédite
                pred_class_id = torch.argmax(probs, dim=1).item()

                # ------------------------------------------------------------
                # RÈGLE CLINIQUE : une région détectée par YOLO ne peut pas être "Normal".
                # Si la classe prédite est Normal (index 0), on la remplace par
                # Bénin (1) ou Malin (2) selon la probabilité la plus élevée.
                # ------------------------------------------------------------
                if pred_class_id == 0:  # Normal
                    if probs[0, 1] >= probs[0, 2]:   # Bénin plus probable que Malin
                        pred_class_id = 1
                    else:
                        pred_class_id = 2
                pred_class_name = CLASS_MAPPING.get(pred_class_id, "Inconnu")

                # 6. Score de malignité (probabilité de la classe Malin) – inchangé
                malignancy_score = prob_malignant

                # 7. Vecteur de caractéristiques (embedding) : on stocke les 3 probabilités
                features = probs.cpu().numpy().flatten().tolist()

                # 8. Mise à jour de la détection
                det["malignancy_score"] = malignancy_score
                det["malignancy_label"] = pred_class_name
                det["feature_vector"] = features
                det["class_probabilities"] = {
                    "normal": prob_normal,
                    "benign": prob_benign,
                    "malignant": prob_malignant,
                }

                logger.debug(
                    f"Classification : {pred_class_name} (score={malignancy_score:.4f}), "
                    f"probas: N={prob_normal:.3f}, B={prob_benign:.3f}, M={prob_malignant:.3f}"
                )

        return detections

    def _extract_patch(self, image: np.ndarray, mask: np.ndarray = None, bbox: List[int] = None) -> np.ndarray:
        """
        Extrait le patch de l'image à partir du masque ou de la bounding box.
        """
        if mask is not None and np.any(mask > 0):
            # Utiliser le masque pour délimiter la région
            coords = np.where(mask > 0)
            if len(coords[0]) == 0:
                return None
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            # Ajouter une petite marge
            pad = 10
            y_min = max(0, y_min - pad)
            y_max = min(image.shape[0], y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(image.shape[1], x_max + pad)
            patch = image[y_min:y_max, x_min:x_max]
            return patch

        elif bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Ajouter une marge de 10% pour le contexte
            w = x2 - x1
            h = y2 - y1
            pad_x = int(0.1 * w)
            pad_y = int(0.1 * h)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(image.shape[1], x2 + pad_x)
            y2 = min(image.shape[0], y2 + pad_y)
            patch = image[y1:y2, x1:x2]
            return patch

        return None

    def _preprocess_patch(self, patch: np.ndarray) -> torch.Tensor:
        """
        Prétraite le patch pour Swin (redimensionnement, normalisation).
        """
        # Convertir en RGB si nécessaire
        if len(patch.shape) == 2:
            patch = np.stack([patch] * 3, axis=-1)
        elif patch.shape[2] == 1:
            patch = np.concatenate([patch] * 3, axis=-1)
        elif patch.shape[2] > 3:
            patch = patch[:, :, :3]

        # Appliquer les transformations
        tensor = self.transform(patch)
        return tensor

    def _set_default_classification(self, det: Dict):
        """
        Définit des valeurs par défaut en cas d'échec de la classification.
        """
        det["malignancy_score"] = 0.5
        det["malignancy_label"] = "Indéterminé"
        det["feature_vector"] = []
        det["class_probabilities"] = {"normal": 0.0, "benign": 0.0, "malignant": 0.0}