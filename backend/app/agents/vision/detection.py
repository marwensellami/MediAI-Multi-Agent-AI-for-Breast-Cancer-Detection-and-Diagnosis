# backend/app/agents/vision/detection.py
import logging
from typing import List, Dict, Any

import numpy as np
from ultralytics import YOLO

from app.ml.model_registry import ModelRegistry
from app.ml.loaders.vision_loader import VisionLoader

logger = logging.getLogger(__name__)


class DetectionAgent:
    """
    Agent de détection des lésions mammaires à l'aide de YOLO.
    Utilise le modèle fine‑tuné s'il est disponible, sinon le modèle générique.
    """

    def __init__(self, confidence_threshold: float = 0.4):
        """
        Args:
            confidence_threshold: Seuil de confiance pour les détections.
        """
        self.agent_id = "detection_agent"
        self.confidence_threshold = confidence_threshold

        # Charger le modèle YOLO (fine‑tuné en priorité)
        self.model: YOLO = self._load_model()

        if self.model is None:
            raise RuntimeError(
                "Aucun modèle YOLO n'a pu être chargé. "
                "Vérifiez que le modèle fine‑tuné existe ou que le modèle générique est disponible."
            )

    def _load_model(self) -> YOLO:
        """
        Charge le modèle YOLO en priorisant le modèle fine‑tuné.
        Si le fine‑tuné n'est pas disponible, charge le modèle générique depuis le registre.
        """
        # 1. Essayer de charger le modèle fine‑tuné
        try:
            model = VisionLoader.load_yolo_finetuned()
            if model is not None:
                logger.info("✅ YOLO fine‑tuné chargé avec succès.")
                return model
        except Exception as e:
            logger.warning(f"⚠️ Échec du chargement du modèle fine‑tuné : {e}")

        # 2. Fallback : charger le modèle générique depuis le registre
        registry = ModelRegistry()
        model = registry.get("yolo")
        if model is not None:
            logger.info("✅ YOLO générique chargé depuis le registre.")
            return model

        # 3. Dernier recours : charger directement depuis VisionLoader (téléchargement)
        logger.info("📥 Chargement de YOLO générique (téléchargement direct)...")
        model = VisionLoader.load_yolo()
        if model is not None:
            # Enregistrer dans le registre pour les prochaines fois
            registry.register("yolo", model)
            logger.info("✅ YOLO générique chargé et enregistré.")
            return model

        raise RuntimeError("Impossible de charger YOLO (aucun modèle disponible).")

    async def process(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Détecte les lésions sur une image RGB uint8.

        Parameters
        ----------
        image : np.ndarray
            Image RGB préparée pour YOLO.

        Returns
        -------
        List[Dict]
            Liste des détections.
        """
        # -----------------------------
        # Normalisation de l'image
        # -----------------------------
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.max() > 0:
                image = image / image.max()
            image = (image * 255).astype(np.uint8)

        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)

        image_height, image_width = image.shape[:2]

        # -----------------------------
        # Inférence YOLO avec le seuil de confiance
        # -----------------------------
        results = self.model(image, verbose=False, conf=self.confidence_threshold)

        detections = []

        if not results:
            logger.info("Aucune détection.")
            return detections

        boxes = results[0].boxes
        if boxes is None:
            logger.info("Aucune détection.")
            return detections

        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < self.confidence_threshold:
                continue

            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Sécurité : s'assurer que les coordonnées sont dans l'image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_width, x2)
            y2 = min(image_height, y2)

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            center_x = x1 + width / 2
            center_y = y1 + height / 2

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "bbox_width_px": width,
                "bbox_height_px": height,
                "center": [center_x, center_y],
                "model_image_shape": (image_height, image_width),
                "confidence": confidence,
                "class_id": class_id,
                "class_name": self.model.names[class_id],
            })

        logger.info(f"🔍 {len(detections)} détection(s) trouvée(s).")
        return detections