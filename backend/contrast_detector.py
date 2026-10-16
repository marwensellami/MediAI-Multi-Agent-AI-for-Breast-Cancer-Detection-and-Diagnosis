# backend/app/agents/vision/contrast_detector.py
import logging
import numpy as np
import cv2
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ContrastDetector:
    """
    Détecteur de lésions basé sur le contraste local et la morphologie.
    """

    def __init__(self, min_area: int = 200, max_area: int = 20000, threshold_scale: float = 0.3):
        """
        Args:
            min_area: surface minimale en pixels pour une région
            max_area: surface maximale en pixels
            threshold_scale: facteur multiplicatif pour le seuil Otsu (plus bas = plus de régions)
        """
        self.min_area = min_area
        self.max_area = max_area
        self.threshold_scale = threshold_scale

    async def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Détecte les régions d'intérêt sur une image en niveaux de gris ou RGB.

        Returns:
            Liste de dictionnaires avec 'bbox' [x1, y1, x2, y2] et 'confidence' (fixée à 0.5)
        """
        # Convertir en niveaux de gris si RGB
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 1. Seuillage adaptatif (Otsu)
        # On applique un seuil sur une version floutée pour réduire le bruit
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Otsu donne un seuil automatique
        thresh_val, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # On ajuste le seuil pour être plus ou moins sensible
        adjusted_thresh = thresh_val * (1 - self.threshold_scale)
        _, thresh = cv2.threshold(blurred, adjusted_thresh, 255, cv2.THRESH_BINARY)

        # 2. Morphologie pour fermer les trous et lisser
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # 3. Extraction des contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            # Boîte englobante
            x, y, w, h = cv2.boundingRect(cnt)
            # Vérifier que la boîte n'est pas trop allongée (pour éviter les artefacts)
            ratio = w / h if h > 0 else 0
            if ratio > 2.5 or ratio < 0.4:
                continue

            # On augmente légèrement la boîte pour inclure le contexte
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray.shape[1], x + w + pad)
            y2 = min(gray.shape[0], y + h + pad)

            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": 0.5,  # confiance fixe, car pas de score de détection
                "class_id": 0,
                "class_name": "suspect_region",
            })

        logger.info(f"🔍 {len(detections)} région(s) suspecte(s) détectée(s) par contraste.")
        return detections