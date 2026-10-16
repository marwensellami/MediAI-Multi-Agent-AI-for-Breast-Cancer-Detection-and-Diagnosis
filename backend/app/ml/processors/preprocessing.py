# backend/app/ml/processors/preprocessing.py
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    @staticmethod
    def create_breast_mask(image: np.ndarray) -> np.ndarray:
        """
        Crée un masque binaire pour isoler le sein (région claire)
        et supprimer le fond noir ou gris.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # 1. Seuillage adaptatif pour séparer le sein du fond
        # Otsu donne un bon seuil automatique
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 2. Nettoyer le masque (enlever le bruit, fermer les trous)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 3. Trouver le plus grand contour (le sein)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest], -1, 255, -1)

        return mask

    @staticmethod
    def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Applique le masque sur l'image (fond mis à 0)."""
        if len(image.shape) == 3:
            # Appliquer le masque sur chaque canal
            masked = image.copy()
            for c in range(3):
                masked[:, :, c] = cv2.bitwise_and(masked[:, :, c], masked[:, :, c], mask=mask)
            return masked
        else:
            return cv2.bitwise_and(image, image, mask=mask)

    @staticmethod
    def preprocess_for_yolo(image: np.ndarray) -> np.ndarray:
        """
        Prétraitement complet pour YOLO :
        - Création du masque du sein
        - Application du masque
        - Retourne l'image avec fond noir
        """
        # Si image en RGB, on prend le canal de luminance pour le masque
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Masque du sein
        mask = Preprocessor.create_breast_mask(gray)

        # Appliquer le masque
        if len(image.shape) == 3:
            masked = image.copy()
            for c in range(3):
                masked[:, :, c] = cv2.bitwise_and(masked[:, :, c], masked[:, :, c], mask=mask)
            return masked
        else:
            return cv2.bitwise_and(image, image, mask=mask)