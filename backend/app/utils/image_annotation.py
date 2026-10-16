import cv2
import numpy as np
import base64
from typing import List, Dict, Any

def draw_lesions_on_image(image: np.ndarray, detections: List[Dict[str, Any]]) -> str:
    """
    Dessine des ellipses avec remplissage semi-transparent autour des lésions.
    Rouge pour Malin, Vert pour Bénin.
    """
    # S'assurer que l'image est en BGR pour pouvoir ajouter de la couleur
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Niveaux de gris → BGR
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
        annotated = image.copy()
    else:
        # Autre cas (RGBA, etc.) → convertir en BGR
        annotated = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    overlay = annotated.copy()  # Calque pour les remplissages

    for det in detections:
        bbox = det.get("bbox")
        label = det.get("malignancy_label", "Inconnu")
        if not bbox or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius_x = (x2 - x1) // 2
        radius_y = (y2 - y1) // 2

        # Choix des couleurs (BGR)
        if label.lower() in ["malin", "maligne", "malignant"]:
            color = (0, 0, 255)      # Rouge
        elif label.lower() in ["bénin", "bénigne", "benign"]:
            color = (0, 255, 0)      # Vert
        else:
            color = (255, 255, 0)    # Jaune pour indéterminé

        # Dessiner une ellipse remplie (semi-transparent) sur le calque
        if radius_x > 0 and radius_y > 0:
            cv2.ellipse(overlay, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, color, -1)
            # Contour épais pour bien délimiter
            cv2.ellipse(annotated, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, color, 2)
        else:
            # Fallback rectangle si l'ellipse est invalide
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Pas de texte → on supprime cv2.putText

    # Fusionner le calque avec l'image de base (transparence 40%)
    alpha = 0.4
    annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)

    # Encoder en PNG
    _, buffer = cv2.imencode('.png', annotated)
    return base64.b64encode(buffer).decode('utf-8')