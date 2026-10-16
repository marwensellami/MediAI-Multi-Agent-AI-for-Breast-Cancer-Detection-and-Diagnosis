# backend/app/utils/anatomy.py
"""
Utilitaires pour convertir des coordonnées en localisations anatomiques.
"""

def get_quadrant(bbox, image_shape):
    """
    Convertit une boîte englobante (x1, y1, x2, y2) en quadrant anatomique.

    Args:
        bbox: liste de 4 entiers [x1, y1, x2, y2]
        image_shape: tuple (hauteur, largeur) de l'image

    Returns:
        str: description du quadrant (ex: "Quadrant supéro-externe")
    """
    if not bbox or len(bbox) < 4:
        return "Localisation inconnue"

    x1, y1, x2, y2 = bbox[:4]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    h, w = image_shape[:2]

    if cx < w / 2 and cy < h / 2:
        return "Quadrant supéro-interne"
    elif cx >= w / 2 and cy < h / 2:
        return "Quadrant supéro-externe"
    elif cx < w / 2 and cy >= h / 2:
        return "Quadrant inféro-interne"
    else:
        return "Quadrant inféro-externe"

def get_side(patient_position: str = None) -> str:
    """
    Retourne le côté du sein (gauche/droit) si disponible.
    """
    if patient_position:
        if "L" in patient_position.upper():
            return "gauche"
        elif "R" in patient_position.upper():
            return "droit"
    return "non spécifié"

def get_view_description(view_code: str) -> str:
    """
    Retourne la description complète d'une vue mammographique.
    
    Args:
        view_code: Code de la vue (CCLT, CCRT, MLOLT, MLORT)
    
    Returns:
        Description complète avec le code entre parenthèses
    """
    view_map = {
        "CCLT": "cranio-caudale du sein gauche (CCLT)",
        "CCRT": "cranio-caudale du sein droit (CCRT)",
        "MLOLT": "médio-latérale oblique du sein gauche (MLOLT)",
        "MLORT": "médio-latérale oblique du sein droit (MLORT)",
    }
    return view_map.get(view_code, view_code or "Non spécifiée")

def get_density_description(density_code: str) -> str:
    """
    Retourne la description complète de la densité mammaire.
    
    Args:
        density_code: Code de densité (F, G, D)
    
    Returns:
        Description complète
    """
    density_map = {
        "F": "sein graisseux (adipeux)",
        "G": "sein graisseux-glandulaire",
        "D": "sein glandulaire dense (très dense)",
    }
    return density_map.get(density_code, density_code or "Non spécifiée")