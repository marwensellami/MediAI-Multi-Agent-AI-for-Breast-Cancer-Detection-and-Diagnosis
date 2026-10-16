# backend/app/utils/birads_utils.py
"""
Utilitaires pour la classification BI-RADS basée sur les scores de malignité.
Centralise les descriptions, seuils et recommandations pour les 6 catégories BI-RADS.
"""

from typing import Dict, Any, Optional

BI_RADS_DATA = [
    {
        "level": "BI-RADS 1",
        "description": "Négatif / Sein strictement normal",
        "score_range": "0 %",
        "recommendation": "Poursuivre le calendrier de dépistage de routine par mammographie.",
        "birads_number": 1,
        "subcategory": None,
    },
    {
        "level": "BI-RADS 2",
        "description": "Anomalie typiquement bénigne (ex: kyste simple)",
        "score_range": "≤ 1 %",
        "recommendation": "Poursuivre le calendrier de dépistage habituel, sans examen intermédiaire.",
        "birads_number": 2,
        "subcategory": None,
    },
    {
        "level": "BI-RADS 3",
        "description": "Anomalie probablement bénigne (à contrôler)",
        "score_range": "≤ 2 %",
        "recommendation": "Planifier un contrôle d'imagerie rapproché à 6 mois pour vérifier la stabilité.",
        "birads_number": 3,
        "subcategory": None,
    },
    {
        "level": "BI-RADS 4A",
        "description": "Anomalie avec faible suspicion de malignité",
        "score_range": "> 2 % à ≤ 10 %",
        "recommendation": "Effectuer une biopsie guidée par imagerie pour écarter formellement le doute.",
        "birads_number": 4,
        "subcategory": "A",
    },
    {
        "level": "BI-RADS 4B",
        "description": "Anomalie avec suspicion modérée de malignité",
        "score_range": "> 10 % à ≤ 50 %",
        "recommendation": "Réaliser obligatoirement une biopsie pour obtenir un diagnostic cellulaire.",
        "birads_number": 4,
        "subcategory": "B",
    },
    {
        "level": "BI-RADS 4C",
        "description": "Anomalie avec forte suspicion de malignité",
        "score_range": "> 50 % à < 95 %",
        "recommendation": "Réaliser le prélèvement rapidement pour confirmer la nature de l'anomalie.",
        "birads_number": 4,
        "subcategory": "C",
    },
    {
        "level": "BI-RADS 5",
        "description": "Anomalie hautement évocatrice de malignité",
        "score_range": "≥ 95 %",
        "recommendation": "Programmer une biopsie immédiate et consulter un chirurgien ou oncologue.",
        "birads_number": 5,
        "subcategory": None,
    },
    {
        "level": "BI-RADS 6",
        "description": "Malignité déjà prouvée (biopsie antérieure)",
        "score_range": "100 %",
        "recommendation": "Valider le calendrier thérapeutique lors de la concertation pluridisciplinaire.",
        "birads_number": 6,
        "subcategory": None,
    },
]


def get_birads_info(score: float, confirmed_malignant: bool = False) -> Dict[str, Any]:
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"Le score de malignité doit être entre 0 et 1, reçu : {score}")

    if confirmed_malignant:
        return {
            "level": "BI-RADS 6",
            "description": "Malignité déjà prouvée (biopsie antérieure)",
            "score_range": "100 %",
            "recommendation": "Valider le calendrier thérapeutique lors de la concertation pluridisciplinaire.",
            "birads_number": 6,
            "subcategory": None,
        }

    # === SEUILS CORRECTS ===
    # BI-RADS 1 : score == 0.0
    if score == 0.0:
        idx = 0
    # BI-RADS 2 : ≤ 1 %
    elif score <= 0.01:
        idx = 1
    # BI-RADS 3 : ≤ 2 %
    elif score <= 0.02:
        idx = 2
    # BI-RADS 4A : > 2 % à ≤ 10 %
    elif score <= 0.10:
        idx = 3
    # BI-RADS 4B : > 10 % à ≤ 50 %
    elif score <= 0.50:
        idx = 4
    # BI-RADS 4C : > 50 % à < 95 %
    elif score < 0.95:
        idx = 5
    # BI-RADS 5 : ≥ 95 %
    else:
        idx = 6

    birads = BI_RADS_DATA[idx]
    return {
        "level": birads["level"],
        "description": birads["description"],
        "score_range": birads["score_range"],
        "recommendation": birads["recommendation"],
        "birads_number": birads["birads_number"],
        "subcategory": birads["subcategory"],
    }


def get_birads_label(score: float, confirmed_malignant: bool = False) -> str:
    return get_birads_info(score, confirmed_malignant)["level"]


def get_birads_number(score: float, confirmed_malignant: bool = False) -> int:
    return get_birads_info(score, confirmed_malignant)["birads_number"]


def is_birads_suspect(score: float, confirmed_malignant: bool = False) -> bool:
    return get_birads_number(score, confirmed_malignant) >= 4


def get_recommendation(score: float, confirmed_malignant: bool = False) -> str:
    return get_birads_info(score, confirmed_malignant)["recommendation"]