# backend/app/ml/loaders/swin_finetuned_loader.py
"""
Chargeur pour le modèle Swin fine‑tuné sur les mammographies.
"""
import torch
import logging
from pathlib import Path

from app.ml.models import load_swin_classifier, SWIN_MODEL_PATH as DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)


def load_finetuned_swin(
    model_path: Path = DEFAULT_MODEL_PATH,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Wrapper pour charger le modèle Swin fine‑tuné.

    Args:
        model_path: Chemin vers le fichier .pth.
        device: 'cuda' ou 'cpu'.

    Returns:
        Le modèle chargé en mode évaluation.
    """
    return load_swin_classifier(model_path, device)


# Pour une utilisation directe dans le pipeline, on peut aussi exposer la fonction
# avec un nom plus explicite.
load_swin_breast_classifier = load_finetuned_swin