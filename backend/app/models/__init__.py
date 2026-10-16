# backend/app/ml/models/__init__.py
"""
Module contenant les classes et fonctions pour les modèles entraînés.
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Chemin par défaut vers le modèle fine‑tuné
DEFAULT_MODEL_PATH = Path(__file__).parent / "swin_breast_classifier.pth"


class SwinClassifier(nn.Module):
    """
    Classifieur Swin avec une tête linéaire pour la classification binaire.
    """
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features
        self.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)  # sortie du backbone
        # Gérer les différents formats de sortie
        if x.dim() == 2:          # (B, C) déjà un vecteur
            pass
        elif x.dim() == 3:        # (B, L, C) : on moyenne sur les patches
            x = x.mean(dim=1)
        elif x.dim() == 4:        # (B, C, H, W) : on moyenne sur H et W
            x = x.mean(dim=[2, 3])
        else:
            raise ValueError(f"Shape inattendue : {x.shape}")
        x = self.fc(x)
        return x


def load_swin_classifier(model_path: Path = DEFAULT_MODEL_PATH, device: str = "cpu") -> nn.Module:
    """
    Charge le modèle Swin fine‑tuné à partir du fichier de poids.

    Args:
        model_path: Chemin vers le fichier .pth contenant les poids.
        device: 'cuda' ou 'cpu'.

    Returns:
        Le modèle chargé en mode évaluation.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Fichier modèle introuvable : {model_path}")

    logger.info(f"🔄 Chargement du modèle fine‑tuné depuis {model_path}...")

    # Créer le backbone
    backbone = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0)
    model = SwinClassifier(backbone, num_classes=2)

    # Charger les poids
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("✅ Modèle Swin fine‑tuné chargé avec succès.")
    return model