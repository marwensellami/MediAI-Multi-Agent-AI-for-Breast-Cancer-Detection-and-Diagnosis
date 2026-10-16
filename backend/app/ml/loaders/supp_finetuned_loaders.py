# backend/app/ml/loaders/finetuned_loaders.py
"""
Chargeurs pour les modèles fine‑tunés (YOLO et Swin).
"""
import logging
import torch
import timm
from pathlib import Path
from typing import Optional
from ultralytics import YOLO

from app.ml.models import YOLO_MODEL_PATH, SWIN_MODEL_PATH
from app.core.config import settings

logger = logging.getLogger(__name__)


def load_yolo_finetuned(
    model_path: Path = YOLO_MODEL_PATH,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Optional[YOLO]:
    """
    Charge le modèle YOLO fine‑tuné sur les mammographies.
    
    Args:
        model_path : Chemin vers le fichier .pt (par défaut yolo_breast.pt)
        device : 'cuda' ou 'cpu'
    
    Returns:
        YOLO : modèle chargé, ou None en cas d'échec
    """
    try:
        if not model_path.exists():
            logger.error(f"❌ Modèle YOLO non trouvé : {model_path}")
            logger.info("ℹ️ Veuillez exécuter l'entraînement YOLO d'abord.")
            return None

        logger.info(f"📥 Chargement du modèle YOLO fine‑tuné depuis {model_path}")
        model = YOLO(str(model_path))
        # Forcer le modèle à utiliser le bon device
        if device == "cuda":
            model.to("cuda")
        logger.info("✅ YOLO fine‑tuné chargé avec succès.")
        return model

    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement de YOLO fine‑tuné : {e}")
        return None


def load_swin_finetuned(
    model_path: Path = SWIN_MODEL_PATH,
    num_classes: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Optional[torch.nn.Module]:
    """
    Charge le modèle Swin fine‑tuné sur les patches de mammographies (3 classes).
    
    Args:
        model_path : Chemin vers le fichier .pth (par défaut swin_breast_3class.pth)
        num_classes : Nombre de classes (3 par défaut)
        device : 'cuda' ou 'cpu'
    
    Returns:
        torch.nn.Module : modèle chargé avec sa tête, ou None en cas d'échec
    """
    try:
        if not model_path.exists():
            logger.error(f"❌ Modèle Swin non trouvé : {model_path}")
            logger.info("ℹ️ Veuillez exécuter l'entraînement Swin d'abord.")
            return None

        logger.info("🔄 Chargement du backbone Swin...")
        backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,  # On ne charge pas les poids d'ImageNet (déjà fine‑tunés)
            num_classes=0,
        )

        # Ajouter la tête de classification
        class SwinWithHead(torch.nn.Module):
            def __init__(self, backbone, num_classes):
                super().__init__()
                self.backbone = backbone
                self.num_features = backbone.num_features
                self.head = torch.nn.Linear(self.num_features, num_classes)

            def forward(self, x):
                features = self.backbone(x)
                # Si le backbone retourne un tuple (features, logits) ou autre
                if isinstance(features, tuple):
                    features = features[0]
                # Si le backbone retourne un tenseur 4D (B, C, H, W)
                if features.dim() == 4:
                    features = features.mean(dim=[2, 3])
                elif features.dim() == 3:
                    features = features.mean(dim=1)
                return self.head(features)

        model = SwinWithHead(backbone, num_classes=num_classes)

        # Charger les poids fine‑tunés
        logger.info(f"📥 Chargement des poids fine‑tunés depuis {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        logger.info("✅ Swin fine‑tuné chargé avec succès.")
        return model

    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement de Swin fine‑tuné : {e}")
        return None


def load_yolo_finetuned_from_registry() -> Optional[YOLO]:
    """
    Charge YOLO fine‑tuné et l'enregistre dans le ModelRegistry.
    À appeler dans init_models.py.
    """
    from app.ml.model_registry import ModelRegistry

    registry = ModelRegistry()
    if registry.is_loaded("yolo_finetuned"):
        return registry.get("yolo_finetuned")

    model = load_yolo_finetuned()
    if model:
        registry.register("yolo_finetuned", model)
    return model


def load_swin_finetuned_from_registry() -> Optional[torch.nn.Module]:
    """
    Charge Swin fine‑tuné et l'enregistre dans le ModelRegistry.
    À appeler dans init_models.py.
    """
    from app.ml.model_registry import ModelRegistry

    registry = ModelRegistry()
    if registry.is_loaded("swin_finetuned"):
        return registry.get("swin_finetuned")

    model = load_swin_finetuned()
    if model:
        registry.register("swin_finetuned", model)
    return model