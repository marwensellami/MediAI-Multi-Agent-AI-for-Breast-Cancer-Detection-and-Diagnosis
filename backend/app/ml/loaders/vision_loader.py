# backend/app/ml/loaders/vision_loader.py
import logging
from typing import Optional
from pathlib import Path

import torch
import timm
import urllib.request
from ultralytics import YOLO

from app.core.config import settings
from app.ml.models import YOLO_MODEL_PATH, SWIN_MODEL_PATH

logger = logging.getLogger(__name__)

# Dossier de cache pour les modèles de vision
CACHE_DIR = Path(settings.MODEL_CACHE_DIR) / "vision"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class VisionLoader:
    """Chargeur centralisé pour tous les modèles de vision."""

    # ============================================================
    # 1. YOLO (modèle générique COCO)
    # ============================================================
    @staticmethod
    def load_yolo(version: str = "yolo11m.pt") -> YOLO:
        model_path = CACHE_DIR / version
        if not model_path.exists():
            logger.info(f"📥 Téléchargement de YOLO {version}...")
            model = YOLO(version)
            model.save(str(model_path))
        else:
            model = YOLO(str(model_path))
        logger.info(f"✅ YOLO {version} chargé avec succès.")
        return model

    # ============================================================
    # 2. Swin Transformer (backbone générique)
    # ============================================================
    @staticmethod
    def load_swin(pretrained: bool = True) -> torch.nn.Module:
        logger.info("🔄 Chargement de Swin Transformer (backbone)...")
        model = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )
        model.eval()
        logger.info("✅ Swin Transformer chargé avec succès.")
        return model

    # ============================================================
    # 3. SAM2 (Segment Anything 2) – DÉSACTIVÉ
    # ============================================================
    # @staticmethod
    # def load_sam2() -> Optional[torch.nn.Module]:
    #     try:
    #         from sam2.build_sam import build_sam2
    #         from sam2.sam2_image_predictor import SAM2ImagePredictor
    #         import sam2
    #
    #         cache_dir = Path(settings.MODEL_CACHE_DIR) / "sam2"
    #         cache_dir.mkdir(parents=True, exist_ok=True)
    #
    #         checkpoint_filename = "sam2_hiera_large.pt"
    #         checkpoint_path = cache_dir / checkpoint_filename
    #
    #         if not checkpoint_path.exists():
    #             logger.info("📥 Téléchargement du checkpoint SAM2...")
    #             url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    #             urllib.request.urlretrieve(url, checkpoint_path)
    #             logger.info("✅ Checkpoint SAM2 téléchargé avec succès.")
    #
    #         sam2_pkg = Path(sam2.__file__).parent
    #         possible_configs = [
    #             sam2_pkg / "configs" / "sam2" / "sam2_hiera_l.yaml",
    #             sam2_pkg / "sam2" / "configs" / "sam2" / "sam2_hiera_l.yaml",
    #             sam2_pkg / "sam2_hiera_l.yaml",
    #             sam2_pkg / "configs" / "sam2_hiera_l.yaml",
    #         ]
    #         config_file = None
    #         for p in possible_configs:
    #             if p.exists():
    #                 config_file = p
    #                 break
    #
    #         if config_file is None:
    #             logger.error("❌ Aucun fichier de configuration SAM2 trouvé.")
    #             return None
    #
    #         logger.info(f"📄 Fichier de configuration : {config_file}")
    #         logger.info("🔄 Construction du modèle SAM2...")
    #
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #         model = build_sam2(
    #             config_file=str(config_file),
    #             checkpoint_path=str(checkpoint_path),
    #             device=device,
    #         )
    #         predictor = SAM2ImagePredictor(model)
    #         logger.info(f"✅ SAM2 chargé avec succès sur {device}.")
    #         return predictor
    #
    #     except ImportError as e:
    #         logger.error(f"❌ SAM2 non installé : {e}")
    #         return None
    #     except Exception as e:
    #         logger.error(f"❌ Erreur lors du chargement de SAM2 : {e}")
    #         return None

    # ============================================================
    # 4. MODÈLES FINE‑TUNÉS (intégrés directement)
    # ============================================================
    @staticmethod
    def load_yolo_finetuned(
        model_path: Path = YOLO_MODEL_PATH,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> Optional[YOLO]:
        """
        Charge le modèle YOLO fine‑tuné sur les mammographies.
        """
        try:
            if not model_path.exists():
                logger.error(f"❌ Modèle YOLO fine‑tuné non trouvé : {model_path}")
                return None

            logger.info(f"📥 Chargement de YOLO fine‑tuné depuis {model_path}")
            model = YOLO(str(model_path))
            if device == "cuda":
                model.to("cuda")
            logger.info("✅ YOLO fine‑tuné chargé avec succès.")
            return model

        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement de YOLO fine‑tuné : {e}")
            return None

    @staticmethod
    def load_swin_finetuned(
        model_path: Path = SWIN_MODEL_PATH,
        num_classes: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> Optional[torch.nn.Module]:
        """
        Charge le modèle Swin fine‑tuné (3 classes) depuis le fichier .pth.
        """
        try:
            if not model_path.exists():
                logger.error(f"❌ Modèle Swin fine‑tuné non trouvé : {model_path}")
                return None

            logger.info(f"🔄 Chargement du modèle Swin fine‑tuné depuis {model_path}...")

            # Créer le backbone
            backbone = timm.create_model(
                "swin_base_patch4_window7_224",
                pretrained=False,  # on ne charge pas les poids d'ImageNet
                num_classes=0,
            )

            # Définir la classe du modèle avec une tête
            class SwinWithHead(torch.nn.Module):
                def __init__(self, backbone, num_classes):
                    super().__init__()
                    self.backbone = backbone
                    # Récupérer le nombre de caractéristiques
                    self.num_features = backbone.num_features
                    self.head = torch.nn.Linear(self.num_features, num_classes)

                def forward(self, x):
                    features = self.backbone(x)
                    # Si le backbone renvoie un tuple, on prend le premier élément
                    if isinstance(features, tuple):
                        features = features[0]
                    # Si features est 4D (B, C, H, W), on moyenne sur H et W
                    if features.dim() == 4:
                        features = features.mean(dim=[2, 3])
                    elif features.dim() == 3:
                        features = features.mean(dim=1)
                    return self.head(features)

            model = SwinWithHead(backbone, num_classes)

            # Charger les poids
            state_dict = torch.load(str(model_path), map_location=device)
            # Si le modèle a été sauvegardé avec un nom différent (ex: 'fc' au lieu de 'head')
            # on essaie de renommer
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("fc."):
                    new_key = key.replace("fc.", "head.")
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            model.load_state_dict(new_state_dict)
            model.to(device)
            model.eval()
            logger.info(f"✅ Swin fine‑tuné ({num_classes} classes) chargé avec succès.")
            return model

        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement de Swin fine‑tuné : {e}")
            return None