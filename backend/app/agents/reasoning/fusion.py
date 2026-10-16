# backend\app\agents\reasoning\fusion.py
import logging
import torch
import numpy as np
import open_clip
from typing import Dict, Any, List, Optional

from app.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class FusionAgent:
    def __init__(self):
        self.agent_id = "fusion_agent"
        self.registry = ModelRegistry()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Charger BioMedCLIP (modèle pré-entraîné)
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                pretrained=True,
                device=self.device,
            )
            self.tokenizer = open_clip.get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
            self.model.eval()
            logger.info("✅ BioMedCLIP chargé avec succès.")
        except Exception as e:
            logger.error(f"❌ Erreur chargement BioMedCLIP: {e}")
            self.model = None
            self.tokenizer = None

    async def process(
        self,
        image_patches: List[np.ndarray],
        clinical_text: str,
        detections: List[Dict],
    ) -> Dict[str, Any]:
        """
        Fusionne les informations :
        - image_patches : liste de patches (numpy uint8 RGB) correspondant aux lésions
        - clinical_text : texte décrivant les données cliniques (ex: "Âge: 55 ans, antécédents: ...")
        - detections : résultats des agents Vision (pour enrichir)
        Retourne un vecteur fusionné (embedding) et des métadonnées.
        """
        if self.model is None:
            logger.warning("BioMedCLIP non disponible, utilisation d'un vecteur nul.")
            return {"fusion_vector": np.zeros(512).tolist(), "error": "BioMedCLIP not loaded"}

        with torch.no_grad():
            # 1. Encoder les images (patches)
            image_embeddings = []
            for patch in image_patches:
                # Convertir en tenseur et prétraiter
                if patch.dtype != np.uint8:
                    patch = (patch / patch.max() * 255).astype(np.uint8)
                # Redimensionner en 224x224 si nécessaire
                if patch.shape[:2] != (224, 224):
                    import cv2
                    patch = cv2.resize(patch, (224, 224))
                # Convertir en RGB si nécessaire
                if len(patch.shape) == 2:
                    patch = np.stack([patch]*3, axis=-1)
                elif patch.shape[2] == 1:
                    patch = np.concatenate([patch]*3, axis=-1)
                elif patch.shape[2] > 3:
                    patch = patch[:, :, :3]
                
                # Prétraitement BioMedCLIP
                image_tensor = self.preprocess(patch).unsqueeze(0).to(self.device)
                img_emb = self.model.encode_image(image_tensor)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)  # Normaliser
                image_embeddings.append(img_emb.cpu().numpy().squeeze())

            # Moyenne des embeddings d'images (si plusieurs patches)
            if image_embeddings:
                avg_image_emb = np.mean(image_embeddings, axis=0)
            else:
                avg_image_emb = np.zeros(512)  # dimension du modèle

            # 2. Encoder le texte clinique
            text_tokens = self.tokenizer([clinical_text]).to(self.device)
            text_emb = self.model.encode_text(text_tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb.cpu().numpy().squeeze()

            # 3. Fusion : concaténation simple
            # On peut aussi faire une moyenne pondérée, mais ici on concatène
            fusion_vector = np.concatenate([avg_image_emb, text_emb])

            # 4. Ajouter des métadonnées (score de confiance, nombre de lésions)
            metadata = {
                "num_detections": len(detections),
                "detection_summary": [{"class": d.get("class_name"), "confidence": d.get("confidence")} for d in detections],
            }

            logger.info(f"🔗 Fusion vectorielle générée (dimension {len(fusion_vector)})")
            return {
                "fusion_vector": fusion_vector.tolist(),
                "metadata": metadata,
                "avg_image_embedding": avg_image_emb.tolist(),
                "text_embedding": text_emb.tolist(),
            }