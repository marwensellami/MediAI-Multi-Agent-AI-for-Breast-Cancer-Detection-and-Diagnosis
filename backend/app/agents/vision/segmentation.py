# backend/app/agents/vision/segmentation.py
import logging
import numpy as np
import cv2
from typing import List, Dict, Any

from app.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class SegmentationAgent:
    def __init__(self):
        self.agent_id = "segmentation_agent"
        self.registry = ModelRegistry()
        self.sam_predictor = self.registry.get("sam2")
        if self.sam_predictor:
            logger.info("✅ SAM2 disponible pour la segmentation précise.")
        else:
            #logger.warning("⚠️ SAM2 non disponible. Masques approximatifs (boîtes).")
            logger.warning("Masques approximatifs (boîtes).")

    async def process(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(detections) == 0:
            return []

        # ============================================================
        # FORCER LE FALLBACK (boîtes) – désactiver SAM2
        # ============================================================
        # if self.sam_predictor:
        #     return await self._run_sam2(image, detections)
        # else:
        return self._generate_box_masks(image, detections)

    async def _run_sam2(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        try:
            # Conversion en RGB si nécessaire
            if len(image.shape) == 2:
                image_rgb = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 1:
                image_rgb = np.concatenate([image] * 3, axis=-1)
            else:
                image_rgb = image

            # Boîtes : s'assurer que c'est un tableau 2D float32
            boxes = [d["bbox"] for d in detections]
            input_boxes = np.array(boxes, dtype=np.float32)
            if input_boxes.ndim == 1:
                input_boxes = input_boxes.reshape(1, -1)
            if input_boxes.ndim != 2 or input_boxes.shape[1] != 4:
                logger.error(f"Format des boîtes invalide : {input_boxes.shape}")
                return self._generate_box_masks(image, detections)

            self.sam_predictor.set_image(image_rgb)
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            for i, det in enumerate(detections):
                if i < masks.shape[0]:
                    mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
                    mask_bin = (mask > 0).astype(np.uint8) * 255
                    det["mask"] = mask_bin
                    if scores is not None and i < len(scores):
                        try:
                            det["segmentation_score"] = scores[i].item() if hasattr(scores[i], 'item') else float(scores[i])
                        except:
                            det["segmentation_score"] = 0.5
                        else:
                            det["segmentation_score"] = 0.5
                else:
                    det["mask"] = None
            return detections

        except Exception as e:
            logger.error(f"❌ Erreur SAM2, fallback : {e}")
            return self._generate_box_masks(image, detections)
    
    def _generate_box_masks(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Fallback amélioré : masques en forme d'ellipse pour des métriques plus réalistes.
        """
        h, w = image.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Masque vide
            mask = np.zeros((h, w), dtype=np.uint8)
            
            width = x2 - x1
            height = y2 - y1
            
            # Vérification des dimensions valides
            if width <= 0 or height <= 0:
                logger.warning(f"⚠️ Bbox invalide : {det['bbox']}")
                det["mask"] = mask
                det["segmentation_score"] = det.get("confidence", 0.5)
                det["segmentation_method"] = "fallback_invalid"
                continue
            
            # ---- 1. PADDING DYNAMIQUE (10% de la plus grande dimension) ----
            pad = int(max(width, height) * 0.10)
            pad = max(1, pad)  # Au moins 1 pixel
            
            # ---- 2. CRÉATION D'UNE ELLIPSE (plus réaliste qu'un rectangle) ----
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Rayons = demi-largeur + padding
            axes_x = (width // 2) + pad
            axes_y = (height // 2) + pad
            
            # Sécurité : les rayons ne doivent pas dépasser les bords de l'image
            axes_x = min(axes_x, center_x, w - center_x)
            axes_y = min(axes_y, center_y, h - center_y)
            
            if axes_x > 0 and axes_y > 0:
                # Dessiner une ellipse pleine sur le masque
                cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
            else:
                # Fallback : rectangle (si l'ellipse est trop petite)
                x1_p = max(0, x1 - pad)
                y1_p = max(0, y1 - pad)
                x2_p = min(w, x2 + pad)
                y2_p = min(h, y2 + pad)
                mask[y1_p:y2_p, x1_p:x2_p] = 255
                logger.debug(f"⚠️ Ellipse impossible, fallback rectangle pour {det['bbox']}")
            
            # ---- 3. ENREGISTREMENT ----
            det["mask"] = mask
            det["segmentation_score"] = det.get("confidence", 0.5)
            det["segmentation_method"] = "ellipse_approximation"  # Pour traçabilité
            
        return detections
    
    
    
    
    """ def _generate_box_masks(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        Fallback : masques rectangulaires.
        h, w = image.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            mask = np.zeros((h, w), dtype=np.uint8)
            pad = 5
            x1_p = max(0, x1 - pad)
            y1_p = max(0, y1 - pad)
            x2_p = min(w, x2 + pad)
            y2_p = min(h, y2 + pad)
            mask[y1_p:y2_p, x1_p:x2_p] = 255
            det["mask"] = mask
            det["segmentation_score"] = det.get("confidence", 0.5)
        return detections  """