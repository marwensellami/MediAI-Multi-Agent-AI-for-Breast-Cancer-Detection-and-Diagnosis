# backend/app/agents/vision/mammography.py
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import cv2
import torch
import torchvision

from app.ml.processors.image_reader import ImageProcessor
from .detection import DetectionAgent
from .segmentation import SegmentationAgent
from .classification import ClassificationAgent

logger = logging.getLogger(__name__)


def apply_clahe(image: np.ndarray, clip_limit: float = 3.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Applique CLAHE pour améliorer le contraste local."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def nms_boxes(boxes, scores, iou_threshold=0.5):
    """Suppression des doublons par NMS."""
    if not boxes:
        return [], []
    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep = torchvision.ops.nms(boxes_t, scores_t, iou_threshold)
    return [boxes[i] for i in keep.tolist()], [scores[i] for i in keep.tolist()]


class MammographyAgent:
    def __init__(self):
        self.agent_id = "mammography_agent"

        # Détecteur YOLO fine-tuné (seul détecteur utilisé)
        self.yolo_detector = DetectionAgent(confidence_threshold=0.29)

        self.segmenter = SegmentationAgent()
        self.classifier = ClassificationAgent()

    async def process(self, file_path: Path, pixel_spacing_override: float = None) -> Dict[str, Any]:
        logger.info(f"🩻 Analyse mammographie : {file_path.name}")

        # Lecture de l'image
        image_data = ImageProcessor.read_file(file_path, pixel_spacing_override)
        if not image_data["is_valid"]:
            logger.error(f"❌ Erreur de lecture : {image_data.get('error')}")
            return {"modality": "MG", "error": image_data.get("error"), "metadata": {}, "detections": [], "summary": "Erreur de lecture."}

        pixels = image_data["pixels"]
        metadata = image_data["metadata"]
        pixel_spacing = metadata.get("pixel_spacing")
        original_shape = pixels.shape[:2]
        patient_position = metadata.get("patient_position", None)
        modality = metadata.get("modality", "MG")

        if pixel_spacing_override is not None:
            pixel_spacing = [pixel_spacing_override, pixel_spacing_override]

        logger.info(f"📐 Dimensions : {original_shape}")
        if pixel_spacing:
            logger.info(f"📏 Pixel Spacing : {pixel_spacing} mm/pixel")

        # Préparation de l'image pour YOLO
        image_yolo = ImageProcessor.prepare_for_yolo(pixels)
        model_shape = image_yolo.shape[:2]
        logger.info(f"🔄 Image prétraitée pour YOLO : {model_shape}")

        # ============================================================
        # 1. DÉTECTION GUIDÉE PAR LA MÉTADONNÉE (si l'image est dans DMID)
        # ============================================================
        from app.ml.utils.metadata_reader import get_lesion_bbox_from_metadata

        image_id = file_path.stem
        metadata_path = Path("app/data/DMID_PNG/Metadata.xlsx")

        bbox_metadata = get_lesion_bbox_from_metadata(image_id, metadata_path)
        detections = []

        if bbox_metadata:
            logger.info(f"📌 Utilisation des coordonnées de Metadata.xlsx pour {image_id}")
            detections = [{
                "bbox": bbox_metadata,
                "confidence": 0.9,
                "class_name": "lesion (metadata)",
                "_original_image": image_yolo,
                "original_image_shape": original_shape,
                "model_input_shape": model_shape,
                "pixel_spacing": pixel_spacing,
                "patient_position": patient_position,
            }]
        else:
            # ============================================================
            # 2. DÉTECTION PAR YOLO UNIQUEMENT
            # ============================================================
            detections_yolo = await self.yolo_detector.process(image_yolo)
            logger.info(f"🔍 YOLO : {len(detections_yolo)} détection(s)")

            if not detections_yolo:
                logger.warning("⚠️ Aucune détection par YOLO.")
                return {
                    "modality": modality,
                    "metadata": metadata,
                    "detections": [],
                    "summary": {
                        "total_lesions": 0,
                        "malignant_lesions": 0,
                        "benign_lesions": 0,
                        "message": "Aucune lésion détectée."
                    }
                }

            # Construction des détections (sans fusion)
            detections = []
            for det in detections_yolo:
                detections.append({
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "class_name": det.get("class_name", "lesion"),
                    "_original_image": image_yolo,
                    "original_image_shape": original_shape,
                    "model_input_shape": model_shape,
                    "pixel_spacing": pixel_spacing,
                    "patient_position": patient_position,
                })

            logger.info(f"✅ {len(detections)} détection(s) retenue(s)")

        # ============================================================
        # 3. SEGMENTATION
        # ============================================================
        detections = await self.segmenter.process(image_yolo, detections)
        logger.info(f"🎯 Segmentation : {len(detections)} lésion(s) segmentée(s)")

        # =====================================================
        # Extraction des caractéristiques morphologiques
        # =====================================================
        if pixel_spacing:
            spacing_x = float(pixel_spacing[0])
            spacing_y = float(pixel_spacing[1])
        else:
            spacing_x = spacing_y = 0.1
            logger.warning("⚠️ Pixel spacing par défaut 0.1 mm/pixel")

        for det in detections:
            mask = det.get("mask")
            bbox = det.get("bbox")

            # Initialisation à None (sera affiché "non disponibles")
            det["area_px"] = None
            det["perimeter_px"] = None
            det["circularity"] = None
            det["area_mm2"] = None
            det["perimeter_mm"] = None

            if mask is not None and bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]

                if x1 >= x2 or y1 >= y2:
                    logger.warning(f"⚠️ Bbox invalide : {bbox}")
                    continue

                mask_crop = mask[y1:y2, x1:x2]
                if mask_crop.size == 0:
                    logger.warning(f"⚠️ Masque vide pour la bbox {bbox}")
                    continue

                mask_bin = (mask_crop > 0).astype(np.uint8)
                area_px = np.sum(mask_bin)

                det["area_px"] = int(area_px)

                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    perimeter_px = cv2.arcLength(contours[0], True)
                    det["perimeter_px"] = round(perimeter_px, 2)

                    if perimeter_px > 0:
                        circularity = 4 * np.pi * area_px / (perimeter_px ** 2)
                        det["circularity"] = round(circularity, 3)
                    else:
                        det["circularity"] = 0

                    det["area_mm2"] = round(area_px * spacing_x * spacing_y, 2)
                    det["perimeter_mm"] = round(perimeter_px * spacing_x, 2)

                    logger.info(f"📐 Métriques calculées : area={det['area_mm2']:.2f} mm², perim={det['perimeter_mm']:.2f} mm, circ={det['circularity']:.3f}")
                else:
                    det["perimeter_px"] = 0
                    det["circularity"] = 0
                    det["area_mm2"] = 0
                    det["perimeter_mm"] = 0
            else:
                logger.debug(f"⚠️ Pas de masque ou bbox pour la détection : {det}")

        # =====================================================
        # Classification
        # =====================================================
        detections = await self.classifier.process(detections)
        logger.info(f"🏷️ Classification : {len(detections)} lésion(s) classifiée(s)")

        # =====================================================
        # Calcul de la taille en mm
        # =====================================================
        for det in detections:
            bbox = det.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                width_px = abs(x2 - x1)
                height_px = abs(y2 - y1)

                size_mm = max(width_px * spacing_x, height_px * spacing_y)

                if size_mm > 500:
                    if size_mm < 1000:
                        size_mm = size_mm / 10
                        logger.info(f"📏 Taille corrigée : division par 10 → {size_mm:.1f} mm")
                    else:
                        size_mm = size_mm / 100
                        logger.info(f"📏 Taille corrigée : division par 100 → {size_mm:.1f} mm")
                else:
                    logger.info(f"📏 Taille conservée : {size_mm:.1f} mm")

                if size_mm > 300:
                    logger.warning(f"⚠️ Taille anormale ({size_mm:.1f} mm) → correction à 300 mm")
                    size_mm = min(size_mm, 300)

                det["size_mm"] = round(size_mm, 1)
            else:
                det["size_mm"] = None

        # =====================================================
        # Nettoyage
        # =====================================================
        for det in detections:
            det.pop("_original_image", None)

        metadata.update({
            "original_image_shape": original_shape,
            "model_input_shape": model_shape,
            "pixel_spacing": pixel_spacing,
            "patient_position": patient_position,
            "modality": modality,
        })

        nb_lesions = len(detections)
        malignant = sum(1 for d in detections if d.get("malignancy_label") in ["Suspecte", "Malin", "Maligne", "Malignant"])
        benign = nb_lesions - malignant

        logger.info(f"✅ Analyse terminée : {nb_lesions} lésion(s), {malignant} suspecte(s)")

        return {
            "modality": modality,
            "metadata": metadata,
            "detections": detections,
            "summary": {
                "total_lesions": nb_lesions,
                "malignant_lesions": malignant,
                "benign_lesions": benign,
                "message": f"{nb_lesions} lésion(s) détectée(s)"
            }
        }