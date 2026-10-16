import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

from app.ml.processors.dicom_reader import DICOMProcessor
from .detection import DetectionAgent
from .segmentation import SegmentationAgent
from .classification import ClassificationAgent

logger = logging.getLogger(__name__)

class UltrasoundAgent:
    def __init__(self):
        self.agent_id = "ultrasound_agent"
        self.detector = DetectionAgent(confidence_threshold=0.6)  # Seuil plus élevé pour écho
        self.segmenter = SegmentationAgent()
        self.classifier = ClassificationAgent()

    async def process(self, dicom_path: Path) -> Dict[str, Any]:
        logger.info(f"🩻 Analyse échographie : {dicom_path.name}")

        dicom_data = DICOMProcessor.read_dicom(dicom_path)
        if not dicom_data["is_valid"]:
            return {"error": dicom_data.get("error", "Invalid DICOM")}

        pixels = dicom_data["pixels"]
        metadata = dicom_data["metadata"]

        # L'échographie est souvent déjà en niveaux de gris, on prépare pour YOLO
        image_yolo = DICOMProcessor.prepare_for_yolo(pixels)

        detections = await self.detector.process(image_yolo)
        if not detections:
            return {
                "modality": "US",
                "metadata": metadata,
                "detections": [],
                "summary": "Aucune lésion détectée.",
            }

        for det in detections:
            det["_original_image"] = image_yolo

        detections = await self.segmenter.process(image_yolo, detections)
        detections = await self.classifier.process(detections)

        for det in detections:
            det.pop("_original_image", None)

        nb_lesions = len(detections)
        malignant = sum(1 for d in detections if d.get("malignancy_label") == "Malin")

        return {
            "modality": "US",
            "metadata": metadata,
            "detections": detections,
            "summary": {
                "total_lesions": nb_lesions,
                "malignant_lesions": malignant,
                "benign_lesions": nb_lesions - malignant,
            }
        }