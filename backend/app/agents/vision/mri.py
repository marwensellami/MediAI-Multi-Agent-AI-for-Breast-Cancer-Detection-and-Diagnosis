import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from app.ml.processors.dicom_reader import DICOMProcessor
from .detection import DetectionAgent
from .segmentation import SegmentationAgent
from .classification import ClassificationAgent

logger = logging.getLogger(__name__)

class MRIAgent:
    def __init__(self):
        self.agent_id = "mri_agent"
        self.detector = DetectionAgent(confidence_threshold=0.4)  # Seuil plus bas pour IRM
        self.segmenter = SegmentationAgent()
        self.classifier = ClassificationAgent()

    async def process(self, dicom_paths: List[Path]) -> Dict[str, Any]:
        """
        Reçoit une liste de fichiers DICOM correspondant à une série 3D.
        """
        logger.info(f"🩻 Analyse IRM : {len(dicom_paths)} coupes")

        all_detections = []
        metadata = None

        for idx, path in enumerate(dicom_paths):
            dicom_data = DICOMProcessor.read_dicom(path)
            if not dicom_data["is_valid"]:
                continue
            if metadata is None:
                metadata = dicom_data["metadata"]

            pixels = dicom_data["pixels"]
            image_yolo = DICOMProcessor.prepare_for_yolo(pixels)

            detections = await self.detector.process(image_yolo)
            if not detections:
                continue

            for det in detections:
                det["slice_index"] = idx
                det["_original_image"] = image_yolo

            # Segmentation + Classification par coupe
            detections = await self.segmenter.process(image_yolo, detections)
            detections = await self.classifier.process(detections)

            for det in detections:
                det.pop("_original_image", None)
                all_detections.append(det)

        # Agrégation simple : on garde les détections avec le score de malignité le plus élevé par localisation (simplifié)
        # Ici, on se contente de toutes les retourner, le raisonneur fera le tri.
        nb_lesions = len(all_detections)
        malignant = sum(1 for d in all_detections if d.get("malignancy_label") == "Malin")

        return {
            "modality": "MRI",
            "metadata": metadata,
            "total_slices": len(dicom_paths),
            "detections": all_detections,
            "summary": {
                "total_lesions": nb_lesions,
                "malignant_lesions": malignant,
                "benign_lesions": nb_lesions - malignant,
            }
        }