# app/services/orchestrator_service.py

import logging
import uuid
import asyncio
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import numpy as np
import cv2

from app.agents.orchestrator import Orchestrator, AgentState
from app.services.progress_tracker import ProgressTracker
from app.services.report_service import ReportService
from app.core.database import AsyncSessionLocal
from app.core.config import settings

logger = logging.getLogger(__name__)


def clean_detection(det: dict) -> dict:
    """
    Nettoie un dictionnaire de détection pour le rendre JSON-serializable.
    Supprime les champs non sérialisables et convertit les types numpy.
    """
    # Champs autorisés (ceux que le frontend attend)
    allowed_keys = {
        "bbox", "confidence", "class_id", "class_name",
        "malignancy_label", "malignancy_score", "size_mm",
        "area_mm2", "perimeter_mm", "circularity",
        "birads_category", "birads_level", "recommendation",
        "modality", "classification", "segmentation_score"
    }
    cleaned = {}
    for k, v in det.items():
        if k not in allowed_keys:
            continue
        # Conversion des types numpy
        if isinstance(v, (np.integer, np.floating, np.bool_)):
            cleaned[k] = v.item()
        elif isinstance(v, np.ndarray):
            cleaned[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            # Si la liste contient des numpy, les convertir
            cleaned[k] = [
                x.item() if isinstance(x, (np.integer, np.floating, np.bool_)) else x
                for x in v
            ]
        else:
            cleaned[k] = v
    return cleaned


class OrchestratorService:
    def __init__(self):
        self.orchestrator = Orchestrator()
        self.active_tasks: Dict[str, ProgressTracker] = {}
        self.report_service = ReportService()

    async def run_diagnosis(
        self,
        patient_id: str,
        clinical_data: Dict[str, Any],
        modalities: Dict[str, Path],
        files: Dict[str, Path],
        pixel_spacing_override: float = None,
    ) -> str:
        task_id = str(uuid.uuid4())
        tracker = ProgressTracker(task_id)
        self.active_tasks[task_id] = tracker
        tracker.set_status("RUNNING")
        asyncio.create_task(
            self._execute(
                task_id,
                patient_id,
                clinical_data,
                modalities,
                files,
                pixel_spacing_override
            )
        )
        return task_id

    async def _execute(
        self,
        task_id: str,
        patient_id: str,
        clinical_data: Dict[str, Any],
        modalities: Dict[str, Path],
        files: Dict[str, Path],
        pixel_spacing_override: float = None,
    ):
        tracker = self.active_tasks.get(task_id)
        if not tracker:
            return

        try:
            tracker.update("Sauvegarde en base", 5, "Enregistrement du patient...")
            study_id = await self._save_patient_and_study(patient_id, clinical_data, modalities, files)

            tracker.update("Préparation de l'état", 10, "Construction de l'état initial...")
            images_dict = {}
            for modality, path in files.items():
                if isinstance(path, Path) and path.is_dir():
                    images_dict[modality] = list(path.glob("*.dcm"))
                elif isinstance(path, Path):
                    images_dict[modality] = [path]
                else:
                    images_dict[modality] = path

            # Vérification de l'existence des fichiers
            for modality, paths in images_dict.items():
                if isinstance(paths, list):
                    for p in paths:
                        if not p.exists():
                            logger.error(f"❌ Fichier non trouvé pour {modality}: {p}")
                            tracker.set_error(f"Fichier non trouvé: {p}")
                            return
                elif isinstance(paths, Path) and not paths.exists():
                    logger.error(f"❌ Fichier non trouvé pour {modality}: {paths}")
                    tracker.set_error(f"Fichier non trouvé: {paths}")
                    return

            logger.info(f"📁 Images chargées: {list(images_dict.keys())}")
            if pixel_spacing_override:
                logger.info(f"📏 Pixel spacing forcé: {pixel_spacing_override} mm/pixel")

            # --- Chargement de l'image originale en base64 (pour le rapport) ---
            original_image_base64 = None
            mg_path = files.get("MG")
            if mg_path:
                try:
                    from app.ml.processors.image_reader import ImageProcessor
                    image_data = ImageProcessor.read_file(mg_path, pixel_spacing_override)
                    if image_data.get("is_valid"):
                        img = image_data.get("pixels")
                        if img is not None:
                            # Convertir en RGB si nécessaire
                            if len(img.shape) == 2:
                                img = np.stack([img]*3, axis=-1)
                            elif img.shape[2] == 1:
                                img = np.concatenate([img]*3, axis=-1)
                            _, buffer = cv2.imencode('.png', img)
                            original_image_base64 = base64.b64encode(buffer).decode('utf-8')
                            logger.info("✅ Image originale encodée en base64")
                        else:
                            logger.warning("⚠️ Image lue mais pixels sont None")
                    else:
                        logger.warning(f"⚠️ Échec lecture image: {image_data.get('error')}")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lors de l'encodage de l'image: {e}")

            initial_state: AgentState = {
                "patient_id": patient_id,
                "clinical_data": clinical_data,
                "modalities": list(modalities.keys()),
                "images": images_dict,
                "detections": {},
                "masks": {},
                "features": {},
                "vision_summary": {},
                "fusion_vector": None,
                "rag_docs": [],
                "kg_knowledge": {},
                "cot_reasoning": None,
                "final_report": None,
                "final_birads": None,
                "final_confidence": None,
                "validation_status": None,
                "current_step": "",
                "errors": [],
                "pixel_spacing_override": pixel_spacing_override,
                "original_image_base64": original_image_base64,  # ajout
            }

            tracker.update("Lancement de l'orchestrateur", 15, "Démarrage du pipeline IA...")
            final_state = await self._run_orchestrator_with_progress(initial_state, tracker)

            tracker.update("Sauvegarde des résultats", 90, "Enregistrement du diagnostic...")
            detections_dict = final_state.get("detections", {})
            if isinstance(detections_dict, dict):
                all_detections = detections_dict.get("all", [])
            else:
                all_detections = []

            # ✅ Nettoyer les détections pour les rendre JSON-serializables
            cleaned_detections = [clean_detection(d) for d in all_detections]

            # Compter les lésions
            total_lesions = len(cleaned_detections)
            malignant = sum(
                1 for d in cleaned_detections
                if d.get("malignancy_label", "").lower() in ["malin", "maligne", "malignant"]
            )
            benign = total_lesions - malignant

            file_name = clinical_data.get("original_filename")

            # Construction de raw_features incluant l'image originale
            raw_features = {
            "detections": cleaned_detections,
            "fusion_vector": clean_features(final_state.get("fusion_vector")),
            "original_image": original_image_base64,  # ← Utiliser la variable locale, pas final_state
            "view": clinical_data.get("view"),
             "density": clinical_data.get("density"),
            }
            diagnosis_id = await self.report_service.save_diagnosis(
                study_id=study_id,
                birads=final_state.get("final_birads"),
                confidence=final_state.get("final_confidence"),
                report=final_state.get("final_report"),
                detections=cleaned_detections,
                features=final_state.get("fusion_vector"),
                file_name=file_name,
                raw_features=raw_features,  # nouvel argument
            )

            tracker.set_result({
                "diagnosis_id": diagnosis_id,
                "report": final_state.get("final_report"),
                "birads": final_state.get("final_birads"),
                "confidence": final_state.get("final_confidence"),
                "status": final_state.get("validation_status"),
                "total_lesions": total_lesions,
                "malignant_lesions": malignant,
                "benign_lesions": benign,
                "detections": cleaned_detections,
                "original_image": final_state.get("original_image_base64"),  # pour l'affichage
            })
            tracker.update("Terminé", 100, "Diagnostic terminé avec succès.")

        except Exception as e:
            logger.exception(f"❌ Erreur lors de l'exécution du diagnostic {task_id}")
            tracker.set_error(str(e))

    async def _run_orchestrator_with_progress(self, state: AgentState, tracker: ProgressTracker) -> AgentState:
        try:
            tracker.update("Exécution du pipeline", 20, "Lancement des agents...")
            final_state = await self.orchestrator.graph.ainvoke(state)
            tracker.update("Pipeline terminé", 85, "Analyse terminée")
            return final_state
        except Exception as e:
            logger.exception(f"❌ Erreur dans l'orchestrateur: {e}")
            tracker.set_error(str(e))
            raise

    async def _save_patient_and_study(
        self,
        patient_id: str,
        clinical_data: Dict,
        modalities: Dict,
        files: Dict
    ) -> int:
        async with AsyncSessionLocal() as db:
            try:
                from app.models.domain.patient import Patient
                patient = Patient(
                    first_name=clinical_data.get("first_name", ""),
                    last_name=clinical_data.get("last_name", ""),
                    age=clinical_data.get("age"),
                    gender=clinical_data.get("gender", "FEMALE"),
                    medical_history=clinical_data.get("history", ""),
                    risk_factors=clinical_data.get("risk_factors", ""),
                )
                db.add(patient)
                await db.commit()
                await db.refresh(patient)

                from app.models.domain.study import Study
                study = Study(
                    patient_id=patient.id,
                    modality=", ".join(modalities.keys()),
                    view=clinical_data.get("view"),
                    description=clinical_data.get("description", "Examen combiné"),
                    study_date=datetime.now(),
                )
                db.add(study)
                await db.commit()
                await db.refresh(study)

                logger.info(f"✅ Patient {patient.id} et étude {study.id} créés")
                return study.id
            except Exception as e:
                logger.exception(f"❌ Erreur lors de la sauvegarde en base: {e}")
                return 1

    async def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        tracker = self.active_tasks.get(task_id)
        if not tracker:
            return None
        return {
            "task_id": tracker.task_id,
            "status": tracker.status,
            "percent": tracker.percent,
            "steps": tracker.steps,
            "error": tracker.error,
        }

    async def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        tracker = self.active_tasks.get(task_id)
        if not tracker:
            return None
        if tracker.status == "COMPLETED" and tracker.result:
            return tracker.result
        return None

    def cleanup_completed_tasks(self, max_age_seconds: int = 3600):
        current_time = datetime.now()
        to_remove = []
        for task_id, tracker in self.active_tasks.items():
            if tracker.status in ["COMPLETED", "FAILED"]:
                if hasattr(tracker, 'created_at'):
                    age = (current_time - tracker.created_at).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(task_id)
                else:
                    to_remove.append(task_id)
        for task_id in to_remove:
            del self.active_tasks[task_id]
            logger.info(f"🧹 Tâche {task_id} nettoyée du cache")


# Fonction utilitaire pour nettoyer les features si nécessaire
def clean_features(features):
    if features is None:
        return None
    if isinstance(features, np.ndarray):
        return features.tolist()
    return features


# Instance globale unique
orchestrator_service = OrchestratorService()