#backend\app\services\report_service.py
import logging
import json
import numpy as np
import base64
import cv2
from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.domain.diagnosis import Diagnosis
from app.models.domain.study import Study
from app.models.domain.patient import Patient
from app.core.database import AsyncSessionLocal
from app.utils.image_annotation import draw_lesions_on_image

logger = logging.getLogger(__name__)

def clean_serializable(obj):
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: clean_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return [clean_serializable(item) for item in obj]
    return obj

class ReportService:
    async def get_or_create_patient(self, db: AsyncSession, patient_id: str, data: Dict):
        from app.models.domain.patient import Patient
        patient = Patient(
            first_name=data.get("first_name", ""),
            last_name=data.get("last_name", ""),
            age=data.get("age"),
            gender=data.get("gender", "FEMALE"),
            medical_history=data.get("history", ""),
            risk_factors=data.get("risk_factors", ""),
        )
        db.add(patient)
        await db.commit()
        await db.refresh(patient)
        return patient

    async def create_study(self, db: AsyncSession, patient_id: int, modalities: str, description: str):
        from app.models.domain.study import Study
        study = Study(
            patient_id=patient_id,
            modality=modalities,
            description=description,
            study_date=datetime.now(),
        )
        db.add(study)
        await db.commit()
        await db.refresh(study)
        return study

    async def save_diagnosis(
            self,
            study_id: int,
            birads: Optional[str],
            confidence: Optional[float],
            report: Optional[str],
            detections: List[Dict],
            features: Optional[List],
            file_name: Optional[str] = None,
            raw_features: Optional[Dict] = None,
        ) -> int:
            async with AsyncSessionLocal() as db:
                clean_detections = clean_serializable(detections)
                clean_features = clean_serializable(features)
                if raw_features is None:
                    raw_features = {
                        "detections": clean_detections,
                        "fusion_vector": clean_features,
                    }
                else:
                    raw_features = clean_serializable(raw_features)
                diagnosis = Diagnosis(
                    study_id=study_id,
                    birads_category=birads,
                    confidence_score=confidence,
                    raw_report=report,
                    raw_features=raw_features,
                    created_at=datetime.now(),
                    file_name=file_name,
                )
                db.add(diagnosis)
                await db.commit()
                await db.refresh(diagnosis)
                logger.info(f"✅ Diagnostic {diagnosis.id} sauvegardé avec image")
                return diagnosis.id

    async def get_diagnosis_report(self, diagnosis_id: int) -> Optional[Dict]:
        async with AsyncSessionLocal() as db:
            query = select(Diagnosis).where(Diagnosis.id == diagnosis_id)
            result = await db.execute(query)
            diagnosis = result.scalar_one_or_none()
            if not diagnosis:
                return None

            detections = diagnosis.raw_features.get("detections", []) if diagnosis.raw_features else []
            original_image_base64 = diagnosis.raw_features.get("original_image", None) if diagnosis.raw_features else None
            view = diagnosis.raw_features.get("view", "Non spécifiée") if diagnosis.raw_features else "Non spécifiée"
            density = diagnosis.raw_features.get("density", None) if diagnosis.raw_features else None
            # ✅ Récupérer le nom du fichier
            file_name = diagnosis.file_name

            annotated_image_base64 = None
            if original_image_base64 and detections:
                try:
                    img_bytes = base64.b64decode(original_image_base64)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        annotated_image_base64 = draw_lesions_on_image(img, detections)
                except Exception as e:
                    logger.error(f"Erreur génération image annotée: {e}")

            return {
                "diagnosis_id": diagnosis.id,
                "report": diagnosis.raw_report,
                "birads": diagnosis.birads_category,
                "confidence": diagnosis.confidence_score,
                "status": "COMPLETED",
                "detections": detections,
                "original_image": original_image_base64,
                "annotated_image": annotated_image_base64,
                "file_name": file_name,  # ✅ Ajout
                "view": view,
                "density": density,
            }

    async def get_diagnoses(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        async with AsyncSessionLocal() as db:
            try:
                query = (
                    select(
                        Diagnosis.id,
                        Diagnosis.birads_category,
                        Diagnosis.confidence_score,
                        Diagnosis.created_at,
                        Diagnosis.file_name,
                        Patient.id.label("patient_id"),
                        Patient.first_name,
                        Patient.last_name,
                    )
                    .join(Study, Diagnosis.study_id == Study.id)
                    .join(Patient, Study.patient_id == Patient.id)
                    .order_by(Diagnosis.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
                result = await db.execute(query)
                rows = result.all()

                output = []
                for row in rows:
                    output.append({
                        "id": row.id,
                        "patient_id": row.patient_id,
                        "patient_name": f"{row.first_name or ''} {row.last_name or ''}".strip() or "Inconnu",
                        "date": row.created_at.isoformat() if row.created_at else "",
                        "birads": row.birads_category,
                        "confidence": row.confidence_score,
                        "status": "Complété",
                        "file_name": row.file_name or "Fichier inconnu",
                    })
                return output
            except Exception as e:
                logger.error(f"❌ Erreur dans get_diagnoses : {e}", exc_info=True)
                return []

    async def get_dashboard_stats(self) -> Dict:
        async with AsyncSessionLocal() as db:
            total_diag = await db.scalar(select(func.count(Diagnosis.id)))
            birads_dist = await db.execute(
                select(Diagnosis.birads_category, func.count(Diagnosis.id))
                .group_by(Diagnosis.birads_category)
            )
            birads_data = {row[0] or "Non classifié": row[1] for row in birads_dist.all()}
            avg_conf = await db.scalar(select(func.avg(Diagnosis.confidence_score)))
            return {
                "total_diagnoses": total_diag or 0,
                "birads_distribution": birads_data,
                "avg_confidence": round(avg_conf or 0, 2),
            }

    