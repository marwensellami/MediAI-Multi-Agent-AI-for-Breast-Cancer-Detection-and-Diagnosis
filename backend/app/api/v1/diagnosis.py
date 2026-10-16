import shutil
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.orchestrator_service import orchestrator_service
from app.services.report_service import ReportService
import uuid
import logging
from app.ml.utils.metadata_reader import get_view_from_metadata
from app.ml.utils.metadata_reader import get_view_from_metadata, get_density_from_metadata
router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("app/data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'.dcm', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'}

def get_safe_filename(base_name: str, original_filename: str) -> str:
    ext = Path(original_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = '.dcm'
        logger.warning(f"Extension non autorisée pour {original_filename}, fallback sur .dcm")
    safe_base = "".join(c for c in base_name if c.isalnum() or c in "._-")
    return f"{safe_base}{ext}"

@router.post("/diagnosis")
async def create_diagnosis(
    patient_id: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    history: str = Form(""),
    risk_factors: str = Form(""),
    mammo_file: UploadFile = File(None),
    us_file: UploadFile = File(None),
    mri_files: list[UploadFile] = File(None),
    pixel_spacing: Optional[float] = Form(None),
):
    if age is None:
        age = 0
    if not patient_id or patient_id.strip() == "":
        patient_id = f"P-{uuid.uuid4().hex[:8].upper()}"

    if not mammo_file and not us_file and not mri_files:
        raise HTTPException(status_code=400, detail="Au moins un fichier est requis")

    modalities = {}
    files = {}

    if mammo_file:
        try:
            filename = get_safe_filename("mammo", mammo_file.filename)
            dest = UPLOAD_DIR / filename
            content = await mammo_file.read()
            with open(dest, "wb") as f:
                f.write(content)
            modalities["MG"] = dest
            files["MG"] = dest
            logger.info(f"Fichier mammo sauvegardé: {dest} (taille: {len(content)} octets)")
        except Exception as e:
            logger.error(f"Erreur mammo: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

    if us_file:
        try:
            filename = get_safe_filename("echo", us_file.filename)
            dest = UPLOAD_DIR / filename
            content = await us_file.read()
            with open(dest, "wb") as f:
                f.write(content)
            modalities["US"] = dest
            files["US"] = dest
            logger.info(f"Fichier echo sauvegardé: {dest}")
        except Exception as e:
            logger.error(f"Erreur echo: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

    if mri_files:
        try:
            mri_dir = UPLOAD_DIR / "mri"
            mri_dir.mkdir(exist_ok=True)
            saved_files = []
            for i, file in enumerate(mri_files):
                filename = get_safe_filename(f"frame_{i:04d}", file.filename)
                dest = mri_dir / filename
                content = await file.read()
                with open(dest, "wb") as out:
                    out.write(content)
                saved_files.append(dest)
            modalities["MRI"] = mri_dir
            files["MRI"] = saved_files
            logger.info(f"{len(saved_files)} fichiers IRM sauvegardés")
        except Exception as e:
            logger.error(f"Erreur IRM: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
    
        if not view or view.strip() == "":
        # Extraire le nom de l'image (sans extension)
            image_id = Path(mammo_file.filename).stem if mammo_file else None
            if image_id:
                view = get_view_from_metadata(image_id)
            else:
                view = "Non spécifiée"
    original_filename = mammo_file.filename if mammo_file else None
    view = None
    density = None
    if original_filename:
        image_name = Path(original_filename).stem
        view = get_view_from_metadata(image_name)
        density = get_density_from_metadata(image_name)
        if view:
            logger.info(f"📷 Vue récupérée depuis Metadata.xlsx: {view}")
        else:
            logger.warning(f"⚠️ Vue non trouvée pour {image_name}")

    clinical_data = {
        "age": age,
        "history": history or "",
        "risk_factors": risk_factors or "",
        "patient_id": patient_id,
        "original_filename": original_filename,
        "view": view,  # ✅ CORRECT
        "density": density,
    }

    # Vérification existence fichiers
    for modality, path in files.items():
        if isinstance(path, list):
            for p in path:
                if not p.exists() or p.stat().st_size == 0:
                    raise HTTPException(status_code=400, detail=f"Fichier invalide pour {modality}")
        elif isinstance(path, Path):
            if not path.exists() or path.stat().st_size == 0:
                raise HTTPException(status_code=400, detail=f"Fichier invalide pour {modality}")

    logger.info(f"Paramètres reçus: Patient ID={patient_id}, Âge={age}, Modalités={list(modalities.keys())}")

    try:
        task_id = await orchestrator_service.run_diagnosis(
            patient_id=patient_id,
            clinical_data=clinical_data,
            modalities=modalities,
            files=files,
            pixel_spacing_override=pixel_spacing,
        )
        logger.info(f"Diagnostic lancé avec task_id: {task_id}")
        return {"task_id": task_id}
    except Exception as e:
        logger.error(f"Erreur lancement: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du lancement: {str(e)}")


@router.get("/diagnosis")
async def list_diagnoses(
    limit: int = 50,
    offset: int = 0,
):
    try:
        report_service = ReportService()
        diagnoses = await report_service.get_diagnoses(limit=limit, offset=offset)
        return diagnoses
    except Exception as e:
        logger.error(f"Erreur list_diagnoses: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diagnosis/{task_id}/status")
async def get_status(task_id: str):
    progress = await orchestrator_service.get_progress(task_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return progress


@router.get("/diagnosis/{task_id}/result")
async def get_result(task_id: str):
    result = await orchestrator_service.get_result(task_id)
    if result is None:
        progress = await orchestrator_service.get_progress(task_id)
        if progress is not None and progress.get("status") != "COMPLETED":
            raise HTTPException(status_code=202, detail="Task still processing")
        raise HTTPException(status_code=404, detail="Task not found or no result available")
    return result


@router.get("/diagnosis/{diagnosis_id}/report")
async def get_diagnosis_report(diagnosis_id: int):
    try:
        report_service = ReportService()
        report = await report_service.get_diagnosis_report(diagnosis_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Diagnosis not found")
        # Log de la taille des images
        logger.info(f"📤 Rapport renvoyé: original_image len={len(report.get('original_image') or '')}, annotated_image len={len(report.get('annotated_image') or '')}")
        return report
    except Exception as e:
        logger.error(f"Erreur get_diagnosis_report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/stats")
async def get_dashboard_stats():
    try:
        report_service = ReportService()
        stats = await report_service.get_dashboard_stats()
        return stats
    except Exception as e:
        logger.error(f"Erreur stats dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/diagnosis/{task_id}")
async def cancel_diagnosis(task_id: str):
    progress = await orchestrator_service.get_progress(task_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if progress.get("status") in ["COMPLETED", "FAILED"]:
        raise HTTPException(status_code=400, detail="Task already completed or failed")
    if task_id in orchestrator_service.active_tasks:
        del orchestrator_service.active_tasks[task_id]
        return {"message": f"Task {task_id} cancelled successfully"}
    else:
        raise HTTPException(status_code=404, detail="Task not found")