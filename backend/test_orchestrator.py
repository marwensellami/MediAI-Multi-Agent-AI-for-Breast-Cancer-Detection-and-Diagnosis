# backend/test_orchestrator.py
import asyncio
import logging
from pathlib import Path

from app.agents.orchestrator import Orchestrator, AgentState
from app.ml.loaders.vision_loader import VisionLoader
from app.ml.loaders.llm_loader import LLMLoader
from app.ml.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def load_models():
    """Charge les modèles nécessaires."""
    reg = ModelRegistry()
    if not reg.is_loaded("yolo"):
        reg.register("yolo", VisionLoader.load_yolo())
    if not reg.is_loaded("swin"):
        reg.register("swin", VisionLoader.load_swin())
    if not reg.is_loaded("llm"):
        _, _, pipeline = LLMLoader.load_model()
        if pipeline:
            reg.register("llm", pipeline)

async def test_orchestrator():
    """Test complet de l'orchestrateur avec des données simulées."""
    logger.info("🧪 Test de l'Orchestrateur...")

    # Charger les modèles
    await load_models()

    # Créer l'orchestrateur
    orchestrator = Orchestrator()

    # Préparer l'état initial (simulé)
    initial_state: AgentState = {
        "patient_id": "P-12345",
        "clinical_data": {
            "age": 55,
            "history": "Fumeuse, antécédents familiaux de cancer du sein",
            "risk_factors": ["tabagisme", "hérédité"],
        },
        "modalities": ["MG", "US"],  # Pas d'IRM ici
        "images": {
            "MG": [Path("app/data/dicom_samples/mammo.dcm")],  # À remplacer par un vrai fichier
            "US": [Path("app/data/dicom_samples/echo.dcm")],
            "MRI": [],
        },
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
    }

    # Exécuter l'orchestrateur
    try:
        final_state = await orchestrator.run(initial_state)
        
        logger.info("\n" + "=" * 60)
        logger.info("📄 RAPPORT FINAL:")
        logger.info("=" * 60)
        
        report = final_state.get("final_report", "Rapport non disponible")
        if report:
            logger.info(report)
        else:
            logger.info("Rapport non disponible")
        
        logger.info("=" * 60)
        
        # Afficher la confiance en toute sécurité
        confidence = final_state.get("final_confidence", 0.0)
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.0
            
        logger.info(f"✅ Statut final: {final_state.get('validation_status', 'UNKNOWN')}")
        logger.info(f"✅ Confiance: {confidence:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Échec du test: {e}")

if __name__ == "__main__":
    asyncio.run(test_orchestrator())