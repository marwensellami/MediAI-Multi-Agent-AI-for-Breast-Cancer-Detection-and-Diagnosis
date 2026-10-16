# backend/test_agents.py
import asyncio
import logging
import numpy as np
from app.ml.loaders.vision_loader import VisionLoader
from app.ml.loaders.llm_loader import LLMLoader
from app.ml.model_registry import ModelRegistry
from app.agents.vision.mammography import MammographyAgent
from app.agents.vision.ultrasound import UltrasoundAgent
from app.agents.vision.mri import MRIAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def load_models():
    """Charge tous les modèles nécessaires dans le registre."""
    reg = ModelRegistry()
    
    # YOLO
    if not reg.is_loaded("yolo"):
        yolo = VisionLoader.load_yolo()
        reg.register("yolo", yolo)
        logger.info("✅ YOLO chargé")
    
    # Swin
    if not reg.is_loaded("swin"):
        swin = VisionLoader.load_swin()
        reg.register("swin", swin)
        logger.info("✅ Swin chargé")
    
    # SAM2 (optionnel)
    if not reg.is_loaded("sam2"):
        sam = VisionLoader.load_sam2()
        if sam:
            reg.register("sam2", sam)
            logger.info("✅ SAM2 chargé")
        else:
            logger.warning("⚠️ SAM2 non chargé (continuons)")
    
    # LLM (optionnel pour les agents vision, mais chargé pour la cohérence)
    if not reg.is_loaded("llm"):
        tokenizer, model, pipeline = LLMLoader.load_model()
        if pipeline:
            reg.register("llm", pipeline)
            logger.info("✅ LLM chargé")

async def test_agents():
    logger.info("🔍 Test des Agents Vision...")
    
    # Charger les modèles
    await load_models()
    
    # Créer les agents
    mammo = MammographyAgent()
    logger.info("✅ MammographyAgent instancié avec succès")
    
    us = UltrasoundAgent()
    logger.info("✅ UltrasoundAgent instancié avec succès")
    
    mri = MRIAgent()
    logger.info("✅ MRIAgent instancié avec succès")
    
    logger.info("🎉 Tous les agents sont prêts !")

if __name__ == "__main__":
    asyncio.run(test_agents())
    
# Ajouter dans test_agents.py après l'instanciation des agents vision
async def test_reasoning():
    from app.agents.reasoning.fusion import FusionAgent
    from app.agents.reasoning.rag_agent import RAGAgent
    from app.agents.reasoning.kg_agent import KGAgent
    from app.agents.reasoning.medical_reasoning import MedicalReasoningAgent

    fusion = FusionAgent()
    rag = RAGAgent()
    kg = KGAgent()
    reasoning = MedicalReasoningAgent()

    # Simuler des données (à remplacer par de vraies données)
    dummy_detections = [{"bbox": [10,20,30,40], "malignancy_label": "Malin", "malignancy_score": 0.8}]
    dummy_metadata = {"age": 55, "history": "Fumeuse", "modalities": ["MG"]}
    fusion_result = await fusion.process([np.zeros((224,224,3), dtype=np.uint8)], "Femme de 55 ans, antécédents familiaux", dummy_detections)
    rag_result = await rag.process("masse spiculée")
    kg_result = await kg.process(["masse spiculée"])
    final_reasoning = await reasoning.process(fusion_result, rag_result, kg_result, dummy_detections, dummy_metadata)
    print(final_reasoning)
    
    # Ajouter une fonction de test pour les agents de sortie
async def test_output_agents():
    from app.agents.output.report_generator import ReportGeneratorAgent
    from app.agents.output.validator import ValidatorAgent
    from app.agents.output.explainability import ExplainabilityAgent

    report_gen = ReportGeneratorAgent()
    validator = ValidatorAgent()
    explainer = ExplainabilityAgent()

    dummy_reasoning = "1. Je vois une masse spiculée. 2. Le KG indique une possibilité de carcinome. 3. Je conclus à BI-RADS 4."
    dummy_detections = [{"bbox": [10,20,30,40], "malignancy_label": "Malin", "malignancy_score": 0.8}]
    dummy_metadata = {"age": 55, "history": "Fumeuse", "modalities": ["MG"]}
    dummy_kg = {"diagnostics": [{"pathology": "Carcinome canalaire", "nature": "Malin", "birads": "BI-RADS 5"}]}

    report = await report_gen.process(dummy_reasoning, dummy_detections, {"modalities": ["MG"]}, dummy_kg, dummy_metadata)
    validated = await validator.process(report, dummy_detections)
    print("Rapport validé:", validated)

    # Explainability (sans image)
    heatmaps = await explainer.process(dummy_detections, np.zeros((224,224,3), dtype=np.uint8))
    print("Heatmaps:", len(heatmaps["heatmaps"]))