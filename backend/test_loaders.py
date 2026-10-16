import logging
from app.ml.loaders.vision_loader import VisionLoader
from app.ml.loaders.llm_loader import LLMLoader
from app.ml.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vision():
    logger.info("=== TEST VISION ===")
    reg = ModelRegistry()
    # YOLO
    yolo = VisionLoader.load_yolo()
    reg.register("yolo", yolo)
    logger.info(f"YOLO chargé: {yolo is not None}")
    # Swin
    swin = VisionLoader.load_swin()
    reg.register("swin", swin)
    logger.info(f"Swin chargé: {swin is not None}")
    # SAM2
    sam = VisionLoader.load_sam2()
    if sam:
        reg.register("sam2", sam)
        logger.info("SAM2 chargé avec succès")
    else:
        logger.warning("SAM2 non chargé (continuons sans)")

def test_llm():
    logger.info("=== TEST LLM ===")
    tokenizer, model, pipeline = LLMLoader.load_model()
    if pipeline:
        logger.info("✅ LLM chargé. Test de génération...")
        response = pipeline("Décris les symptômes d'un carcinome canalaire.", max_new_tokens=50)
        logger.info(f"Réponse: {response[0]['generated_text']}")
    else:
        logger.error("❌ LLM non chargé.")

if __name__ == "__main__":
    test_vision()
    test_llm()