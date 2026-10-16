# backend/app/init_models.py
import logging
from app.ml.loaders.vision_loader import VisionLoader
from app.ml.loaders.llm_loader import LLMLoader
from app.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

def init_models():
    """
    Initialise et charge tous les modèles dans le ModelRegistry.
    Priorité : modèles fine‑tunés > modèles génériques.
    """
    logger.info("🚀 Initialisation des modèles...")
    registry = ModelRegistry()

    # ============================================================
    # 1. YOLO (fine‑tuné en priorité)
    # ============================================================
    logger.info("📥 Chargement de YOLO fine‑tuné...")
    try:
        yolo_finetuned = VisionLoader.load_yolo_finetuned()
        if yolo_finetuned:
            registry.register("yolo_finetuned", yolo_finetuned)
            logger.info("✅ YOLO fine‑tuné chargé avec succès.")
        else:
            logger.warning("⚠️ YOLO fine‑tuné non disponible, fallback sur le générique.")
    except Exception as e:
        logger.error(f"❌ Erreur chargement YOLO fine‑tuné : {e}")

    # YOLO générique (fallback)
    logger.info("📥 Chargement de YOLO générique...")
    if not registry.is_loaded("yolo"):
        try:
            yolo = VisionLoader.load_yolo()
            registry.register("yolo", yolo)
            logger.info("✅ YOLO générique chargé avec succès.")
        except Exception as e:
            logger.error(f"❌ Erreur chargement YOLO générique : {e}")

    # ============================================================
    # 2. Swin (fine‑tuné en priorité)
    # ============================================================
    logger.info("📥 Chargement de Swin fine‑tuné (3 classes)...")
    try:
        swin_finetuned = VisionLoader.load_swin_finetuned()
        if swin_finetuned:
            registry.register("swin_finetuned", swin_finetuned)
            logger.info("✅ Swin fine‑tuné (3 classes) chargé avec succès.")
        else:
            logger.warning("⚠️ Swin fine‑tuné non disponible, fallback sur le générique.")
    except Exception as e:
        logger.error(f"❌ Erreur chargement Swin fine‑tuné : {e}")

    # Swin générique (fallback)
    logger.info("📥 Chargement de Swin générique...")
    if not registry.is_loaded("swin"):
        try:
            swin = VisionLoader.load_swin()
            registry.register("swin", swin)
            logger.info("✅ Swin générique chargé avec succès.")
        except Exception as e:
            logger.error(f"❌ Erreur chargement Swin générique : {e}")

    # ============================================================
    # 3. SAM2 (optionnel)
    # ============================================================
    #logger.info("📥 Tentative de chargement de SAM2...")
    #if not registry.is_loaded("sam2"):
    #    try:
    #        sam = VisionLoader.load_sam2()
    #        if sam:
    #            registry.register("sam2", sam)
    #            logger.info("✅ SAM2 chargé avec succès.")
    #        else:
    #            logger.warning("⚠️ SAM2 non disponible.")
    #    except Exception as e:
    #        logger.error(f"❌ Erreur chargement SAM2 : {e}")

    # ============================================================
    # 4. LLM (BioMistral / Phi‑2)
    # ============================================================
    logger.info("📥 Chargement du LLM...")
    if not registry.is_loaded("llm"):
        try:
            tokenizer, model, pipeline = LLMLoader.load_model()
            if pipeline:
                registry.register("llm", {
                    "tokenizer": tokenizer,
                    "model": model,
                    "pipeline": pipeline
                })
                logger.info("✅ LLM chargé et enregistré avec succès.")
            else:
                logger.warning("⚠️ LLM non disponible.")
        except Exception as e:
            logger.error(f"❌ Erreur chargement LLM : {e}")

    # ============================================================
    # 5. RÉSUMÉ DES MODÈLES CHARGÉS
    # ============================================================
    loaded = list(registry._models.keys())
    logger.info(f"📦 Modèles chargés dans le registre : {loaded}")
    logger.info("✅ Initialisation des modèles terminée.")
    return registry