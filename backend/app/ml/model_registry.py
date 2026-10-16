import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Singleton pour stocker et récupérer les modèles chargés."""
    _instance: Optional["ModelRegistry"] = None
    _models: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._models = {}
        return cls._instance

    def register(self, key: str, model: Any):
        """Enregistre un modèle en mémoire."""
        self._models[key] = model
        logger.info(f"📦 Modèle '{key}' enregistré dans le registre.")

    def get(self, key: str) -> Optional[Any]:
        """Récupère un modèle s'il existe."""
        return self._models.get(key)

    def is_loaded(self, key: str) -> bool:
        """Vérifie si un modèle est chargé."""
        return key in self._models

    def clear(self):
        """Vide le registre (utile pour les tests)."""
        self._models.clear()
        logger.warning("🧹 Registre des modèles vidé.")