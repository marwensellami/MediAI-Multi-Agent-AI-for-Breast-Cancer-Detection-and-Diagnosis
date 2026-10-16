import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import settings

logger = logging.getLogger(__name__)

_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> Optional[QdrantClient]:
    """
    Retourne une instance unique du client Qdrant.
    Retourne None si la connexion échoue.
    """
    global _qdrant_client
    if _qdrant_client is None:
        try:
            _qdrant_client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=10,
            )
            # Test de connexion via la récupération des collections
            _qdrant_client.get_collections()
            logger.info(f"✅ Qdrant connecté avec succès à {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        except Exception as e:
            logger.error(f"❌ Impossible de se connecter à Qdrant : {e}")
            _qdrant_client = None
    return _qdrant_client