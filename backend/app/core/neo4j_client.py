import logging
from typing import Optional

from neo4j import AsyncGraphDatabase, AsyncDriver

from app.core.config import settings

logger = logging.getLogger(__name__)

_driver: Optional[AsyncDriver] = None


async def get_neo4j_driver() -> Optional[AsyncDriver]:
    """
    Retourne une instance unique du driver Neo4j.
    Retourne None si la connexion échoue (ex: Neo4j non démarré).
    """
    global _driver
    if _driver is None:
        try:
            _driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                max_connection_lifetime=3600,
            )
            # Test rapide de connexion
            async with _driver.session() as session:
                await session.run("RETURN 1")
            logger.info(f"✅ Neo4j connecté avec succès à {settings.NEO4J_URI}")
        except Exception as e:
            logger.error(f"❌ Impossible de se connecter à Neo4j : {e}")
            _driver = None
    return _driver


async def close_neo4j_driver():
    """Ferme proprement la connexion Neo4j."""
    global _driver
    if _driver:
        await _driver.close()
        _driver = None
        logger.info("🔒 Connexion Neo4j fermée")