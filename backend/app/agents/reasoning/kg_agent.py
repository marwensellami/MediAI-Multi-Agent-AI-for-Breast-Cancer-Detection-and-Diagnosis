import logging
from typing import List, Dict, Any

from app.core.neo4j_client import get_neo4j_driver
from app.utils.helpers import safe_str

logger = logging.getLogger(__name__)

class KGAgent:
    def __init__(self):
        self.agent_id = "kg_agent"

    async def process(self, entities: List[str]) -> Dict[str, Any]:
        """
        Prend une liste d'entités et retourne les diagnostics différentiels et recommandations.
        """
        driver = await get_neo4j_driver()
        if not driver:
            logger.warning("Neo4j non disponible")
            return {"diagnostics": [], "recommendations": []}

        results = {"diagnostics": [], "recommendations": []}
        async with driver.session() as session:
            for entity in entities:
                if not entity or entity is None:
                    continue
                try:
                    query = """
                    MATCH (p:Pathologie)-[:ASSOCIE_A|INDICATEUR]->(b:BI_RADS)
                    WHERE p.name CONTAINS $entity OR b.description CONTAINS $entity
                    RETURN p.name AS pathology, p.nature AS nature, b.label AS birads, b.description AS birads_desc
                    LIMIT 5
                    """
                    result = await session.run(query, entity=safe_str(entity))
                    async for record in result:
                        if record:
                            results["diagnostics"].append({
                                "pathology": safe_str(record.get("pathology")),
                                "nature": safe_str(record.get("nature")),
                                "birads": safe_str(record.get("birads")),
                                "birads_desc": safe_str(record.get("birads_desc")),
                            })
                except Exception as e:
                    logger.error(f"Erreur requête Neo4j pour {entity}: {e}")
                    continue

            # Récupérer des recommandations générales
            try:
                for level in [4, 5]:
                    rec_query = """
                    MATCH (b:BI_RADS {level: $level})
                    RETURN b.recommendation AS recommendation
                    """
                    result = await session.run(rec_query, level=level)
                    async for record in result:
                        if record and record.get("recommendation"):
                            results["recommendations"].append(safe_str(record["recommendation"]))
            except Exception as e:
                logger.error(f"Erreur récupération recommandations: {e}")

        return results