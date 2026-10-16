import asyncio
import logging
from typing import List, Dict, Any

from app.core.neo4j_client import get_neo4j_driver, close_neo4j_driver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Données sources ---
BIRADS_DATA: List[Dict[str, Any]] = [
    {"level": 0, "label": "BI-RADS 0", "description": "Évaluation incomplète. Nécessite des examens supplémentaires."},
    {"level": 1, "label": "BI-RADS 1", "description": "Négatif. Pas d'anomalie détectée."},
    {"level": 2, "label": "BI-RADS 2", "description": "Bénin. Découverte typiquement bénigne."},
    {"level": 3, "label": "BI-RADS 3", "description": "Probablement bénin. Recommandation : suivi court-terme (6 mois)."},
    {"level": 4, "label": "BI-RADS 4", "description": "Suspicion modérée. Biopsie recommandée. Sous-catégories A, B, C."},
    {"level": 5, "label": "BI-RADS 5", "description": "Très suspect de malignité. Biopsie obligatoire."},
    {"level": 6, "label": "BI-RADS 6", "description": "Malignité confirmée par biopsie avant traitement."},
]

PATHOLOGIES_DATA: List[Dict[str, Any]] = [
    {"name": "Kyste", "nature": "Bénin", "description": "Lésion liquidienne bénigne"},
    {"name": "Fibroadénome", "nature": "Bénin", "description": "Tumeur bénigne du tissu conjonctif et glandulaire"},
    {"name": "Carcinome canalaire", "nature": "Malin", "description": "Cancer du sein le plus fréquent"},
    {"name": "Carcinome lobulaire", "nature": "Malin", "description": "Cancer du sein d'origine lobulaire"},
]

RELATIONS: List[Dict[str, str]] = [
    {"pathology": "Kyste", "birads": "BI-RADS 2", "type": "ASSOCIE_A"},
    {"pathology": "Fibroadénome", "birads": "BI-RADS 3", "type": "ASSOCIE_A"},
    {"pathology": "Carcinome canalaire", "birads": "BI-RADS 5", "type": "INDICATEUR"},
    {"pathology": "Carcinome lobulaire", "birads": "BI-RADS 5", "type": "INDICATEUR"},
]


async def build_knowledge_graph():
    """Construction du graphe de connaissances."""
    driver = await get_neo4j_driver()
    if not driver:
        logger.error("⛔ Neo4j non disponible. Arrêt du build.")
        return

    async with driver.session() as session:
        logger.info("🚀 Début de la construction du Knowledge Graph...")

        # 1. Nettoyer les anciennes données
        await session.run("MATCH (n) DETACH DELETE n")
        logger.info("🧹 Base de données nettoyée.")

        # 2. Créer les nœuds BI-RADS
        for bir in BIRADS_DATA:
            await session.run(
                """
                CREATE (b:BI_RADS {level: $level, label: $label, description: $description})
                """,
                **bir
            )
        logger.info(f"✅ {len(BIRADS_DATA)} nœuds BI-RADS créés.")

        # 3. Créer les nœuds Pathologie
        for patho in PATHOLOGIES_DATA:
            await session.run(
                """
                CREATE (p:Pathologie {name: $name, nature: $nature, description: $description})
                """,
                **patho
            )
        logger.info(f"✅ {len(PATHOLOGIES_DATA)} nœuds Pathologie créés.")

        # 4. Créer les relations
        for rel in RELATIONS:
            # Utiliser une variable locale pour éviter les problèmes de formatage
            relation_type = rel["type"]
            query = f"""
                MATCH (p:Pathologie {{name: $pathology}})
                MATCH (b:BI_RADS {{label: $birads}})
                CREATE (p)-[:{relation_type}]->(b)
            """
            await session.run(
                query,
                pathology=rel["pathology"],
                birads=rel["birads"]
            )
        logger.info(f"✅ {len(RELATIONS)} relations créées.")

        # 5. Vérification rapide
        result = await session.run("MATCH (n) RETURN count(n) as total")
        total = await result.single()
        logger.info(f"🎉 Graphe terminé ! Nombre total de nœuds : {total['total']}")

    await close_neo4j_driver()


if __name__ == "__main__":
    asyncio.run(build_knowledge_graph())