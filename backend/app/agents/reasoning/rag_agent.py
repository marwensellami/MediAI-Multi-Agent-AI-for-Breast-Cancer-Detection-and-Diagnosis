# backend\app\agents\reasoning\rag_agent.py
import logging
from typing import List, Dict, Any

from app.rag.retriever import Retriever  # nous allons le créer
from app.core.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)

class RAGAgent:
    def __init__(self):
        self.agent_id = "rag_agent"
        self.retriever = None
        self.client = get_qdrant_client()
        if self.client:
            self.retriever = Retriever(client=self.client)
            logger.info("✅ RAG Agent prêt")
        else:
            logger.warning("⚠️ Qdrant non disponible, RAG désactivé")

    async def process(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Interroge la base vectorielle avec la requête clinique.
        Retourne les documents les plus pertinents.
        """
        if not self.retriever:
            return []

        try:
            results = self.retriever.retrieve(query, top_k=top_k)
            return results
        except Exception as e:
            logger.error(f"❌ Erreur RAG: {e}")
            return []
        
