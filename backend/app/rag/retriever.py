import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import settings

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, client: QdrantClient = None):
        self.client = client or QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        self.collection_name = "medical_knowledge"
        self.encoder = SentenceTransformer("BAAI/bge-m3", device="cpu")
        self.encoder.max_seq_length = 512

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Recherche les documents les plus pertinents."""
        if not self.client:
            return []

        # Encoder la requête
        query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()

        # Recherche dans Qdrant
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
            results = []
            for hit in search_result:
                results.append({
                    "text": hit.payload.get("text", ""),
                    "source": hit.payload.get("source", "unknown"),
                    "page": hit.payload.get("page", 0),
                    "score": hit.score,
                })
            return results
        except Exception as e:
            logger.error(f"Erreur recherche Qdrant: {e}")
            return []