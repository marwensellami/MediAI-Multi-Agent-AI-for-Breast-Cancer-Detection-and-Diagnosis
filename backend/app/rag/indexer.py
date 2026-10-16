import asyncio
import os
import logging
import uuid
from typing import List, Dict, Any
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models as qmodels

from app.core.qdrant_client import get_qdrant_client
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DOCUMENTS_DIR = Path(__file__).parent.parent / "data" / "documents"
COLLECTION_NAME = "medical_knowledge"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
VECTOR_SIZE = 1024  # Taille des embeddings de BGE-M3


class MedicalIndexer:
    def __init__(self):
        self.client = get_qdrant_client()
        if not self.client:
            raise RuntimeError("Impossible de se connecter à Qdrant. Assurez-vous qu'il tourne.")

        logger.info(f"📥 Chargement du modèle d'embeddings {EMBEDDING_MODEL_NAME}...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        # Note : si vous avez un GPU, passez device="cuda" pour accélérer.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        self._ensure_collection()

    def _ensure_collection(self):
        """Crée la collection Qdrant si elle n'existe pas."""
        collections = self.client.get_collections().collections
        existing_names = [c.name for c in collections]

        if COLLECTION_NAME not in existing_names:
            logger.info(f"🆕 Création de la collection '{COLLECTION_NAME}'...")
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=VECTOR_SIZE,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(f"✅ Collection '{COLLECTION_NAME}' créée.")
        else:
            logger.info(f"ℹ️ Collection '{COLLECTION_NAME}' existe déjà.")

    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extrait le texte page par page d'un PDF."""
        try:
            reader = PdfReader(pdf_path)
            chunks = []
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    chunks.append({
                        "page": page_num,
                        "text": text.strip(),
                        "source": pdf_path.name,
                    })
            return chunks
        except Exception as e:
            logger.error(f"❌ Erreur lecture {pdf_path.name} : {e}")
            return []

    def index_documents(self):
        """Parcourt le dossier, découpe et indexe dans Qdrant."""
        if not DOCUMENTS_DIR.exists():
            logger.error(f"📁 Dossier {DOCUMENTS_DIR} introuvable. Créez-le et placez vos PDF.")
            return

        pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"⚠️ Aucun PDF trouvé dans {DOCUMENTS_DIR}")
            return

        total_chunks = 0
        for pdf_file in pdf_files:
            logger.info(f"📖 Traitement de {pdf_file.name}...")
            page_chunks = self.extract_text_from_pdf(pdf_file)

            for chunk_data in page_chunks:
                # Découpage en sous-chunks (sémantiques)
                sub_chunks = self.text_splitter.split_text(chunk_data["text"])
                for sub_chunk in sub_chunks:
                    if len(sub_chunk.strip()) < 20:  # Ignorer les fragments trop petits
                        continue

                    # Encodage avec BGE-M3
                    embedding = self.encoder.encode(sub_chunk, normalize_embeddings=True).tolist()

                    # Générer un ID numérique valide pour Qdrant
                    point_id = abs(hash(sub_chunk + str(chunk_data['page']))) % (2**63 - 1)

                    # Préparation du point à insérer
                    point = qmodels.PointStruct(
                        id=point_id,  # Utiliser un entier au lieu d'une chaîne
                        vector=embedding,
                        payload={
                            "text": sub_chunk,
                            "source": pdf_file.name,
                            "page": chunk_data["page"],
                        },
                    )
                    try:
                        self.client.upsert(collection_name=COLLECTION_NAME, points=[point])
                        total_chunks += 1
                    except Exception as e:
                        logger.error(f"❌ Erreur lors de l'insertion du chunk : {e}")

            logger.info(f"✅ {pdf_file.name} traité avec succès.")

        logger.info(f"🎉 Indexation terminée. {total_chunks} chunks insérés dans Qdrant.")


async def run_indexing():
    """Point d'entrée asynchrone pour l'indexation."""
    try:
        indexer = MedicalIndexer()
        indexer.index_documents()
    except Exception as e:
        logger.exception(f"⛔ Échec de l'indexation : {e}")


if __name__ == "__main__":
    asyncio.run(run_indexing())