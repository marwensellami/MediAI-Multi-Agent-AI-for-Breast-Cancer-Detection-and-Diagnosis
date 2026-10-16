#backend/app/core/config.py

import os
from typing import Optional

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Charge les variables depuis le fichier .env
load_dotenv()

class Settings(BaseSettings):
    # --- Database (SQLite pour le développement) ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./breast_cancer.db?timeout=30")
    DATABASE_SYNC_URL: str = os.getenv("DATABASE_SYNC_URL", "sqlite:///./breast_cancer.db?timeout=30")

    # --- Neo4j (optionnel) ---
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

    # --- Qdrant (optionnel) ---
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

    # --- Redis (optionnel) ---
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # --- Models ---
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "./models_cache")

    # --- App Metadata ---
    APP_NAME: str = "Breast Cancer Agentic AI"
    APP_VERSION: str = "1.0.0"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()