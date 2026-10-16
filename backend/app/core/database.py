from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from typing import AsyncGenerator
from app.core.config import settings

Base = declarative_base()

# Moteur de connexion ASYNC vers PostgreSQL
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,               # Mettre True pour voir les requêtes SQL
    future=True,
    pool_pre_ping=True,
    pool_size=5,              # Nombre de connexions en pool
    max_overflow=10,          # Connexions supplémentaires en cas de pic
)

# Fabrique de sessions asynchrones
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Dépendance FastAPI : récupère une session DB pour chaque requête
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()