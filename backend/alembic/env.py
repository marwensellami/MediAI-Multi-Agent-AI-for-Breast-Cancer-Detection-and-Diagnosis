import asyncio
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# IMPORTANT : Importer le module des modèles
from app.models.domain import Base
from app.core.config import settings

# Cible des métadonnées
target_metadata = Base.metadata

# Interpréter le fichier alembic.ini
config = context.config

# Surcharger l'URL de la DB (SQLite pour les migrations)
config.set_main_option("sqlalchemy.url", settings.DATABASE_SYNC_URL)

# Configuration des logs
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()