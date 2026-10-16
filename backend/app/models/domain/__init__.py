# Ce fichier permet d'importer facilement tous les modèles en une seule fois
from app.models.domain.patient import Patient
from app.models.domain.study import Study
from app.models.domain.diagnosis import Diagnosis

# Exporte la Base de SQLAlchemy pour Alembic
# Nous la définissons dans database.py mais nous l'importons ici pour centraliser
from app.core.database import Base

__all__ = [
    "Base",
    "Patient",
    "Study",
    "Diagnosis",
]