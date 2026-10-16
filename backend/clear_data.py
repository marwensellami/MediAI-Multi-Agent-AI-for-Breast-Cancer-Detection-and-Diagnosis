# backend/clear_data.py
import asyncio
from app.core.database import AsyncSessionLocal
from app.models.domain.patient import Patient
from app.models.domain.study import Study
from app.models.domain.diagnosis import Diagnosis
from sqlalchemy import text

async def clear_all_data():
    async with AsyncSessionLocal() as session:
        # Supprimer toutes les lignes des tables (ordre inverse des dépendances)
        await session.execute(text("DELETE FROM diagnoses"))
        await session.execute(text("DELETE FROM studies"))
        await session.execute(text("DELETE FROM patients"))
        
        # Réinitialiser les compteurs d'auto-incrémentation (SQLite)
        await session.execute(text("DELETE FROM sqlite_sequence WHERE name IN ('diagnoses', 'studies', 'patients')"))
        
        await session.commit()
        print("✅ Toutes les données ont été supprimées. Les compteurs ont été réinitialisés.")

if __name__ == "__main__":
    asyncio.run(clear_all_data())