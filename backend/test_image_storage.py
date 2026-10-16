#!/usr/bin/env python
"""
Test de stockage et récupération de l'image originale.
Exécutez avec : python test_image_storage.py
"""

import asyncio
import base64
import json
import sys
from pathlib import Path
import httpx
from sqlalchemy import select, func
from app.core.database import AsyncSessionLocal
from app.models.domain.diagnosis import Diagnosis

async def check_database():
    """Vérifie la présence de l'image dans la base de données."""
    print("🔍 Vérification dans la base de données...")
    async with AsyncSessionLocal() as db:
        # Récupérer le dernier diagnostic
        query = select(Diagnosis).order_by(Diagnosis.id.desc()).limit(1)
        result = await db.execute(query)
        diagnosis = result.scalar_one_or_none()
        if not diagnosis:
            print("❌ Aucun diagnostic trouvé.")
            return None

        print(f"✅ Diagnostic trouvé : ID {diagnosis.id}")
        raw = diagnosis.raw_features or {}
        original_image = raw.get("original_image")
        if original_image:
            print(f"✅ original_image présent : longueur {len(original_image)} caractères.")
            # Décoder pour vérifier que c'est valide
            try:
                img_bytes = base64.b64decode(original_image)
                print(f"   - Taille décodée : {len(img_bytes)} octets")
                print(f"   - Début du base64 : {original_image[:50]}...")
                # On peut vérifier le type de l'image
                if img_bytes[:4] == b'\x89PNG':
                    print("   - ✅ Format PNG détecté.")
                else:
                    print("   - ⚠️  Format non reconnu comme PNG.")
            except Exception as e:
                print(f"   - ❌ Erreur de décodage : {e}")
        else:
            print("❌ original_image absent dans raw_features.")

        # Vérifier les détections
        detections = raw.get("detections", [])
        print(f"   - {len(detections)} détection(s) stockée(s).")

        return diagnosis.id

async def check_api(diagnosis_id: int):
    """Teste l'endpoint API /report."""
    print(f"\n🌐 Test de l'API /diagnosis/{diagnosis_id}/report ...")
    url = f"http://localhost:8001/api/v1/diagnosis/{diagnosis_id}/report"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                original = data.get("original_image")
                annotated = data.get("annotated_image")
                print(f"✅ API OK")
                print(f"   - original_image : {len(original) if original else 0} caractères")
                print(f"   - annotated_image : {len(annotated) if annotated else 0} caractères")
                # Vérifier que le frontend pourra afficher
                if original and original.startswith("data:image"):
                    print("   - ⚠️  Le frontend attend 'data:image/png;base64,...' mais l'API renvoie du base64 brut.")
                    print("   -   ✅ Le frontend doit préfixer avec 'data:image/png;base64,' dans ReportViewer.")
                elif original:
                    print("   - ✅ L'image est renvoyée en base64 brut (à préfixer côté frontend).")
                else:
                    print("   - ❌ Aucune image originale renvoyée par l'API.")
            else:
                print(f"❌ Erreur API : {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Erreur de connexion à l'API : {e}")

async def main():
    diagnosis_id = await check_database()
    if diagnosis_id:
        await check_api(diagnosis_id)
    print("\n✅ Test terminé.")

if __name__ == "__main__":
    asyncio.run(main())