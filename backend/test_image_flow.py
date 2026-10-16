#!/usr/bin/env python
"""
Test complet du flux de l'image :
- Encodage
- Passage dans l'état
- Stockage en base
- Récupération via API
"""

import asyncio
import base64
import json
import sys
from pathlib import Path
import httpx
from sqlalchemy import select
from app.core.database import AsyncSessionLocal
from app.models.domain.diagnosis import Diagnosis

async def test_image_flow():
    print("🔍 Test complet du flux de l'image...")
    async with AsyncSessionLocal() as db:
        # 1. Récupérer le dernier diagnostic
        query = select(Diagnosis).order_by(Diagnosis.id.desc()).limit(1)
        result = await db.execute(query)
        diagnosis = result.scalar_one_or_none()
        if not diagnosis:
            print("❌ Aucun diagnostic trouvé.")
            return

        print(f"✅ Diagnostic ID: {diagnosis.id}")
        raw = diagnosis.raw_features or {}

        # 2. Vérifier la présence de l'image
        img_b64 = raw.get("original_image")
        if not img_b64:
            print("❌ ❌ ❌ IMAGE ABSENTE de raw_features")
            print(f"Contenu de raw_features: {list(raw.keys())}")
            return

        print(f"✅ Image présente: {len(img_b64)} caractères")

        # 3. Vérifier que le base64 est valide
        try:
            img_bytes = base64.b64decode(img_b64)
            print(f"   - Décodage OK: {len(img_bytes)} octets")
            # Vérifier le magic number PNG
            if img_bytes[:4] == b'\x89PNG':
                print("   - ✅ Format PNG valide")
            else:
                print(f"   - ⚠️ Format non PNG (début: {img_bytes[:4]})")
        except Exception as e:
            print(f"   - ❌ Échec décodage: {e}")
            return

        # 4. Vérifier les détections
        detections = raw.get("detections", [])
        print(f"   - {len(detections)} détections")

        # 5. Test de l'API
        print("\n🌐 Test de l'API /report...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(f"http://localhost:8001/api/v1/diagnosis/{diagnosis.id}/report")
                if resp.status_code == 200:
                    data = resp.json()
                    api_img = data.get("original_image")
                    if api_img:
                        print(f"✅ API renvoie l'image: {len(api_img)} caractères")
                        # Vérifier que c'est la même que celle en base
                        if api_img == img_b64:
                            print("   - ✅ Image identique à celle en base")
                        else:
                            print("   - ⚠️ Image différente de celle en base")
                    else:
                        print("❌ API ne renvoie pas d'image")
                else:
                    print(f"❌ API error {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"❌ Erreur API: {e}")

if __name__ == "__main__":
    asyncio.run(test_image_flow())