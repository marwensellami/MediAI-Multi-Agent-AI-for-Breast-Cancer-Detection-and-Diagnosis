import base64
from sqlalchemy import select
from app.core.database import AsyncSessionLocal
from app.models.domain.diagnosis import Diagnosis
import asyncio

async def decode_latest():
    async with AsyncSessionLocal() as db:
        query = select(Diagnosis).order_by(Diagnosis.id.desc()).limit(1)
        result = await db.execute(query)
        diag = result.scalar_one_or_none()
        if not diag:
            print("Aucun diagnostic")
            return
        raw = diag.raw_features or {}
        img = raw.get("original_image")
        if not img:
            print("Pas d'image")
            return
        try:
            data = base64.b64decode(img)
            with open("test_image.png", "wb") as f:
                f.write(data)
            print(f"✅ Image sauvegardée sous test_image.png ({len(data)} octets)")
        except Exception as e:
            print(f"❌ Erreur: {e}")

asyncio.run(decode_latest())