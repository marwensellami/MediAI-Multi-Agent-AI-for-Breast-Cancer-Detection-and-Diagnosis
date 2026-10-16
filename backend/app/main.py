# backend/app/main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import router as api_router
from app.api.v1.websocket import websocket_handler
from app.init_models import init_models
from app.core.database import engine, Base

# ✅ Import obligatoire pour enregistrer tous les modèles
from app.models.domain import *  # Patient, Study, Diagnosis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Breast Cancer AI Framework", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Démarrage de l'application...")
    # ✅ Créer les tables si elles n'existent pas
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Tables vérifiées/créées")
    try:
        init_models()
        logger.info("✅ Modèles chargés avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement des modèles: {e}")

app.include_router(api_router, prefix="/api")

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket, task_id: str):
    await websocket_handler(websocket, task_id)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)