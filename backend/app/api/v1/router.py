# backend/app/api/v1/router.py
from fastapi import APIRouter
from . import diagnosis, websocket, chat

router = APIRouter(prefix="/v1")

router.include_router(diagnosis.router)   # routes sous /v1/diagnosis/...
router.include_router(chat.router)        # routes sous /v1/chat/...

# WebSocket est monté séparément dans main.py