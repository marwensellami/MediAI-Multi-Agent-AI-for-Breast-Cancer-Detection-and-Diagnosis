from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.websockets import WebSocket

class WebSocketCorsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Si c'est une requête WebSocket (upgrade)
        if request.headers.get("upgrade", "").lower() == "websocket":
            # On laisse passer sans modification
            response = await call_next(request)
            return response
        # Pour les autres requêtes, on applique le CORS standard
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response