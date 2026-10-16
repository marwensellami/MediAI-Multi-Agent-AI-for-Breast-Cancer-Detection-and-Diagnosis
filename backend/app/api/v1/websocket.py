#backend/app/api/v1/websocket.py
import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from app.services.orchestrator_service import orchestrator_service

async def websocket_handler(websocket: WebSocket, task_id: str):
    # Accepter la connexion immédiatement
    await websocket.accept()
    print(f"✅ WebSocket accepté pour task {task_id}")
    
    tracker = orchestrator_service.active_tasks.get(task_id)
    if not tracker:
        await websocket.send_text(json.dumps({"error": "Task not found"}))
        await websocket.close()
        return

    queue = asyncio.Queue()
    tracker.subscribe(queue)

    try:
        while True:
            data = await queue.get()
            await websocket.send_text(json.dumps(data))
    except WebSocketDisconnect:
        print(f"🔌 WebSocket déconnecté pour task {task_id}")
        pass
    finally:
        tracker.unsubscribe(queue)