# backend/test_websocket.py
import asyncio
import websockets
import json

async def test_ws():
    task_id = "550e8400-e29b-41d4-a716-446655440000"  # Votre task_id
    uri = f"ws://localhost:8000/ws/{task_id}"
    
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"📊 Progression: {data['percent']}% - {data['status']}")
            if data['status'] in ['COMPLETED', 'FAILED']:
                break

asyncio.run(test_ws())