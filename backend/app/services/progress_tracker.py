import asyncio
from typing import List, Dict, Any

class ProgressTracker:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.steps = []
        self.percent = 0
        self.status = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
        self.result = None
        self.error = None
        self._subscribers = []  # List of asyncio.Queue pour WebSocket

    def update(self, step: str, percent: int, message: str = ""):
        """Met à jour la progression et notifie les abonnés."""
        self.steps.append({"step": step, "percent": percent, "message": message})
        self.percent = percent
        self._notify()

    def set_status(self, status: str):
        self.status = status
        self._notify()

    def set_result(self, result):
        self.result = result
        self.status = "COMPLETED"
        self.percent = 100
        self._notify()

    def set_error(self, error: str):
        self.error = error
        self.status = "FAILED"
        self._notify()

    def _notify(self):
        """Envoie l'état actuel à tous les abonnés WebSocket."""
        data = {
            "task_id": self.task_id,
            "status": self.status,
            "percent": self.percent,
            "steps": self.steps,
            "error": self.error,
        }
        for queue in self._subscribers:
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

    def subscribe(self, queue: asyncio.Queue):
        self._subscribers.append(queue)

    def unsubscribe(self, queue: asyncio.Queue):
        if queue in self._subscribers:
            self._subscribers.remove(queue)