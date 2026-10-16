# backend/app/services/__init__.py
from .orchestrator_service import OrchestratorService
from .report_service import ReportService
from .progress_tracker import ProgressTracker

# Créer une instance unique du service
orchestrator_service = OrchestratorService()

__all__ = [
    "OrchestratorService",
    "ReportService",
    "ProgressTracker",
    "orchestrator_service",
]