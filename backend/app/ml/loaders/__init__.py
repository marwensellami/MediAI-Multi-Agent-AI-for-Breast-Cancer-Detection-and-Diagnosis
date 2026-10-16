# backend/app/ml/loaders/__init__.py
"""
Modules de chargement des modèles : vision, LLM, etc.
"""


from .vision_loader import VisionLoader
from .llm_loader import LLMLoader

__all__ = ["VisionLoader", "LLMLoader"]