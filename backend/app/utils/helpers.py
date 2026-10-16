# backend/app/utils/helpers.py
from typing import Any

def safe_str(value: Any) -> str:
    """Convertit une valeur en string, gère les None et les types complexes."""
    if value is None:
        return "Non disponible"
    if isinstance(value, list):
        return ", ".join([safe_str(v) for v in value])
    if isinstance(value, dict):
        return str(value)
    return str(value)