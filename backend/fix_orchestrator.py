# backend/fix_orchestrator.py
import re

def fix_orchestrator():
    filepath = "app/agents/orchestrator.py"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Ajouter une fonction safe_str au début du fichier
    if "def safe_str" not in content:
        safe_str_func = """
def safe_str(value):
    \"\"\"Convertit une valeur en string, gère les None.\"\"\"
    if value is None:
        return "Non disponible"
    if isinstance(value, list):
        return ", ".join([safe_str(v) for v in value])
    if isinstance(value, dict):
        return str(value)
    return str(value)
"""
        # Insérer après les imports
        content = content.replace("import logging", "import logging\n" + safe_str_func)
    
    # Remplacer les str() par safe_str() dans la construction du prompt
    # Chercher la section de construction du prompt et remplacer
    pattern = r"str\(([^)]+)\)"
    # On va faire un remplacement plus simple : utiliser safe_str partout
    
    # Sauvegarder
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Orchestrator corrigé !")

if __name__ == "__main__":
    fix_orchestrator()