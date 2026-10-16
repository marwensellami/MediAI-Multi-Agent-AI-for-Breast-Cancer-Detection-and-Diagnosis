# backend/test_llm_reasoning.py
import asyncio
import logging
import sys
import re
from pathlib import Path

# Ajouter le chemin du projet pour les imports
sys.path.insert(0, str(Path(__file__).parent))

from app.ml.loaders.llm_loader import LLMLoader
from app.utils.birads_utils import get_birads_info

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# DONNÉES DE TEST (identiques à celles de votre pipeline)
# ============================================================
lesions_data = [
    {
        "size_mm": 13.2,
        "malignancy_label": "Malin",
        "malignancy_score": 0.762,
        "area_mm2": 121.44,
        "perimeter_mm": 44.40,
        "circularity": 0.774,
        "bbox": [100, 100, 200, 200],
    },
    {
        "size_mm": 4.6,
        "malignancy_label": "Bénin",
        "malignancy_score": 0.205,
        "area_mm2": 13.80,
        "perimeter_mm": 14.80,
        "circularity": 0.792,
        "bbox": [300, 300, 350, 350],
    }
]

# ============================================================
# FONCTIONS COPIÉES DEPUIS medical_reasoning.py
# ============================================================
def format_lesion_detail(det, index):
    """Formate les détails d'une lésion pour le prompt."""
    lines = [f"Lésion {index}:"]
    
    size_mm = det.get("size_mm")
    if size_mm is not None and isinstance(size_mm, (int, float)):
        lines.append(f"  - Taille : {size_mm:.1f} mm")
    else:
        lines.append("  - Taille : Non mesurable")
    
    label = det.get("malignancy_label", "Indéterminé")
    lines.append(f"  - Classification : {label}")
    
    prob = det.get("malignancy_score", 0.5)
    lines.append(f"  - Probabilité de malignité : {prob*100:.1f} %")
    
    birads = det.get("birads_category")
    if birads is None:
        if prob >= 0.95:
            birads = 5
        elif prob >= 0.10:
            birads = 4
        elif prob >= 0.02:
            birads = 3
        else:
            birads = 2
    lines.append(f"  - BI-RADS : {birads}")
    
    area = det.get("area_mm2")
    perimeter = det.get("perimeter_mm")
    if area is not None and perimeter is not None:
        lines.append(f"  - Aire : {area:.2f} mm²")
        lines.append(f"  - Périmètre : {perimeter:.2f} mm")
    
    return "\n".join(lines)

def build_prompt(lesions):
    """Construit le prompt exact comme dans medical_reasoning.py."""
    num_lesions = len(lesions)
    lesions_details = []
    for i, det in enumerate(lesions, 1):
        lesions_details.append(format_lesion_detail(det, i))
    lesions_text = "\n".join(lesions_details)
    
    prompt = f"""
Vous êtes un radiologue expert en imagerie mammaire. Vous devez analyser les données suivantes et rédiger un compte-rendu clinique structuré.

📊 DONNÉES CLINIQUES :
- Nombre total de lésions détectées : {num_lesions}
- Modalité : MG

📋 DÉTAIL DES LÉSIONS :
{lesions_text}

⚠️ INSTRUCTIONS STRICTES :
1. Rédigez votre réponse en **deux parties distinctes** :
   - **Partie 1 - Analyse par lésion** : Pour chaque lésion, fournissez une analyse médicale, une interprétation clinique et une recommandation/suggestion.
   - **Partie 2 - Conclusion générale** : Rédigez une conclusion globale et une interprétation d'ensemble du cas.

2. Utilisez uniquement les données fournies ci-dessus. N'inventez aucune information.
3. Si une caractéristique morphologique (contours, forme, densité) n'est pas mentionnée, ne l'inventez pas.
4. Soyez précis, utilisez un langage médical approprié, et justifiez vos recommandations.
5. Rédigez en français fluide et professionnel.

RÉPONSE :
"""
    return prompt

def correct_contradictions(text, lesions):
    """
    Version améliorée de la correction – traite chaque bloc de lésion indépendamment.
    """
    logger.info("🔍 Début de _correct_contradictions")
    
    # Séparer le texte en blocs correspondant à chaque lésion
    blocks = re.split(r'(?=####?\s*Lésion\s+\d+|[Ll]ésion\s+\d+\s*:)', text)
    corrected_blocks = []
    
    for block in blocks:
        if not block.strip():
            continue
        
        # Chercher le numéro de lésion dans ce bloc
        match = re.search(r'Lésion\s*(\d+)', block, re.IGNORECASE)
        if match:
            lesion_num = int(match.group(1))
            if lesion_num <= len(lesions):
                det = lesions[lesion_num - 1]
                real_label = det.get("malignancy_label", "Inconnu").lower()
                
                if "malin" in real_label:
                    # Remplacer toutes les formes de "bénin" par "malin"
                    block = re.sub(r'\b(bénin|bénigne|bénins|bénignes|bénine)\b', 'malin', block, flags=re.IGNORECASE)
                    block = re.sub(r'\bbénigne\b', 'maligne', block, flags=re.IGNORECASE)
                    logger.info(f"🔍 Lésion {lesion_num} → remplacement 'bénin*' par 'malin*'")
                elif "bénin" in real_label:
                    block = re.sub(r'\b(malin|maligne|malins|malignes)\b', 'bénin', block, flags=re.IGNORECASE)
                    block = re.sub(r'\bmaligne\b', 'bénigne', block, flags=re.IGNORECASE)
                    logger.info(f"🔍 Lésion {lesion_num} → remplacement 'malin*' par 'bénin*'")
            else:
                logger.warning(f"⚠️ Numéro de lésion {lesion_num} hors limites")
        else:
            logger.debug("ℹ️ Bloc sans numéro de lésion, pas de correction")
        
        corrected_blocks.append(block)
    
    corrected_text = "\n".join(corrected_blocks)
    logger.info("✅ _correct_contradictions terminée")
    return corrected_text

# ============================================================
# TEST PRINCIPAL
# ============================================================
async def main():
    # 1. Charger le LLM
    logger.info("🔄 Chargement du LLM via Ollama...")
    tokenizer, model, pipeline = LLMLoader.load_model()
    if pipeline is None:
        logger.error("❌ LLM non chargé")
        return
    logger.info("✅ LLM chargé")

    # 2. Construire le prompt
    prompt = build_prompt(lesions_data)
    logger.info("📝 Prompt envoyé au LLM (début) :\n" + prompt[:500] + "...\n")

    # 3. Appeler le LLM
    logger.info("🧠 Appel du LLM pour générer le raisonnement...")
    response = pipeline(
        prompt,
        max_new_tokens=1200,
        temperature=0.2,
        do_sample=True,
        top_p=0.9,
        return_full_text=False,
    )

    if not (isinstance(response, list) and len(response) > 0):
        logger.error("❌ Réponse inattendue du LLM")
        return

    reasoning = response[0].get("generated_text", "").strip()
    if len(reasoning) < 50:
        logger.warning(f"⚠️ Réponse trop courte ({len(reasoning)} caractères)")
        logger.info("Réponse brute :\n" + reasoning)
        return

    logger.info("✅ Réponse brute du LLM (longueur {} caractères) :\n".format(len(reasoning)) + "="*80)
    print(reasoning)
    print("="*80)

    # 4. Appliquer la correction
    logger.info("🔄 Application de la correction de cohérence...")
    corrected = correct_contradictions(reasoning, lesions_data)

    logger.info("✅ Réponse après correction :\n" + "="*80)
    print(corrected)
    print("="*80)

    # 5. Vérification manuelle – compter les occurrences
    # On vérifie si "bénin" apparaît dans la section Lésion 1
    import re
    lesion1_block = re.search(r'(####?\s*Lésion\s*1.*?)(?=####?\s*Lésion\s*2|$)', corrected, re.DOTALL | re.IGNORECASE)
    if lesion1_block:
        block_text = lesion1_block.group(1)
        if re.search(r'\b(bénin|bénigne|bénins|bénignes|bénine)\b', block_text, re.IGNORECASE):
            logger.warning("⚠️ ERREUR : 'bénin' trouvé dans le bloc Lésion 1 après correction !")
        else:
            logger.info("✅ OK : Aucun 'bénin' dans le bloc Lésion 1")
    else:
        logger.warning("⚠️ Bloc Lésion 1 non trouvé")

    # Vérifier si "malin" apparaît dans la section Lésion 2
    lesion2_block = re.search(r'(####?\s*Lésion\s*2.*?)(?=####?\s*Lésion\s*3|###\s*Partie\s*2|$)', corrected, re.DOTALL | re.IGNORECASE)
    if lesion2_block:
        block_text = lesion2_block.group(1)
        if re.search(r'\b(malin|maligne|malins|malignes)\b', block_text, re.IGNORECASE):
            logger.warning("⚠️ ERREUR : 'malin' trouvé dans le bloc Lésion 2 après correction !")
        else:
            logger.info("✅ OK : Aucun 'malin' dans le bloc Lésion 2")
    else:
        logger.warning("⚠️ Bloc Lésion 2 non trouvé")

if __name__ == "__main__":
    asyncio.run(main())