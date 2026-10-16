# backend/app/agents/reasoning/medical_reasoning.py
from importlib import metadata
import logging
import re
from typing import Dict, Any, List, Optional
from app.utils.anatomy import get_density_description, get_view_description
from app.ml.model_registry import ModelRegistry
from app.utils.birads_utils import get_birads_info  # ← import centralisé
from app.utils.anatomy import get_view_description, get_density_description
logger = logging.getLogger(__name__)

class MedicalReasoningAgent:
    def __init__(self):
        self.agent_id = "medical_reasoning_agent"
        self.registry = ModelRegistry()
        llm_data = self.registry.get("llm")
        if llm_data and isinstance(llm_data, dict):
            self.llm_pipeline = llm_data.get("pipeline")
        else:
            self.llm_pipeline = None

        if not self.llm_pipeline:
            logger.critical("⚠️ Aucun LLM n'est disponible. Le raisonnement clinique sera désactivé.")

    def _correct_contradictions(self, text: str, lesions: List[Dict]) -> str:
        """
        Corrige les labels erronés en traitant chaque lésion dans son propre bloc.
        Remplace 'bénin' par 'malin' (et inversement) dans le bloc correspondant.
        """
        logger.info("🔍 Début de _correct_contradictions")
        # Séparer le texte en blocs selon les motifs "Lésion X"
        blocks = re.split(r'(?=####?\s*Lésion\s+\d+|[Ll]ésion\s+\d+\s*:)', text)
        corrected_blocks = []

        for block in blocks:
            if not block.strip():
                continue

            match = re.search(r'Lésion\s*(\d+)', block, re.IGNORECASE)
            if match:
                lesion_num = int(match.group(1))
                if lesion_num <= len(lesions):
                    det = lesions[lesion_num - 1]
                    real_label = det.get("malignancy_label", "Inconnu").lower()

                    if "malin" in real_label:
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

    def _clean_reasoning(self, reasoning: str) -> str:
        """Nettoie les retours à la ligne et espaces en trop."""
        reasoning = re.sub(r'\n\s*\n', '\n\n', reasoning)
        return reasoning.strip()

    def _format_lesion_detail(self, det: Dict, index: int) -> str:
        """Formate les détails d'une lésion avec BI‑RADS complet."""
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

        # BI‑RADS complet via la fonction centralisée
        birads_info = get_birads_info(prob)
        lines.append(f"  - BI-RADS : {birads_info['level']}")  # ex: BI-RADS 4C

        area = det.get("area_mm2")
        perimeter = det.get("perimeter_mm")
        if area is not None and perimeter is not None:
            lines.append(f"  - Aire : {area:.2f} mm²")
            lines.append(f"  - Périmètre : {perimeter:.2f} mm")

        circularity = det.get("circularity")
        if circularity is not None:
            lines.append(f"  - Circularité : {circularity:.3f}")

        margin = det.get("margin")
        if margin:
            lines.append(f"  - Contours : {margin}")

        shape = det.get("shape")
        if shape:
            lines.append(f"  - Forme : {shape}")

        return "\n".join(lines)

    def format_classifications_only(self, lesions: List[Dict]) -> str:
        """
        Retourne un texte listant uniquement la classification et le BI‑RADS de chaque lésion.
        Exemple de sortie :
            Lésion 1 : Malin (BI-RADS 4C)
            Lésion 2 : Bénin (BI-RADS 4B)
        """
        lines = []
        for i, det in enumerate(lesions, 1):
            label = det.get("malignancy_label", "Inconnu")
            prob = det.get("malignancy_score", 0.5)
            birads = get_birads_info(prob)["level"]  # ex: "BI-RADS 4C"
            lines.append(f"Lésion {i} : {label} ({birads})")
        return "\n".join(lines)
    
    async def process(
        self,
        fusion_data: Dict[str, Any],
        rag_docs: List[Dict],
        kg_data: Dict[str, Any],
        detections: List[Dict],
        metadata: Dict,
    ) -> str:
        # ---- 1. Filtrer les détections pour ne garder que les lésions ----
        lesions = []
        for det in detections:
            label = det.get("malignancy_label", "Normal")
            if label not in ["Normal", "Indéterminé", "Inconnu"]:
                lesions.append(det)

        num_lesions = len(lesions)
        if num_lesions == 0:
            return "Aucune lésion détectée. Examen normal."

        # ---- 2. Vérifier la disponibilité du LLM ----
        if not self.llm_pipeline or not callable(self.llm_pipeline):
            logger.critical("⛔ LLM non disponible, impossible de générer le raisonnement clinique.")
            return "Raisonnement clinique non disponible (LLM non chargé). Veuillez vérifier la configuration."

        # ---- 3. Préparer les données pour le prompt ----
        lesions_details = []
        for i, det in enumerate(lesions, 1):
            lesions_details.append(self._format_lesion_detail(det, i))
        lesions_text = "\n".join(lesions_details)
        classifications_only = self.format_classifications_only(lesions)
        logger.info(f"📝 Classifications :\n{classifications_only}")
        
        age = metadata.get("age") or "Non renseigné"
        view = metadata.get("view") or "Non spécifiée"
        view_display = get_view_description(view)
        density = metadata.get("density") or "Non spécifiée"
        density_display = get_density_description(density)
        history = metadata.get("history") or "Aucun antécédent"
        risk_factors = metadata.get("risk_factors") or "Aucun facteur de risque"
        # On ajoute l'instruction de rappeler explicitement classification et BI-RADS
        prompt = f"""
Vous êtes un radiologue expert en imagerie mammaire. Vous devez analyser les DONNÉES CLINIQUES et les DÉTAILS DES LÉSIONS {lesions_text} suivantes et rédiger un compte-rendu clinique structuré.

📊 DONNÉES CLINIQUES :
- Âge : {age} ans
- Vue mammographique : {view}
- Densité mammaire : {density_display}
- Antécédents : {history}
- Facteurs de risque : {risk_factors}
- Nombre total de lésions détectées : {num_lesions}
- Modalité : {", ".join(metadata.get("modalities", ["MG"]))}

📋 DÉTAIL DES LÉSIONS :
{lesions_text}

⚠️ INSTRUCTIONS STRICTES :
1. Rédigez votre réponse en **deux parties distinctes** :
   - ** 1) - Analyse par lésion, Pour chaque lésion, fournissez Une Analyse médicale qui rappelle explicitement premièrement sa probabilité de malignité, sa classification qu'elle est bien mentionnéé dans {lesions_text} et dans {classifications_only} , sa taille et son BI RADS exact dans une paragraphe, Une interprétation clinique et finalement Une Recommandation.

   - ** 2) - Conclusion générale : Rédigez une conclusion globale et une interprétation d'ensemble du cas.


2. Utilisez uniquement les données fournies ci-dessus. N'inventez aucune autre information.
3. Si une caractéristique morphologique (contours, forme, densité) n'est pas mentionnée, ne l'inventez pas.
4. Soyez précis, utilisez un langage médical approprié, et justifiez vos recommandations.
5. Rédigez en français fluide et professionnel.

RÉPONSE :
"""



        # ---- 5. Appel au LLM ----
        try:
            logger.info("🧠 Appel du LLM pour générer le raisonnement clinique structuré...")
            # ✅ Déterminer max_new_tokens en fonction du nombre de lésions
            if num_lesions <= 3:
                max_tokens = 1200
            elif num_lesions == 4:
                max_tokens = 1500
            elif num_lesions == 5:
                max_tokens = 1800
            elif num_lesions == 6:
                max_tokens = 2100
            elif num_lesions == 7:
                max_tokens = 2400
            elif num_lesions == 8:
                max_tokens = 2700
            else:  # 9 lésions et plus
                max_tokens = 3000
                
                
            response = self.llm_pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                return_full_text=False,
            )

            if isinstance(response, list) and len(response) > 0:
                reasoning = response[0].get("generated_text", "").strip()
                if len(reasoning) > 50:
                    logger.info("✅ Raisonnement clinique généré avec succès.")
                    logger.debug(f"🔍 Texte avant correction : {reasoning[:500]}...")
                    reasoning = self._correct_contradictions(reasoning, lesions)
                    logger.info("✅ Corrections de cohérence appliquées.")
                    reasoning = self._clean_reasoning(reasoning)
                    logger.debug(f"🔍 Texte après correction : {reasoning[:500]}...")

                    # Ajout des références
                    references = list(set([doc.get("source", "") for doc in rag_docs[:3] if doc.get("source")]))
                    if references:
                        reasoning += "\n\n📚 RÉFÉRENCES :\n" + "\n".join(f"- {ref}" for ref in references)

                    return reasoning
                else:
                    logger.warning(f"⚠️ Réponse du LLM trop courte ({len(reasoning)} caractères).")
                    return "Le raisonnement généré est trop court. Veuillez réessayer."
            else:
                logger.error(f"❌ Réponse du LLM au format inattendu : {type(response)}")
                return "Erreur : le format de la réponse du LLM est inattendu."

        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération du raisonnement par le LLM : {e}", exc_info=True)
            return f"Erreur lors de la génération du raisonnement clinique : {str(e)}"