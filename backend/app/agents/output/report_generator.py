# backend/app/agents/output/report_generator.py
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from matplotlib import lines
from networkx import density
from app.utils.anatomy import get_quadrant, get_side, get_view_description
from app.utils.birads_utils import get_birads_info
from app.utils.anatomy import get_quadrant, get_side, get_view_description, get_density_description
logger = logging.getLogger(__name__)


class ReportGeneratorAgent:
    def __init__(self, language: str = "fr"):
        self.agent_id = "report_generator_agent"
        self.language = language
    
    async def process(
        self,
        reasoning_text: Optional[str],
        detections: List[Dict],
        fusion_metadata: Dict,
        kg_info: Dict,
        clinical_metadata: Dict,
    ) -> Dict[str, Any]:
        """
        Génère un rapport structuré à partir des données.
        Chaque lésion est présentée individuellement avec ses paramètres,
        sa synthèse et sa recommandation.
        """
        # ============================================================
        # LOG POUR DIAGNOSTIC
        # ============================================================
        logger.info(f"📊 {len(detections)} détections reçues par ReportGenerator")
        for i, det in enumerate(detections):
            label = det.get("malignancy_label", "NON DEFINI")
            score = det.get("malignancy_score", 0)
            logger.info(f"   Détection {i+1}: label='{label}', score={score:.3f}")

        # ============================================================
        # FILTRAGE : garder uniquement les lésions (pas les "Normal")
        # ============================================================
        lesions = []
        for det in detections:
            label = det.get("malignancy_label", "")
            label_clean = label.lower().strip()
            if label_clean in ["normal", "indéterminé", "inconnu", ""]:
                logger.info(f"⛔ Détection ignorée (label='{label}')")
                continue
            lesions.append(det)

        num_lesions = len(lesions)
        logger.info(f"✅ {num_lesions} lésions conservées après filtrage")

        # ============================================================
        # STATISTIQUES
        # ============================================================
        malignant = 0
        benign = 0
        for det in lesions:
            label = det.get("malignancy_label", "").lower()
            if label in ["malin", "maligne", "malignant"]:
                malignant += 1
            elif label in ["bénin", "bénigne", "benign"]:
                benign += 1
            else:
                malignant += 1

        max_score = 0.0
        if lesions:
            max_score = max((d.get("malignancy_score", 0) for d in lesions), default=0)

        # ============================================================
        # CONFIANCE GLOBALE
        # ============================================================
        if num_lesions == 0:
            confidence = 0.95
        else:
            if max_score >= 0.95:
                confidence = 0.95
            elif max_score >= 0.50:
                confidence = 0.90
            elif max_score >= 0.10:
                confidence = 0.85
            elif max_score >= 0.02:
                confidence = 0.75
            else:
                confidence = 0.70

        # ============================================================
        # CONSTRUCTION DU RAPPORT
        # ============================================================
        lines = []
        lines = []
        lines.append("📋 # RAPPORT RADIOLOGIQUE MAMMAIRE")
        lines.append("")
        lines.append(f"**Patient** : {clinical_metadata.get('patient_id', 'Inconnu')}")
        lines.append("")
        lines.append(f"**Âge** : {clinical_metadata.get('age') or 'Non renseigné'}")
        lines.append("")
        lines.append(f"**Modalités** : {', '.join(clinical_metadata.get('modalities', ['MG']))}")
        lines.append("")
        view = clinical_metadata.get("view", "Non spécifiée")
        view_display = get_view_description(view)
        density = clinical_metadata.get("density", "Non spécifiée")
        density_display = get_density_description(density)
        if view and view != "Non spécifiée":
            lines.append(f"**Vue** : {view_display}")
            lines.append("")
        if density and density != "Non spécifiée":
            lines.append(f"**Densité mammaire** : {density_display}")
            lines.append("")
        lines.append(f"**Date** : {datetime.now().strftime('%d/%m/%Y')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        if num_lesions == 0:
            lines.append("## Observations")
            lines.append("")
            lines.append("Aucune lésion suspecte détectée sur les images fournies.")
            lines.append("")
            lines.append("## Synthèse")
            lines.append("")
            lines.append("Examen normal. Aucun signe de cancer.")
            lines.append("")
            lines.append("## Classification BI-RADS")
            lines.append("")
            lines.append("**BI-RADS 1** (Négatif) – Probabilité de malignité : 0 %")
            lines.append("")
            lines.append("## Recommandation")
            lines.append("")
            lines.append("Poursuivre le calendrier de dépistage de routine par mammographie.")
        else:
            lines.append("## Observations")
            lines.append("")
            for i, det in enumerate(lesions, 1):
                # Récupération des données
                bbox = det.get("bbox", [0, 0, 0, 0])
                quadrant = get_quadrant(bbox, clinical_metadata.get("image_shape", (512, 512)))
                side = get_side(clinical_metadata.get("patient_position"))
                size_mm = det.get("size_mm", "Non disponible")
                prob = det.get("malignancy_score", 0.5)
                prob_percent = round(prob * 100, 1)
                label = det.get("malignancy_label", "Indéterminé")
                birads_info = get_birads_info(prob)
                birads_level = birads_info["level"]
                birads_desc = birads_info["description"]
                birads_score_range = birads_info["score_range"]
                birads_rec = birads_info["recommendation"]
                area = det.get("area_mm2")
                perimeter = det.get("perimeter_mm")
                circularity = det.get("circularity")

                lines.append(f"### Lésion {i}")
                lines.append("")
                lines.append(f"- **Localisation** : {quadrant} ({side})")
                #view = clinical_metadata.get("view", "Non spécifiée")
                #lines.append(f"    - Vue : {view}")
                if size_mm is not None and isinstance(size_mm, (int, float)):
                    lines.append(f"- **Taille estimée** : {size_mm:.1f} mm")
                else:
                    lines.append("- **Taille estimée** : Non mesurable")
                lines.append(f"- **Classification** : {label}")
                lines.append(f"- **Probabilité de malignité** : {prob_percent} %")
                lines.append(f"- **BI-RADS** : {birads_level} ({birads_desc})")
                lines.append(f"- **Seuil de risque** : {birads_score_range}")

                if area is not None and perimeter is not None:
                    lines.append("")
                    lines.append("**Paramètres morphologiques :**")
                    lines.append(f"  - **Aire** : {area:.2f} mm²")
                    lines.append(f"  - **Périmètre** : {perimeter:.2f} mm")
                    if circularity is not None:
                        lines.append(f"  - **Circularité** : {circularity:.3f}")

                if birads_info["birads_number"] >= 4:
                    interp = "Lésion suspecte nécessitant une investigation approfondie."
                elif birads_info["birads_number"] == 3:
                    interp = "Lésion probablement bénigne, mais un contrôle est recommandé."
                else:
                    interp = "Lésion bénigne, sans risque significatif."

                lines.append("")
                lines.append(f"**Synthèse** : {interp}")
                lines.append("")
                lines.append(f"**Recommandation** : {birads_rec}")
                lines.append("")
                lines.append("---")
                lines.append("")

            # ---- Conclusion ----
            lines.append("## Conclusion")
            lines.append("")
            lines.append(
                    f"  {num_lesions} {'lésion' if num_lesions <= 1 else 'lésions'} "
                    f"{'détectée' if num_lesions <= 1 else 'détectées'}, "
                    f"dont {malignant} {'maligne' if malignant <= 1 else 'malignes'} "
                    f"et {benign} {'bénigne' if benign <= 1 else 'bénignes'}."
                    )
            if malignant > 0:
                lines.append("- Présence d'au moins une lésion maligne. Une prise en charge rapide est recommandée.")
            elif benign > 0 and malignant == 0:
                lines.append("- Les lésions détectées sont bénignes. Aucune anomalie maligne n'est identifiée.")
            else:
                lines.append("- Les lésions détectées ne présentent pas de caractère malin évident.")
            lines.append("- La concordance entre les résultats de l'imagerie et les données cliniques doit être évaluée par un radiologue.")
            lines.append("")

            # ---- Confiance ----
            lines.append("## Confiance du système")
            lines.append("")
            lines.append(f"- **Confiance globale** : {round(confidence*100)} %")
            yolo_conf = round(max((d.get("confidence", 0) for d in lesions), default=0) * 100)
            lines.append(f"- **Confiance détection YOLO** : {yolo_conf} %")

        # ---- Avertissement ----
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("> **NB** : Ce rapport est une aide à la décision et ne remplace pas l'avis d'un radiologue.")

        # ---- Ajout du raisonnement clinique s'il est fourni ----
        if reasoning_text and reasoning_text.strip():
            lines.append("")
            lines.append("## Raisonnement clinique détaillé")
            lines.append("")
            lines.append(reasoning_text)

        report_text = "\n".join(lines)

        logger.info(f"📋 Rapport généré : {num_lesions} lésions affichées")

        # BI-RADS global (pour l'API)
        global_birads = max((get_birads_info(d.get("malignancy_score", 0))["birads_number"] for d in lesions), default=1)
        global_label = f"BI-RADS {global_birads}"

        return {
            "report_text": report_text,
            "birads_category": global_birads,
            "birads_label": global_label,
            "total_lesions": num_lesions,
            "malignant_lesions": malignant,
            "benign_lesions": benign,
            "confidence_score": confidence,
        } 