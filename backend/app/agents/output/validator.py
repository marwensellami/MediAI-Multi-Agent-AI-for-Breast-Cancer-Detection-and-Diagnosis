import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ValidatorAgent:
    def __init__(self):
        self.agent_id = "validator_agent"

    async def process(self, report: Dict[str, Any], detections: List[Dict]) -> Dict[str, Any]:
        """
        Valide la cohérence du rapport par rapport aux détections.
        Retourne le rapport enrichi d'un score de confiance ajusté et d'alertes.
        """
        birads = report.get("birads_category", 0)
        malignant_count = report.get("malignant_lesions", 0)
        total_lesions = report.get("total_lesions", 0)

        alerts = []
        confidence = report.get("confidence_score", 0.8)

        # Règle 1 : BI-RADS 5 doit avoir au moins une lésion maligne
        if birads >= 5 and malignant_count == 0:
            alerts.append("Incohérence : BI-RADS 5 sans lésion maligne détectée.")
            confidence *= 0.7

        # Règle 2 : BI-RADS 1 doit avoir 0 lésion
        if birads == 1 and total_lesions > 0:
            alerts.append("Incohérence : BI-RADS 1 avec lésions détectées.")
            confidence *= 0.6

        # Règle 3 : Si pas de lésion, BI-RADS doit être 1
        if total_lesions == 0 and birads > 1:
            alerts.append("Avertissement : Pas de lésion mais BI-RADS > 1.")
            confidence *= 0.8

        # Mettre à jour le rapport
        report["validation_alerts"] = alerts
        report["confidence_score"] = confidence

        logger.info(f"✅ Validation terminée, score de confiance ajusté à {confidence:.2f}")
        return report