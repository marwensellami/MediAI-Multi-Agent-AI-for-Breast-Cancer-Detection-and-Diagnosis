# backend/app/services/validation_service.py
import logging

logger = logging.getLogger(__name__)

class ValidationService:
    @staticmethod
    def validate_report(detections, reasoning):
        """
        Vérifie la cohérence entre les détections et le raisonnement.
        Retourne un raisonnement corrigé et une liste d'alertes.
        """
        alerts = []
        for det in detections:
            label = det.get("malignancy_label")
            if label == "Malin":
                if "bénin" in reasoning.lower():
                    alerts.append(f"Contradiction : lésion classée maligne, mais raisonnement mentionne 'bénin'.")
                    reasoning = reasoning.replace("bénin", "malin")
                    reasoning = reasoning.replace("bénigne", "maligne")
            elif label == "Bénin":
                if "malin" in reasoning.lower():
                    alerts.append(f"Contradiction : lésion classée bénigne, mais raisonnement mentionne 'malin'.")
                    reasoning = reasoning.replace("malin", "bénin")
                    reasoning = reasoning.replace("maligne", "bénigne")

        # Supprimer les phrases trop vagues (ex: "compatible avec une lésion bénigne" si maligne)
        if "Malin" in [d.get("malignancy_label") for d in detections]:
            if "compatible avec une lésion bénigne" in reasoning:
                reasoning = reasoning.replace("compatible avec une lésion bénigne", "suspecte")
                alerts.append("Ajustement automatique : remplacement de 'compatible bénigne' par 'suspecte'.")

        return reasoning, alerts
    
    
