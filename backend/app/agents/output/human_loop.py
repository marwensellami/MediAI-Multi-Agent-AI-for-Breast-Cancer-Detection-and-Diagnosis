import logging
from typing import Dict, Any, Optional
from datetime import datetime
from app.core.database import AsyncSessionLocal
from app.models.domain.diagnosis import Diagnosis

logger = logging.getLogger(__name__)

class HumanLoopAgent:
    def __init__(self):
        self.agent_id = "human_loop_agent"

    async def process_feedback(self, diagnosis_id: int, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reçoit le feedback du radiologue (ex: correction du BI-RADS, nouvelles annotations).
        Stocke ces données pour un futur fine-tuning.
        """
        async with AsyncSessionLocal() as session:
            try:
                # Récupérer le diagnostic
                diagnosis = await session.get(Diagnosis, diagnosis_id)
                if not diagnosis:
                    return {"error": "Diagnosis not found"}

                # Mettre à jour avec le feedback
                # On pourrait stocker le feedback dans un champ JSON dédié
                # Pour l'instant, on le sauvegarde dans raw_features
                if diagnosis.raw_features is None:
                    diagnosis.raw_features = {}
                diagnosis.raw_features["feedback"] = {
                    "timestamp": datetime.now().isoformat(),
                    "corrections": feedback_data,
                }
                await session.commit()

                logger.info(f"📝 Feedback reçu pour le diagnostic {diagnosis_id}")
                return {"status": "Feedback enregistré", "diagnosis_id": diagnosis_id}
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'enregistrement du feedback: {e}")
                return {"error": str(e)}

    async def prepare_for_active_learning(self):
        """
        Récupère les cas avec feedback pour les utiliser dans un futur fine-tuning.
        """
        # Cette méthode serait utilisée par un script externe pour exporter les données
        # d'entraînement.
        pass