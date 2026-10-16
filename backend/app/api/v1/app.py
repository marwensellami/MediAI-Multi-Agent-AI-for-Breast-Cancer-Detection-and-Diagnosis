# backend/app/api/v1/chat.py
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.services.orchestrator_service import orchestrator_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chatbot"])

# ============================================================
# 1. MODÈLES DE DONNÉES
# ============================================================

class ChatRequest(BaseModel):
    """Requête de chat envoyée par le frontend."""
    task_id: str
    question: str
    history: Optional[List[Dict[str, str]]] = None  # historique optionnel

class ChatResponse(BaseModel):
    """Réponse du chatbot."""
    answer: str
    task_id: str

# ============================================================
# 2. FONCTIONS UTILITAIRES
# ============================================================

def build_chat_prompt(report: str, question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Construit le prompt complet pour le LLM :
    - Système : rôle et limites
    - Historique (si fourni)
    - Rapport médical
    - Question de l'utilisateur
    """
    system_prompt = """
Vous êtes un assistant spécialisé en imagerie mammaire. Vous êtes conçu pour aider les radiologues à comprendre le rapport généré par le système d'IA.

RÈGLES STRICTES :
1. Vous ne disposez que du rapport ci-dessous. Ne faites appel à aucune connaissance extérieure.
2. Répondez UNIQUEMENT en vous basant sur les informations présentes dans ce rapport.
3. Si une information n'est pas dans le rapport, dites : "Je ne trouve pas cette information dans le rapport."
4. Vous ne devez PAS donner d'avis médical personnel. Vous vous contentez d'expliquer ce qui est écrit dans le rapport.
5. Rappelez toujours que le rapport final fait foi et que votre rôle est d'aider à sa compréhension, pas de le remplacer.
6. Utilisez un langage clair et professionnel, adapté à un radiologue.
"""

    prompt = f"{system_prompt}\n\n"

    # Ajout de l'historique (si fourni)
    if history:
        prompt += "📜 HISTORIQUE DE LA CONVERSATION :\n"
        for entry in history:
            role = "Utilisateur" if entry.get("role") == "user" else "Assistant"
            prompt += f"{role} : {entry.get('content')}\n"
        prompt += "\n"

    # Ajout du rapport
    prompt += f"📄 RAPPORT MÉDICAL :\n{report}\n\n"

    # Question de l'utilisateur
    prompt += f"❓ QUESTION DE L'UTILISATEUR :\n{question}\n\n"

    prompt += "RÉPONSE DE L'ASSISTANT :"

    return prompt


async def get_report_by_task_id(task_id: str) -> str:
    """
    Récupère le rapport final associé à un task_id.
    """
    result = await orchestrator_service.get_result(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Rapport non trouvé ou tâche non terminée.")

    report = result.get("report")
    if not report:
        raise HTTPException(status_code=404, detail="Rapport non disponible.")

    return report


def get_ollama_pipeline():
    """
    Récupère le pipeline Ollama (le même que celui utilisé pour le rapport).
    """
    from app.ml.model_registry import ModelRegistry
    registry = ModelRegistry()
    llm_data = registry.get("llm")
    if llm_data and isinstance(llm_data, dict):
        pipeline = llm_data.get("pipeline")
        if pipeline and callable(pipeline):
            return pipeline
    raise HTTPException(status_code=503, detail="LLM non disponible. Vérifiez Ollama.")


# ============================================================
# 3. ENDPOINTS
# ============================================================

@router.post("/", response_model=ChatResponse)
async def chat_with_report(request: ChatRequest):
    """
    Endpoint de chat. Reçoit une question sur un rapport et renvoie une réponse.
    """
    # 1. Récupérer le rapport
    report = await get_report_by_task_id(request.task_id)

    # 2. Récupérer le pipeline LLM (Ollama)
    try:
        llm_pipeline = get_ollama_pipeline()
    except HTTPException as e:
        raise e

    # 3. Construire le prompt
    prompt = build_chat_prompt(
        report=report,
        question=request.question,
        history=request.history,
    )

    # 4. Appeler le LLM
    try:
        logger.info(f"🧠 Chatbot - Question pour task {request.task_id}: {request.question[:50]}...")
        response = llm_pipeline(
            prompt,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            return_full_text=False,
        )
        

        if isinstance(response, list) and len(response) > 0:
            answer = response[0].get("generated_text", "").strip()
        else:
            answer = "Désolé, je n'ai pas pu générer de réponse."

        logger.info(f"✅ Chatbot - Réponse générée pour task {request.task_id}")
        return ChatResponse(answer=answer, task_id=request.task_id)

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'appel au LLM : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur du LLM : {str(e)}")


@router.get("/health")
async def chat_health():
    """Vérifie la disponibilité du chatbot."""
    try:
        get_ollama_pipeline()
        return {"status": "healthy", "message": "Chatbot disponible"}
    except HTTPException:
        return {"status": "unavailable", "message": "LLM non disponible"}