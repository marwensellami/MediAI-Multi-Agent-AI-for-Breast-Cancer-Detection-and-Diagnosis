# backend/app/ml/loaders/llm_loader.py
import logging
import requests
import json
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class LLMLoader:
    # Modèle Ollama à utiliser
    DEFAULT_MODEL_NAME = "qwen2.5:7b-instruct"
    # URL de l'API Ollama (par défaut localhost:11434)
    OLLAMA_API_URL = "http://localhost:11434/api/generate"

    @staticmethod
    def load_model(model_name: str = DEFAULT_MODEL_NAME):
        """
        Charge un modèle via Ollama et retourne une fonction callable.
        Retourne (tokenizer, model, pipeline) – pour compatibilité, tokenizer et model sont None.
        """
        try:
            # Vérifier que le modèle est disponible dans Ollama
            # On peut lancer 'ollama list' ou faire une requête de test
            logger.info(f"🔄 Utilisation du modèle Ollama : {model_name}")

            # Définir une fonction pipeline qui envoie une requête à Ollama
            def ollama_pipeline(
                prompt: str,
                max_new_tokens: int = 1200,
                temperature: float = 0.2,
                do_sample: bool = True,
                top_p: float = 0.9,
                return_full_text: bool = False,
                **kwargs
            ) -> list:
                """
                Appelle l'API Ollama pour générer une réponse.
                Retourne une liste de dict avec 'generated_text' pour rester compatible.
                """
                # Construction du payload
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                }
                # On retire les paramètres non supportés par Ollama
                try:
                    response = requests.post(
                        LLMLoader.OLLAMA_API_URL,
                        json=payload,
                        timeout=120  # temps d'attente pour la génération
                    )
                    if response.status_code != 200:
                        logger.error(f"❌ Erreur Ollama : {response.status_code} - {response.text}")
                        return [{"generated_text": "Erreur de génération (Ollama)."}]

                    data = response.json()
                    generated_text = data.get("response", "").strip()
                    # Si on veut le texte complet (prompt + réponse) ou seulement la réponse
                    if return_full_text:
                        full_text = prompt + generated_text
                        return [{"generated_text": full_text}]
                    else:
                        return [{"generated_text": generated_text}]

                except requests.exceptions.RequestException as e:
                    logger.error(f"❌ Erreur de connexion à Ollama : {e}")
                    return [{"generated_text": "Erreur de connexion à Ollama."}]

            logger.info(f"✅ LLM {model_name} chargé via Ollama.")
            # Pour compatibilité avec le code existant, on retourne (None, None, pipeline_callable)
            return None, None, ollama_pipeline

        except Exception as e:
            logger.error(f"❌ Échec du chargement du LLM via Ollama : {e}")
            return None, None, None