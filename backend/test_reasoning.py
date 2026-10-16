# backend/test_reasoning.py
import asyncio
import numpy as np
from app.agents.reasoning.fusion import FusionAgent
from app.agents.reasoning.rag_agent import RAGAgent
from app.agents.reasoning.kg_agent import KGAgent
from app.agents.reasoning.medical_reasoning import MedicalReasoningAgent
from init_models import init_models

async def test_reasoning():
    print("🔍 Test des Agents de Raisonnement...")
    
    # 1. Initialiser les modèles
    print("📥 Chargement des modèles...")
    registry = init_models()
    
    # 2. Créer les agents
    print("🤖 Création des agents...")
    fusion = FusionAgent()
    print("✅ FusionAgent créé")
    
    rag = RAGAgent()
    print("✅ RAGAgent créé")
    
    kg = KGAgent()
    print("✅ KGAgent créé")
    
    reasoning = MedicalReasoningAgent()
    print("✅ MedicalReasoningAgent créé")
    
    # 3. Simuler des données
    dummy_patches = [np.zeros((224, 224, 3), dtype=np.uint8)]
    dummy_clinical_text = "Femme de 55 ans, antécédents familiaux de cancer du sein"
    dummy_detections = [
        {
            "bbox": [100, 150, 200, 250],
            "malignancy_label": "Malin",
            "malignancy_score": 0.78,
            "class_name": "lesion"
        }
    ]
    dummy_metadata = {"age": 55, "history": "Antécédents familiaux", "modalities": ["MG"]}
    
    # 4. Exécuter le pipeline
    print("🔄 Exécution du pipeline de raisonnement...")
    
    fusion_result = await fusion.process(dummy_patches, dummy_clinical_text, dummy_detections)
    print(f"✅ Fusion: vecteur de {len(fusion_result['fusion_vector'])} dimensions")
    
    rag_result = await rag.process("masse spiculée", top_k=3)
    print(f"✅ RAG: {len(rag_result)} documents trouvés")
    
    kg_result = await kg.process(["masse spiculée"])
    print(f"✅ KG: {len(kg_result.get('diagnostics', []))} diagnostics différentiels")
    
    final_reasoning = await reasoning.process(
        fusion_result,
        rag_result,
        kg_result,
        dummy_detections,
        dummy_metadata
    )
    
    print("\n" + "="*60)
    print("🧠 RAISONNEMENT CLINIQUE GÉNÉRÉ:")
    print("="*60)
    print(final_reasoning)
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_reasoning())