# backend/app/agents/orchestrator.py
import logging
import asyncio
from typing import TypedDict, List, Dict, Any, Optional, Literal
from pathlib import Path
from unittest import result

from langgraph.graph import StateGraph, END

from app.agents.vision.mammography import MammographyAgent
# Les imports suivants ont été supprimés :
# from app.agents.vision.ultrasound import UltrasoundAgent
# from app.agents.vision.mri import MRIAgent
from app.agents.reasoning.fusion import FusionAgent
from app.agents.reasoning.rag_agent import RAGAgent
from app.agents.reasoning.kg_agent import KGAgent
from app.agents.reasoning.medical_reasoning import MedicalReasoningAgent
from app.utils.helpers import safe_str
from app.services.validation_service import ValidationService


logger = logging.getLogger(__name__)

# ============================================================
# 1. DÉFINITION DE L'ÉTAT (AGENT STATE)
# ============================================================
class AgentState(TypedDict, total=False):
    """État partagé entre tous les agents du pipeline."""
    patient_id: str
    clinical_data: Dict[str, Any]
    modalities: List[str]
    images: Dict[str, List[Path]]
    detections: Dict[str, List[Dict]]
    masks: Dict[str, List[Dict]]
    features: Dict[str, List[Dict]]
    vision_summary: Dict[str, Any]
    fusion_vector: Optional[List[float]]
    rag_docs: List[Dict[str, Any]]
    kg_knowledge: Dict[str, Any]
    cot_reasoning: Optional[str]
    final_report: Optional[str]
    final_birads: Optional[str]
    final_confidence: Optional[float]
    validation_status: Optional[str]
    current_step: str
    errors: List[str]
    pixel_spacing_override: Optional[float]


# ============================================================
# 2. CLASSE ORCHESTRATEUR
# ============================================================
class Orchestrator:
    def __init__(self):
        self.graph = None
        self._build_graph()
        logger.info("🧠 Orchestrateur initialisé avec LangGraph")

    def _build_graph(self):
        """Construit le graphe d'état avec les nœuds et les arêtes conditionnelles."""
        workflow = StateGraph(AgentState)

        # Définition des nœuds
        workflow.add_node("load_data", self._load_data)
        # Suppression des nœuds "route_modality", "analyze_us", "analyze_mri"
        workflow.add_node("analyze_mammo", self._analyze_mammo)
        workflow.add_node("aggregate_vision", self._aggregate_vision)
        workflow.add_node("fusion", self._run_fusion)
        workflow.add_node("rag", self._run_rag)
        workflow.add_node("kg", self._run_kg)
        workflow.add_node("reasoning", self._run_reasoning)
        workflow.add_node("validate", self._validate)
        workflow.add_node("generate_report", self._generate_report)
        workflow.add_node("error_handler", self._error_handler)

        workflow.set_entry_point("load_data")
        # Enchaînement direct : chargement -> mammographie -> agrégation -> fusion -> ...
        workflow.add_edge("load_data", "analyze_mammo")
        workflow.add_edge("analyze_mammo", "aggregate_vision")

        workflow.add_edge("aggregate_vision", "fusion")
        workflow.add_edge("fusion", "rag")
        workflow.add_edge("rag", "kg")
        workflow.add_edge("kg", "reasoning")
        workflow.add_edge("reasoning", "validate")

        workflow.add_conditional_edges(
            "validate",
            self._decide_validation,
            {"valid": "generate_report", "invalid": "error_handler"}
        )

        workflow.add_edge("generate_report", END)
        workflow.add_edge("error_handler", END)

        self.graph = workflow.compile()
        logger.info("✅ Graphe LangGraph compilé avec succès")

    def _decide_validation(self, state: AgentState) -> str:
        status = state.get("validation_status", "VALIDATED")
        return "valid" if status == "VALIDATED" else "invalid"

    # ============================================================
    # 3. NŒUDS DU GRAPHE
    # ============================================================

    async def _load_data(self, state: AgentState) -> AgentState:
        logger.info(f"📂 Chargement des données pour patient {state.get('patient_id', 'inconnu')}")
        state["current_step"] = "load_data"
        state["errors"] = state.get("errors", [])

        # Initialisation des champs requis
        state.setdefault("detections", {})
        state.setdefault("images", {})
        state.setdefault("clinical_data", {})
        state.setdefault("modalities", [])
        state.setdefault("rag_docs", [])
        state.setdefault("kg_knowledge", {})
        state.setdefault("vision_summary", {})
        state.setdefault("pixel_spacing_override", None)

        return state

    # ---------- VISION (uniquement mammographie) ----------
    async def _analyze_mammo(self, state: AgentState) -> AgentState:
        if "MG" not in state.get("modalities", []):
            logger.info("🩻 Pas de mammographie à analyser, passage...")
            return state

        logger.info("🩻 Lancement de l'analyse mammographique...")
        try:
            agent = MammographyAgent()
            image_paths = state["images"].get("MG", [])

            if not image_paths:
                images = state["images"].get("MG")
                if images and isinstance(images, Path):
                    image_paths = [images]
                else:
                    raise ValueError("Aucun fichier DICOM pour la mammographie")

            pixel_spacing_override = state.get("pixel_spacing_override")
            if pixel_spacing_override:
                logger.info(f"📏 Pixel spacing forcé pour la mammographie : {pixel_spacing_override} mm/pixel")

            if isinstance(image_paths, list) and image_paths:
                result = await agent.process(
                    image_paths[0],
                    pixel_spacing_override=pixel_spacing_override
                )
            elif isinstance(image_paths, Path):
                result = await agent.process(
                    image_paths,
                    pixel_spacing_override=pixel_spacing_override
                )
            else:
                raise ValueError("Format d'image non supporté pour la mammographie")

            state["detections"]["MG"] = result.get("detections", [])
            summary = result.get("summary", {})
            if isinstance(summary, dict):
                state["vision_summary"] = summary
            else:
                state["vision_summary"] = {"total_lesions": 0, "message": str(summary)}
            state["current_step"] = "analyze_mammo"

            logger.info(f"✅ Mammographie analysée : {len(state['detections']['MG'])} lésion(s) détectée(s)")

        except Exception as e:
            logger.error(f"❌ Erreur mammographie: {e}", exc_info=True)
            state["errors"].append(f"Mammo: {safe_str(e)}")
            state["vision_summary"] = {"total_lesions": 0, "error": str(e)}
        return state

    async def _aggregate_vision(self, state: AgentState) -> AgentState:
        # Agrégation des résultats Vision (uniquement MG)
        all_detections = []
        detections_dict = state.get("detections", {})

        if not isinstance(detections_dict, dict):
            detections_dict = {}

        for modality, dets in detections_dict.items():
            if modality == "all":
                continue
            if dets and isinstance(dets, list):
                for d in dets:
                    if isinstance(d, dict):
                        d["modality"] = modality
                        all_detections.append(d)

        state["detections"]["all"] = all_detections
        state["current_step"] = "aggregate_vision"
        return state

    # ---------- RAISONNEMENT ----------
    async def _run_fusion(self, state: AgentState) -> AgentState:
        try:
            agent = FusionAgent()
            clinical_data = state.get("clinical_data", {})
            clinical_text = f"Âge: {safe_str(clinical_data.get('age', 'inconnu'))}, Antécédents: {safe_str(clinical_data.get('history', 'aucun'))}"
            detections = state["detections"].get("all", [])

            result = await agent.process([], clinical_text, detections)
            state["fusion_vector"] = result.get("fusion_vector")
            state["current_step"] = "fusion"
        except Exception as e:
            logger.error(f"❌ Erreur fusion: {e}", exc_info=True)
            state["errors"].append(f"Fusion: {safe_str(e)}")
        return state

    async def _run_rag(self, state: AgentState) -> AgentState:
        logger.info("📚 Recherche RAG...")
        try:
            agent = RAGAgent()
            summary = state.get("vision_summary", {})
            if not isinstance(summary, dict):
                summary = {}

            nb_lesions = 0
            # La vision_summary ne contient plus que les données MG
            if isinstance(summary, dict) and "total_lesions" in summary:
                nb_lesions = summary.get("total_lesions", 0)

            query = f"Cancer du sein, {nb_lesions} lésions détectées. Symptômes et recommandations."
            state["rag_docs"] = await agent.process(query, top_k=5) or []
            state["current_step"] = "rag"
            logger.info(f"📚 {len(state['rag_docs'])} documents trouvés")
        except Exception as e:
            logger.error(f"❌ Erreur RAG: {e}", exc_info=True)
            state["errors"].append(f"RAG: {safe_str(e)}")
            state["rag_docs"] = []
        return state

    async def _run_kg(self, state: AgentState) -> AgentState:
        logger.info("🕸️ Interrogation du Knowledge Graph...")
        try:
            agent = KGAgent()
            entities = ["masse", "microcalcification", "tumeur"]
            detections = state["detections"].get("all", [])
            for det in detections:
                class_name = det.get("class_name", "")
                if class_name:
                    entities.append(class_name)

            state["kg_knowledge"] = await agent.process(entities) or {"diagnostics": [], "recommendations": []}
            state["current_step"] = "kg"
        except Exception as e:
            logger.error(f"❌ Erreur KG: {e}", exc_info=True)
            state["errors"].append(f"KG: {safe_str(e)}")
            state["kg_knowledge"] = {"diagnostics": [], "recommendations": []}
        return state

    async def _run_reasoning(self, state: AgentState) -> AgentState:
        logger.info("🧠 Génération du raisonnement clinique...")
        try:
            agent = MedicalReasoningAgent()

            fusion_data = {"fusion_vector": state.get("fusion_vector", [])}
            metadata = state.get("clinical_data", {})
            metadata["modalities"] = state.get("modalities", [])
            if "view" in state.get("clinical_data", {}):
                metadata["view"] = state["clinical_data"]["view"]
            if "density" in state.get("clinical_data", {}):
                metadata["density"] = state["clinical_data"]["density"]

            detections = state["detections"].get("all", [])
            rag_docs = state.get("rag_docs", []) or []
            kg_knowledge = state.get("kg_knowledge", {"diagnostics": [], "recommendations": []}) or {"diagnostics": [], "recommendations": []}

            result = await agent.process(fusion_data, rag_docs, kg_knowledge, detections, metadata)
            state["cot_reasoning"] = result if result else "Raisonnement non disponible"

            # Validation de la cohérence
            validated_reasoning, alerts = ValidationService.validate_report(
                detections=detections,
                reasoning=state["cot_reasoning"]
            )
            if alerts:
                state["errors"].extend(alerts)
                logger.warning(f"⚠️ Alertes de validation : {alerts}")
            state["cot_reasoning"] = validated_reasoning
            state["current_step"] = "reasoning"

        except Exception as e:
            logger.error(f"❌ Erreur raisonnement: {e}", exc_info=True)
            state["errors"].append(f"Reasoning: {safe_str(e)}")
            state["cot_reasoning"] = "Raisonnement non disponible en raison d'une erreur."

        return state

    # ---------- SORTIE ----------
    async def _validate(self, state: AgentState) -> AgentState:
        logger.info("✅ Validation des résultats...")
        detections = state["detections"].get("all", [])
        total = len(detections)

        if total == 0:
            state["final_birads"] = "BI-RADS 1 (Négatif)"
            state["validation_status"] = "VALIDATED"
            state["final_confidence"] = 0.95
            state["current_step"] = "validate"
            logger.info("✅ Aucune lésion détectée → BI-RADS 1 (Négatif)")
            return state

        malignant = sum(1 for d in detections if d.get("malignancy_label") in ["Suspecte", "Malin", "Maligne", "Malignant"])
        if malignant > 0:
            max_score = max((d.get("malignancy_score", 0) for d in detections), default=0)
            if max_score < 0.1 and malignant > 0:
                state["errors"].append("Avertissement: Lésion maligne détectée mais score de malignité très faible.")
                confidence = 0.7
            else:
                confidence = 0.85
        else:
            confidence = 0.75

        state["final_confidence"] = confidence
        state["validation_status"] = "VALIDATED"
        state["current_step"] = "validate"
        return state

    async def _generate_report(self, state: AgentState) -> AgentState:
        logger.info("📝 Génération du rapport final...")
        from app.agents.output.report_generator import ReportGeneratorAgent
        agent = ReportGeneratorAgent()

        detections = state["detections"].get("all", [])
        clinical_metadata = state.get("clinical_data", {})
        clinical_metadata["patient_id"] = state.get("patient_id", "Inconnu")
        clinical_metadata["modalities"] = state.get("modalities", [])
        view = state.get("clinical_data", {}).get("view")
        if view:
            clinical_metadata["view"] = view
        kg_info = state.get("kg_knowledge", {})
        reasoning_text = state.get("cot_reasoning", "")

        result = await agent.process(
            reasoning_text=reasoning_text,
            detections=detections,
            fusion_metadata={},
            kg_info=kg_info,
            clinical_metadata=clinical_metadata,
        )

        state["final_report"] = result.get("report_text", "Rapport non disponible")
        state["final_birads"] = result.get("birads_label", "BI-RADS 0")
        state["final_confidence"] = result.get("confidence_score", 0.0)
        state["current_step"] = "generate_report"
        state["total_lesions"] = result.get("total_lesions", 0)
        state["malignant_lesions"] = result.get("malignant_lesions", 0)
        state["benign_lesions"] = result.get("benign_lesions", 0)
        return state

    async def _error_handler(self, state: AgentState) -> AgentState:
        errors = state.get("errors", ["Erreur inconnue"])
        logger.error(f"🚨 Pipeline interrompu. Erreurs: {errors}")
        state["final_report"] = f"Erreur lors de l'analyse : {', '.join([safe_str(e) for e in errors])}"
        state["validation_status"] = "ERROR"
        state["final_confidence"] = 0.0
        state["final_birads"] = "BI-RADS 0"
        return state

    # ============================================================
    # 4. MÉTHODE PUBLIQUE D'EXÉCUTION
    # ============================================================
    async def run(self, initial_state: AgentState) -> AgentState:
        logger.info("🚀 Démarrage de l'orchestrateur...")

        initial_state.setdefault("detections", {})
        initial_state.setdefault("images", {})
        initial_state.setdefault("clinical_data", {})
        initial_state.setdefault("modalities", [])
        initial_state.setdefault("rag_docs", [])
        initial_state.setdefault("kg_knowledge", {})
        initial_state.setdefault("vision_summary", {})
        initial_state.setdefault("errors", [])
        initial_state.setdefault("pixel_spacing_override", None)

        if initial_state.get("pixel_spacing_override"):
            logger.info(f"📏 Pixel spacing forcé pour toute l'analyse : {initial_state['pixel_spacing_override']} mm/pixel")

        final_state = await self.graph.ainvoke(initial_state)
        logger.info(f"✅ Orchestrateur terminé. Statut: {final_state.get('validation_status', 'UNKNOWN')}")
        return final_state