#backend/app/models/domain/diagnosis.py

from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, DateTime, JSON, func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    study_id = Column(Integer, ForeignKey("studies.id", ondelete="CASCADE"), nullable=False, index=True)

    # Résultats structurés
    birads_category = Column(String(10), nullable=True)          # Ex: "BI-RADS 4"
    birads_score = Column(Integer, nullable=True)                # 0 à 6
    diagnosis_label = Column(String(50), nullable=True)          # "Bénin", "Malin", "Suspect"
    confidence_score = Column(Float, nullable=True)              # 0.0 à 1.0

    # Localisation (ex: "sein droit, quadrant supéro-externe")
    location = Column(String(255), nullable=True)

    # Rapport complet généré par le LLM (en texte brut ou Markdown)
    raw_report = Column(Text, nullable=True)

    # Données techniques / intermédiaires stockées en JSON
    raw_features = Column(JSON, nullable=True)                   # Embeddings, features brutes
    segmentation_masks = Column(JSON, nullable=True)             # URLs ou chemins vers les masques

    # Metadonnées d'exécution
    processing_time_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relation inverse
    study = relationship("Study", back_populates="diagnoses")

    file_name = Column(String(255), nullable=True)

    def __repr__(self):
        return f"<Diagnosis(id={self.id}, birads={self.birads_category}, score={self.confidence_score})>"