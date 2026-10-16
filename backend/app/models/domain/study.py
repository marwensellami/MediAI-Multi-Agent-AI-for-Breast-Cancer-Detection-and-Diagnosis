#backend/app/models/domain/study.py

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Study(Base):
    __tablename__ = "studies"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)

    # Modalité : "MG" (Mammographie), "US" (Échographie), "MRI" (IRM)
    modality = Column(String(10), nullable=False)
    view = Column(String(10), nullable=True)
    study_date = Column(DateTime(timezone=True), nullable=True)
    description = Column(Text, nullable=True)  # Description libre de l'examen
    raw_dicom_paths = Column(Text, nullable=True)  # Stocke les chemins des fichiers DICOM (JSON ou séparés par des virgules)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations
    patient = relationship("Patient", back_populates="studies")
    diagnoses = relationship("Diagnosis", back_populates="study", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Study(id={self.id}, modality={self.modality}, patient_id={self.patient_id})>"