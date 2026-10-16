#backend/app/models/domain/patient.py

from sqlalchemy import Column, Integer, String, Text, DateTime, func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(10), nullable=True, default="FEMALE")
    medical_history = Column(Text, nullable=True)  # Antécédents médicaux (texte libre)
    risk_factors = Column(Text, nullable=True)    # Facteurs de risque (tabac, hérédité...)

    # Date de création automatique
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relations avec les autres tables
    studies = relationship("Study", back_populates="patient", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Patient(id={self.id}, age={self.age})>"