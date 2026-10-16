# backend/create_dicom_test.py
import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import ExplicitVRLittleEndian
import os
from pathlib import Path

def create_test_dicom(filename, modality="MG", shape=(512, 512)):
    """Crée un fichier DICOM de test avec une lésion simulée."""
    
    # Créer une image avec une lésion
    image = np.zeros(shape, dtype=np.uint16)
    
    # Ajouter une lésion (cercle)
    center = (shape[0]//2, shape[1]//2)
    radius = 50
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (i - center[0])**2 + (j - center[1])**2 < radius**2:
                image[i, j] = 2000  # Valeur DICOM typique pour une lésion
            else:
                image[i, j] = 500   # Fond

    # Créer le dataset DICOM
    ds = Dataset()
    
    # Métadonnées obligatoires
    ds.file_meta = Dataset()
    ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1.2'  # Mammographie
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    # Métadonnées patient
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"
    ds.PatientSex = "F"
    ds.PatientBirthDate = "19700101"
    
    # Métadonnées étude
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.StudyDate = "20240101"
    ds.StudyTime = "120000"
    ds.StudyDescription = "Breast Cancer Screening"
    
    # Métadonnées série
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesNumber = 1
    ds.Modality = modality
    ds.SeriesDescription = "Mammography Test"
    
    # Métadonnées image
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.1.2'
    
    # Pixel data
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = shape[0]
    ds.Columns = shape[1]
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # Unsigned
    ds.PixelSpacing = [0.1, 0.1]  # mm par pixel
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    
    # Ajouter les pixels
    ds.PixelData = image.tobytes()
    
    # Sauvegarder
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(filename, write_like_original=False)
    
    print(f"✅ Fichier DICOM créé: {filename}")

def create_all_test_dicoms():
    """Crée tous les fichiers DICOM de test."""
    
    # Créer le dossier
    dicom_dir = Path("app/data/dicom_samples")
    dicom_dir.mkdir(parents=True, exist_ok=True)
    
    # Mammographie (avec lésion)
    create_test_dicom(
        dicom_dir / "mammo.dcm",
        modality="MG",
        shape=(512, 512)
    )
    
    # Échographie (avec lésion)
    create_test_dicom(
        dicom_dir / "echo.dcm",
        modality="US",
        shape=(400, 300)
    )
    
    print("\n🎉 Tous les fichiers DICOM de test ont été créés !")

if __name__ == "__main__":
    create_all_test_dicoms()