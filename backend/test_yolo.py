import numpy as np
from ultralytics import YOLO
from app.ml.processors.dicom_reader import DICOMProcessor
from pathlib import Path

# Charger le modèle
model = YOLO("yolo11m.pt")

# Lire un DICOM
dicom = DICOMProcessor.read_dicom(Path("app/data/uploads/mammo.dcm"))
if not dicom["is_valid"]:
    print("DICOM invalide")
    exit()

pixels = dicom["pixels"]
# Préparer pour YOLO (doit être RGB uint8)
image = DICOMProcessor.prepare_for_yolo(pixels)
print(f"Image shape: {image.shape}, dtype: {image.dtype}")

# Inférence avec seuil bas
results = model(image, verbose=True, conf=0.1)
for r in results:
    boxes = r.boxes
    if boxes is not None:
        print(f"Nombre de boîtes : {len(boxes)}")
        for box in boxes:
            print(f"  score: {box.conf[0].item():.3f}, classe: {model.names[int(box.cls[0])]}")
    else:
        print("Aucune boîte")