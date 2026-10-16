# backend/visualize_mask.py
import cv2
import numpy as np  # <-- AJOUTÉ
from pathlib import Path

from app.ml.processors.image_reader import ImageProcessor
from app.ml.processors.preprocessing import Preprocessor

def visualize_mask(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Fichier non trouvé : {image_path}")
        return

    print(f"📂 Chargement de {image_path.name}...")

    data = ImageProcessor.read_file(image_path)
    if not data["is_valid"]:
        print("❌ Erreur de lecture")
        return

    pixels = data["pixels"]

    # Convertir en uint8 RGB pour l'affichage
    if pixels.dtype != np.uint8:
        if pixels.max() > 0:
            pixels = (pixels / pixels.max() * 255).astype(np.uint8)
        else:
            pixels = pixels.astype(np.uint8)

    if len(pixels.shape) == 2:
        image_rgb = np.stack([pixels] * 3, axis=-1)
    elif pixels.shape[2] == 1:
        image_rgb = np.concatenate([pixels] * 3, axis=-1)
    else:
        image_rgb = pixels[:, :, :3]

    # Masque
    mask = Preprocessor.create_breast_mask(pixels)
    cv2.imwrite("breast_mask.png", mask)
    print("💾 Masque sauvegardé sous 'breast_mask.png'")

    # Image masquée
    masked = Preprocessor.apply_mask(image_rgb, mask)
    cv2.imwrite("breast_masked.png", masked)
    print("💾 Image masquée sauvegardée sous 'breast_masked.png'")

    # Comparaison
    comparison = np.hstack([image_rgb, masked])
    cv2.imwrite("breast_comparison.png", comparison)
    print("💾 Comparaison sauvegardée sous 'breast_comparison.png'")

    print("✅ Visualisation terminée.")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "app/data/dicom_samples/mammo.png"
    visualize_mask(path)