# backend/test_breast_mask.py
import cv2
import numpy as np
from pathlib import Path

from app.ml.processors.image_reader import ImageProcessor
from app.ml.processors.preprocessing import Preprocessor

def test_breast_mask(image_path: str):
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Fichier non trouvé : {image_path}")
        return

    print(f"📂 Chargement de {image_path.name}...")

    # 1. Lecture de l'image
    data = ImageProcessor.read_file(image_path)
    if not data["is_valid"]:
        print(f"❌ Erreur : {data.get('error')}")
        return

    pixels = data["pixels"]
    # Convertir en RGB pour l'affichage
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

    # Sauvegarde de l'originale
    cv2.imwrite("mask_original.png", image_rgb)
    print("💾 Image originale sauvegardée sous 'mask_original.png'")

    # 2. Création du masque du sein (niveaux de gris)
    print("🔄 Création du masque du sein...")
    mask = Preprocessor.create_breast_mask(pixels)
    cv2.imwrite("breast_mask_binary.png", mask)
    print("💾 Masque binaire sauvegardé sous 'breast_mask_binary.png'")

    # 3. Application du masque sur l'image RGB
    print("🔄 Application du masque...")
    masked_image = Preprocessor.apply_mask(image_rgb, mask)
    cv2.imwrite("breast_masked.png", masked_image)
    print("💾 Image masquée sauvegardée sous 'breast_masked.png'")

    # 4. Superposition du masque sur l'original (en vert)
    overlay = image_rgb.copy()
    mask_colored = np.zeros_like(image_rgb)
    mask_colored[:, :, 1] = mask  # canal vert
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    cv2.imwrite("breast_mask_overlay.png", overlay)
    print("💾 Masque superposé (vert) sauvegardé sous 'breast_mask_overlay.png'")

    # 5. Comparaison côte à côte
    h, w = image_rgb.shape[:2]
    comparison = np.hstack([image_rgb, masked_image])
    cv2.imwrite("breast_comparison.png", comparison)
    print("💾 Comparaison (original | masquée) sauvegardée sous 'breast_comparison.png'")

    # 6. Statistiques du masque
    area_breast = np.sum(mask > 0)
    total_pixels = mask.size
    percent = (area_breast / total_pixels) * 100
    print(f"\n📊 Statistiques du masque :")
    print(f"   - Surface du sein : {area_breast} pixels")
    print(f"   - Total pixels : {total_pixels}")
    print(f"   - Pourcentage : {percent:.1f}% de l'image")

    print("\n✅ Test terminé !")
    print("📁 Fichiers générés :")
    print("   - mask_original.png (image originale)")
    print("   - breast_mask_binary.png (masque binaire blanc/noir)")
    print("   - breast_masked.png (image avec fond noir)")
    print("   - breast_mask_overlay.png (masque superposé en vert)")
    print("   - breast_comparison.png (original | masquée)")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else input("Entrez le chemin du fichier : ")
    test_breast_mask(path)