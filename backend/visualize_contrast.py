# backend/visualize_contrast.py
import asyncio
import cv2
import numpy as np
import argparse
from pathlib import Path

from app.ml.processors.image_reader import ImageProcessor
from app.ml.detectors.contrast_detector import ContrastDetector


async def visualize_contrast_detections(
    image_path: str,
    min_area: int = 200,
    max_area: int = 20000,
    threshold_scale: float = 0.3,
    output_path: str = None,
):
    """
    Charge une image, détecte les lésions avec le détecteur par contraste,
    et génère une image annotée avec des rectangles rouges autour des régions suspectes.

    Args:
        image_path: Chemin de l'image (PNG, JPG, DICOM)
        min_area: Surface minimale en pixels pour une région
        max_area: Surface maximale en pixels
        threshold_scale: Sensibilité du seuil (0.1 = plus de détections, 0.5 = moins)
        output_path: Nom du fichier de sortie (optionnel)
    """
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

    # Conversion en RGB uint8 pour l'affichage
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

    # Sauvegarde de l'originale (pour référence)
    cv2.imwrite("contrast_original.png", image_rgb)
    print("💾 Originale sauvegardée sous 'contrast_original.png'")

    # 2. Détection par contraste
    print(f"🔍 Détection par contraste (min_area={min_area}, max_area={max_area}, scale={threshold_scale})...")
    detector = ContrastDetector(
        min_area=min_area,
        max_area=max_area,
        threshold_scale=threshold_scale
    )
    detections = await detector.detect(image_rgb)

    print(f"   → {len(detections)} région(s) suspecte(s) détectée(s).")

    if not detections:
        print("⚠️ Aucune région suspecte détectée.")
        # Sauvegarder l'image sans annotation
        if output_path is None:
            output_path = f"{image_path.stem}_contrast_no_detection.png"
        cv2.imwrite(output_path, image_rgb)
        print(f"💾 Image sauvegardée sous : {output_path}")
        return

    # 3. Annoter l'image
    annotated = image_rgb.copy()
    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["bbox"]
        confidence = det.get("confidence", 0.5)

        # Rectangle rouge
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Étiquette avec numéro
        label = f"Région {i}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 255), -1)
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Petit cercle au centre
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

    # 4. Ajouter un résumé en bas de l'image
    h, w = annotated.shape[:2]
    summary = f"Régions suspectes : {len(detections)}"
    cv2.putText(annotated, summary, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 5. Sauvegarde
    if output_path is None:
        output_path = f"{image_path.stem}_contrast_detections.png"
    cv2.imwrite(output_path, annotated)
    print(f"💾 Image annotée sauvegardée sous : {output_path}")

    # 6. Résumé des détections
    print("\n📊 Résumé des régions détectées :")
    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["bbox"]
        width = x2 - x1
        height = y2 - y1
        print(f"   Région {i}: position=({x1},{y1}), taille={width}x{height} pixels")

    print(f"\n💡 Pour ajuster la détection, modifiez les paramètres :")
    print(f"   --min_area  (actuel: {min_area})")
    print(f"   --max_area  (actuel: {max_area})")
    print(f"   --scale     (actuel: {threshold_scale})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Détecte les régions suspectes par contraste sur une mammographie")
    parser.add_argument("image", help="Chemin de l'image (PNG, JPG, DICOM)")
    parser.add_argument("--min_area", type=int, default=200, help="Surface minimale en pixels (défaut: 200)")
    parser.add_argument("--max_area", type=int, default=20000, help="Surface maximale en pixels (défaut: 20000)")
    parser.add_argument("--scale", type=float, default=0.3, help="Sensibilité du seuil (0.1-0.5, défaut: 0.3)")
    parser.add_argument("--output", type=str, default=None, help="Nom du fichier de sortie")
    args = parser.parse_args()

    asyncio.run(visualize_contrast_detections(
        image_path=args.image,
        min_area=args.min_area,
        max_area=args.max_area,
        threshold_scale=args.scale,
        output_path=args.output,
    ))