# backend/visualize_detections.py
import asyncio
import cv2
import numpy as np
import argparse
from pathlib import Path

from app.ml.loaders.vision_loader import VisionLoader
from app.ml.processors.image_reader import ImageProcessor

async def visualize_detections(
    image_path: str,
    yolo_conf: float = 0.1,
    output_path: str = None,
):
    """
    Charge une image, détecte les lésions avec YOLO, et génère une image annotée
    avec des rectangles rouges autour des détections.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Fichier non trouvé : {image_path}")
        return

    print(f"📂 Chargement de {image_path.name}...")
    data = ImageProcessor.read_file(image_path)
    if not data["is_valid"]:
        print(f"❌ Erreur : {data.get('error')}")
        return

    pixels = data["pixels"]
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

    # Charger YOLO
    print(f"📥 Chargement de YOLO...")
    yolo = VisionLoader.load_yolo()
    print("✅ YOLO chargé.")

    # Détection
    print(f"🔍 Détection avec seuil {yolo_conf}...")
    results = yolo(image_rgb, verbose=False, conf=yolo_conf)
    detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_name": yolo.names[int(box.cls[0])],
            })

    print(f"   → {len(detections)} détection(s) trouvée(s).")

    if not detections:
        print("⚠️ Aucune détection. Aucune annotation à afficher.")
        return

    # Annoter l'image
    annotated = image_rgb.copy()
    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        class_name = det["class_name"]

        # Rectangle rouge
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Étiquette avec numéro et score
        label = f"Lésion {i}: {class_name} ({conf:.2f})"
        # Fond pour le texte
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 255), -1)
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Petit cercle au centre (optionnel)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

    # Sauvegarde
    if output_path is None:
        base_name = image_path.stem
        output_path = f"{base_name}_detections_annotated.png"
    cv2.imwrite(output_path, annotated)
    print(f"💾 Image annotée sauvegardée sous : {output_path}")

    # Résumé des détections
    print(f"📊 Résumé des détections :")
    for i, det in enumerate(detections, start=1):
        print(f"   {i}: conf={det['confidence']:.3f}, classe={det['class_name']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise les détections YOLO sur une image")
    parser.add_argument("image", help="Chemin de l'image")
    parser.add_argument("--conf", type=float, default=0.1, help="Seuil de confiance YOLO (0.0 - 1.0)")
    parser.add_argument("--output", type=str, default=None, help="Nom du fichier de sortie")
    args = parser.parse_args()

    asyncio.run(visualize_detections(
        image_path=args.image,
        yolo_conf=args.conf,
        output_path=args.output,
    ))