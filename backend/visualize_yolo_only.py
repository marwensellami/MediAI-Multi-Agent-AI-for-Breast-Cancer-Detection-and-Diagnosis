# backend/visualize_yolo_only.py
import asyncio
import cv2
import numpy as np
from pathlib import Path

from app.ml.loaders.vision_loader import VisionLoader
from app.ml.processors.image_reader import ImageProcessor

async def visualize_yolo_only(image_path: str):
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
    cv2.imwrite("yolo_original.png", image_rgb)
    print("💾 Originale sauvegardée sous 'yolo_original.png'")

    # 2. Chargement de YOLO
    print("📥 Chargement de YOLO...")
    yolo = VisionLoader.load_yolo()
    print("✅ YOLO chargé.")

    # 3. Détection YOLO avec différents seuils de confiance
    thresholds = [0.1, 0.3, 0.5, 0.7]
    for threshold in thresholds:
        print(f"\n🔍 Détection avec seuil {threshold}...")
        results = yolo(image_rgb, verbose=False, conf=threshold)
        detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": int(box.cls[0]),
                    "class_name": yolo.names[int(box.cls[0])],
                })
        print(f"   → {len(detections)} détection(s) trouvée(s).")

        # Copie pour dessiner les boîtes
        img_copy = image_rgb.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            class_name = det["class_name"]
            # Dessiner la boîte
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Afficher la classe et le score
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img_copy, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Sauvegarder l'image avec les boîtes
        filename = f"yolo_threshold_{int(threshold*10)}.png"
        cv2.imwrite(filename, img_copy)
        print(f"💾 Image avec boîtes sauvegardée sous '{filename}'")

    # 4. Image avec toutes les détections (seuil bas 0.1)
    print("\n📦 Détections finales (seuil 0.1) :")
    results = yolo(image_rgb, verbose=False, conf=0.1)
    detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": int(box.cls[0]),
                "class_name": yolo.names[int(box.cls[0])],
            })
    print(f"   → {len(detections)} détection(s) au total.")

    # 5. Résumé des classes détectées
    classes = {}
    for det in detections:
        class_name = det["class_name"]
        if class_name not in classes:
            classes[class_name] = 0
        classes[class_name] += 1
    print("\n📊 Classes détectées :")
    for class_name, count in classes.items():
        print(f"   - {class_name}: {count}")

    print("\n✅ Visualisation YOLO terminée.")
    print("📁 Fichiers générés :")
    print("   - yolo_original.png (image originale)")
    print("   - yolo_threshold_1.png (seuil 0.1)")
    print("   - yolo_threshold_3.png (seuil 0.3)")
    print("   - yolo_threshold_5.png (seuil 0.5)")
    print("   - yolo_threshold_7.png (seuil 0.7)")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else input("Entrez le chemin du fichier : ")
    asyncio.run(visualize_yolo_only(path))