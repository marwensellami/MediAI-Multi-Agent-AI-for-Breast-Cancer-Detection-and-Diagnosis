# visualize_masks.py
import asyncio
import sys
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import torch
from app.ml.processors.image_reader import ImageProcessor
from app.ml.loaders.vision_loader import VisionLoader
from app.ml.model_registry import ModelRegistry

async def visualize_masks(image_path: str):
    # 1. Charger l'image
    image_data = ImageProcessor.read_file(Path(image_path))
    pixels = image_data["pixels"]
    image_yolo = ImageProcessor.prepare_for_yolo(pixels)  # RGB uint8 (1024,1024,3)

    # 2. Charger YOLO fine-tuné
    registry = ModelRegistry()
    yolo_model = registry.get("yolo_finetuned")
    if yolo_model is None:
        yolo_model = VisionLoader.load_yolo_finetuned()
        if yolo_model is None:
            print("❌ YOLO fine-tuné non trouvé")
            return
        registry.register("yolo_finetuned", yolo_model)
    print("✅ YOLO chargé")

    # 3. Détection YOLO
    results = yolo_model(image_yolo, verbose=False, conf=0.29)
    detections = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(image_yolo.shape[1], x2); y2 = min(image_yolo.shape[0], y2)
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                })
    if not detections:
        print("❌ Aucune détection")
        return
    print(f"✅ {len(detections)} détections")

    # 4. Définir les deux méthodes de masque
    def old_rectangle_mask(image, bbox):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        pad = 5
        x1_p = max(0, x1 - pad)
        y1_p = max(0, y1 - pad)
        x2_p = min(w, x2 + pad)
        y2_p = min(h, y2 + pad)
        mask[y1_p:y2_p, x1_p:x2_p] = 255
        return mask

    def new_ellipse_mask(image, bbox):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return mask
        pad = int(max(width, height) * 0.10)
        pad = max(1, pad)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        axes_x = (width // 2) + pad
        axes_y = (height // 2) + pad
        axes_x = min(axes_x, center_x, w - center_x)
        axes_y = min(axes_y, center_y, h - center_y)
        if axes_x > 0 and axes_y > 0:
            cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
        else:
            # Fallback rectangle
            x1_p = max(0, x1 - pad)
            y1_p = max(0, y1 - pad)
            x2_p = min(w, x2 + pad)
            y2_p = min(h, y2 + pad)
            mask[y1_p:y2_p, x1_p:x2_p] = 255
        return mask

    # 5. Créer une image pour la visualisation
    img_old = image_yolo.copy()
    img_new = image_yolo.copy()

    # Couleurs (BGR)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

    for i, det in enumerate(detections):
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox

        # Ancienne méthode (rectangle)
        mask_old = old_rectangle_mask(image_yolo, bbox)
        contours_old, _ = cv2.findContours(mask_old, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_old, contours_old, -1, RED, 2)
        cv2.rectangle(img_old, (x1, y1), (x2, y2), BLUE, 1)  # Boîte YOLO en bleu

        # Nouvelle méthode (ellipse)
        mask_new = new_ellipse_mask(image_yolo, bbox)
        contours_new, _ = cv2.findContours(mask_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_new, contours_new, -1, GREEN, 2)
        cv2.rectangle(img_new, (x1, y1), (x2, y2), BLUE, 1)  # Boîte YOLO en bleu

        # Ajouter un texte pour identifier la lésion
        cv2.putText(img_old, f"L{i+1} (rectangle)", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
        cv2.putText(img_new, f"L{i+1} (ellipse)", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

    # 6. Afficher et sauvegarder les images
    # Redimensionner pour l'affichage (si trop grande)
    def resize_for_display(img, max_height=800):
        h, w = img.shape[:2]
        if h > max_height:
            ratio = max_height / h
            new_w = int(w * ratio)
            return cv2.resize(img, (new_w, max_height))
        return img

    img_old_resized = resize_for_display(img_old)
    img_new_resized = resize_for_display(img_new)

    cv2.imshow("Ancienne methode (rectangle)", img_old_resized)
    cv2.imshow("Nouvelle methode (ellipse)", img_new_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Sauvegarde
    output_dir = Path("app/data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "mask_rectangle.jpg"), img_old)
    cv2.imwrite(str(output_dir / "mask_ellipse.jpg"), img_new)
    print(f"✅ Images sauvegardées dans {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_masks.py <image_path>")
        sys.exit(1)
    asyncio.run(visualize_masks(sys.argv[1]))