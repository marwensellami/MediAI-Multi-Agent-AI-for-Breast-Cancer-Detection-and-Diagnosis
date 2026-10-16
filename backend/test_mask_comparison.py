# test_mask_comparison_direct.py
import asyncio
import sys
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from app.ml.processors.image_reader import ImageProcessor
from app.ml.models import SwinClassifier
from app.ml.loaders.vision_loader import VisionLoader
from app.ml.model_registry import ModelRegistry

async def test_masks(image_path: str):
    # 1. Charger l'image
    image_data = ImageProcessor.read_file(Path(image_path))
    pixels = image_data["pixels"]
    image_yolo = ImageProcessor.prepare_for_yolo(pixels)
    original_shape = pixels.shape[:2]

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
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(image_yolo.shape[1], x2); y2 = min(image_yolo.shape[0], y2)
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": yolo_model.names[cls_id],
                    "_original_image": image_yolo,
                    "original_image_shape": original_shape,
                })
    if not detections:
        print("❌ Aucune détection")
        return
    print(f"✅ {len(detections)} détections")

    # 4. Charger Swin fine-tuné
    swin_model = registry.get("swin_finetuned")
    if swin_model is None:
        swin_model = VisionLoader.load_swin_finetuned()
        if swin_model is None:
            print("❌ Swin fine-tuné non trouvé")
            return
        registry.register("swin_finetuned", swin_model)
    print("✅ Swin chargé")

    # 5. Classification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    swin_model = swin_model.to(device).eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            patch = image_yolo[y1:y2, x1:x2]
            if patch.size == 0:
                det["malignancy_label"] = "Inconnu"
                det["malignancy_score"] = 0.5
                continue
            patch_rgb = patch
            if patch_rgb.shape[2] == 1:
                patch_rgb = np.concatenate([patch_rgb]*3, axis=2)
            elif patch_rgb.shape[2] > 3:
                patch_rgb = patch_rgb[:,:,:3]
            tensor = transform(patch_rgb).unsqueeze(0).to(device)
            logits = swin_model(tensor)
            probs = torch.softmax(logits, dim=1)
            prob_normal, prob_benign, prob_malignant = probs[0].tolist()
            pred_class = torch.argmax(probs, dim=1).item()
            class_map = {0: "Normal", 1: "Bénin", 2: "Malin"}
            det["malignancy_score"] = prob_malignant
            det["malignancy_label"] = class_map.get(pred_class, "Inconnu")
            if det["malignancy_label"] == "Normal":
                if prob_benign >= prob_malignant:
                    det["malignancy_label"] = "Bénin"
                else:
                    det["malignancy_label"] = "Malin"
            print(f"Lésion: label={det['malignancy_label']}, score={det['malignancy_score']:.3f}")

    # 6. Définir les deux méthodes de masquage
    def old_generate_box_masks(image, detections):
        h, w = image.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            mask = np.zeros((h, w), dtype=np.uint8)
            pad = 5
            x1_p = max(0, x1 - pad)
            y1_p = max(0, y1 - pad)
            x2_p = min(w, x2 + pad)
            y2_p = min(h, y2 + pad)
            mask[y1_p:y2_p, x1_p:x2_p] = 255
            det["mask"] = mask
            det["segmentation_score"] = det.get("confidence", 0.5)
        return detections

    def new_generate_box_masks(image, detections):
        h, w = image.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            mask = np.zeros((h, w), dtype=np.uint8)
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                det["mask"] = mask
                det["segmentation_score"] = det.get("confidence", 0.5)
                continue
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
                x1_p = max(0, x1 - pad)
                y1_p = max(0, y1 - pad)
                x2_p = min(w, x2 + pad)
                y2_p = min(h, y2 + pad)
                mask[y1_p:y2_p, x1_p:x2_p] = 255
            det["mask"] = mask
            det["segmentation_score"] = det.get("confidence", 0.5)
        return detections

    # Appliquer les deux méthodes
    detections_old = [d.copy() for d in detections]
    detections_new = [d.copy() for d in detections]
    detections_old = old_generate_box_masks(image_yolo, detections_old)
    detections_new = new_generate_box_masks(image_yolo, detections_new)

    # Calcul des métriques
    def compute_metrics(detections):
        results = []
        for det in detections:
            mask = det.get("mask")
            if mask is None:
                results.append((None, None, None))
                continue
            mask_bin = (mask > 0).astype(np.uint8)
            area_px = np.sum(mask_bin)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perimeter_px = cv2.arcLength(contours[0], True)
                circ = 4 * np.pi * area_px / (perimeter_px ** 2) if perimeter_px > 0 else 0
                results.append((area_px, perimeter_px, circ))
            else:
                results.append((area_px, 0, 0))
        return results

    print("\n🔍 Comparaison des métriques (pixels)")
    print("=" * 70)
    print(f"{'Lésion':<8} {'Méthode':<12} {'Aire (px)':<12} {'Périmètre (px)':<15} {'Circularité':<12}")
    print("-" * 70)

    for i in range(len(detections)):
        old_metrics = compute_metrics(detections_old)[i]
        new_metrics = compute_metrics(detections_new)[i]
        print(f"Lésion {i+1:<4} Ancienne   {old_metrics[0] if old_metrics[0] is not None else 'N/A':<12} {old_metrics[1] if old_metrics[1] is not None else 'N/A':<15} {old_metrics[2] if old_metrics[2] is not None else 'N/A':<12.4f}")
        print(f"          Nouvelle   {new_metrics[0] if new_metrics[0] is not None else 'N/A':<12} {new_metrics[1] if new_metrics[1] is not None else 'N/A':<15} {new_metrics[2] if new_metrics[2] is not None else 'N/A':<12.4f}")
        print("-" * 70)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_mask_comparison_direct.py <image_path>")
        sys.exit(1)
    asyncio.run(test_masks(sys.argv[1]))