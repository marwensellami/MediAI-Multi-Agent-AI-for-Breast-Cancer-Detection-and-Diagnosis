# backend/test_single_image.py
import sys
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Imports directs
import importlib.util

spec_detection = importlib.util.spec_from_file_location(
    "detection", "app/agents/vision/detection.py"
)
detection_module = importlib.util.module_from_spec(spec_detection)
spec_detection.loader.exec_module(detection_module)
DetectionAgent = detection_module.DetectionAgent

spec_segmentation = importlib.util.spec_from_file_location(
    "segmentation", "app/agents/vision/segmentation.py"
)
segmentation_module = importlib.util.module_from_spec(spec_segmentation)
spec_segmentation.loader.exec_module(segmentation_module)
SegmentationAgent = segmentation_module.SegmentationAgent

spec_classification = importlib.util.spec_from_file_location(
    "classification", "app/agents/vision/classification.py"
)
classification_module = importlib.util.module_from_spec(spec_classification)
spec_classification.loader.exec_module(classification_module)
ClassificationAgent = classification_module.ClassificationAgent

from app.ml.processors.image_reader import ImageProcessor

DEFAULT_CONF = 0.3

def load_image(image_path: Path):
    data = ImageProcessor.read_file(image_path)
    if not data["is_valid"]:
        raise RuntimeError(f"Erreur lecture : {data.get('error')}")
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
    return image_rgb, data["metadata"]

def draw_results(image: np.ndarray, detections: List[Dict], output_path: Path):
    overlay = image.copy()
    for det in detections:
        bbox = det.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        label = det.get("malignancy_label", "?")
        prob = det.get("malignancy_score", 0.0)
        if label.lower() in ["malin", "maligne", "malignant"]:
            color = (0, 0, 255)
        elif label.lower() in ["bénin", "bénigne", "benign"]:
            color = (0, 255, 0)
        else:
            color = (255, 255, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {prob:.2f}"
        cv2.putText(overlay, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        mask = det.get("mask")
        if mask is not None:
            mask_bin = (mask > 0).astype(np.uint8)
            overlay[mask_bin > 0, :] = (overlay[mask_bin > 0, :] * 0.5 + np.array(color, dtype=np.uint8) * 0.5)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"💾 Image annotée sauvegardée sous {output_path}")

def print_results(detections: List[Dict]):
    if not detections:
        print("⚠️ Aucune lésion détectée.")
        return
    print(f"\n📊 {len(detections)} lésion(s) détectée(s) :")
    for i, det in enumerate(detections, 1):
        bbox = det.get("bbox", [0,0,0,0])
        label = det.get("malignancy_label", "?")
        prob = det.get("malignancy_score", 0.0)
        size = det.get("size_mm", "N/A")
        print(f"  {i}: label={label}, prob={prob:.3f}, taille={size} mm, bbox={bbox}")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device : {device}")

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Fichier introuvable : {image_path}")
        sys.exit(1)
    print(f"📂 Chargement de {image_path.name}...")
    try:
        image_rgb, metadata = load_image(image_path)
        print(f"📐 Dimensions : {image_rgb.shape}")
    except Exception as e:
        print(f"❌ Erreur de chargement : {e}")
        sys.exit(1)

    # Détection
    print("🔍 Détection YOLO...")
    detector = DetectionAgent(confidence_threshold=args.conf)
    detections = await detector.process(image_rgb)
    print(f"   → {len(detections)} détection(s) brute(s)")

    # Segmentation
    print("🔄 Segmentation SAM2...")
    #print("🔄 Masques approximatifs (boîtes) ")
    segmenter = SegmentationAgent()
    detections = await segmenter.process(image_rgb, detections)
    print(f"   → {len(detections)} détection(s) après segmentation")

    # ============================================================
    # AJOUTER L'IMAGE ORIGINALE POUR LA CLASSIFICATION
    # ============================================================
    for det in detections:
        det["_original_image"] = image_rgb

    # Classification
    print("🏷️ Classification Swin...")
    classifier = ClassificationAgent()
    detections = await classifier.process(detections)
    print(f"   → {len(detections)} détection(s) classifiée(s)")

    # Calcul des tailles
    pixel_spacing = metadata.get("pixel_spacing", [0.1, 0.1])
    spacing = pixel_spacing[0] if pixel_spacing else 0.1
    for det in detections:
        bbox = det.get("bbox")
        if bbox and len(bbox) == 4:
            width_px = bbox[2] - bbox[0]
            height_px = bbox[3] - bbox[1]
            size_mm = max(width_px, height_px) * spacing
            det["size_mm"] = round(size_mm, 1)
        else:
            det["size_mm"] = None

    print_results(detections)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.parent / f"{image_path.stem}_annotated.png"
    draw_results(image_rgb, detections, output_path)

    print("✅ Pipeline terminé.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())