# backend/test_contrast_params.py
import asyncio
import cv2
import numpy as np
from pathlib import Path
from app.ml.processors.image_reader import ImageProcessor
from app.ml.detectors.contrast_detector import ContrastDetector

async def test_params(image_path, min_area, max_area, threshold_scale):
    data = ImageProcessor.read_file(image_path)
    if not data["is_valid"]:
        print("Erreur")
        return
    pixels = data["pixels"]
    if pixels.dtype != np.uint8:
        pixels = (pixels / pixels.max() * 255).astype(np.uint8)
    if len(pixels.shape) == 2:
        image = np.stack([pixels] * 3, axis=-1)
    else:
        image = pixels[:, :, :3]

    detector = ContrastDetector(min_area=min_area, max_area=max_area, threshold_scale=threshold_scale)
    detections = await detector.detect(image)

    # Dessiner les détections
    overlay = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(f"test_{min_area}_{threshold_scale}.png", overlay)
    print(f"Paramètres : min_area={min_area}, scale={threshold_scale} → {len(detections)} détections")
    return len(detections)

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "app/data/DMID_PNG/1024/TIFF/IMG014.png"
    # Testez différentes combinaisons
    asyncio.run(test_params(path, 100, 25000, 0.12))
    asyncio.run(test_params(path, 80, 30000, 0.10))
    asyncio.run(test_params(path, 150, 20000, 0.15))