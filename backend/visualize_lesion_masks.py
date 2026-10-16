# backend/visualize_lesion_masks.py
import asyncio
import cv2
import numpy as np
from pathlib import Path

from app.ml.loaders.vision_loader import VisionLoader
from app.ml.processors.image_reader import ImageProcessor

async def visualize_lesion_masks(image_path: str):
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

    cv2.imwrite("lesion_original.png", image_rgb)
    print("💾 Originale sauvegardée sous 'lesion_original.png'")

    # --- YOLO ---
    print("📥 Chargement de YOLO...")
    yolo = VisionLoader.load_yolo()
    print("✅ YOLO chargé.")

    print("🔍 Détection (seuil 0.1)...")
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
            })
    print(f"   → {len(detections)} détection(s) trouvée(s).")

    if not detections:
        print("⚠️ Aucune lésion détectée.")
        return

    # --- SAM2 ---
    print("📥 Chargement de SAM2...")
    sam = VisionLoader.load_sam2()
    if sam is None:
        print("❌ SAM2 non chargé.")
        return
    print("✅ SAM2 chargé.")

    print("🔄 Segmentation SAM2...")
    sam.set_image(image_rgb)
    input_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    masks, scores, _ = sam.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # Combinaison des masques
    combined_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
    for i in range(masks.shape[0]):
        mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
        mask_bin = (mask > 0).astype(np.uint8)
        combined_mask = cv2.bitwise_or(combined_mask, mask_bin)

    # Masque final : lésions en blanc, fond noir
    lesion_only = combined_mask * 255
    cv2.imwrite("lesion_mask.png", lesion_only)
    print("💾 Masque des lésions (blanc sur noir) sauvegardé sous 'lesion_mask.png'")

    # Superposition sur l'original (vert)
    overlay = image_rgb.copy()
    overlay[combined_mask > 0] = (0, 255, 0)
    cv2.imwrite("lesion_overlay.png", overlay)
    print("💾 Superposition sur original sauvegardée sous 'lesion_overlay.png'")

    print("✅ Visualisation terminée.")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else input("Entrez le chemin du fichier : ")
    asyncio.run(visualize_lesion_masks(path))