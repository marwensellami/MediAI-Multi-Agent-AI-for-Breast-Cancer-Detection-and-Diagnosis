# backend/test_sam2_standalone.py
import asyncio
import cv2
import numpy as np
import sys
from pathlib import Path

from app.ml.processors.image_reader import ImageProcessor
from app.ml.loaders.vision_loader import VisionLoader

async def detect_yolo(image_rgb, confidence_threshold=0.3):
    from ultralytics import YOLO
    yolo = VisionLoader.load_yolo_finetuned()
    if yolo is None:
        yolo = VisionLoader.load_yolo()
    
    results = yolo(image_rgb, verbose=False, conf=confidence_threshold)
    detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_name": yolo.names[int(box.cls[0])],
            })
    return detections

async def test_sam2_direct(image_path: str):
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

    print("📥 Chargement de SAM2...")
    sam = VisionLoader.load_sam2()
    if sam is None:
        print("❌ SAM2 n'a pas pu être chargé.")
        return
    print("✅ SAM2 chargé avec succès.")

    print("🔍 Détection YOLO...")
    detections = await detect_yolo(image_rgb, confidence_threshold=0.3)
    print(f"   → {len(detections)} détection(s) brute(s)")

    if not detections:
        print("⚠️ Aucune lésion détectée. Arrêt.")
        return

    print("🔄 Segmentation SAM2...")
    input_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    sam.set_image(image_rgb)
    masks, scores, _ = sam.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    print(f"   → {masks.shape[0]} masque(s) généré(s)")

    print("\n📊 RÉSULTATS :")
    for i, det in enumerate(detections):
        bbox = det["bbox"]
        print(f"  Lésion {i+1}: bbox={bbox}")
        print(f"    - Confiance YOLO: {det['confidence']:.3f}")
        if i < len(masks):
            mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
            score = scores[i] if scores is not None else 0.5
            # ============================================================
            # CORRECTION : gérer le cas où score est un tableau NumPy
            # ============================================================
            if isinstance(score, np.ndarray):
                score_val = float(score[0]) if score.size > 0 else 0.0
            else:
                score_val = float(score)
            area = np.sum(mask > 0)
            print(f"    - Score SAM2: {score_val:.3f}")
            print(f"    - Surface du masque: {area} px²")

    print("\n🖼️ Sauvegarde des images...")
    overlay = image_rgb.copy()
    for i in range(masks.shape[0]):
        mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
        mask_bin = (mask > 0).astype(np.uint8)
        overlay[mask_bin > 0, :] = (overlay[mask_bin > 0, :] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5)

        bbox = detections[i]["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(overlay, f"YOLO: {detections[i]['confidence']:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imwrite("sam2_result.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("💾 Image avec masques sauvegardée sous 'sam2_result.png'")

    for i in range(masks.shape[0]):
        mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
        mask_bin = (mask * 255).astype(np.uint8)
        cv2.imwrite(f"sam2_mask_{i+1}.png", mask_bin)
        print(f"💾 Masque {i+1} sauvegardé sous 'sam2_mask_{i+1}.png'")

    print("\n✅ Test SAM2 terminé.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else input("Entrez le chemin du fichier : ")
    asyncio.run(test_sam2_direct(path))