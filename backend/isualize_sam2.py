# backend/visualize_sam2.py
import asyncio
import cv2
import numpy as np
from pathlib import Path

from app.ml.loaders.vision_loader import VisionLoader
from app.ml.processors.image_reader import ImageProcessor
from app.agents.vision.segmentation import SegmentationAgent

async def visualize_segmentation(image_path: str):
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
        pixels = (pixels / pixels.max() * 255).astype(np.uint8)
    if len(pixels.shape) == 2:
        image_rgb = np.stack([pixels] * 3, axis=-1)
    elif pixels.shape[2] == 1:
        image_rgb = np.concatenate([pixels] * 3, axis=-1)
    else:
        image_rgb = pixels[:, :, :3]

    # 2. Chargement de YOLO
    print("📥 Chargement de YOLO...")
    yolo = VisionLoader.load_yolo()
    print("✅ YOLO chargé.")

    # 3. Détection YOLO
    print("🔍 Détection...")
    results = yolo(image_rgb, verbose=False, conf=0.3)
    detections = []
    boxes = results[0].boxes if results and len(results) > 0 else None
    if boxes is not None:
        for box in boxes:
            conf = float(box.conf[0])
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": int(box.cls[0]),
                "class_name": yolo.names[int(box.cls[0])],
            })
    print(f"   → {len(detections)} détection(s) trouvée(s).")

    if not detections:
        print("⚠️ Aucune lésion détectée.")
        return

    # 4. Chargement de SAM2
    print("📥 Chargement de SAM2...")
    sam = VisionLoader.load_sam2()
    if sam is None:
        print("❌ SAM2 non chargé.")
        return
    print("✅ SAM2 chargé.")

    # 5. Segmentation avec SAM2
    print("🔄 Segmentation...")
    seg_agent = SegmentationAgent()  # utilise le registre, mais on peut aussi utiliser sam directement
    # On va utiliser le prédicteur directement pour éviter le fallback
    detections = await seg_agent.process(image_rgb, detections)  # fonctionne si sam est dans le registre
    # Mais pour être sûr, on peut utiliser sam directement :
    # sam.set_image(image_rgb)
    # input_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    # masks, scores, _ = sam.predict(box=input_boxes, multimask_output=False)
    # for i, det in enumerate(detections):
    #     if i < len(masks):
    #         mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
    #         det["mask"] = (mask > 0).astype(np.uint8) * 255
    #         det["segmentation_score"] = float(scores[i]) if scores is not None else 0.5

    # 6. Superposition des masques
    overlay = image_rgb.copy()
    for det in detections:
        mask = det.get("mask")
        if mask is not None:
            mask_bin = (mask > 0).astype(np.uint8)
            overlay[mask_bin > 0] = (0, 255, 0)  # vert

        bbox = det.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            conf = det.get("confidence", 0)
            cv2.putText(overlay, f"YOLO: {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            seg_score = det.get("segmentation_score")
            if seg_score:
                cv2.putText(overlay, f"SAM2: {seg_score:.2f}", (x1, y1-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    cv2.imwrite("segmented_overlay.png", overlay)
    print("💾 Image avec masques sauvegardée sous 'segmented_overlay.png'")

    # Fusion transparente
    alpha = 0.5
    blended = cv2.addWeighted(image_rgb, 1-alpha, overlay, alpha, 0)
    cv2.imwrite("segmented_blended.png", blended)
    print("💾 Image fusionnée sauvegardée sous 'segmented_blended.png'")

    print("✅ Visualisation terminée.")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else input("Entrez le chemin du fichier : ")
    asyncio.run(visualize_segmentation(path))