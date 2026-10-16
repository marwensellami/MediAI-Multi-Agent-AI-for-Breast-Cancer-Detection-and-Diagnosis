# backend/test_sam2_with_comparison.py
import asyncio
import cv2
import numpy as np
import sys
from pathlib import Path

from app.ml.processors.image_reader import ImageProcessor
from app.ml.loaders.vision_loader import VisionLoader

# Import des agents pour la classification (et éventuellement la détection)
import importlib.util

def load_agent(module_name, class_name):
    """Charge une classe depuis un module sans passer par __init__ (évite les imports circulaires)."""
    spec = importlib.util.spec_from_file_location(
        module_name,
        f"app/agents/vision/{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

ClassificationAgent = load_agent("classification", "ClassificationAgent")

# ------------------- Détection YOLO directe (synchrone) -------------------
def detect_yolo(image_rgb, confidence_threshold=0.3):
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

# ------------------- Masque approximatif à partir d'une boîte -------------------
def create_box_mask(bbox, shape):
    """Crée un masque binaire (0/1) avec 1 à l'intérieur du rectangle."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = map(int, bbox)
    mask[y1:y2, x1:x2] = 1
    return mask

# ------------------- Fonction principale de comparaison -------------------
async def compare_segmentation(image_path: str):
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

    print("🔍 Détection YOLO...")
    detections = detect_yolo(image_rgb, confidence_threshold=0.3)
    print(f"   → {len(detections)} détection(s) brute(s)")

    if not detections:
        print("⚠️ Aucune lésion détectée. Arrêt.")
        return

    # ------------------- 1. Masques approximatifs (boîtes) -------------------
    print("\n🟦 MASQUES APPROXIMATIFS (boîtes) :")
    detections_box = []
    for det in detections:
        det_copy = det.copy()
        det_copy["mask"] = create_box_mask(det["bbox"], image_rgb.shape)
        detections_box.append(det_copy)
    print(f"   → {len(detections_box)} masques boîtes générés")

    # ------------------- 2. Masques SAM2 (précis) -------------------
    print("\n🟩 MASQUES SAM2 :")
    sam = VisionLoader.load_sam2()
    if sam is None:
        print("⚠️ SAM2 non disponible → fallback sur les boîtes pour SAM2 aussi.")
        detections_sam = [d.copy() for d in detections_box]  # copie des masques boîtes
    else:
        sam.set_image(image_rgb)
        input_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        masks, scores, _ = sam.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        detections_sam = []
        for i, det in enumerate(detections):
            det_copy = det.copy()
            if i < masks.shape[0]:
                mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
                # seuillage à 0.5 pour obtenir un masque binaire
                det_copy["mask"] = (mask > 0.5).astype(np.uint8)
                # score SAM2 (gestion robuste)
                score_val = np.asarray(scores[i]).item() if scores is not None else 0.5
                det_copy["sam_score"] = score_val
            else:
                # fallback si moins de masques que de détections
                det_copy["mask"] = create_box_mask(det["bbox"], image_rgb.shape)
                det_copy["sam_score"] = 0.0
            detections_sam.append(det_copy)
        print(f"   → {len(detections_sam)} masques SAM2 générés")

    # ------------------- 3. Classification Swin (sur les deux jeux) -------------------
    print("\n🏷️ Classification Swin...")
    classifier = ClassificationAgent()
    detections_box = await classifier.process(detections_box)
    detections_sam = await classifier.process(detections_sam)

    # ------------------- 4. Affichage des résultats -------------------
    print("\n" + "="*60)
    print("📊 RÉSULTATS COMPARATIFS")
    print("="*60)

    print("\n🟦 MASQUES APPROXIMATIFS (boîtes) :")
    for i, det in enumerate(detections_box, 1):
        label = det.get("malignancy_label", "?")
        score = det.get("malignancy_score", 0.0)
        bbox = det.get("bbox", [0,0,0,0])
        area = np.sum(det.get("mask", np.zeros((1,1))) > 0)
        print(f"  Lésion {i}: label={label}, score={score:.3f}, bbox={bbox}, surface={area} px²")

    print("\n🟩 MASQUES SAM2 :")
    for i, det in enumerate(detections_sam, 1):
        label = det.get("malignancy_label", "?")
        score = det.get("malignancy_score", 0.0)
        bbox = det.get("bbox", [0,0,0,0])
        sam_score = det.get("sam_score", 0.0)
        area = np.sum(det.get("mask", np.zeros((1,1))) > 0)
        print(f"  Lésion {i}: label={label}, score={score:.3f}, bbox={bbox}, SAM2={sam_score:.3f}, surface={area} px²")

    # ------------------- 5. Génération des images comparatives -------------------
    print("\n🖼️ Génération des images comparatives...")

    def draw_segmentation(image, detections, title, color):
        overlay = image.copy()
        for det in detections:
            mask = det.get("mask")
            if mask is not None:
                mask_bin = (mask > 0).astype(np.uint8)
                overlay[mask_bin > 0, :] = (overlay[mask_bin > 0, :] * 0.5 + np.array(color, dtype=np.uint8) * 0.5)
            bbox = det.get("bbox")
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                label = det.get("malignancy_label", "?")
                prob = det.get("malignancy_score", 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {prob:.2f}"
                cv2.putText(overlay, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(overlay, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return overlay

    img_box = draw_segmentation(image_rgb, detections_box, "Masques approximatifs (boîtes)", (0, 0, 255))
    img_sam = draw_segmentation(image_rgb, detections_sam, "SAM2 (précis)", (0, 255, 0))

    # Assemblage côte à côte
    h, w = image_rgb.shape[:2]
    if w > 1200:
        scale = 1200 / (2 * w)
        new_w = int(2 * w * scale)
        new_h = int(h * scale)
        img_box_resized = cv2.resize(img_box, (new_w // 2, new_h))
        img_sam_resized = cv2.resize(img_sam, (new_w // 2, new_h))
        comparison = np.hstack([img_box_resized, img_sam_resized])
    else:
        comparison = np.hstack([img_box, img_sam])

    cv2.imwrite("segmentation_comparison.png", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print("💾 Comparaison sauvegardée sous 'segmentation_comparison.png'")

    # Sauvegarde des masques individuels
    for i, det in enumerate(detections_box):
        if det.get("mask") is not None:
            cv2.imwrite(f"mask_box_{i+1}.png", det["mask"] * 255)
    for i, det in enumerate(detections_sam):
        if det.get("mask") is not None:
            cv2.imwrite(f"mask_sam_{i+1}.png", det["mask"] * 255)
    print("💾 Masques individuels sauvegardés (mask_box_*.png, mask_sam_*.png)")

    # ------------------- 6. Statistiques globales -------------------
    print("\n📊 STATISTIQUES :")
    print(f"  Nombre de lésions (boîtes)  : {len(detections_box)}")
    print(f"  Nombre de lésions (SAM2)    : {len(detections_sam)}")
    areas_box = [np.sum(det.get("mask", np.zeros((1,1))) > 0) for det in detections_box if det.get("mask") is not None]
    areas_sam = [np.sum(det.get("mask", np.zeros((1,1))) > 0) for det in detections_sam if det.get("mask") is not None]
    if areas_box:
        print(f"  Surface moyenne (boîtes)    : {np.mean(areas_box):.0f} px²")
    if areas_sam:
        print(f"  Surface moyenne (SAM2)      : {np.mean(areas_sam):.0f} px²")

    print("\n✅ Comparaison terminée.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else input("Entrez le chemin du fichier : ")
    asyncio.run(compare_segmentation(path))