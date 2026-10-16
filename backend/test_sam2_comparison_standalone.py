# backend/test_sam2_comparison_standalone.py
import asyncio
import cv2
import numpy as np
import sys
from pathlib import Path

from app.ml.processors.image_reader import ImageProcessor
from app.ml.loaders.vision_loader import VisionLoader

# ============================================================
# 1. FONCTION DE DÉTECTION YOLO
# ============================================================
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


# ============================================================
# 2. FONCTION DE SEGMENTATION PAR BOÎTES (fallback)
# ============================================================
def generate_box_masks(image, detections):
    """Génère des masques rectangulaires à partir des boîtes."""
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


# ============================================================
# 3. FONCTION DE SEGMENTATION SAM2
# ============================================================
async def segment_sam2(image_rgb, detections):
    """Segmentation précise avec SAM2."""
    sam = VisionLoader.load_sam2()
    if sam is None:
        print("❌ SAM2 non disponible, utilisation du fallback.")
        return generate_box_masks(image_rgb, detections), "fallback"

    input_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    sam.set_image(image_rgb)
    masks, scores, _ = sam.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    for i, det in enumerate(detections):
        if i < masks.shape[0]:
            mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
            det["mask"] = (mask > 0).astype(np.uint8) * 255
            score = scores[i] if scores is not None else 0.5
            if isinstance(score, np.ndarray):
                det["segmentation_score"] = float(score[0]) if score.size > 0 else 0.5
            else:
                det["segmentation_score"] = float(score)
        else:
            det["mask"] = None
            det["segmentation_score"] = 0.0

    return detections, "sam2"


# ============================================================
# 4. FONCTION DE CLASSIFICATION SWIN (simplifiée)
# ============================================================
async def classify_lesions(detections, image_rgb):
    """Classification simplifiée sans ClassificationAgent."""
    import torch
    from app.ml.loaders.vision_loader import VisionLoader
    
    # Charger Swin fine-tuné
    model = VisionLoader.load_swin_finetuned()
    if model is None:
        print("⚠️ Swin fine-tuné non disponible, classification factice.")
        for det in detections:
            det["malignancy_label"] = "Indéterminé"
            det["malignancy_score"] = 0.5
        return detections

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Transformations
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        for det in detections:
            mask = det.get("mask")
            bbox = det.get("bbox")
            
            # Extraire le patch
            if mask is not None and np.any(mask > 0):
                coords = np.where(mask > 0)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    pad = 10
                    y_min = max(0, y_min - pad)
                    y_max = min(image_rgb.shape[0], y_max + pad)
                    x_min = max(0, x_min - pad)
                    x_max = min(image_rgb.shape[1], x_max + pad)
                    patch = image_rgb[y_min:y_max, x_min:x_max]
                else:
                    continue
            elif bbox:
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                pad_x = int(0.1 * w)
                pad_y = int(0.1 * h)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(image_rgb.shape[1], x2 + pad_x)
                y2 = min(image_rgb.shape[0], y2 + pad_y)
                patch = image_rgb[y1:y2, x1:x2]
            else:
                continue

            if patch.size == 0:
                continue

            # Prétraiter et classer
            input_tensor = transform(patch).unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            
            # Classes: 0=Normal, 1=Bénin, 2=Malin
            pred_class = torch.argmax(probs, dim=1).item()
            class_names = ["Normal", "Bénin", "Malin"]
            det["malignancy_label"] = class_names[pred_class]
            det["malignancy_score"] = float(probs[0, 2])  # probabilité Malin

    return detections


# ============================================================
# 5. FONCTION PRINCIPALE DE COMPARAISON
# ============================================================
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

    # ============================================================
    # DÉTECTION YOLO
    # ============================================================
    print("🔍 Détection YOLO...")
    detections = await detect_yolo(image_rgb, confidence_threshold=0.3)
    print(f"   → {len(detections)} détection(s) brute(s)")

    if not detections:
        print("⚠️ Aucune lésion détectée. Arrêt.")
        return

    # Copier les détections pour les deux méthodes
    detections_box = [d.copy() for d in detections]
    detections_sam = [d.copy() for d in detections]

    # ============================================================
    # MÉTHODE 1 : MASQUES APPROXIMATIFS (BOÎTES)
    # ============================================================
    print("\n🟦 MÉTHODE 1 : Masques approximatifs (boîtes)")
    detections_box = generate_box_masks(image_rgb, detections_box)
    detections_box = await classify_lesions(detections_box, image_rgb)
    print(f"   → {len(detections_box)} détection(s) classifiée(s)")

    # ============================================================
    # MÉTHODE 2 : SAM2 (MASQUES PRÉCIS)
    # ============================================================
    print("\n🟩 MÉTHODE 2 : SAM2 (masques précis)")
    detections_sam, method = await segment_sam2(image_rgb, detections_sam)
    detections_sam = await classify_lesions(detections_sam, image_rgb)
    print(f"   → {len(detections_sam)} détection(s) classifiée(s)")

    # ============================================================
    # AFFICHAGE DES RÉSULTATS
    # ============================================================
    print("\n" + "="*60)
    print("📊 RÉSULTATS COMPARATIFS")
    print("="*60)

    print("\n🟦 MASQUES APPROXIMATIFS (boîtes) :")
    for i, det in enumerate(detections_box, 1):
        label = det.get("malignancy_label", "?")
        score = det.get("malignancy_score", 0)
        bbox = det.get("bbox", [0,0,0,0])
        mask_area = np.sum(det.get("mask", np.zeros((1,1))) > 0) if det.get("mask") is not None else 0
        print(f"  Lésion {i}: label={label}, score={score:.3f}, bbox={bbox}, surface={mask_area} px²")

    print("\n🟩 SAM2 (masques précis) :")
    for i, det in enumerate(detections_sam, 1):
        label = det.get("malignancy_label", "?")
        score = det.get("malignancy_score", 0)
        bbox = det.get("bbox", [0,0,0,0])
        mask_area = np.sum(det.get("mask", np.zeros((1,1))) > 0) if det.get("mask") is not None else 0
        print(f"  Lésion {i}: label={label}, score={score:.3f}, bbox={bbox}, surface={mask_area} px²")

    # ============================================================
    # GÉNÉRATION DES IMAGES COMPARATIVES
    # ============================================================
    print("\n🖼️ Génération des images...")

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

    img_box = draw_segmentation(image_rgb, detections_box, "Boîtes (fallback)", (0, 0, 255))
    img_sam = draw_segmentation(image_rgb, detections_sam, "SAM2 (précis)", (0, 255, 0))

    # Comparaison côte à côte
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

    # Masques individuels
    for i, det in enumerate(detections_box):
        if det.get("mask") is not None:
            cv2.imwrite(f"mask_box_{i+1}.png", det["mask"])
    for i, det in enumerate(detections_sam):
        if det.get("mask") is not None:
            cv2.imwrite(f"mask_sam_{i+1}.png", det["mask"])

    print("💾 Masques individuels sauvegardés (mask_box_*.png, mask_sam_*.png)")

    # ============================================================
    # STATISTIQUES
    # ============================================================
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