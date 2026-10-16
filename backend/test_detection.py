# backend/test_detection.py
import asyncio
import cv2
import numpy as np
import sys
from pathlib import Path

from app.ml.processors.image_reader import ImageProcessor
from app.ml.loaders.vision_loader import VisionLoader

# Import des agents
import importlib.util

def load_agent(module_name, class_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        f"app/agents/vision/{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

DetectionAgent = load_agent("detection", "DetectionAgent")
SegmentationAgent = load_agent("segmentation", "SegmentationAgent")
ClassificationAgent = load_agent("classification", "ClassificationAgent")

def normalize_label(label):
    """Normalise le label pour comparaison insensible à la casse et aux accents"""
    if not label:
        return ""
    label_lower = label.lower()
    # Gérer les accents
    if 'é' in label_lower:
        label_lower = label_lower.replace('é', 'e')
    if 'è' in label_lower:
        label_lower = label_lower.replace('è', 'e')
    return label_lower

async def test_boxes(image_path: str):
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
    metadata = data["metadata"]
    pixel_spacing = metadata.get("pixel_spacing", [0.1, 0.1])
    spacing = pixel_spacing[0] if pixel_spacing else 0.1

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

    print(f"📏 Pixel spacing : {spacing} mm/pixel")

    # 1. Détection YOLO
    print("🔍 Détection YOLO...")
    detector = DetectionAgent(confidence_threshold=0.3)
    detections = await detector.process(image_rgb)
    print(f"   → {len(detections)} détection(s) brute(s)")

    if not detections:
        print("⚠️ Aucune lésion détectée. Arrêt.")
        return

    for det in detections:
        det["_original_image"] = image_rgb

    # 2. Segmentation par boîtes (fallback)
    print("\n🟦 Segmentation par masques approximatifs (boîtes)")
    seg = SegmentationAgent()
    seg.sam_predictor = None  # force le fallback
    detections_box = await seg.process(image_rgb, detections.copy())
    print(f"   → {len(detections_box)} masques boîtes générés")

    # 3. Classification Swin
    print("\n🏷️ Classification Swin...")
    classifier = ClassificationAgent()
    detections_box = await classifier.process(detections_box)

    # ============================================================
    # 4. CALCUL DE LA TAILLE EN MM (CORRECTION AJOUTÉE)
    # ============================================================
    print("\n📏 Calcul des tailles...")
    for det in detections_box:
        bbox = det.get("bbox")
        if bbox and len(bbox) == 4:
            width_px = abs(bbox[2] - bbox[0])
            height_px = abs(bbox[3] - bbox[1])
            
            # Taille brute (plus grande dimension)
            size_mm = max(width_px, height_px) * spacing
            
            # Correction adaptative pour les valeurs aberrantes
            if size_mm > 500:
                if size_mm < 1000:
                    size_mm = size_mm / 10
                else:
                    size_mm = size_mm / 100
            
            # Sécurité médicale : taille max 300 mm
            if size_mm > 300:
                size_mm = 300
            
            det["size_mm"] = round(size_mm, 1)
        else:
            det["size_mm"] = None

    # 5. Affichage des résultats
    print("\n" + "="*60)
    print("📊 RÉSULTATS (masques approximatifs)")
    print("="*60)
    for i, det in enumerate(detections_box, 1):
        label = det.get("malignancy_label", "?")
        score = det.get("malignancy_score", 0)
        bbox = det.get("bbox", [0,0,0,0])
        size_mm = det.get("size_mm", "N/A")
        area = np.sum(det.get("mask", np.zeros((1,1))) > 0) if det.get("mask") is not None else 0
        print(f"  Lésion {i}: label={label}, score={score:.3f}, taille={size_mm} mm, bbox={bbox}, surface={area} px²")

    # 6. Sauvegarde des images
    print("\n🖼️ Sauvegarde des images...")
    overlay = image_rgb.copy()
    for det in detections_box:
        mask = det.get("mask")
        label = det.get("malignancy_label", "?")
        label_normalized = normalize_label(label)
        size_mm = det.get("size_mm", "N/A")
        
        # Définir la couleur : ROUGE pour malin, VERT pour bénin
        if label_normalized == "malin":
            color = np.array([255, 0, 0], dtype=np.uint8)  # Rouge (BGR)
            bbox_color = (0, 0, 255)  # Rouge pour le rectangle
            label_type = "MALIN"
        else:
            color = np.array([0, 255, 0], dtype=np.uint8)  # Vert (BGR)
            bbox_color = (0, 255, 0)  # Vert pour le rectangle
            label_type = "BENIN"
        
        # Appliquer le masque avec la couleur appropriée
        if mask is not None:
            mask_bin = (mask > 0).astype(np.uint8)
            overlay[mask_bin > 0, :] = (overlay[mask_bin > 0, :] * 0.5 + color * 0.5)
        
        # Dessiner la boîte englobante
        bbox = det.get("bbox")
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            prob = det.get("malignancy_score", 0)
            
            # Rectangle avec la couleur appropriée
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bbox_color, 2)
            
            # Fond pour le texte
            text = f"{label} {prob:.2f}  {size_mm}mm"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(overlay, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 0, 0), -1)
            
            # Texte avec la couleur appropriée
            cv2.putText(overlay, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
    
    # Légende en bas de l'image
    cv2.putText(overlay, "Masques approximatifs (boites)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(overlay, "■ Malin (Rouge)", (10, overlay.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(overlay, "■ Benin (Vert)", (10, overlay.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite("segmentation_boxes.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("💾 Image sauvegardee sous 'segmentation_boxes.png'")

    # Sauvegarde des masques individuels
    for i, det in enumerate(detections_box, 1):
        if det.get("mask") is not None:
            label = det.get("malignancy_label", "?")
            label_normalized = normalize_label(label)
            
            if label_normalized == "malin":
                mask_filename = f"mask_box_{i+1}_MALIN.png"
            else:
                mask_filename = f"mask_box_{i+1}_BENIN.png"
            
            cv2.imwrite(mask_filename, det["mask"] * 255)
            print(f"💾 Masque sauvegardé : {mask_filename}")

    # 7. Sauvegarde du rapport JSON avec les tailles
    import json
    from datetime import datetime
    
    report = {
        "image": str(image_path),
        "date": datetime.now().isoformat(),
        "pixel_spacing": spacing,
        "detections": []
    }
    
    for i, det in enumerate(detections_box, 1):
        report["detections"].append({
            "id": i,
            "label": det.get("malignancy_label", "?"),
            "score": det.get("malignancy_score", 0),
            "size_mm": det.get("size_mm"),
            "bbox": det.get("bbox"),
            "surface_px": int(np.sum(det.get("mask", np.zeros((1,1))) > 0)) if det.get("mask") is not None else 0
        })
    
    with open("detection_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("💾 Rapport JSON sauvegardé sous 'detection_report.json'")

    # 8. Statistiques
    print("\n📊 STATISTIQUES :")
    print(f"  Nombre de lesions : {len(detections_box)}")
    
    malin_count = 0
    benin_count = 0
    unknown_count = 0
    sizes = []
    
    for d in detections_box:
        label = d.get("malignancy_label", "")
        label_normalized = normalize_label(label)
        size_mm = d.get("size_mm")
        if size_mm is not None:
            sizes.append(size_mm)
        
        if label_normalized == "malin":
            malin_count += 1
        elif label_normalized == "benin":
            benin_count += 1
        else:
            unknown_count += 1
    
    print(f"  Malin (rouge) : {malin_count}")
    print(f"  Benin (vert) : {benin_count}")
    if unknown_count > 0:
        print(f"  Non classifié : {unknown_count}")
    
    if sizes:
        print(f"  Taille moyenne : {np.mean(sizes):.1f} mm")
        print(f"  Taille min : {np.min(sizes):.1f} mm")
        print(f"  Taille max : {np.max(sizes):.1f} mm")
    
    areas = [np.sum(det.get("mask", np.zeros((1,1))) > 0) for det in detections_box if det.get("mask") is not None]
    if areas:
        print(f"  Surface moyenne : {np.mean(areas):.0f} px²")

    print("\n✅ Test des masques approximatifs termine.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else input("Entrez le chemin du fichier : ")
    asyncio.run(test_boxes(path))