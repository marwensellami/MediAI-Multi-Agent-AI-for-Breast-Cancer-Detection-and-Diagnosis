# backend/test_full_pipeline_standalone.py
import asyncio
import cv2
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime

# ============================================================
# 1. IMPORTS DIRECTS (sans passer par app.agents.__init__)
# ============================================================
from app.ml.processors.image_reader import ImageProcessor
from app.ml.loaders.vision_loader import VisionLoader

# Chargement direct des modules
import importlib.util

def load_agent(module_name, class_name):
    """Charge une classe depuis un module sans passer par __init__."""
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

# ============================================================
# 2. FONCTION PRINCIPALE
# ============================================================
async def test_full_pipeline(image_path: str, output_path: str = None, conf_threshold: float = 0.3):
    """
    Test complet du pipeline avec visualisation finale.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Fichier non trouvé : {image_path}")
        return None

    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_pipeline_result.png"

    print("="*60)
    print("🔬 PIPELINE COMPLET - DIAGNOSTIC MAMMAIRE")
    print("="*60)
    print(f"📂 Image : {image_path.name}")
    print(f"📏 Seuil de confiance : {conf_threshold}")
    print(f"🖥️ Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*60)

    # ============================================================
    # 1. CHARGEMENT DE L'IMAGE
    # ============================================================
    print("📥 Chargement de l'image...")
    data = ImageProcessor.read_file(image_path)
    if not data["is_valid"]:
        print(f"❌ Erreur : {data.get('error')}")
        return None

    pixels = data["pixels"]
    metadata = data["metadata"]
    pixel_spacing = metadata.get("pixel_spacing", [0.1, 0.1])

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

    print(f"✅ Image chargée : {image_rgb.shape}")
    print(f"📏 Pixel spacing : {pixel_spacing}")

    # ============================================================
    # 2. DÉTECTION YOLO
    # ============================================================
    print("\n🔍 Étape 1 : Détection YOLO...")
    detector = DetectionAgent(confidence_threshold=conf_threshold)
    detections = await detector.process(image_rgb)
    print(f"   → {len(detections)} détection(s) brute(s)")

    if not detections:
        print("⚠️ Aucune lésion détectée.")
        result_img = image_rgb.copy()
        cv2.putText(result_img, "AUCUNE LESION DETECTEE", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(str(output_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        print(f"💾 Image sauvegardée sous : {output_path}")
        return {"detections": [], "summary": "Aucune lésion"}

    for det in detections:
        det["_original_image"] = image_rgb

    # ============================================================
    # 3. SEGMENTATION
    # ============================================================
    print("\n🎯 Étape 2 : Segmentation...")
    segmenter = SegmentationAgent()
    detections = await segmenter.process(image_rgb, detections)
    print(f"   → {len(detections)} détection(s) segmentée(s)")

    # ============================================================
    # 4. CLASSIFICATION SWIN
    # ============================================================
    print("\n🏷️ Étape 3 : Classification Swin...")
    classifier = ClassificationAgent()
    detections = await classifier.process(detections)
    print(f"   → {len(detections)} détection(s) classifiée(s)")

    # ============================================================
    # 5. CALCUL DES TAILLES
    # ============================================================
    spacing = pixel_spacing[0] if pixel_spacing else 0.1
    for det in detections:
        bbox = det.get("bbox")
        if bbox and len(bbox) == 4:
            width_px = abs(bbox[2] - bbox[0])
            height_px = abs(bbox[3] - bbox[1])
            size_mm = max(width_px, height_px) * spacing
            if size_mm > 500:
                if size_mm < 1000:
                    size_mm = size_mm / 10
                else:
                    size_mm = size_mm / 100
            if size_mm > 300:
                size_mm = 300
            det["size_mm"] = round(size_mm, 1)
        else:
            det["size_mm"] = None
        det.pop("_original_image", None)

    # ============================================================
    # 6. AFFICHAGE DES RÉSULTATS
    # ============================================================
    print("\n" + "="*60)
    print("📊 RÉSULTATS DU DIAGNOSTIC")
    print("="*60)

    for i, det in enumerate(detections, 1):
        bbox = det.get("bbox", [0,0,0,0])
        label = det.get("malignancy_label", "Indéterminé")
        score = det.get("malignancy_score", 0.0)
        size = det.get("size_mm", "N/A")
        conf = det.get("confidence", 0.0)
        seg_score = det.get("segmentation_score", 0.0)
        
        print(f"\n🔹 Lésion {i}:")
        print(f"   - Classification : {label}")
        print(f"   - Probabilité de malignité : {score*100:.1f}%")
        print(f"   - Taille estimée : {size} mm")
        print(f"   - Confiance YOLO : {conf*100:.1f}%")
        print(f"   - Score segmentation : {seg_score:.3f}")
        print(f"   - Localisation : {bbox}")

    malignant = sum(1 for d in detections if d.get("malignancy_label") in ["Malin", "Maligne", "Malignant"])
    benign = len(detections) - malignant
    max_score = max((d.get("malignancy_score", 0) for d in detections), default=0)

    print("\n" + "-"*60)
    print("📈 SYNTHÈSE :")
    print(f"   - Total lésions : {len(detections)}")
    print(f"   - Malignes : {malignant}")
    print(f"   - Bénignes : {benign}")
    print(f"   - Score max de malignité : {max_score*100:.1f}%")
    if max_score >= 0.95:
        birads = "5"
    elif max_score >= 0.10:
        birads = "4"
    elif max_score >= 0.02:
        birads = "3"
    else:
        birads = "2"
    print(f"   - BI-RADS suggéré : {birads}")
    print("="*60)

    # ============================================================
    # 7. GÉNÉRATION DE L'IMAGE ANNOTÉE
    # ============================================================
    print("\n🖼️ Génération de l'image finale...")

    result_img = image_rgb.copy()
    h, w = result_img.shape[:2]

    cv2.rectangle(result_img, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(result_img, f"Diagnostic Mammaire - {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    colors = {
        "Malin": (0, 0, 255),
        "Maligne": (0, 0, 255),
        "Malignant": (0, 0, 255),
        "Bénin": (0, 255, 0),
        "Bénigne": (0, 255, 0),
        "Benign": (0, 255, 0),
        "Normal": (255, 255, 0),
        "Indéterminé": (255, 165, 0)
    }

    for i, det in enumerate(detections, 1):
        bbox = det.get("bbox")
        if not bbox:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        label = det.get("malignancy_label", "Indéterminé")
        score = det.get("malignancy_score", 0.0)
        size = det.get("size_mm", "N/A")
        color = colors.get(label, (255, 255, 255))

        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)

        mask = det.get("mask")
        if mask is not None:
            mask_bin = (mask > 0).astype(np.uint8)
            color_transparent = np.array(color, dtype=np.uint8)
            for c in range(3):
                result_img[:, :, c] = np.where(
                    mask_bin > 0,
                    (result_img[:, :, c] * 0.6 + color_transparent[c] * 0.4).astype(np.uint8),
                    result_img[:, :, c]
                )

        label_text = f"#{i} {label} {score*100:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(y1 - 10, 30)
        cv2.rectangle(result_img, (x1, y_text - text_h - 8), (x1 + text_w + 10, y_text + 4), color, -1)
        cv2.putText(result_img, label_text, (x1 + 5, y_text - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        size_text = f"Taille: {size} mm"
        cv2.putText(result_img, size_text, (x1, y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        seg_score = det.get("segmentation_score")
        if seg_score:
            seg_text = f"Seg: {seg_score:.2f}"
            cv2.putText(result_img, seg_text, (x1, y2 + 38), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 200), 1)

        conf = det.get("confidence", 0)
        conf_text = f"YOLO: {conf*100:.1f}%"
        cv2.putText(result_img, conf_text, (x2 - 100, y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)

    summary_text = f"Lésions: {len(detections)} | Malignes: {malignant} | Bénignes: {benign} | BI-RADS: {birads} | Confiance: {max_score*100:.1f}%"
    cv2.rectangle(result_img, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(result_img, summary_text, (10, h - 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(str(output_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print(f"💾 Image annotée sauvegardée sous : {output_path}")

    # ============================================================
    # 8. SAUVEGARDE DU RAPPORT JSON
    # ============================================================
    report_json = {
        "image": str(image_path),
        "date": datetime.now().isoformat(),
        "pixel_spacing": pixel_spacing,
        "detections": [
            {
                "id": i,
                "bbox": det.get("bbox"),
                "label": det.get("malignancy_label"),
                "score": det.get("malignancy_score"),
                "size_mm": det.get("size_mm"),
                "confidence_yolo": det.get("confidence"),
                "segmentation_score": det.get("segmentation_score")
            }
            for i, det in enumerate(detections, 1)
        ],
        "summary": {
            "total": len(detections),
            "malignant": malignant,
            "benign": benign,
            "max_score": max_score,
            "birads": birads
        }
    }

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)
    print(f"💾 Rapport JSON sauvegardé sous : {json_path}")

    print("\n✅ Pipeline terminé avec succès !")
    return report_json

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_full_pipeline_standalone.py <image_path> [--output output.png] [--conf 0.3]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = None
    conf = 0.3

    for i, arg in enumerate(sys.argv):
        if arg == "--output" and i+1 < len(sys.argv):
            output_path = sys.argv[i+1]
        elif arg == "--conf" and i+1 < len(sys.argv):
            try:
                conf = float(sys.argv[i+1])
            except:
                pass

    asyncio.run(test_full_pipeline(image_path, output_path, conf))