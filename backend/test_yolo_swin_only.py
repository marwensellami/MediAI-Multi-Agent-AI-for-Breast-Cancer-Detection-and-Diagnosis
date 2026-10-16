# backend/test_yolo_swin_only.py
import asyncio
import cv2
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime

from app.ml.processors.image_reader import ImageProcessor
from app.ml.loaders.vision_loader import VisionLoader

# ============================================================
# IMPORTS DIRECTS (sans passer par app.agents.__init__)
# ============================================================
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
ClassificationAgent = load_agent("classification", "ClassificationAgent")

async def test_yolo_swin_only(image_path: str, output_dir: str = None, conf_threshold: float = 0.3):
    """
    Test YOLO fine‑tuné + Swin fine‑tuné (sans SAM2).
    Génère 3 images :
    - yolo_only.png : Résultat YOLO uniquement (boîtes)
    - swin_only.png : Résultat Swin uniquement (classification sur patches)
    - final_result.png : Résultat combiné
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Fichier non trouvé : {image_path}")
        return None

    if output_dir is None:
        output_dir = image_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    base_name = image_path.stem

    print("="*60)
    print("🔬 TEST YOLO fine‑tuné + Swin fine‑tuné (SANS SAM2)")
    print("="*60)
    print(f"📂 Image : {image_path.name}")
    print(f"📏 Seuil de confiance : {conf_threshold}")
    print(f"📁 Dossier de sortie : {output_dir}")
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
    print("\n🔍 Étape 1 : Détection YOLO fine‑tuné...")
    detector = DetectionAgent(confidence_threshold=conf_threshold)
    detections = await detector.process(image_rgb)
    print(f"   → {len(detections)} détection(s) brute(s)")

    if not detections:
        print("⚠️ Aucune lésion détectée.")
        # Image "aucune détection"
        img_clean = image_rgb.copy()
        cv2.putText(img_clean, "AUCUNE LESION DETECTEE", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        output_path = output_dir / f"{base_name}_no_detection.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(img_clean, cv2.COLOR_RGB2BGR))
        print(f"💾 Image sauvegardée sous : {output_path}")
        return {"detections": [], "summary": "Aucune lésion"}

    # Ajouter l'image originale aux détections
    for det in detections:
        det["_original_image"] = image_rgb

    # ============================================================
    # 3. CLASSIFICATION SWIN
    # ============================================================
    print("\n🏷️ Étape 2 : Classification Swin fine‑tuné...")
    classifier = ClassificationAgent()
    detections = await classifier.process(detections)
    print(f"   → {len(detections)} détection(s) classifiée(s)")

    # ============================================================
    # 4. CALCUL DES TAILLES
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
    # 5. AFFICHAGE DES RÉSULTATS
    # ============================================================
    print("\n" + "="*60)
    print("📊 RÉSULTATS")
    print("="*60)

    for i, det in enumerate(detections, 1):
        bbox = det.get("bbox", [0,0,0,0])
        label = det.get("malignancy_label", "Indéterminé")
        score = det.get("malignancy_score", 0.0)
        size = det.get("size_mm", "N/A")
        conf = det.get("confidence", 0.0)
        print(f"\n🔹 Lésion {i}:")
        print(f"   - Classification Swin : {label}")
        print(f"   - Probabilité de malignité : {score*100:.1f}%")
        print(f"   - Taille estimée : {size} mm")
        print(f"   - Confiance YOLO : {conf*100:.1f}%")
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
    # 6. GÉNÉRATION DES IMAGES
    # ============================================================
    print("\n🖼️ Génération des images...")

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

    # ============================================================
    # 6.1 IMAGE YOLO UNIQUEMENT (boîtes uniquement)
    # ============================================================
    print("   📸 1/3 : Image YOLO uniquement...")
    img_yolo = image_rgb.copy()
    for det in detections:
        bbox = det.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        conf = det.get("confidence", 0)
        class_name = det.get("class_name", "lesion")
        cv2.rectangle(img_yolo, (x1, y1), (x2, y2), (255, 255, 0), 2)
        label = f"{class_name} {conf*100:.1f}%"
        cv2.putText(img_yolo, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(img_yolo, "YOLO fine-tuned (détection)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    yolo_path = output_dir / f"{base_name}_yolo_only.png"
    cv2.imwrite(str(yolo_path), cv2.cvtColor(img_yolo, cv2.COLOR_RGB2BGR))
    print(f"      💾 {yolo_path.name}")

    # ============================================================
    # 6.2 IMAGE SWIN UNIQUEMENT (patchs classifiés)
    # ============================================================
    print("   📸 2/3 : Image Swin uniquement...")
    img_swin = image_rgb.copy()
    for det in detections:
        bbox = det.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        label = det.get("malignancy_label", "Indéterminé")
        score = det.get("malignancy_score", 0)
        color = colors.get(label, (255, 255, 255))

        # Afficher le patch Swin (si possible, on le découpe)
        patch = det.get("_patch", None)
        if patch is not None and isinstance(patch, np.ndarray):
            # Superposer le patch redimensionné
            try:
                patch_resized = cv2.resize(patch, (bbox[2]-bbox[0], bbox[3]-bbox[1]))
                img_swin[y1:y2, x1:x2] = patch_resized
            except:
                pass

        # Cercle de couleur pour la classification
        cv2.circle(img_swin, (int((x1+x2)/2), int((y1+y2)/2)), 15, color, -1)
        label_text = f"{label} {score*100:.1f}%"
        cv2.putText(img_swin, label_text, (x1, y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(img_swin, "Swin fine-tuned (classification)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    swin_path = output_dir / f"{base_name}_swin_only.png"
    cv2.imwrite(str(swin_path), cv2.cvtColor(img_swin, cv2.COLOR_RGB2BGR))
    print(f"      💾 {swin_path.name}")

    # ============================================================
    # 6.3 IMAGE FINALE COMBINÉE (YOLO + Swin)
    # ============================================================
    print("   📸 3/3 : Image finale combinée...")
    img_final = image_rgb.copy()
    h, w = img_final.shape[:2]

    # En-tête
    cv2.rectangle(img_final, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(img_final, f"YOLO + Swin - {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for i, det in enumerate(detections, 1):
        bbox = det.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        label = det.get("malignancy_label", "Indéterminé")
        score = det.get("malignancy_score", 0)
        size = det.get("size_mm", "N/A")
        conf = det.get("confidence", 0)
        color = colors.get(label, (255, 255, 255))

        # Boîte YOLO
        cv2.rectangle(img_final, (x1, y1), (x2, y2), color, 3)

        # Label principal
        label_text = f"#{i} {label} {score*100:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(y1 - 10, 30)
        cv2.rectangle(img_final, (x1, y_text - text_h - 8), (x1 + text_w + 10, y_text + 4), color, -1)
        cv2.putText(img_final, label_text, (x1 + 5, y_text - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Taille
        cv2.putText(img_final, f"Taille: {size} mm", (x1, y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Confiance YOLO
        cv2.putText(img_final, f"YOLO: {conf*100:.1f}%", (x2 - 100, y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)

    # Pied de page
    summary_text = f"Lésions: {len(detections)} | Malignes: {malignant} | Bénignes: {benign} | BI-RADS: {birads}"
    cv2.rectangle(img_final, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(img_final, summary_text, (10, h - 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    final_path = output_dir / f"{base_name}_final_result.png"
    cv2.imwrite(str(final_path), cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR))
    print(f"      💾 {final_path.name}")

    # ============================================================
    # 7. SAUVEGARDE DU RAPPORT JSON
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
                "confidence_yolo": det.get("confidence")
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

    json_path = output_dir / f"{base_name}_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)
    print(f"💾 Rapport JSON sauvegardé sous : {json_path}")

    print("\n✅ Test terminé avec succès !")
    print(f"\n📁 Fichiers générés dans {output_dir}:")
    print(f"   1. {yolo_path.name} - Détection YOLO uniquement")
    print(f"   2. {swin_path.name} - Classification Swin uniquement")
    print(f"   3. {final_path.name} - Résultat final combiné")
    print(f"   4. {json_path.name} - Rapport JSON")

    return report_json

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_yolo_swin_only.py <image_path> [--output output_dir] [--conf 0.3]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = None
    conf = 0.3

    for i, arg in enumerate(sys.argv):
        if arg == "--output" and i+1 < len(sys.argv):
            output_dir = sys.argv[i+1]
        elif arg == "--conf" and i+1 < len(sys.argv):
            try:
                conf = float(sys.argv[i+1])
            except:
                pass

    asyncio.run(test_yolo_swin_only(image_path, output_dir, conf))