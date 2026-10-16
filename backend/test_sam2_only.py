# backend/test_sam2_only_standalone.py
import asyncio
import cv2
import numpy as np
import sys
from pathlib import Path

# ============================================================
# IMPORTS DIRECTS (sans passer par app.agents.__init__)
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

async def test_sam2_only(image_path: str, output_path: str = None):
    """
    Test SAM2 uniquement : détection YOLO + segmentation SAM2.
    Génère une image avec le masque SAM2 superposé en vert.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Fichier non trouvé : {image_path}")
        return

    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_sam2_only.png"
    else:
        output_path = Path(output_path)

    print("="*60)
    print("🎯 TEST SAM2 UNIQUEMENT - SEGMENTATION PRÉCISE")
    print("="*60)
    print(f"📂 Image : {image_path.name}")
    print("-"*60)

    # ============================================================
    # 1. CHARGEMENT DE L'IMAGE
    # ============================================================
    print("📥 Chargement de l'image...")
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

    print(f"✅ Image chargée : {image_rgb.shape}")

    # ============================================================
    # 2. DÉTECTION YOLO (pour avoir des boîtes)
    # ============================================================
    print("🔍 Détection YOLO...")
    detector = DetectionAgent(confidence_threshold=0.3)
    detections = await detector.process(image_rgb)
    print(f"   → {len(detections)} détection(s) brute(s)")

    if not detections:
        print("⚠️ Aucune lésion détectée.")
        result_img = image_rgb.copy()
        cv2.putText(result_img, "AUCUNE LESION DETECTEE", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(str(output_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        print(f"💾 Image sauvegardée sous : {output_path}")
        return

    # ============================================================
    # 3. CHARGEMENT DE SAM2
    # ============================================================
    print("📥 Chargement de SAM2...")
    sam = VisionLoader.load_sam2()
    if sam is None:
        print("❌ SAM2 n'a pas pu être chargé.")
        return
    print("✅ SAM2 chargé avec succès.")

    # ============================================================
    # 4. SEGMENTATION SAM2
    # ============================================================
    print("🔄 Segmentation SAM2...")
    input_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    
    # Définir l'image dans le prédicteur
    sam.set_image(image_rgb)
    
    # Prédire les masques
    masks, scores, _ = sam.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    print(f"   → {masks.shape[0]} masque(s) généré(s)")

    # ============================================================
    # 5. AFFICHAGE DES RÉSULTATS
    # ============================================================
    print("\n📊 RÉSULTATS SAM2 :")
    for i, det in enumerate(detections):
        bbox = det["bbox"]
        print(f"  Lésion {i+1}: bbox={bbox}")
        print(f"    - Confiance YOLO: {det['confidence']:.3f}")
        if i < len(masks):
            mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
            score = scores[i] if scores is not None else 0.5
            if isinstance(score, np.ndarray):
                score_val = float(score[0]) if score.size > 0 else 0.0
            else:
                score_val = float(score)
            area = np.sum(mask > 0)
            print(f"    - Score SAM2: {score_val:.3f}")
            print(f"    - Surface du masque: {area} px²")

    # ============================================================
    # 6. GÉNÉRATION DE L'IMAGE FINALE
    # ============================================================
    print("\n🖼️ Génération de l'image SAM2...")

    result_img = image_rgb.copy()
    h, w = result_img.shape[:2]

    # En-tête
    cv2.rectangle(result_img, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(result_img, "SAM2 - Segmentation Precise", (10, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Pour chaque lésion
    for i in range(masks.shape[0]):
        mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
        mask_bin = (mask > 0).astype(np.uint8)
        
        # 1. SUPERPOSITION DU MASQUE SAM2 (vert semi-transparent)
        color_sam = np.array([0, 255, 0], dtype=np.uint8)
        for c in range(3):
            result_img[:, :, c] = np.where(
                mask_bin > 0,
                (result_img[:, :, c] * 0.5 + color_sam[c] * 0.5).astype(np.uint8),
                result_img[:, :, c]
            )

        # 2. CONTOURS DU MASQUE SAM2
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)

        # 3. BOÎTE YOLO (en jaune pour comparaison)
        bbox = detections[i]["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 200, 0), 1)

        # 4. LABELS
        label = f"Lésion {i+1} (SAM2)"
        cv2.putText(result_img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Score SAM2
        score = scores[i] if scores is not None else 0.5
        if isinstance(score, np.ndarray):
            score_val = float(score[0]) if score.size > 0 else 0.0
        else:
            score_val = float(score)
        score_text = f"Score: {score_val:.3f}"
        cv2.putText(result_img, score_text, (x1, y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Surface
        area = np.sum(mask > 0)
        area_text = f"Surface: {area} px²"
        cv2.putText(result_img, area_text, (x1, y2 + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 5. LÉGENDE
    legend_y = h - 60
    cv2.rectangle(result_img, (10, legend_y), (200, h - 10), (0, 0, 0), -1)
    cv2.putText(result_img, "LEGENDE:", (20, legend_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(result_img, "--- Masque SAM2 (vert)", (20, legend_y + 38), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(result_img, "--- Bbox YOLO (jaune)", (20, legend_y + 54), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    # 6. SAUVEGARDE
    cv2.imwrite(str(output_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print(f"💾 Image SAM2 sauvegardée sous : {output_path}")

    # 7. SAUVEGARDE DES MASQUES BRUTS
    for i in range(masks.shape[0]):
        mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
        mask_bin = (mask * 255).astype(np.uint8)
        mask_path = image_path.parent / f"{image_path.stem}_sam2_mask_{i+1}.png"
        cv2.imwrite(str(mask_path), mask_bin)
        print(f"💾 Masque brut {i+1} sauvegardé sous : {mask_path}")

    print("\n✅ Test SAM2 terminé.")
    print("="*60)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else input("Entrez le chemin du fichier : ")
    output = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(test_sam2_only(path, output))