# backend/test_sam2_vs_yolo.py
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

async def test_sam2_vs_yolo(image_path: str, output_path: str = None):
    """
    Compare visuellement SAM2 (masque précis) vs YOLO (boîte rectangulaire).
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Fichier non trouvé : {image_path}")
        return

    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_sam2_vs_yolo.png"
    else:
        output_path = Path(output_path)

    print("="*70)
    print("🎯 SAM2 vs YOLO - COMPARAISON VISUELLE")
    print("="*70)
    print(f"📂 Image : {image_path.name}")
    print("-"*70)

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
    # 2. DÉTECTION YOLO
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
    sam.set_image(image_rgb)
    masks, scores, _ = sam.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    print(f"   → {masks.shape[0]} masque(s) généré(s)")

    # ============================================================
    # 5. GÉNÉRATION DE L'IMAGE DE COMPARAISON
    # ============================================================
    print("\n🖼️ Génération de l'image de comparaison...")

    h, w = image_rgb.shape[:2]

    # Créer 3 images côte à côte
    img_original = image_rgb.copy()
    img_yolo = image_rgb.copy()
    img_sam2 = image_rgb.copy()
    img_comparison = image_rgb.copy()

    # --- 5.1 Image 1 : YOLO seul (boîte rectangulaire) ---
    for i, det in enumerate(detections):
        bbox = det["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        # Boîte YOLO
        cv2.rectangle(img_yolo, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # Remplir l'intérieur en bleu clair (transparent)
        overlay = img_yolo.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 200), -1)
        cv2.addWeighted(overlay, 0.3, img_yolo, 0.7, 0, img_yolo)
        # Texte
        cv2.putText(img_yolo, f"YOLO #{i+1}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # --- 5.2 Image 2 : SAM2 seul (masque précis) ---
    for i in range(masks.shape[0]):
        mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
        mask_bin = (mask > 0).astype(np.uint8)
        # Masque SAM2 en vert
        color_sam = np.array([0, 255, 0], dtype=np.uint8)
        for c in range(3):
            img_sam2[:, :, c] = np.where(
                mask_bin > 0,
                (img_sam2[:, :, c] * 0.5 + color_sam[c] * 0.5).astype(np.uint8),
                img_sam2[:, :, c]
            )
        # Contours du masque
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_sam2, contours, -1, (0, 255, 0), 2)
        cv2.putText(img_sam2, f"SAM2 #{i+1}", (10, 30 + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- 5.3 Image 3 : Comparaison superposée (YOLO + SAM2) ---
    for i in range(masks.shape[0]):
        mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
        mask_bin = (mask > 0).astype(np.uint8)
        # SAM2 en vert
        color_sam = np.array([0, 255, 0], dtype=np.uint8)
        for c in range(3):
            img_comparison[:, :, c] = np.where(
                mask_bin > 0,
                (img_comparison[:, :, c] * 0.4 + color_sam[c] * 0.6).astype(np.uint8),
                img_comparison[:, :, c]
            )
        # Contours SAM2 en vert fluo
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_comparison, contours, -1, (0, 255, 0), 3)

        # Boîte YOLO en rouge
        bbox = detections[i]["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_comparison, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Texte avec les deux labels
        cv2.putText(img_comparison, f"YOLO (rouge) vs SAM2 (vert)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img_comparison, f"Lésion {i+1}", (x1, y1 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- 5.4 Légende ---
    for img, title in [(img_original, "Original"), (img_yolo, "YOLO (boîte)"), 
                        (img_sam2, "SAM2 (masque)"), (img_comparison, "Superposition")]:
        cv2.rectangle(img, (0, 0), (w, 35), (0, 0, 0), -1)
        cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Combiner les 4 images côte à côte
    # Redimensionner pour que chaque image fasse la même hauteur (on garde le ratio)
    target_h = 400
    scale = target_h / h
    new_w = int(w * scale)

    def resize_img(img):
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    img_original_resized = resize_img(img_original)
    img_yolo_resized = resize_img(img_yolo)
    img_sam2_resized = resize_img(img_sam2)
    img_comparison_resized = resize_img(img_comparison)

    # Assemblage en grille 2x2
    top_row = np.hstack([img_original_resized, img_yolo_resized])
    bottom_row = np.hstack([img_sam2_resized, img_comparison_resized])
    final_image = np.vstack([top_row, bottom_row])

    # Ajouter une bordure
    cv2.rectangle(final_image, (0, 0), (final_image.shape[1]-1, final_image.shape[0]-1), (255, 255, 255), 2)

    # Sauvegarde
    cv2.imwrite(str(output_path), cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    print(f"💾 Image de comparaison sauvegardée sous : {output_path}")
    print(f"   - En haut à gauche : Original")
    print(f"   - En haut à droite : YOLO (boîte rectangulaire)")
    print(f"   - En bas à gauche : SAM2 (masque précis)")
    print(f"   - En bas à droite : Superposition (YOLO rouge + SAM2 vert)")

    # ============================================================
    # 6. SAUVEGARDE DES MASQUES BRUTS
    # ============================================================
    for i in range(masks.shape[0]):
        mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
        mask_bin = (mask * 255).astype(np.uint8)
        mask_path = image_path.parent / f"{image_path.stem}_sam2_mask_{i+1}.png"
        cv2.imwrite(str(mask_path), mask_bin)
        print(f"💾 Masque brut {i+1} sauvegardé sous : {mask_path}")

    print("\n✅ Test terminé.")
    print("="*70)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else input("Entrez le chemin du fichier : ")
    output = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(test_sam2_vs_yolo(path, output))