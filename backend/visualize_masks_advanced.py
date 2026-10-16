# backend/visualize_masks_advanced.py
import asyncio
import cv2
import numpy as np
from pathlib import Path

from app.ml.loaders.vision_loader import VisionLoader
from app.ml.processors.image_reader import ImageProcessor

async def visualize_masks(image_path: str):
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

    # YOLO
    yolo = VisionLoader.load_yolo()
    results = yolo(image_rgb, verbose=False, conf=0.1)
    detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({"bbox": [x1, y1, x2, y2]})

    if not detections:
        print("⚠️ Aucune lésion détectée.")
        return

    # SAM2
    sam = VisionLoader.load_sam2()
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

    # --- 1. Masque binaire (blanc sur noir) ---
    cv2.imwrite("mask_binary.png", combined_mask * 255)
    print("💾 Masque binaire sauvegardé sous 'mask_binary.png'")

    # --- 2. Masque coloré superposé (ROUGE, comme l'exemple) ---
    overlay_red = image_rgb.copy()
    overlay_red[combined_mask > 0] = (0, 0, 255)  # Rouge
    alpha = 0.5
    blended = cv2.addWeighted(image_rgb, 1 - alpha, overlay_red, alpha, 0)
    cv2.imwrite("mask_overlay_red.png", blended)
    print("💾 Superposition rouge sauvegardée sous 'mask_overlay_red.png'")

    # --- 3. Masque coloré superposé (VERT) ---
    overlay_green = image_rgb.copy()
    overlay_green[combined_mask > 0] = (0, 255, 0)  # Vert
    blended_green = cv2.addWeighted(image_rgb, 1 - alpha, overlay_green, alpha, 0)
    cv2.imwrite("mask_overlay_green.png", blended_green)
    print("💾 Superposition verte sauvegardée sous 'mask_overlay_green.png'")

    # --- 4. Masque avec contours (pour voir précisément les bords) ---
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = image_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 255), 2)  # Jaune
    cv2.imwrite("mask_contours.png", contour_img)
    print("💾 Contours sauvegardés sous 'mask_contours.png'")

    # --- 5. Masque en niveaux de gris (pour voir la zone) ---
    gray_mask = (combined_mask * 255).astype(np.uint8)
    cv2.imwrite("mask_gray.png", gray_mask)
    print("💾 Masque niveaux de gris sauvegardé sous 'mask_gray.png'")

    print("\n✅ Tous les masques ont été générés.")
    print("📁 Fichiers créés :")
    print("   - mask_binary.png (blanc sur noir)")
    print("   - mask_overlay_red.png (rouge superposé)")
    print("   - mask_overlay_green.png (vert superposé)")
    print("   - mask_contours.png (contours en jaune)")
    print("   - mask_gray.png (niveaux de gris)")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else input("Entrez le chemin du fichier : ")
    asyncio.run(visualize_masks(path))