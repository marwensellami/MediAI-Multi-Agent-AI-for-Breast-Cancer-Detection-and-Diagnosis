# backend/test_segmentation_visual.py
import asyncio
import cv2
import numpy as np
from pathlib import Path

from app.ml.processors.image_reader import ImageProcessor
from app.agents.vision.detection import DetectionAgent
from app.agents.vision.segmentation import SegmentationAgent

async def visualize_segmentation(image_path: str):
    """Charge une image, détecte les lésions, segmente avec SAM2 et superpose les masques."""
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Fichier non trouvé : {image_path}")
        return

    print(f"📂 Chargement de {image_path.name}...")
    # 1. Lecture de l'image
    data = ImageProcessor.read_file(image_path)
    if not data["is_valid"]:
        print(f"❌ Erreur de lecture : {data.get('error')}")
        return

    pixels = data["pixels"]
    # Conversion en RGB uint8 pour l'affichage
    if pixels.dtype != np.uint8:
        pixels = (pixels / pixels.max() * 255).astype(np.uint8)
    if len(pixels.shape) == 2:
        image_rgb = np.stack([pixels] * 3, axis=-1)
    elif pixels.shape[2] == 1:
        image_rgb = np.concatenate([pixels] * 3, axis=-1)
    else:
        image_rgb = pixels[:, :, :3]

    # Sauvegarde de l'image originale pour référence
    cv2.imwrite("original_image.png", image_rgb)
    print("💾 Image originale sauvegardée sous 'original_image.png'")

    # 2. Détection YOLO
    print("🔍 Détection YOLO...")
    detector = DetectionAgent(confidence_threshold=0.3)  # seuil bas pour capturer plus de lésions
    detections = await detector.process(image_rgb)
    print(f"   → {len(detections)} détection(s) trouvée(s).")

    if not detections:
        print("⚠️ Aucune lésion détectée. Arrêt.")
        return

    # 3. Segmentation SAM2
    print("🔄 Segmentation SAM2...")
    segmenter = SegmentationAgent()
    detections = await segmenter.process(image_rgb, detections)

    # 4. Superposition des masques sur l'image
    overlay = image_rgb.copy()
    for det in detections:
        mask = det.get("mask")
        if mask is not None:
            # Le masque est en uint8 (0-255), on le binarise
            mask_bin = (mask > 0).astype(np.uint8)
            # Couleur verte pour le masque
            overlay[mask_bin > 0] = (0, 255, 0)  # vert

            # Ajouter la boîte englobante
            bbox = det.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)  # bleu
                # Afficher le score de confiance
                conf = det.get("confidence", 0)
                label = f"YOLO: {conf:.2f}"
                cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                # Afficher le score de segmentation (si disponible)
                seg_score = det.get("segmentation_score")
                if seg_score:
                    label_seg = f"SAM2: {seg_score:.2f}"
                    cv2.putText(overlay, label_seg, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # Sauvegarde du résultat
    cv2.imwrite("segmented_overlay.png", overlay)
    print("💾 Image avec masques et boîtes sauvegardée sous 'segmented_overlay.png'")

    # Superposition avec transparence (pour mieux voir l'image)
    alpha = 0.5
    blended = cv2.addWeighted(image_rgb, 1-alpha, overlay, alpha, 0)
    cv2.imwrite("segmented_blended.png", blended)
    print("💾 Image fusionnée (transparente) sauvegardée sous 'segmented_blended.png'")

    print("✅ Visualisation terminée. Ouvrez les fichiers pour vérifier la segmentation.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Entrez le chemin du fichier (PNG, JPG, DICOM) : ")
    asyncio.run(visualize_segmentation(path))