# backend/test_sam2_direct.py
import asyncio
import numpy as np
import cv2
from pathlib import Path

# On importe directement le loadeur SAM2 sans passer par l'orchestrateur
from app.ml.loaders.vision_loader import VisionLoader
from app.ml.processors.dicom_reader import DICOMProcessor

async def test_sam2():
    print("🧪 TEST SAM2 - SEGMENTATION (DIRECT)")
    print("=" * 40)

    # ============================================================
    # 1. Chargement de SAM2 via le loader (sans registre)
    # ============================================================
    print("📥 Chargement de SAM2...")
    sam = VisionLoader.load_sam2()
    if sam is None:
        print("❌ Échec du chargement de SAM2.")
        return
    print("✅ SAM2 chargé avec succès.")

    # ============================================================
    # 2. Chargement de l'image (DICOM)
    # ============================================================
    dicom_path = Path("app/data/dicom_samples/mammo.dcm")
    if not dicom_path.exists():
        # Création d'une image synthétique
        print("⚠️ Aucun DICOM trouvé, création d'une image synthétique.")
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(image, (256, 256), 80, (255, 255, 255), -1)
        cv2.imwrite("test_image.png", image)
        print("✅ Image synthétique créée : test_image.png")
    else:
        print(f"📂 Chargement du DICOM : {dicom_path}")
        result = DICOMProcessor.read_dicom(dicom_path)
        if not result["is_valid"]:
            print("❌ Erreur lecture DICOM.")
            return
        pixels = result["pixels"]
        # Normalisation et conversion en RGB uint8
        if pixels.dtype != np.uint8:
            if pixels.max() > 0:
                pixels = (pixels / pixels.max() * 255).astype(np.uint8)
            else:
                pixels = pixels.astype(np.uint8)
        if len(pixels.shape) == 2:
            image = np.stack([pixels] * 3, axis=-1)
        elif pixels.shape[2] == 1:
            image = np.concatenate([pixels] * 3, axis=-1)
        else:
            image = pixels[:, :, :3]
        print(f"✅ Image chargée : {image.shape}")

    # ============================================================
    # 3. Définition d'une boîte de test
    # ============================================================
    h, w = image.shape[:2]
    box_x1, box_y1 = w//3, h//3
    box_x2, box_y2 = 2*w//3, 2*h//3
    input_box = np.array([[box_x1, box_y1, box_x2, box_y2]], dtype=np.float32)
    print(f"📦 Boîte de test : {input_box[0].tolist()}")

    # ============================================================
    # 4. Segmentation SAM2
    # ============================================================
    print("🔄 Lancement de la segmentation SAM2...")
    # SAM2 prédit sur l'image
    sam.set_image(image)
    masks, scores, _ = sam.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False,
    )
    # masks shape: (N, 1, H, W) ou (N, H, W)
    if masks is not None and len(masks) > 0:
        mask = masks[0, 0, :, :] if masks.ndim == 4 else masks[0, :, :]
        mask_bin = (mask > 0).astype(np.uint8) * 255
        print(f"✅ Masque généré !")
        print(f"   - Forme : {mask_bin.shape}")
        print(f"   - Pixels > 0 : {np.sum(mask_bin > 0)}")
        # Sauvegarde
        cv2.imwrite("mask_sam2_direct.png", mask_bin)
        print("💾 Masque sauvegardé sous 'mask_sam2_direct.png'")
        # Superposition
        overlay = image.copy()
        overlay[mask_bin > 0] = (0, 255, 0)  # vert
        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        cv2.imwrite("mask_overlay_direct.png", blended)
        print("💾 Superposition sauvegardée sous 'mask_overlay_direct.png'")
        print("✅ TEST RÉUSSI !")
    else:
        print("❌ Aucun masque produit.")

if __name__ == "__main__":
    asyncio.run(test_sam2())