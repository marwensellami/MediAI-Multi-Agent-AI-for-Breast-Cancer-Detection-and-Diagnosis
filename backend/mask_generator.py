# backend/mask_generator_auto.py
import asyncio
import cv2
import numpy as np
import argparse
from pathlib import Path

from app.ml.loaders.vision_loader import VisionLoader
from app.ml.processors.image_reader import ImageProcessor

# Dossiers à explorer automatiquement
SEARCH_DIRS = [
    Path("app/data/dicom_samples"),
    Path("app/data/uploads"),
]

def find_file(filename: str) -> Path:
    """Cherche un fichier dans les dossiers de recherche et le répertoire courant."""
    # 1. Vérifier si le chemin absolu existe
    p = Path(filename)
    if p.exists():
        return p

    # 2. Chercher dans les dossiers prédéfinis
    for dir_path in SEARCH_DIRS:
        candidate = dir_path / p.name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Fichier '{filename}' introuvable. Veuillez fournir un chemin complet.")

async def generate_mask(
    image_ref: str,
    yolo_conf: float = 0.1,
    output_type: str = "overlay_red",
    alpha: float = 0.5,
):
    try:
        image_path = find_file(image_ref)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Dossiers vérifiés : " + ", ".join(str(d) for d in SEARCH_DIRS))
        return

    print(f"📂 Chargement de {image_path.name} depuis {image_path.parent}...")
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
    print(f"🔍 Détection YOLO (seuil={yolo_conf})...")
    yolo = VisionLoader.load_yolo()
    results = yolo(image_rgb, verbose=False, conf=yolo_conf)
    detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
            })
    print(f"   → {len(detections)} détection(s)")

    if not detections:
        print("⚠️ Aucune détection. Masque vide.")
        return

    # SAM2
    print("🔄 Segmentation SAM2...")
    sam = VisionLoader.load_sam2()
    sam.set_image(image_rgb)
    input_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    masks, _, _ = sam.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    combined_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
    for i in range(masks.shape[0]):
        mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
        mask_bin = (mask > 0).astype(np.uint8)
        combined_mask = cv2.bitwise_or(combined_mask, mask_bin)

    # Génération de la sortie
    output_dir = Path.cwd()
    base_name = image_path.stem

    if output_type == "binary":
        out_img = combined_mask * 255
        out_name = f"{base_name}_mask_binary.png"
    elif output_type == "overlay_red":
        overlay = image_rgb.copy()
        overlay[combined_mask > 0] = (0, 0, 255)
        out_img = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)
        out_name = f"{base_name}_mask_overlay_red.png"
    elif output_type == "overlay_green":
        overlay = image_rgb.copy()
        overlay[combined_mask > 0] = (0, 255, 0)
        out_img = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)
        out_name = f"{base_name}_mask_overlay_green.png"
    elif output_type == "contours":
        out_img = image_rgb.copy()
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out_img, contours, -1, (0, 255, 255), 2)
        out_name = f"{base_name}_mask_contours.png"
    elif output_type == "gray":
        out_img = (combined_mask * 255).astype(np.uint8)
        out_name = f"{base_name}_mask_gray.png"
    else:
        print(f"❌ Type de sortie inconnu : {output_type}")
        return

    cv2.imwrite(str(output_dir / out_name), out_img)
    print(f"💾 Masque sauvegardé sous : {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère un masque de segmentation avec YOLO + SAM2")
    parser.add_argument("image", help="Nom du fichier ou chemin (ex: mammo.png)")
    parser.add_argument("--conf", type=float, default=0.1, help="Seuil YOLO (0.0 - 1.0)")
    parser.add_argument("--type", choices=["binary", "overlay_red", "overlay_green", "contours", "gray"],
                        default="overlay_red", help="Type de masque")
    parser.add_argument("--alpha", type=float, default=0.5, help="Opacité (0.0 - 1.0)")
    args = parser.parse_args()

    asyncio.run(generate_mask(
        image_ref=args.image,
        yolo_conf=args.conf,
        output_type=args.type,
        alpha=args.alpha,
    ))