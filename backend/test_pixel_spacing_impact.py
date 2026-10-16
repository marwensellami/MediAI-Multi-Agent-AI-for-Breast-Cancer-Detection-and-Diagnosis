# backend/test_pixel_spacing_impact.py
"""
Comparaison de l’impact du Pixel Spacing (0.1 vs 0.05 mm/pixel)
sur les mesures de taille et la classification.

Exécution :
    python test_pixel_spacing_impact.py --output_dir ./pixel_spacing_comparison
"""

import sys
import argparse
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
import torchvision.transforms as transforms

from app.ml.loaders.vision_loader import VisionLoader
from app.ml.processors.image_reader import ImageProcessor
from app.ml.training.train_swin_classifier import SwinClassifier
import timm

# ============================================
# CHEMINS PAR DÉFAUT
# ============================================
YOLO_MODEL_PATH = Path("app/ml/models/yolo_breast.pt")
SWIN_MODEL_PATH = Path("app/ml/models/swin_breast_3class.pth")
TIFF_DIR = Path("app/data/DMID_PNG/1024/TIFF")

def load_yolo(model_path=YOLO_MODEL_PATH):
    from ultralytics import YOLO
    return YOLO(str(model_path))

def load_swin(model_path=SWIN_MODEL_PATH, device="cuda"):
    backbone = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=0)
    model = SwinClassifier(backbone, num_classes=3)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def preprocess_patch(patch, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(patch)

def extract_patch(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]

def classify_patch(model, patch, device="cuda"):
    tensor = preprocess_patch(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(1).item()
        prob = probs[0, pred_class].item()
    class_names = ["Normal", "Benign", "Malignant"]
    return class_names[pred_class], prob

def process_image(image_path, pixel_spacing_mm, yolo, swin, sam_predictor, device, conf_threshold):
    """Exécute le pipeline sur une image avec un pixel_spacing donné."""
    # Lecture avec le pixel_spacing forcé
    data = ImageProcessor.read_file(image_path, pixel_spacing_mm=pixel_spacing_mm)
    if not data["is_valid"]:
        return None, f"Erreur lecture: {data.get('error')}"

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
    results = yolo(image_rgb, conf=conf_threshold, verbose=False)
    detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = "Benign" if class_id == 0 else "Malignant"
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "yolo_class": class_name,
            })

    if not detections:
        return [], image_rgb, data["metadata"]["pixel_spacing_mm"]

    # SAM2 segmentation (if available)
    if sam_predictor is not None:
        sam_predictor.set_image(image_rgb)
        input_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        masks, scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        for i, det in enumerate(detections):
            if i < masks.shape[0]:
                mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
                det["mask"] = (mask > 0).astype(np.uint8)
                if scores is not None and i < len(scores):
                    det["segmentation_score"] = float(scores[i].item()) if hasattr(scores[i], 'item') else float(scores[i])
                else:
                    det["segmentation_score"] = 0.5

    # Swin classification
    for det in detections:
        patch = extract_patch(image_rgb, det["bbox"])
        if patch is not None:
            swin_class, swin_prob = classify_patch(swin, patch, device)
            det["swin_class"] = swin_class
            det["swin_prob"] = swin_prob
        else:
            det["swin_class"] = "Unknown"
            det["swin_prob"] = 0.0

    # Ajouter la taille en mm (utiliser le pixel_spacing réel de la métadonnée)
    pixel_spacing = data["metadata"]["pixel_spacing_mm"]
    for det in detections:
        bbox = det["bbox"]
        width_px = bbox[2] - bbox[0]
        height_px = bbox[3] - bbox[1]
        det["size_mm"] = max(width_px, height_px) * pixel_spacing
        det["pixel_spacing_used"] = pixel_spacing

    return detections, image_rgb, pixel_spacing

def main():
    parser = argparse.ArgumentParser(description="Comparaison de l'impact du Pixel Spacing (0.1 vs 0.05)")
    parser.add_argument("--input_dir", default=TIFF_DIR, help="Dossier contenant les images TIFF")
    parser.add_argument("--yolo", default=YOLO_MODEL_PATH, help="Chemin du modèle YOLO fine-tuné")
    parser.add_argument("--swin", default=SWIN_MODEL_PATH, help="Chemin du modèle Swin fine-tuné")
    parser.add_argument("--conf", type=float, default=0.29, help="Seuil de confiance YOLO")
    parser.add_argument("--output_dir", default="./pixel_spacing_comparison", help="Dossier de sortie")
    parser.add_argument("--device", default="cuda", help="cpu ou cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Utilisation du device : {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chargement des modèles
    print("📥 Chargement des modèles...")
    yolo = load_yolo(args.yolo)
    swin_model = load_swin(args.swin, device)
    sam_predictor = VisionLoader.load_sam2()
    if sam_predictor is None:
        print("⚠️ SAM2 non disponible – segmentation ignorée.")
    else:
        print("✅ SAM2 chargé.")

    # Récupération des images
    input_dir = Path(args.input_dir)
    image_files = sorted(input_dir.glob("*.png"))
    if not image_files:
        image_files = sorted(input_dir.glob("*.tiff"))
    print(f"📂 {len(image_files)} images trouvées.")

    # Liste des pixel spacings à tester
    pixel_spacings = [0.1, 0.05]

    results = []

    for img_path in tqdm(image_files, desc="Traitement des images"):
        for ps in pixel_spacings:
            detections, image_rgb, used_ps = process_image(
                img_path, ps, yolo, swin_model, sam_predictor, device, args.conf
            )
            if detections is None:
                continue
            # Enregistrer les résultats
            for i, det in enumerate(detections):
                results.append({
                    "image": img_path.name,
                    "pixel_spacing": used_ps,
                    "detection_index": i+1,
                    "yolo_class": det.get("yolo_class"),
                    "yolo_conf": det.get("confidence"),
                    "swin_class": det.get("swin_class"),
                    "swin_prob": det.get("swin_prob"),
                    "bbox_width_px": det["bbox"][2] - det["bbox"][0],
                    "bbox_height_px": det["bbox"][3] - det["bbox"][1],
                    "size_mm": det.get("size_mm"),
                })

    # Sauvegarde du CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / "pixel_spacing_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Rapport sauvegardé sous {csv_path}")

    # Statistiques comparatives
    if not df.empty:
        print("\n📊 COMPARAISON DES PIXEL SPACINGS")
        for ps in pixel_spacings:
            sub = df[df["pixel_spacing"] == ps]
            print(f"\n🔹 Pixel spacing = {ps} mm/pixel")
            print(f"   - Nombre de détections : {len(sub)}")
            if not sub.empty:
                print(f"   - Taille moyenne (mm) : {sub['size_mm'].mean():.2f}")
                print(f"   - Écart-type : {sub['size_mm'].std():.2f}")
                print(f"   - Classes Swin :")
                for cls in sub["swin_class"].unique():
                    cnt = (sub["swin_class"] == cls).sum()
                    print(f"        {cls}: {cnt} ({cnt/len(sub)*100:.1f}%)")
        # Comparaison directe sur les images communes
        print("\n🔹 COMPARAISON POUR LES IMAGES AVEC DÉTECTIONS DANS LES DEUX CAS")
        # Grouper par image
        grouped = df.groupby("image")
        common_images = []
        for img, group in grouped:
            if set(group["pixel_spacing"]) == set(pixel_spacings):
                common_images.append(img)
        if common_images:
            print(f"   Images communes : {len(common_images)}")
            # Calculer la différence de taille moyenne pour ces images
            diff_sizes = []
            for img in common_images:
                sub = df[df["image"] == img]
                size_01 = sub[sub["pixel_spacing"] == 0.1]["size_mm"].mean()
                size_005 = sub[sub["pixel_spacing"] == 0.05]["size_mm"].mean()
                if not np.isnan(size_01) and not np.isnan(size_005):
                    diff_sizes.append(size_01 - size_005)
            if diff_sizes:
                print(f"   Différence moyenne de taille (0.1 - 0.05) : {np.mean(diff_sizes):.2f} mm")
        else:
            print("   Aucune image commune avec détections dans les deux cas.")
    else:
        print("⚠️ Aucune détection enregistrée.")

    print("\n✅ Analyse terminée.")

if __name__ == "__main__":
    main()