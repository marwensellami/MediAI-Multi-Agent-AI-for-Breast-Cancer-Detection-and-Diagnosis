# backend/test_all_images_pipeline.py
"""
Exécute le pipeline complet (YOLO → SAM2 → Swin) sur toutes les images du dossier TIFF.
Génère un rapport CSV et optionnellement des images annotées.

Utilisation :
    python test_all_images_pipeline.py --output_dir ./results --save_images
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

def process_image(image_path, yolo, swin, sam_predictor, device, conf_threshold):
    """Exécute le pipeline sur une image et retourne les détections."""
    data = ImageProcessor.read_file(image_path)
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

    # YOLO detection
    results = yolo(image_rgb, conf=conf_threshold, verbose=False)
    detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = "Benign" if class_id == 0 else "Malignant"
            detections.append({"bbox": [x1, y1, x2, y2],
                               "confidence": conf,
                               "yolo_class": class_name})

    if not detections:
        return [], image_rgb

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

    return detections, image_rgb

def main():
    parser = argparse.ArgumentParser(description="Test pipeline complet sur toutes les images DMID")
    parser.add_argument("--input_dir", default=TIFF_DIR, help="Dossier contenant les images TIFF")
    parser.add_argument("--yolo", default=YOLO_MODEL_PATH, help="Chemin du modèle YOLO fine-tuné")
    parser.add_argument("--swin", default=SWIN_MODEL_PATH, help="Chemin du modèle Swin fine-tuné")
    parser.add_argument("--conf", type=float, default=0.29, help="Seuil de confiance YOLO")
    parser.add_argument("--output_dir", default="./pipeline_results", help="Dossier de sortie pour les résultats")
    parser.add_argument("--save_images", action="store_true", help="Sauvegarder les images annotées")
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

    results = []
    img_dir = output_dir / "annotated"
    if args.save_images:
        img_dir.mkdir(exist_ok=True)

    for img_path in tqdm(image_files, desc="Traitement des images"):
        start_time = time.time()
        detections, image_rgb = process_image(img_path, yolo, swin_model, sam_predictor, device, args.conf)
        elapsed = time.time() - start_time

        # Enregistrer les résultats
        row = {
            "image": img_path.name,
            "num_detections": len(detections) if detections else 0,
            "detection_details": [],
            "elapsed_seconds": elapsed
        }
        if detections:
            for i, det in enumerate(detections):
                row["detection_details"].append({
                    "index": i+1,
                    "yolo_class": det.get("yolo_class"),
                    "yolo_conf": det.get("confidence"),
                    "swin_class": det.get("swin_class"),
                    "swin_prob": det.get("swin_prob")
                })
        results.append(row)

        # Sauvegarde de l'image annotée
        if args.save_images and detections:
            overlay = image_rgb.copy()
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                swin_class = det.get("swin_class", "?")
                swin_prob = det.get("swin_prob", 0.0)
                color = (0, 255, 0) if swin_class == "Benign" else (0, 0, 255) if swin_class == "Malignant" else (255, 255, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                label = f"{swin_class} {swin_prob:.2f}"
                cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if "mask" in det:
                    mask = det["mask"]
                    overlay[mask > 0, :] = overlay[mask > 0, :] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
            out_path = img_dir / f"{img_path.stem}_annotated.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Sauvegarde du rapport CSV
    rows_csv = []
    for r in results:
        if r["detection_details"]:
            for d in r["detection_details"]:
                rows_csv.append({
                    "image": r["image"],
                    "detection_index": d["index"],
                    "yolo_class": d["yolo_class"],
                    "yolo_conf": d["yolo_conf"],
                    "swin_class": d["swin_class"],
                    "swin_prob": d["swin_prob"],
                    "elapsed_seconds": r["elapsed_seconds"]
                })
        else:
            rows_csv.append({
                "image": r["image"],
                "detection_index": 0,
                "yolo_class": "",
                "yolo_conf": "",
                "swin_class": "",
                "swin_prob": "",
                "elapsed_seconds": r["elapsed_seconds"]
            })

    df = pd.DataFrame(rows_csv)
    csv_path = output_dir / "pipeline_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Rapport sauvegardé sous {csv_path}")

    # Statistiques
    total_images = len(results)
    images_with_detections = sum(1 for r in results if r["num_detections"] > 0)
    total_detections = sum(r["num_detections"] for r in results)
    avg_time = df["elapsed_seconds"].mean() if not df.empty else 0

    print("\n📊 RÉSUMÉ :")
    print(f"   Images traitées : {total_images}")
    print(f"   Images avec détection : {images_with_detections}")
    print(f"   Total détections : {total_detections}")
    print(f"   Temps moyen par image : {avg_time:.2f} s")
    print(f"   Résultats sauvegardés dans {output_dir}")

if __name__ == "__main__":
    main()