# backend/test_full_pipeline.py
"""
Test complet du pipeline : YOLO (fine‑tuné) → SAM2 → Swin (fine‑tuné)
Affiche les détections, les masques et les classifications.

Utilisation :
    python test_full_pipeline.py --image chemin/vers/mammo.png --output resultat.png
"""

import sys
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms

# Imports du projet
from app.ml.loaders.vision_loader import VisionLoader
from app.ml.processors.image_reader import ImageProcessor
from app.ml.training.train_swin_classifier import SwinClassifier
import timm

# ============================================
# CHEMINS PAR DÉFAUT
# ============================================
YOLO_MODEL_PATH = Path("app/ml/models/yolo_breast.pt")
SWIN_MODEL_PATH = Path("app/ml/models/swin_breast_3class.pth")

# ============================================
# FONCTIONS
# ============================================

def load_yolo(model_path=YOLO_MODEL_PATH):
    """Charge le modèle YOLO fine‑tuné."""
    from ultralytics import YOLO
    return YOLO(str(model_path))

def load_swin(model_path=SWIN_MODEL_PATH, device="cuda"):
    """Charge le modèle Swin fine‑tuné (3 classes)."""
    backbone = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=0)
    model = SwinClassifier(backbone, num_classes=3)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def preprocess_patch(patch, target_size=(224, 224)):
    """Redimensionne et normalise un patch pour Swin."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(patch)

def extract_patch(image, bbox):
    """Extrait un patch à partir d'une boîte [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]

def classify_patch(model, patch, device="cuda"):
    """Classifie un patch avec Swin, retourne (classe, probabilité)."""
    tensor = preprocess_patch(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(1).item()
        prob = probs[0, pred_class].item()
    class_names = ["Normal", "Benign", "Malignant"]
    return class_names[pred_class], prob

def main():
    parser = argparse.ArgumentParser(description="Test du pipeline complet YOLO+SAM2+Swin")
    parser.add_argument("--image", required=True, help="Chemin de l'image (DICOM, PNG, JPG)")
    parser.add_argument("--yolo", default=YOLO_MODEL_PATH, help="Chemin du modèle YOLO fine‑tuné")
    parser.add_argument("--swin", default=SWIN_MODEL_PATH, help="Chemin du modèle Swin fine‑tuné")
    parser.add_argument("--conf", type=float, default=0.3, help="Seuil de confiance YOLO")
    parser.add_argument("--output", default=None, help="Fichier de sortie pour l'image annotée")
    parser.add_argument("--device", default="cuda", help="cpu ou cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Utilisation du device : {device}")

    # 1. Charger les modèles
    print("📥 Chargement de YOLO...")
    yolo = load_yolo(args.yolo)
    print("✅ YOLO chargé.")

    print("📥 Chargement de Swin...")
    swin_model = load_swin(args.swin, device)
    print("✅ Swin chargé.")

    print("📥 Chargement de SAM2...")
    sam2_predictor = VisionLoader.load_sam2()
    if sam2_predictor is None:
        print("⚠️ SAM2 non disponible, la segmentation sera ignorée.")
    else:
        print("✅ SAM2 chargé.")

    # 2. Lire l'image
    print(f"📂 Lecture de l'image : {args.image}")
    data = ImageProcessor.read_file(args.image)
    if not data["is_valid"]:
        print("❌ Erreur de lecture de l'image.")
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

    print(f"📐 Dimensions de l'image : {image_rgb.shape}")

    # 3. Détection YOLO
    print("🔍 Détection YOLO...")
    results = yolo(image_rgb, conf=args.conf, verbose=False)
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
    print(f"   → {len(detections)} détection(s) trouvée(s).")

    if not detections:
        print("⚠️ Aucune lésion détectée.")
        # Sauvegarder l'image originale
        if args.output:
            cv2.imwrite(args.output, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            print(f"💾 Image sauvegardée sous {args.output}")
        return

    # 4. Pour chaque détection : segmentation SAM2 + classification Swin
    overlay = image_rgb.copy()
    if sam2_predictor is not None:
        print("🔄 Segmentation SAM2...")
        sam2_predictor.set_image(image_rgb)
        input_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # Ajouter les masques aux détections
        for i, det in enumerate(detections):
            if i < masks.shape[0]:
                mask = masks[i, 0, :, :] if masks.ndim == 4 else masks[i, :, :]
                mask_bin = (mask > 0).astype(np.uint8)
                det["mask"] = mask_bin
                if scores is not None and i < len(scores):
                    # Si c'est un tenseur PyTorch, on utilise .item()
                    if hasattr(scores[i], 'item'):
                        det["segmentation_score"] = scores[i].item()
                    else:
                        # Sinon, on tente la conversion standard
                        det["segmentation_score"] = float(scores[i])
                else:
                    det["segmentation_score"] = 0.5
        print("✅ Segmentation terminée.")

    # 5. Classification Swin
    print("🏷️ Classification Swin...")
    for det in detections:
        bbox = det["bbox"]
        patch = extract_patch(image_rgb, bbox)
        if patch is not None:
            swin_class, swin_prob = classify_patch(swin_model, patch, device)
            det["swin_class"] = swin_class
            det["swin_prob"] = swin_prob
            print(f"   Lésion {detections.index(det)+1}: YOLO={det['yolo_class']} (conf={det['confidence']:.2f}) → Swin: {swin_class} (prob={swin_prob:.3f})")
        else:
            det["swin_class"] = "Unknown"
            det["swin_prob"] = 0.0
            print(f"   Lésion {detections.index(det)+1}: Patch invalide (bbox hors limites)")

    # 6. Annoter l'image
    print("🖼️ Génération de l'image annotée...")
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        swin_class = det.get("swin_class", "?")
        swin_prob = det.get("swin_prob", 0.0)
        # Couleur selon la classe
        color = (0, 255, 0) if swin_class == "Benign" else (0, 0, 255) if swin_class == "Malignant" else (255, 255, 0)
        # Rectangle
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        # Étiquette
        label = f"{swin_class} {swin_prob:.2f}"
        cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Si SAM2 a produit un masque, on le superpose en transparence
        if "mask" in det:
            mask = det["mask"]
            overlay[mask > 0, :] = overlay[mask > 0, :] * 0.5 + np.array(color, dtype=np.uint8) * 0.5

    # 7. Sauvegarde
    if args.output is None:
        base = Path(args.image).stem
        output_path = f"{base}_pipeline_result.png"
    else:
        output_path = args.output
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"💾 Image annotée sauvegardée sous {output_path}")

    # 8. Résumé
    print("\n📊 RÉSUMÉ DES DÉTECTIONS :")
    for i, det in enumerate(detections, 1):
        print(f"   {i}: YOLO={det['yolo_class']} (conf={det['confidence']:.2f}) → Swin: {det.get('swin_class', '?')} (prob={det.get('swin_prob', 0.0):.3f})")

    print("✅ Pipeline terminé.")

if __name__ == "__main__":
    main()