# backend/test_segmentation.py
import asyncio
import cv2
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime
import json

from app.ml.processors.image_reader import ImageProcessor
from app.ml.loaders.vision_loader import VisionLoader

# Import des agents
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
SegmentationAgent = load_agent("segmentation", "SegmentationAgent")
ClassificationAgent = load_agent("classification", "ClassificationAgent")

def normalize_label(label):
    """Normalise le label pour comparaison insensible à la casse et aux accents"""
    if not label:
        return ""
    label_lower = label.lower()
    # Gérer les accents
    if 'é' in label_lower:
        label_lower = label_lower.replace('é', 'e')
    if 'è' in label_lower:
        label_lower = label_lower.replace('è', 'e')
    return label_lower

def setup_output_directory():
    """Crée le dossier de sortie backend/detection_result"""
    output_dir = Path("backend/detection_result")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def log_message(log_file, message, console=True):
    """Écrit un message dans le fichier log et optionnellement dans la console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    
    if console:
        print(message)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(formatted_message + '\n')

async def test_boxes(image_path: str, output_dir: Path, log_file: Path):
    image_path = Path(image_path)
    if not image_path.exists():
        log_message(log_file, f"❌ Fichier non trouvé : {image_path}")
        return None

    log_message(log_file, f"\n{'='*60}")
    log_message(log_file, f"📂 Traitement de {image_path.name}...")
    
    # Lire l'image
    data = ImageProcessor.read_file(image_path)
    if not data["is_valid"]:
        log_message(log_file, f"❌ Erreur : {data.get('error')}")
        return None

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

    # 1. Détection YOLO
    log_message(log_file, "🔍 Détection YOLO...")
    detector = DetectionAgent(confidence_threshold=0.3)
    detections = await detector.process(image_rgb)
    log_message(log_file, f"   → {len(detections)} détection(s) brute(s)")

    if not detections:
        log_message(log_file, "⚠️ Aucune lésion détectée. Arrêt.")
        return None

    for det in detections:
        det["_original_image"] = image_rgb

    # 2. Segmentation par boîtes (fallback)
    log_message(log_file, "\n🟦 Segmentation par masques approximatifs (boîtes)")
    seg = SegmentationAgent()
    seg.sam_predictor = None  # force le fallback
    detections_box = await seg.process(image_rgb, detections.copy())
    log_message(log_file, f"   → {len(detections_box)} masques boîtes générés")

    # 3. Classification Swin
    log_message(log_file, "\n🏷️ Classification Swin...")
    classifier = ClassificationAgent()
    detections_box = await classifier.process(detections_box)

    # 4. Affichage des résultats détaillés
    log_message(log_file, "\n" + "="*60)
    log_message(log_file, "📊 RÉSULTATS (masques approximatifs)")
    log_message(log_file, "="*60)
    for i, det in enumerate(detections_box, 1):
        label = det.get("malignancy_label", "?")
        score = det.get("malignancy_score", 0)
        bbox = det.get("bbox", [0,0,0,0])
        area = np.sum(det.get("mask", np.zeros((1,1))) > 0) if det.get("mask") is not None else 0
        message = f"  Lésion {i}: label={label}, score={score:.3f}, bbox={bbox}, surface={area} px²"
        log_message(log_file, message)

    # 5. Sauvegarde des images et résultats
    log_message(log_file, "\n🖼️ Sauvegarde des résultats...")
    
    # Créer un dossier pour cette image
    image_name = image_path.stem
    image_output_dir = output_dir / image_name
    image_output_dir.mkdir(exist_ok=True)
    
    # Sauvegarder les métadonnées
    results_data = {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "timestamp": datetime.now().isoformat(),
        "total_lesions": len(detections_box),
        "lesions": []
    }
    
    overlay = image_rgb.copy()
    lesion_data = []
    
    for i, det in enumerate(detections_box, 1):
        mask = det.get("mask")
        label = det.get("malignancy_label", "?")
        label_normalized = normalize_label(label)
        bbox = det.get("bbox", [0,0,0,0])
        score = det.get("malignancy_score", 0)
        
        # Définir la couleur : ROUGE pour malin, VERT pour bénin
        if label_normalized == "malin":
            color = np.array([255, 0, 0], dtype=np.uint8)  # Rouge (BGR)
            bbox_color = (0, 0, 255)  # Rouge pour le rectangle
            label_type = "MALIN"
        else:
            color = np.array([0, 255, 0], dtype=np.uint8)  # Vert (BGR)
            bbox_color = (0, 255, 0)  # Vert pour le rectangle
            label_type = "BENIN"
        
        # Appliquer le masque avec la couleur appropriée
        if mask is not None:
            mask_bin = (mask > 0).astype(np.uint8)
            overlay[mask_bin > 0, :] = (overlay[mask_bin > 0, :] * 0.5 + color * 0.5)
            
            # Sauvegarder le masque individuel
            mask_filename = image_output_dir / f"lesion_{i}_{label_type}.png"
            cv2.imwrite(str(mask_filename), mask * 255)
            log_message(log_file, f"  💾 Masque sauvegardé : {mask_filename}")
        
        # Dessiner la boîte englobante
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Rectangle avec la couleur appropriée
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bbox_color, 2)
            
            # Fond pour le texte
            text = f"{label} {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(overlay, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 0, 0), -1)
            
            # Texte avec la couleur appropriée
            cv2.putText(overlay, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
        
        # Stocker les données de la lésion
        lesion_info = {
            "id": i,
            "label": label,
            "label_normalized": label_normalized,
            "score": float(score),
            "bbox": [int(x) for x in bbox] if bbox else None,
            "area": int(area) if mask is not None else 0,
            "type": label_type
        }
        lesion_data.append(lesion_info)
        results_data["lesions"].append(lesion_info)
    
    # Légende en bas de l'image
    cv2.putText(overlay, f"Resultats - {image_path.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(overlay, "■ Malin (Rouge)", (10, overlay.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(overlay, "■ Benin (Vert)", (10, overlay.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Sauvegarder l'image annotée
    output_image_path = image_output_dir / f"{image_name}_annotated.png"
    cv2.imwrite(str(output_image_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    log_message(log_file, f"  💾 Image annotée sauvegardée : {output_image_path}")
    
    # Sauvegarder les métadonnées en JSON
    json_path = image_output_dir / f"{image_name}_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    log_message(log_file, f"  💾 Données sauvegardées : {json_path}")
    
    # 6. Statistiques
    log_message(log_file, "\n📊 STATISTIQUES :")
    log_message(log_file, f"  Nombre de lesions : {len(detections_box)}")
    
    # Statistiques par type avec normalisation
    malin_count = 0
    benin_count = 0
    unknown_count = 0
    
    for d in detections_box:
        label = d.get("malignancy_label", "")
        label_normalized = normalize_label(label)
        
        if label_normalized == "malin":
            malin_count += 1
        elif label_normalized == "benin":
            benin_count += 1
        else:
            unknown_count += 1
            log_message(log_file, f"⚠️ Label non reconnu : '{label}' (normalisé: '{label_normalized}')")
    
    log_message(log_file, f"  Malin (rouge) : {malin_count}")
    log_message(log_file, f"  Benin (vert) : {benin_count}")
    if unknown_count > 0:
        log_message(log_file, f"  Non classifié : {unknown_count}")
    
    areas = [np.sum(det.get("mask", np.zeros((1,1))) > 0) for det in detections_box if det.get("mask") is not None]
    if areas:
        log_message(log_file, f"  Surface moyenne : {np.mean(areas):.0f} px²")
        if len(areas) > 1:
            log_message(log_file, f"  Surface min : {np.min(areas):.0f} px²")
            log_message(log_file, f"  Surface max : {np.max(areas):.0f} px²")
    
    log_message(log_file, f"\n✅ Traitement de {image_path.name} terminé.")
    
    return results_data

async def process_folder(folder_path: str):
    """Traite toutes les images du dossier spécifié"""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"❌ Dossier non trouvé : {folder_path}")
        return
    
    # Créer le dossier de sortie
    output_dir = setup_output_directory()
    log_file = output_dir / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Initialiser le fichier log
    log_message(log_file, f"🚀 DÉBUT DU TRAITEMENT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(log_file, f"📁 Dossier source : {folder_path}")
    log_message(log_file, f"📁 Dossier de sortie : {output_dir}")
    log_message(log_file, "="*60, console=False)
    
    # Trouver toutes les images
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(folder.glob(f'*{ext}'))
        image_files.extend(folder.glob(f'*{ext.upper()}'))
    
    # Dédupliquer
    image_files = list(set(image_files))
    
    if not image_files:
        log_message(log_file, f"❌ Aucune image trouvée dans {folder_path}")
        return
    
    log_message(log_file, f"📊 {len(image_files)} image(s) trouvée(s)")
    log_message(log_file, "="*60)
    
    # Traiter chaque image
    all_results = []
    successful = 0
    failed = 0
    
    for i, image_file in enumerate(image_files, 1):
        log_message(log_file, f"\n📌 Traitement {i}/{len(image_files)} : {image_file.name}")
        try:
            result = await test_boxes(image_file, output_dir, log_file)
            if result:
                all_results.append(result)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            log_message(log_file, f"❌ Erreur lors du traitement de {image_file.name}: {str(e)}")
            failed += 1
    
    # Résumé final
    log_message(log_file, "\n" + "="*60)
    log_message(log_file, "📊 RÉSUMÉ FINAL")
    log_message(log_file, "="*60)
    log_message(log_file, f"✅ Traitement terminé : {successful} réussie(s), {failed} échec(s)")
    log_message(log_file, f"📁 Résultats sauvegardés dans : {output_dir}")
    log_message(log_file, f"📄 Log sauvegardé dans : {log_file}")
    
    print(f"\n✅ Traitement terminé !")
    print(f"📁 Résultats sauvegardés dans : {output_dir}")
    print(f"📄 Log disponible dans : {log_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            asyncio.run(process_folder(path))
        else:
            # Mode fichier unique (backward compatibility)
            output_dir = setup_output_directory()
            log_file = output_dir / f"single_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            asyncio.run(test_boxes(path, output_dir, log_file))
    else:
        print("Usage: python test_segmentation.py <dossier_images>")
        print("Exemple: python test_segmentation.py app/data/DMID_PNG/1024/TIFF/")
        sys.exit(1)