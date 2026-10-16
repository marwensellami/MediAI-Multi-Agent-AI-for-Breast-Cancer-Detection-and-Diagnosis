# backend/test_pipeline_on_dmid.py
import sys
import logging
import torch
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import timm
import torchvision.transforms as transforms

from app.ml.training.train_swin_classifier import SwinClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# CHEMINS
# ============================================
BASE_DIR = Path(__file__).parent  # backend/
DMID_DIR = BASE_DIR / "app" / "data" / "DMID_PNG"
TIFF_DIR = DMID_DIR / "1024" / "TIFF"
METADATA_PATH = DMID_DIR / "Metadata.xlsx"
YOLO_MODEL_PATH = BASE_DIR / "app" / "ml" / "models" / "yolo_breast.pt"
SWIN_MODEL_PATH = BASE_DIR / "app" / "ml" / "models" / "swin_breast_3class.pth"


class PipelineTester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()
        self._load_metadata()

    def _load_models(self):
        # Charger YOLO
        logger.info("📥 Chargement de YOLO...")
        self.yolo = YOLO(str(YOLO_MODEL_PATH))
        logger.info("✅ YOLO chargé.")

        # Charger Swin
        logger.info("📥 Chargement de Swin...")
        backbone = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=0)
        self.swin_model = SwinClassifier(backbone, num_classes=3)
        state_dict = torch.load(SWIN_MODEL_PATH, map_location=self.device)
        self.swin_model.load_state_dict(state_dict)
        self.swin_model.to(self.device)
        self.swin_model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info("✅ Swin chargé.")

    def _load_metadata(self):
        """Charge Metadata.xlsx et identifie les images à tester."""
        df = pd.read_excel(METADATA_PATH, header=None)
        df_data = df.iloc[31:].copy()

        # Dictionnaire pour stocker les lignes par image
        image_rows = {}
        for idx, row in df_data.iterrows():
            img_name = str(row[0]).strip()
            if img_name not in image_rows:
                image_rows[img_name] = []
            # Colonne 5 = classe
            classe = str(row[4]).strip().upper()
            if classe in ['B', 'M', 'N']:
                image_rows[img_name].append({
                    'class': classe,
                    'x': row[5] if not pd.isna(row[5]) else None,
                    'y': row[6] if not pd.isna(row[6]) else None,
                    'radius': row[7] if not pd.isna(row[7]) else None,
                })
            else:
                # Si la colonne 5 est vide, on considère que c'est une image normale
                # (elle n'aura pas de ligne dans image_rows)
                pass

        # Déterminer les images à tester
        self.test_images = {}  # img_name -> 'type' ('lesion' ou 'normal')
        for img_name, rows in image_rows.items():
            has_b_or_m = any(r['class'] in ['B', 'M'] for r in rows)
            has_n = any(r['class'] == 'N' for r in rows)
            if has_b_or_m:
                self.test_images[img_name] = {'type': 'lesion', 'rows': rows}
            elif has_n:
                # Si seulement N, on exclut
                continue
            else:
                # Aucune ligne (image normale)
                self.test_images[img_name] = {'type': 'normal', 'rows': []}

        # Ajouter les images qui ne sont pas dans le dictionnaire (fichiers TIFF sans entrée)
        # Ce sont des normales
        for img_path in TIFF_DIR.glob("*.png"):
            img_name = img_path.stem
            if img_name not in self.test_images:
                self.test_images[img_name] = {'type': 'normal', 'rows': []}

        logger.info(f"📊 Images à tester : {len(self.test_images)}")
        lesion_count = sum(1 for v in self.test_images.values() if v['type'] == 'lesion')
        normal_count = sum(1 for v in self.test_images.values() if v['type'] == 'normal')
        logger.info(f"   - Avec lésions (B/M) : {lesion_count}")
        logger.info(f"   - Normales : {normal_count}")

    def extract_patch(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        patch = image[y1:y2, x1:x2]
        patch = cv2.resize(patch, (224, 224))
        return patch

    def classify_patch(self, patch):
        tensor = self.transform(patch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.swin_model(tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = outputs.argmax(1).item()
            prob = probs[0, pred_class].item()
        class_names = ["Normal", "Benign", "Malignant"]
        return class_names[pred_class], prob

    def process_image(self, img_path):
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Impossible de charger {img_path}")
            return None
        results = self.yolo(image, conf=0.29, verbose=False)
        detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = "Benign" if class_id == 0 else "Malignant"
                patch = self.extract_patch(image, (x1, y1, x2, y2))
                if patch is not None:
                    swin_class, swin_prob = self.classify_patch(patch)
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'yolo_class': class_name,
                        'yolo_conf': conf,
                        'swin_class': swin_class,
                        'swin_prob': swin_prob,
                    })
        return detections

    def run(self, output_file=None):
        logger.info(f"📂 Traitement de {len(self.test_images)} images...")

        results_summary = []
        total_images = 0
        images_with_detections = 0
        total_detections = 0

        # Pour les images avec lésions, on compte les correspondances
        lesion_images = [img for img, info in self.test_images.items() if info['type'] == 'lesion']
        normal_images = [img for img, info in self.test_images.items() if info['type'] == 'normal']

        # Traiter les images avec lésions
        for img_name in tqdm(lesion_images, desc="Images avec lésions"):
            img_path = TIFF_DIR / f"{img_name}.png"
            if not img_path.exists():
                img_path = TIFF_DIR / f"{img_name}.tiff"
                if not img_path.exists():
                    continue
            detections = self.process_image(img_path)
            if detections is None:
                continue
            total_images += 1
            if len(detections) > 0:
                images_with_detections += 1
                total_detections += len(detections)

            gt_rows = self.test_images[img_name]['rows']
            gt_classes = [r['class'] for r in gt_rows if r['class'] in ['B', 'M']]
            pred_classes = [d['swin_class'] for d in detections]
            # Vérifier si le nombre et les classes correspondent
            match = False
            if len(pred_classes) == len(gt_classes):
                gt_mapped = ['Benign' if c=='B' else 'Malignant' for c in gt_classes]
                if sorted(pred_classes) == sorted(gt_mapped):
                    match = True

            results_summary.append({
                'image': img_name,
                'type': 'lesion',
                'num_detections': len(detections),
                'gt_lesions': len(gt_classes),
                'match': match,
                'detections': detections,
                'gt': gt_rows,
            })

        # Traiter les images normales
        for img_name in tqdm(normal_images, desc="Images normales"):
            img_path = TIFF_DIR / f"{img_name}.png"
            if not img_path.exists():
                img_path = TIFF_DIR / f"{img_name}.tiff"
                if not img_path.exists():
                    continue
            detections = self.process_image(img_path)
            if detections is None:
                continue
            total_images += 1
            if len(detections) > 0:
                images_with_detections += 1
                total_detections += len(detections)

            # Une image normale ne devrait pas avoir de détections
            match = (len(detections) == 0)
            results_summary.append({
                'image': img_name,
                'type': 'normal',
                'num_detections': len(detections),
                'gt_lesions': 0,
                'match': match,
                'detections': detections,
                'gt': [],
            })

        # Statistiques
        logger.info("\n" + "="*50)
        logger.info("📊 RÉSUMÉ DU PIPELINE SUR DMID")
        logger.info(f"   Images traitées : {total_images}")
        logger.info(f"   Images avec détection : {images_with_detections}")
        logger.info(f"   Total détections : {total_detections}")

        # Taux de correspondance pour les images avec lésions
        lesion_matches = [r['match'] for r in results_summary if r['type'] == 'lesion' and r['gt_lesions'] > 0]
        if lesion_matches:
            match_rate = sum(lesion_matches) / len(lesion_matches) * 100
            logger.info(f"   Taux de correspondance (lésions) : {match_rate:.1f}%")
        else:
            logger.info("   Aucune image avec lésions.")

        # Taux de faux positifs sur les images normales
        normal_results = [r for r in results_summary if r['type'] == 'normal']
        if normal_results:
            false_positives = sum(1 for r in normal_results if r['num_detections'] > 0)
            fp_rate = false_positives / len(normal_results) * 100
            logger.info(f"   Faux positifs sur les images normales : {false_positives}/{len(normal_results)} ({fp_rate:.1f}%)")

        # Exemples
        logger.info("\n📋 Exemples de résultats :")
        for r in results_summary[:5]:
            logger.info(f"   {r['image']} ({r['type']}) : {r['num_detections']} détections (GT: {r['gt_lesions']}) - Match: {r['match']}")

        if output_file:
            df = pd.DataFrame([{
                'image': r['image'],
                'type': r['type'],
                'num_detections': r['num_detections'],
                'gt_lesions': r['gt_lesions'],
                'match': r['match']
            } for r in results_summary])
            df.to_csv(output_file, index=False)
            logger.info(f"💾 Rapport sauvegardé sous {output_file}")

        return results_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test pipeline YOLO + Swin sur DMID (sans images N)")
    parser.add_argument("--output", type=str, default=None, help="Fichier CSV de sortie")
    args = parser.parse_args()

    tester = PipelineTester()
    tester.run(output_file=args.output)