# backend/app/ml/training/prepare_yolo_dataset.py
import os
import cv2
import numpy as np
import pandas as pd
import shutil
import random
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# CHEMINS
# ============================================
BASE_DIR = Path(__file__).parent.parent.parent.parent  # backend/
DMID_DIR = BASE_DIR / "app" / "data" / "DMID_PNG"
METADATA_PATH = DMID_DIR / "Metadata.xlsx"
TIFF_DIR = DMID_DIR / "1024" / "TIFF"
MASKS_DIR = DMID_DIR / "1024" / "Masks"
YOLO_DIR = BASE_DIR / "yolo_dataset"
IMAGES_TRAIN = YOLO_DIR / "images" / "train"
IMAGES_VAL = YOLO_DIR / "images" / "val"
LABELS_TRAIN = YOLO_DIR / "labels" / "train"
LABELS_VAL = YOLO_DIR / "labels" / "val"

for d in [YOLO_DIR, IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def clean_class(value):
    if pd.isna(value):
        return ''
    return str(value).strip().upper()

def clean_coord(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def get_connected_components(mask):
    """Retourne une liste de masques binaires pour chaque composant connexe."""
    num_labels, labels = cv2.connectedComponents(mask)
    components = []
    for i in range(1, num_labels):
        comp = (labels == i).astype(np.uint8) * 255
        if np.sum(comp) > 50:  # ignorer les très petits composants (bruit)
            components.append(comp)
    return components

def mask_to_yolo_bbox(mask, img_shape):
    """
    Calcule la boîte englobante du masque et la convertit au format YOLO.
    La boîte est recadrée aux limites de l'image.
    Retourne (x_center, y_center, width, height) normalisés entre 0 et 1.
    """
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    h, w = img_shape[:2]

    # --- RECADRAGE AUX LIMITES DE L'IMAGE ---
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w - 1, x_max)
    y_max = min(h - 1, y_max)

    # Si la boîte devient invalide, on la rejette
    if x_max <= x_min or y_max <= y_min:
        return None

    x_center = (x_min + x_max) / 2.0 / w
    y_center = (y_min + y_max) / 2.0 / h
    width = (x_max - x_min) / w
    height = (y_max - y_min) / h

    # Sécurité finale (pour arrondis)
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    if width <= 0 or height <= 0:
        return None
    return x_center, y_center, width, height


def coord_to_yolo_bbox(center_x, center_y, radius, img_shape):
    """Convertit les coordonnées (X,Y, rayon) en boîte YOLO, avec recadrage."""
    h, w = img_shape[:2]
    half = radius
    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(w - 1, center_x + half)
    y2 = min(h - 1, center_y + half)

    if x2 <= x1 or y2 <= y1:
        return None

    x_center = (x1 + x2) / 2.0 / w
    y_center = (y1 + y2) / 2.0 / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h

    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    if width <= 0 or height <= 0:
        return None
    return x_center, y_center, width, height

# ============================================
# SCRIPT PRINCIPAL
# ============================================

def main():
    logger.info("📂 Lecture du fichier Metadata.xlsx...")
    df = pd.read_excel(METADATA_PATH, header=None)
    logger.info(f"Nombre total de lignes : {len(df)}")
    df_data = df.iloc[31:].copy()
    logger.info(f"Lignes analysées (à partir de la ligne 32) : {len(df_data)}")

    # Grouper les lignes par image
    image_rows = {}
    for idx, row in df_data.iterrows():
        img_name = str(row[0]).strip()
        if img_name not in image_rows:
            image_rows[img_name] = []
        image_rows[img_name].append(row)

    # Décider quelles images inclure
    included_images = []  # liste de (img_name, has_lesion)
    for img_name, rows in image_rows.items():
        # Vérifier s'il y a au moins une ligne B ou M
        has_b_or_m = any(clean_class(r[4]) in ['B', 'M'] for r in rows)
        if has_b_or_m:
            included_images.append((img_name, True))
        else:
            # Vérifier si l'image est normale (pas de N, pas de ligne)
            has_n = any(clean_class(r[4]) == 'N' for r in rows)
            if has_n:
                # Exclure les images avec uniquement des N
                logger.info(f"⏭️ Exclusion de {img_name} (uniquement lésions non définies N)")
                continue
            else:
                # Image normale (pas de lésion)
                included_images.append((img_name, False))

    logger.info(f"📊 Nombre d'images incluses : {len(included_images)}")

    # Split train/val (80/20)
    random.seed(42)
    random.shuffle(included_images)
    split_idx = int(0.8 * len(included_images))
    train_images = included_images[:split_idx]
    val_images = included_images[split_idx:]
    logger.info(f"📊 Train : {len(train_images)}, Val : {len(val_images)}")

    # Traitement des images
    processed = {"train": 0, "val": 0}

    for subset, image_list in [("train", train_images), ("val", val_images)]:
        images_dir = IMAGES_TRAIN if subset == "train" else IMAGES_VAL
        labels_dir = LABELS_TRAIN if subset == "train" else LABELS_VAL

        for img_name, has_lesion in image_list:
            # Copier l'image
            src_img = TIFF_DIR / f"{img_name}.png"
            if not src_img.exists():
                src_img = TIFF_DIR / f"{img_name}.tiff"
                if not src_img.exists():
                    logger.warning(f"Image introuvable pour {img_name}")
                    continue
            dst_img = images_dir / f"{img_name}.png"
            shutil.copy2(src_img, dst_img)

            if has_lesion:
                # Générer les boîtes pour les lésions B et M
                # Récupérer les lignes avec B ou M
                lesion_rows = [r for r in image_rows[img_name] if clean_class(r[4]) in ['B', 'M']]
                # Charger le masque
                mask_file = MASKS_DIR / f"{img_name}.png"
                if mask_file.exists():
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    if mask is not None and np.any(mask > 0):
                        components = get_connected_components(mask)
                        # Associer chaque composant à la ligne la plus proche
                        boxes = []
                        for comp in components:
                            coords = np.where(comp > 0)
                            cy = np.mean(coords[0])
                            cx = np.mean(coords[1])
                            # Trouver la ligne la plus proche (distance minimale)
                            min_dist = float('inf')
                            best_row = None
                            for row in lesion_rows:
                                x_meta = clean_coord(row[5])
                                y_meta = clean_coord(row[6])
                                if x_meta is None or y_meta is None:
                                    continue
                                dist = np.sqrt((cx - x_meta)**2 + (cy - y_meta)**2)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_row = row
                            if best_row is not None:
                                classe = clean_class(best_row[4])
                                class_id = 0 if classe == 'B' else 1
                                # Calculer la boîte à partir du composant
                                bbox = mask_to_yolo_bbox(comp, (mask.shape[0], mask.shape[1]))
                                if bbox is not None:
                                    x_center, y_center, width, height = bbox
                                    boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                        # Si on n'a pas de boîtes, utiliser les coordonnées (fallback)
                        if not boxes:
                            logger.warning(f"⚠️ Pas de boîtes à partir du masque pour {img_name}, fallback coordonnées")
                            for row in lesion_rows:
                                x = clean_coord(row[5])
                                y = clean_coord(row[6])
                                radius = clean_radius(row[7])  # on a déjà clean_radius dans le script précédent
                                # on va le définir localement
                                if x is None or y is None or radius is None:
                                    continue
                                bbox = coord_to_yolo_bbox(x, y, radius, (mask.shape[0], mask.shape[1]))
                                if bbox is not None:
                                    classe = clean_class(row[4])
                                    class_id = 0 if classe == 'B' else 1
                                    x_center, y_center, width, height = bbox
                                    boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    else:
                        # Masque absent ou vide, fallback sur coordonnées
                        boxes = []
                        for row in lesion_rows:
                            x = clean_coord(row[5])
                            y = clean_coord(row[6])
                            radius = clean_radius(row[7])
                            if x is None or y is None or radius is None:
                                continue
                            # On a besoin de la taille de l'image
                            img = cv2.imread(str(src_img))
                            if img is None:
                                continue
                            bbox = coord_to_yolo_bbox(x, y, radius, img.shape)
                            if bbox is not None:
                                classe = clean_class(row[4])
                                class_id = 0 if classe == 'B' else 1
                                x_center, y_center, width, height = bbox
                                boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                else:
                    # Pas de masque, fallback coordonnées
                    boxes = []
                    for row in lesion_rows:
                        x = clean_coord(row[5])
                        y = clean_coord(row[6])
                        radius = clean_radius(row[7])
                        if x is None or y is None or radius is None:
                            continue
                        img = cv2.imread(str(src_img))
                        if img is None:
                            continue
                        bbox = coord_to_yolo_bbox(x, y, radius, img.shape)
                        if bbox is not None:
                            classe = clean_class(row[4])
                            class_id = 0 if classe == 'B' else 1
                            x_center, y_center, width, height = bbox
                            boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                # Écrire le fichier .txt
                label_path = labels_dir / f"{img_name}.txt"
                with open(label_path, 'w') as f:
                    for line in boxes:
                        f.write(line + '\n')
                if not boxes:
                    logger.warning(f"⚠️ Aucune boîte générée pour {img_name} (pourtant B/M)")
            else:
                # Image normale : fichier .txt vide
                label_path = labels_dir / f"{img_name}.txt"
                with open(label_path, 'w') as f:
                    pass  # fichier vide

            processed[subset] += 1

    logger.info(f"✅ Traitement terminé : {processed}")

    # ============================================
    # CRÉATION DE dataset.yaml
    # ============================================
    yaml_path = YOLO_DIR / "dataset.yaml"
    yaml_content = f"""
path: {YOLO_DIR.absolute().as_posix()}
train: images/train
val: images/val

nc: 2
names: ['benign', 'malignant']
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    logger.info(f"📄 Fichier dataset.yaml créé : {yaml_path}")

    logger.info("✅ Script terminé avec succès !")

# ============================================
# DÉFINIR clean_radius (fallback)
# ============================================
def clean_radius(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == '-' or s == '':
        return None
    try:
        return float(s)
    except ValueError:
        return None

if __name__ == "__main__":
    main()