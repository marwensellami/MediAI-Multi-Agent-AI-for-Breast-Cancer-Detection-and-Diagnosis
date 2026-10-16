# backend/app/ml/training/test_extract.py
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# CHEMINS
# ============================================
BASE_DIR = Path(__file__).parent.parent.parent.parent
DMID_DIR = BASE_DIR / "app" / "data" / "DMID_PNG"
METADATA_PATH = DMID_DIR / "Metadata.xlsx"
TIFF_DIR = DMID_DIR / "1024" / "TIFF"
MASKS_DIR = DMID_DIR / "1024" / "Masks"
OUTPUT_DIR = BASE_DIR / "app" / "data" / "lesion_patches_test"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BENIGN_DIR = OUTPUT_DIR / "benign"
MALIGNANT_DIR = OUTPUT_DIR / "malignant"
for d in [BENIGN_DIR, MALIGNANT_DIR]:
    d.mkdir(exist_ok=True)

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

def get_breast_mask(image_gray):
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask, dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask

def extract_patch_from_mask_component(image, mask_component, target_size=(224, 224)):
    coords = np.where(mask_component > 0)
    if len(coords[0]) == 0:
        return None
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    margin = int(0.1 * max(y_max - y_min, x_max - x_min))
    margin = max(margin, 10)
    y1 = max(0, y_min - margin)
    y2 = min(image.shape[0], y_max + margin)
    x1 = max(0, x_min - margin)
    x2 = min(image.shape[1], x_max + margin)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    if len(patch.shape) == 3:
        patch = cv2.resize(patch, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        patch = cv2.resize(patch, target_size, interpolation=cv2.INTER_LINEAR)
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
    return patch

def extract_patch_from_coords(image, center_x, center_y, radius, target_size=(224, 224)):
    h, w = image.shape[:2]
    cx = int(round(center_x))
    cy = int(round(center_y))
    rad = int(round(radius))
    if rad < 1:
        rad = max(h, w) // 20
    half_side = rad
    x1 = max(0, cx - half_side)
    y1 = max(0, cy - half_side)
    x2 = min(w, cx + half_side)
    y2 = min(h, cy + half_side)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    if len(patch.shape) == 3:
        patch = cv2.resize(patch, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        patch = cv2.resize(patch, target_size, interpolation=cv2.INTER_LINEAR)
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
    return patch

def get_connected_components(mask):
    num_labels, labels = cv2.connectedComponents(mask)
    components = []
    for i in range(1, num_labels):
        comp = (labels == i).astype(np.uint8) * 255
        if np.sum(comp) > 50:
            components.append(comp)
    return components

def extract_patches_for_image(img, rows, mask=None):
    """
    Extrait les patches pour une image donnée.
    rows : liste de pandas.Series (chaque ligne = une lésion)
    """
    patches = []
    processed_indices = []  # on stocke les indices des lignes déjà traitées

    # 1. Si un masque est fourni, tenter d'extraire par composant
    if mask is not None and np.any(mask > 0):
        components = get_connected_components(mask)
        logger.info(f"  {len(components)} composant(s) trouvé(s) dans le masque.")
        for comp in components:
            coords = np.where(comp > 0)
            cy = np.mean(coords[0])
            cx = np.mean(coords[1])
            # Trouver la ligne la plus proche non encore traitée
            min_dist = float('inf')
            best_row = None
            best_idx = None
            for idx, row in rows.iterrows():
                if idx in processed_indices:
                    continue
                classe = clean_class(row[4])
                if classe not in ['B', 'M']:
                    continue
                x_meta = clean_coord(row[5])
                y_meta = clean_coord(row[6])
                if x_meta is None or y_meta is None:
                    continue
                dist = np.sqrt((cx - x_meta)**2 + (cy - y_meta)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_row = row
                    best_idx = idx
            if best_row is not None:
                classe = clean_class(best_row[4])
                patch = extract_patch_from_mask_component(img, comp)
                if patch is not None:
                    processed_indices.append(best_idx)
                    patches.append((patch, classe))
                    logger.info(f"  ✅ Patch extrait du masque pour {best_row[0]} (classe {classe})")
                else:
                    logger.warning(f"  Échec extraction du patch pour {best_row[0]}")
            else:
                logger.warning(f"  Aucune ligne correspondante pour un composant.")

    # 2. Pour les lignes restantes (non traitées), utiliser les coordonnées
    remaining_rows = []
    for idx, row in rows.iterrows():
        if idx not in processed_indices and clean_class(row[4]) in ['B', 'M']:
            remaining_rows.append((idx, row))

    if remaining_rows:
        logger.info(f"  {len(remaining_rows)} ligne(s) restante(s) -> extraction par coordonnées.")
        for idx, row in remaining_rows:
            classe = clean_class(row[4])
            x = clean_coord(row[5])
            y = clean_coord(row[6])
            radius = clean_radius(row[7])
            if x is None or y is None or radius is None or radius <= 0:
                logger.warning(f"  Coordonnées invalides pour {row[0]}, ignoré.")
                continue
            patch = extract_patch_from_coords(img, x, y, radius)
            if patch is not None:
                processed_indices.append(idx)
                patches.append((patch, classe))
                logger.info(f"  ✅ Patch extrait par coordonnées pour {row[0]} (classe {classe})")
            else:
                logger.warning(f"  Échec extraction coordonnées pour {row[0]}")

    return patches

# ============================================
# SCRIPT PRINCIPAL
# ============================================
def main():
    logger.info("📂 Lecture du fichier Metadata.xlsx...")
    df = pd.read_excel(METADATA_PATH, header=None)
    logger.info(f"Nombre total de lignes : {len(df)}")

    target_images = ['IMG140', 'IMG065']
    df_data = df.iloc[31:].copy()
    mask = df_data.iloc[:, 0].astype(str).str.strip().isin(target_images)
    selected_rows = df_data[mask]

    logger.info(f"Lignes sélectionnées : {len(selected_rows)}")
    if selected_rows.empty:
        logger.error("Aucune ligne correspondante trouvée.")
        return

    for idx, row in selected_rows.iterrows():
        logger.info(f"  {row[0]} - classe: {clean_class(row[4])}, coord: {row[5]}, {row[6]}, rayon: {row[7]}")

    image_rows = {}
    for idx, row in selected_rows.iterrows():
        img_name = str(row[0]).strip()
        if img_name not in image_rows:
            image_rows[img_name] = pd.DataFrame(columns=row.index)  # structure vide
        # on ajoute la ligne à la liste (on utilise une liste de Series)
        # Pour simplifier, on stocke une liste de Series
    # On refait le groupement proprement
    image_rows = {}
    for idx, row in selected_rows.iterrows():
        img_name = str(row[0]).strip()
        if img_name not in image_rows:
            image_rows[img_name] = []
        image_rows[img_name].append(row)

    count_benign = 0
    count_malignant = 0
    patch_counter = {}

    for img_name, rows in image_rows.items():
        logger.info(f"🔄 Traitement de {img_name}...")
        img_file = TIFF_DIR / f"{img_name}.png"
        if not img_file.exists():
            img_file = TIFF_DIR / f"{img_name}.tiff"
            if not img_file.exists():
                logger.warning(f"Fichier image non trouvé : {img_name}")
                continue
        img = cv2.imread(str(img_file))
        if img is None:
            logger.warning(f"Impossible de charger l'image : {img_file}")
            continue

        mask_file = MASKS_DIR / f"{img_name}.png"
        mask = None
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None and np.any(mask > 0):
                logger.info(f"  Masque trouvé pour {img_name}")
            else:
                mask = None
                logger.info(f"  Masque vide pour {img_name}")
        else:
            logger.info(f"  Masque absent pour {img_name}")

        # Convertir la liste de rows en DataFrame pour itérer avec iterrows() plus facilement
        # Mais on a déjà une liste, on peut itérer directement avec un index
        # On va créer un petit DataFrame pour utiliser iterrows
        rows_df = pd.DataFrame(rows)

        patches = extract_patches_for_image(img, rows_df, mask)

        if not patches:
            logger.warning(f"  Aucun patch extrait pour {img_name}")
            continue

        for patch, classe in patches:
            dest_dir = BENIGN_DIR if classe == 'B' else MALIGNANT_DIR
            if classe == 'B':
                count_benign += 1
            else:
                count_malignant += 1
            if img_name not in patch_counter:
                patch_counter[img_name] = 1
            else:
                patch_counter[img_name] += 1
            suffix = patch_counter[img_name]
            out_filename = f"{img_name}_{suffix}.png"
            out_path = dest_dir / out_filename
            cv2.imwrite(str(out_path), patch)
            logger.info(f"  💾 Patch sauvegardé : {out_path}")

    logger.info("="*50)
    logger.info("📊 RÉSUMÉ DE L'EXTRACTION DE TEST")
    logger.info(f"   - Bénins : {count_benign}")
    logger.info(f"   - Malins : {count_malignant}")
    logger.info("="*50)
    logger.info("✅ Test terminé.")

if __name__ == "__main__":
    main()