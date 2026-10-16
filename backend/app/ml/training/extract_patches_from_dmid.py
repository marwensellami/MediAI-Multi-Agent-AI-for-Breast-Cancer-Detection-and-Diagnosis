# backend/app/ml/training/extract_patches_from_dmid.py
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
BASE_DIR = Path(__file__).parent.parent.parent.parent  # backend/
DMID_DIR = BASE_DIR / "app" / "data" / "DMID_PNG"
METADATA_PATH = DMID_DIR / "Metadata.xlsx"
TIFF_DIR = DMID_DIR / "1024" / "TIFF"
MASKS_DIR = DMID_DIR / "1024" / "Masks"
OUTPUT_DIR = BASE_DIR / "app" / "data" / "lesion_patches"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BENIGN_DIR = OUTPUT_DIR / "benign"
MALIGNANT_DIR = OUTPUT_DIR / "malignant"
NORMAL_DIR = OUTPUT_DIR / "normal"
for d in [BENIGN_DIR, MALIGNANT_DIR, NORMAL_DIR]:
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
    """Crée un masque du sein à partir d'une image en niveaux de gris."""
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

def extract_patch_from_bbox(image, bbox, target_size=(224, 224)):
    """Extrait un patch à partir d'une boîte (x1, y1, x2, y2) et le redimensionne."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    # Ajouter une marge de 10%
    h, w = patch.shape[:2]
    margin_h = int(0.1 * h)
    margin_w = int(0.1 * w)
    # on ne peut pas ajouter de marge car on a déjà la boîte exacte
    # on redimensionne simplement
    if len(patch.shape) == 3:
        patch = cv2.resize(patch, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        patch = cv2.resize(patch, target_size, interpolation=cv2.INTER_LINEAR)
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
    return patch

def extract_patch_from_mask_component(image, mask_component, target_size=(224, 224)):
    """Extrait un patch à partir d'un composant du masque (une lésion)."""
    coords = np.where(mask_component > 0)
    if len(coords[0]) == 0:
        return None
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    # Ajouter une marge de 10%
    margin = int(0.1 * max(y_max - y_min, x_max - x_min))
    margin = max(margin, 10)
    y1 = max(0, y_min - margin)
    y2 = min(image.shape[0], y_max + margin)
    x1 = max(0, x_min - margin)
    x2 = min(image.shape[1], x_max + margin)
    return extract_patch_from_bbox(image, (x1, y1, x2, y2), target_size)

def extract_patch_from_coords(image, center_x, center_y, radius, target_size=(224, 224)):
    """Extrait un patch à partir des coordonnées et du rayon."""
    h, w = image.shape[:2]
    cx = int(round(center_x))
    cy = int(round(center_y))
    rad = int(round(radius))
    if rad < 1:
        rad = max(h, w) // 20  # fallback
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
    """Retourne une liste de masques binaires pour chaque composant connexe."""
    num_labels, labels = cv2.connectedComponents(mask)
    components = []
    for i in range(1, num_labels):
        comp = (labels == i).astype(np.uint8) * 255
        if np.sum(comp) > 50:  # ignorer les très petits composants (bruit)
            components.append(comp)
    return components

def extract_random_normal_patches(image_gray, mask, num_patches=12, patch_size=(224, 224)):
    h, w = image_gray.shape
    patches = []
    for _ in range(num_patches * 3):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            break
        idx = np.random.randint(len(xs))
        cx, cy = xs[idx], ys[idx]
        half = patch_size[0] // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        if x2 - x1 < patch_size[0] or y2 - y1 < patch_size[1]:
            continue
        patch = image_gray[y1:y2, x1:x2]
        if patch.shape[0] != patch_size[0] or patch.shape[1] != patch_size[1]:
            patch = cv2.resize(patch, patch_size, interpolation=cv2.INTER_LINEAR)
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
        patches.append(patch_rgb)
        if len(patches) >= num_patches:
            break
    return patches

def main():
    logger.info("📂 Lecture du fichier Metadata.xlsx...")
    df = pd.read_excel(METADATA_PATH, header=None)
    logger.info(f"Nombre total de lignes : {len(df)}")
    df_data = df.iloc[31:].copy()
    logger.info(f"Lignes analysées (à partir de la ligne 32) : {len(df_data)}")

    # Nettoyage des colonnes
    col5 = df_data.iloc[:, 4].astype(str).str.strip().str.upper()
    col5 = col5.replace('NAN', '')
    col4 = df_data.iloc[:, 3].astype(str).str.strip().str.upper()

    normal_images = set()
    count_benign = 0
    count_malignant = 0
    count_normal = 0
    patch_counter = {}  # pour numéroter les patches par image

    # Dictionnaire pour stocker les lignes par image
    image_rows = {}
    for idx, row in df_data.iterrows():
        img_name = str(row[0]).strip()
        if img_name not in image_rows:
            image_rows[img_name] = []
        image_rows[img_name].append(row)

    logger.info("🔄 Extraction des patches de lésions...")
    for img_name, rows in image_rows.items():
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

        # Vérifier si c'est une image normale (aucune ligne avec classe B ou M)
        has_lesion = any(clean_class(r[4]) in ['B', 'M'] for r in rows)
        if not has_lesion:
            # Image normale
            normal_images.add(img_name)
            continue

        # Charger le masque
        mask_file = MASKS_DIR / f"{img_name}.png"
        mask = None
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None and np.any(mask > 0):
                # Trouver les composants connexes
                components = get_connected_components(mask)
                logger.info(f"🔍 {img_name} : {len(components)} composant(s) trouvé(s)")
                # Pour chaque composant, trouver la ligne la plus proche
                for comp in components:
                    # Coordonnées du centroïde du composant
                    coords = np.where(comp > 0)
                    cy = np.mean(coords[0])
                    cx = np.mean(coords[1])
                    # Trouver la ligne la plus proche
                    min_dist = float('inf')
                    best_row = None
                    for row in rows:
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
                    if best_row is not None:
                        classe = clean_class(best_row[4])
                        # Extraire le patch du composant
                        patch = extract_patch_from_mask_component(img, comp)
                        if patch is None:
                            logger.warning(f"Échec extraction patch pour {img_name} (classe {classe})")
                            continue
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
            else:
                # Masque absent ou vide : utiliser les coordonnées
                for row in rows:
                    classe = clean_class(row[4])
                    if classe not in ['B', 'M']:
                        continue
                    x = clean_coord(row[5])
                    y = clean_coord(row[6])
                    radius = clean_radius(row[7])
                    if x is None or y is None or radius is None or radius <= 0:
                        continue
                    patch = extract_patch_from_coords(img, x, y, radius)
                    if patch is None:
                        logger.warning(f"Échec extraction coordonnées pour {img_name} (classe {classe})")
                        continue
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

    logger.info(f"✅ Patches bénins : {count_benign}")
    logger.info(f"✅ Patches malins : {count_malignant}")

    # ==========================================
    # EXTRAIRE LES PATCHES NORMAUX
    # ==========================================
    logger.info("🔄 Extraction des patches normaux (tissu sain)...")
    for img_name in normal_images:
        img_file = TIFF_DIR / f"{img_name}.png"
        if not img_file.exists():
            img_file = TIFF_DIR / f"{img_name}.tiff"
            if not img_file.exists():
                logger.warning(f"Fichier image normal non trouvé : {img_name}")
                continue
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        breast_mask = get_breast_mask(gray)
        num_patches = 12
        patches = extract_random_normal_patches(gray, breast_mask, num_patches=num_patches, patch_size=(224, 224))
        for i, patch in enumerate(patches):
            out_filename = f"{img_name}_{i+1}.png"
            out_path = NORMAL_DIR / out_filename
            cv2.imwrite(str(out_path), patch)
            count_normal += 1

    logger.info(f"✅ Patches normaux : {count_normal}")

    logger.info("\n" + "="*50)
    logger.info("📊 RÉSUMÉ DE L'EXTRACTION")
    logger.info(f"   - Bénins : {count_benign}")
    logger.info(f"   - Malins : {count_malignant}")
    logger.info(f"   - Normaux : {count_normal}")
    logger.info(f"   - Total : {count_benign + count_malignant + count_normal}")
    logger.info("="*50)
    logger.info("✅ Extraction terminée !")

if __name__ == "__main__":
    main()