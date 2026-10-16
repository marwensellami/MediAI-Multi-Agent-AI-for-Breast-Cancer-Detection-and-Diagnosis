# backend/app/ml/utils/metadata_reader.py
import pandas as pd
from pathlib import Path

def get_lesion_bbox_from_metadata(image_id: str, metadata_path: Path) -> list:
    """
    Récupère les coordonnées (bbox) d'une lésion à partir du Metadata.xlsx.
    Retourne [x1, y1, x2, y2] ou None.
    """
    if not metadata_path.exists():
        return None

    df = pd.read_excel(metadata_path, header=2)
    df.columns = ['image', 'view', 'tissue', 'type', 'class', 'x', 'y', 'radius']
    df['image'] = df['image'].astype(str).str.strip()

    # Filtrer pour l'image donnée, classes B ou M, avec coordonnées
    rows = df[(df['image'] == image_id) & (df['class'].isin(['B', 'M'])) & (df['x'].notna())]
    if rows.empty:
        return None

    # Prendre la première lésion (ou la plus grande)
    row = rows.iloc[0]
    x = int(row['x'])
    y = int(row['y'])
    radius = int(row['radius']) if not pd.isna(row['radius']) else 100

    # Boîte centrée
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = x + radius
    y2 = y + radius

    return [x1, y1, x2, y2]

def get_view_from_metadata(image_name: str) -> str:
    """
    Récupère la vue (CCLT, CCRT, MLOLT, MLORT) depuis le fichier Metadata.xlsx.
    """
    metadata_path = Path("app/data/DMID_PNG/Metadata.xlsx")
    
    if not metadata_path.exists():
        return None
    
    try:
        df = pd.read_excel(metadata_path, header=0)
        # La colonne A est l'identifiant de l'image
        # La colonne B est la vue
        # On commence à la ligne 32 (index 31 en Python)
        for idx, row in df.iterrows():
            if idx < 31:  # Ignorer l'en-tête et les lignes de description
                continue
            img_id = str(row.iloc[0]).strip()
            if img_id == image_name:
                view = str(row.iloc[1]).strip()
                if view and view != "-" and view != "nan":
                    return view
    except Exception as e:
        print(f"⚠️ Erreur de lecture des métadonnées: {e}")
    
    return None

def get_density_from_metadata(image_name: str) -> str:
    """
    Récupère la densité mammaire depuis Metadata.xlsx (colonne 3).
    
    Args:
        image_name: Nom de l'image (ex: "IMG001")
    
    Returns:
        Code de densité : F (Fatty), G (Fatty-glandular), D (Dense-glandular)
    """
    metadata_path = Path("app/data/DMID_PNG/Metadata.xlsx")
    if not metadata_path.exists():
        return None
    try:
        df = pd.read_excel(metadata_path, header=0)
        for idx, row in df.iterrows():
            if idx < 31:
                continue
            img_id = str(row.iloc[0]).strip()
            if img_id == image_name:
                density = str(row.iloc[2]).strip()  # Colonne C (index 2)
                if density and density != "-" and density != "nan":
                    return density
    except Exception as e:
        print(f"⚠️ Erreur de lecture des métadonnées: {e}")
    return None