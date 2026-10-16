# test_image_reader.py
import sys
from pathlib import Path
import logging

# Ajouter le chemin du projet si nécessaire
sys.path.append(str(Path(__file__).parent / "app"))

from app.ml.processors.image_reader import ImageProcessor

# Configurer les logs pour voir les détails
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# Chemin de l'image à tester
image_path = Path(r"C:\Users\msi\OneDrive\breast-cancer-ai-framework\backend\app\data\DMID_PNG\1024\TIFF\IMG001.png")

# Lire l'image avec calcul automatique du pixel spacing
result = ImageProcessor.read_file(image_path)

if result["is_valid"]:
    pixels = result["pixels"]
    metadata = result["metadata"]
    
    print("\n✅ IMAGE CHARGÉE AVEC SUCCÈS")
    print(f"   - Dimensions : {metadata['rows']} x {metadata['columns']}")
    print(f"   - Type : {metadata['dtype']}")
    print(f"   - DPI détecté : {metadata.get('dpi')}")
    print(f"   - Pixel spacing : {metadata['pixel_spacing_mm']:.4f} mm/pixel")
    print(f"   - Taille réelle de l'image : {metadata['columns'] * metadata['pixel_spacing_mm']:.2f} mm x {metadata['rows'] * metadata['pixel_spacing_mm']:.2f} mm")
else:
    print(f"❌ Erreur : {result.get('error')}")