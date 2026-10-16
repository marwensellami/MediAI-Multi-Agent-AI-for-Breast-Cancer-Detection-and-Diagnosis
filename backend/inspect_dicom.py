# backend/inspect_dicom.py
import pydicom
from pathlib import Path
import json
import numpy as np

def inspect_dicom(file_path):
    """
    Affiche en détail toutes les métadonnées d'un fichier DICOM.
    """
    print(f"\n{'='*60}")
    print(f"🔍 INSPECTION DU FICHIER DICOM : {file_path}")
    print('='*60)

    try:
        ds = pydicom.dcmread(file_path, force=True)
    except Exception as e:
        print(f"❌ Erreur de lecture : {e}")
        return

    # 1. Informations générales
    print("\n📋 INFORMATIONS GÉNÉRALES")
    print("-" * 40)
    modality = ds.get('Modality', 'Inconnue')
    print(f"  Modality : {modality}")
    print(f"  Patient ID : {ds.get('PatientID', 'Non renseigné')}")
    print(f"  Patient Name : {ds.get('PatientName', 'Non renseigné')}")
    print(f"  Study Date : {ds.get('StudyDate', 'Non renseigné')}")
    print(f"  Study Description : {ds.get('StudyDescription', 'Non renseigné')}")
    print(f"  Series Description : {ds.get('SeriesDescription', 'Non renseigné')}")
    print(f"  Rows : {ds.get('Rows', 'Non renseigné')}")
    print(f"  Columns : {ds.get('Columns', 'Non renseigné')}")
    print(f"  Photometric Interpretation : {ds.get('PhotometricInterpretation', 'Non renseigné')}")
    print(f"  Samples per Pixel : {ds.get('SamplesPerPixel', 'Non renseigné')}")
    print(f"  Bits Allocated : {ds.get('BitsAllocated', 'Non renseigné')}")
    print(f"  Pixel Representation : {ds.get('PixelRepresentation', 'Non renseigné')}")

    # 2. Pixel Spacing (tous les tags possibles)
    print("\n📏 PIXEL SPACING")
    print("-" * 40)
    tags_pixel_spacing = [
        ('PixelSpacing', '0028,0030'),
        ('ImagerPixelSpacing', '0018,1164'),
        ('NominalScannedPixelSpacing', '0018,2010'),
        ('PixelSpacingCalibrationType', '0018,2002'),
        ('PixelSpacingCalibrationDescription', '0018,2004'),
    ]
    pixel_spacing_found = False
    for name, tag in tags_pixel_spacing:
        value = ds.get(name)
        if value is not None:
            print(f"  {name} ({tag}) : {value}")
            pixel_spacing_found = True
    if not pixel_spacing_found:
        print("  ⚠️ Aucun tag de PixelSpacing trouvé.")
        # Vérification dans les tags bruts (au cas où)
        for elem in ds:
            if 'spacing' in elem.name.lower():
                print(f"  🔍 Tag potentiel : {elem.tag} - {elem.name} = {elem.value}")

    # 3. Tous les tags (brut)
    print("\n📚 LISTE COMPLÈTE DES TAGS DICOM")
    print("-" * 40)
    for elem in ds:
        print(f"  {elem.tag} : {elem.name} = {elem.value}")

    # 4. Analyse de l'image
    print("\n🖼️ ANALYSE DE L'IMAGE")
    print("-" * 40)
    if hasattr(ds, 'pixel_array'):
        arr = ds.pixel_array
        print(f"  Shape : {arr.shape}")
        print(f"  Type : {arr.dtype}")
        print(f"  Min : {arr.min()}, Max : {arr.max()}")
        print(f"  Mean : {arr.mean():.2f}, Std : {arr.std():.2f}")
        # Vérifier si l'image est en niveaux de gris ou couleur
        if len(arr.shape) == 3:
            if arr.shape[2] == 3:
                print("  Image couleur (RGB)")
            else:
                print(f"  Image multicanaux ({arr.shape[2]} canaux)")
        else:
            print("  Image en niveaux de gris")
    else:
        print("  ⚠️ Pas de pixel_array trouvé (image compressée ?)")

    print("\n" + "="*60)

if __name__ == "__main__":
    # Vous pouvez spécifier le chemin en argument ou le modifier ici
    # Ex: python inspect_dicom.py app/data/uploads/mammo.dcm
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Entrez le chemin du fichier DICOM : ")
    inspect_dicom(Path(path))