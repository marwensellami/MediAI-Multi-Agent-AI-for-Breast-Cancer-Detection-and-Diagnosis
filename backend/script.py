#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse complète d'une image (PNG, JPG, JPEG, TIFF).
Calcule la distance physique mm/pixel.
Permet de spécifier le DPI via argument ou saisie interactive.
"""

import os
import sys
import argparse
from PIL import Image, ExifTags

# ----------------------------------------------------------------------
# Lecture du chunk pHYs pour PNG (via pypng si installé)
# ----------------------------------------------------------------------
def get_phys_from_png_pypng(image_path):
    try:
        import png
    except ImportError:
        return (None, None)

    try:
        with open(image_path, 'rb') as f:
            reader = png.Reader(file=f)
            width, height, pixels, meta = reader.asDirect()
            if 'physical' in meta:
                phys = meta['physical']
                if len(phys) >= 2:
                    ppm_x, ppm_y = phys[0], phys[1]
                    return (ppm_x, ppm_y)
    except Exception as e:
        print(f"  (lecture pHYs : {e})")
    return (None, None)

# ----------------------------------------------------------------------
# Extraction DPI
# ----------------------------------------------------------------------
def get_dpi_from_image(image_path):
    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Impossible d'ouvrir l'image : {e}")

    dpi_x = dpi_y = None

    # 1. info générale
    if 'dpi' in img.info:
        dpi = img.info['dpi']
        if isinstance(dpi, tuple) and len(dpi) == 2:
            dpi_x, dpi_y = dpi
        elif isinstance(dpi, (int, float)):
            dpi_x = dpi_y = dpi

    # 2. EXIF
    if dpi_x is None or dpi_y is None:
        try:
            exif = img.getexif()
            if exif:
                x_res = exif.get(0x011A)
                y_res = exif.get(0x011B)
                res_unit = exif.get(0x0128)
                if x_res and y_res:
                    if isinstance(x_res, tuple) and len(x_res) == 2:
                        x_dpi = x_res[0] / x_res[1] if x_res[1] != 0 else None
                    else:
                        x_dpi = x_res
                    if isinstance(y_res, tuple) and len(y_res) == 2:
                        y_dpi = y_res[0] / y_res[1] if y_res[1] != 0 else None
                    else:
                        y_dpi = y_res
                    if x_dpi is not None and y_dpi is not None:
                        if res_unit == 3:  # cm -> pouces
                            x_dpi *= 2.54
                            y_dpi *= 2.54
                        dpi_x, dpi_y = x_dpi, y_dpi
        except Exception:
            pass

    # 3. PNG pHYs
    if (dpi_x is None or dpi_y is None) and img.format == 'PNG':
        ppm_x, ppm_y = get_phys_from_png_pypng(image_path)
        if ppm_x is not None and ppm_y is not None:
            dpi_x = ppm_x * 0.0254
            dpi_y = ppm_y * 0.0254

    # si l'un est None, on prend l'autre
    if dpi_x is None and dpi_y is not None:
        dpi_x = dpi_y
    elif dpi_y is None and dpi_x is not None:
        dpi_y = dpi_x

    return (dpi_x, dpi_y)

# ----------------------------------------------------------------------
# Affichage détaillé
# ----------------------------------------------------------------------
def get_exif_dict(image):
    exif_data = {}
    try:
        exif = image.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                exif_data[tag_name] = value
    except Exception:
        pass
    return exif_data

def print_image_details(image_path, dpi_override=None):
    if not os.path.exists(image_path):
        print(f"Erreur : le fichier {image_path} n'existe pas.")
        return

    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Impossible d'ouvrir l'image : {e}")
        return

    print("=" * 60)
    print(f"Fichier : {image_path}")
    print(f"Format : {img.format}")
    print(f"Mode (couleur) : {img.mode}")
    print(f"Dimensions (largeur x hauteur) : {img.width} x {img.height} pixels")
    print(f"Nombre de canaux : {len(img.getbands())}")
    print(f"Taille du fichier : {os.path.getsize(image_path) / 1024:.2f} Ko")

    if img.mode == 'P' and img.palette:
        print("Palette : présente")

    if img.info:
        print("Informations complémentaires (info) :")
        for key, value in img.info.items():
            if key != 'dpi':
                print(f"  {key} : {value}")

    # Extraction DPI
    dpi_x, dpi_y = get_dpi_from_image(image_path)

    # Si un DPI est passé en override, on l'utilise
    if dpi_override is not None:
        dpi_x = dpi_y = dpi_override
        print(f"Résolution (DPI) : {dpi_override:.2f} (spécifiée par l'utilisateur)")
    elif dpi_x is not None and dpi_y is not None:
        print(f"Résolution : X = {dpi_x:.2f} DPI, Y = {dpi_y:.2f} DPI")
        if img.format == 'PNG' and 'dpi' not in img.info:
            print("  (résolution extraite du chunk pHYs du PNG)")
    else:
        print("Résolution (DPI) : non trouvée dans les métadonnées ni dans le chunk pHYs.")

    # EXIF
    exif = get_exif_dict(img)
    if exif:
        print("\nMétadonnées EXIF :")
        for tag, value in sorted(exif.items()):
            if isinstance(value, bytes):
                value = "<données binaires>"
            elif isinstance(value, tuple) and len(value) > 10:
                value = f"<tuple de {len(value)} éléments>"
            print(f"  {tag} : {value}")
    else:
        print("\nAucune métadonnée EXIF trouvée.")

    # Calcul mm/pixel
    if dpi_x is not None and dpi_y is not None:
        if abs(dpi_x - dpi_y) > 0.1:
            print(f"\nNote : les DPI X et Y diffèrent ({dpi_x:.2f} vs {dpi_y:.2f}). Utilisation de la moyenne.")
            dpi = (dpi_x + dpi_y) / 2.0
        else:
            dpi = dpi_x

        mm_per_pixel = 25.4 / dpi
        print(f"\nDistance physique entre deux pixels adjacents : {mm_per_pixel:.6f} mm/pixel")
        print(f"  (soit {1 / mm_per_pixel:.2f} pixels/mm)")
        width_mm = img.width * mm_per_pixel
        height_mm = img.height * mm_per_pixel
        print(f"Taille approximative de l'image : {width_mm:.2f} x {height_mm:.2f} mm")
    else:
        print("\nImpossible de calculer la distance physique car les DPI sont inconnus.")
        # Demande interactive si aucun override
        if dpi_override is None:
            try:
                rep = input("Souhaitez-vous entrer manuellement un DPI ? (o/N) : ").strip().lower()
                if rep in ('o', 'oui', 'y', 'yes'):
                    dpi_str = input("Entrez le DPI (valeur par défaut 96) : ").strip()
                    if dpi_str == '':
                        dpi_man = 96.0
                    else:
                        dpi_man = float(dpi_str)
                    mm_per_pixel = 25.4 / dpi_man
                    print(f"Distance physique (avec DPI={dpi_man:.2f}) : {mm_per_pixel:.6f} mm/pixel")
                    print(f"Taille approximative : {img.width * mm_per_pixel:.2f} x {img.height * mm_per_pixel:.2f} mm")
                else:
                    print("Aucun DPI fourni. Fin de l'analyse.")
            except Exception as e:
                print(f"Erreur de saisie : {e}")
        else:
            print("Aucun DPI spécifié et pas de saisie interactive (utilisation de --dpi requis).")

    print("=" * 60)

# ----------------------------------------------------------------------
# Point d'entrée avec arguments
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyse d'image et calcul mm/pixel")
    parser.add_argument("image", nargs="?", default=None,
                        help="Chemin vers l'image (si omis, utilise le chemin par défaut)")
    parser.add_argument("--dpi", type=float, default=None,
                        help="Spécifier manuellement le DPI (ignore les métadonnées)")
    args = parser.parse_args()

    if args.image:
        image_path = args.image
    else:
        # Chemin par défaut (modifiez-le ou passez-le en argument)
        image_path = r"C:\Users\msi\OneDrive\breast-cancer-ai-framework\backend\app\data\DMID_PNG\1024\TIFF\IMG001.png"

    print_image_details(image_path, dpi_override=args.dpi)

if __name__ == "__main__":
    main()