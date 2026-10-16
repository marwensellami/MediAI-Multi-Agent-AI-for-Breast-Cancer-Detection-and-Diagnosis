"""
Test : Comparaison entre np.where(image > 0) et Otsu + Morphologie
sur une mammographie normale (sans lésion).

Objectif : Montrer visuellement pourquoi Otsu est préférable.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_PATH = Path(r"C:\Users\msi\OneDrive\breast-cancer-ai-framework\backend\app\data\DMID_PNG\1024\TIFF\IMG002.png")

OUTPUT_DIR = Path(r"C:\Users\msi\OneDrive\breast-cancer-ai-framework\backend\app\data\mask_comparison_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FONCTION 1 : Méthode naïve (image > 0)
# ============================================================
def mask_naive(image_gray):
    """
    Méthode naïve : considère comme 'sein' tout pixel > 0.
    """
    mask = (image_gray > 0).astype(np.uint8) * 255
    return mask


# ============================================================
# FONCTION 2 : Méthode Otsu + Morphologie
# ============================================================
def mask_otsu(image_gray):
    """
    Méthode robuste : seuillage d'Otsu + nettoyage morphologique.
    """
    # 1. Seuillage d'Otsu
    _, mask = cv2.threshold(
        image_gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 2. Nettoyage morphologique
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


# ============================================================
# FONCTION 3 : Statistiques d'un masque
# ============================================================
def mask_stats(mask, image_gray, label):
    """
    Calcule et affiche les statistiques d'un masque.
    """
    total_pixels = mask.size
    pixels_inclus = np.sum(mask > 0)
    pourcentage = 100 * pixels_inclus / total_pixels

    # Valeurs d'intensité des pixels inclus
    if pixels_inclus > 0:
        valeurs_incluses = image_gray[mask > 0]
        vmin = valeurs_incluses.min()
        vmax = valeurs_incluses.max()
        vmean = valeurs_incluses.mean()
    else:
        vmin = vmax = vmean = 0

    # Valeurs d'intensité des pixels EXCLUS
    if pixels_inclus < total_pixels:
        valeurs_exclues = image_gray[mask == 0]
        vmin_ex = valeurs_exclues.min()
        vmax_ex = valeurs_exclues.max()
        vmean_ex = valeurs_exclues.mean()
    else:
        vmin_ex = vmax_ex = vmean_ex = 0

    print(f"\n{'=' * 60}")
    print(f"  STATISTIQUES : {label}")
    print(f"{'=' * 60}")
    print(f"  Pixels inclus     : {pixels_inclus:>10,}  ({pourcentage:>6.2f} %)")
    print(f"  Pixels exclus     : {total_pixels - pixels_inclus:>10,}  ({100 - pourcentage:>6.2f} %)")
    print(f"  ---")
    print(f"  Intensité INCLUS  : min={vmin:>3}, max={vmax:>3}, moy={vmean:>6.2f}")
    print(f"  Intensité EXCLUS  : min={vmin_ex:>3}, max={vmax_ex:>3}, moy={vmean_ex:>6.2f}")

    return {
        "label": label,
        "pixels_inclus": pixels_inclus,
        "pourcentage": pourcentage,
        "vmin": vmin, "vmax": vmax, "vmean": vmean,
        "vmin_ex": vmin_ex, "vmax_ex": vmax_ex, "vmean_ex": vmean_ex,
    }


# ============================================================
# FONCTION 4 : Test d'échantillonnage aléatoire
# ============================================================
def test_sampling(mask, image_gray, num_samples=1000):
    """
    Simule 1000 tirages aléatoires dans le masque et
    regarde la valeur d'intensité moyenne du patch extrait.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return {"mean_intensity": 0, "min_intensity": 0, "max_intensity": 0}

    idx = np.random.randint(0, len(xs), size=num_samples)
    sampled_values = image_gray[ys[idx], xs[idx]]

    return {
        "mean_intensity": sampled_values.mean(),
        "min_intensity": sampled_values.min(),
        "max_intensity": sampled_values.max(),
        "pct_dark": 100 * np.sum(sampled_values < 10) / num_samples,
    }


# ============================================================
# SCRIPT PRINCIPAL
# ============================================================
def main():
    print(f"\n{'#' * 60}")
    print(f"#  TEST : np.where(image > 0)  vs  Otsu + Morphologie")
    print(f"#  Image : {IMAGE_PATH.name}")
    print(f"{'#' * 60}")

    # 1. Vérifier que l'image existe
    if not IMAGE_PATH.exists():
        print(f"\n❌ ERREUR : Image introuvable : {IMAGE_PATH}")
        return

    # 2. Charger l'image
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"\n❌ ERREUR : Impossible de charger l'image.")
        return

    print(f"\n✅ Image chargée : {image.shape}  (dtype={image.dtype})")

    # 3. Statistiques globales de l'image
    print(f"\n📊 Statistiques de l'image :")
    print(f"   min  = {image.min()}")
    print(f"   max  = {image.max()}")
    print(f"   mean = {image.mean():.2f}")
    print(f"   médiane = {np.median(image):.2f}")

    # 4. Histogramme (pour voir la distribution)
    hist = np.bincount(image.flatten(), minlength=256)
    print(f"\n📊 Distribution des intensités (extrait) :")
    for i in range(0, 60, 5):
        count = hist[i:i + 5].sum()
        pct = 100 * count / image.size
        bar = "█" * int(pct / 2)
        print(f"   [{i:>3}-{i+4:>3}] : {count:>9,}  ({pct:>5.2f} %)  {bar}")

    # 5. Appliquer les deux méthodes
    mask_naive_result = mask_naive(image)
    mask_otsu_result = mask_otsu(image)

    # 6. Statistiques des masques
    stats_naive = mask_stats(mask_naive_result, image, "MÉTHODE NAÏVE (image > 0)")
    stats_otsu = mask_stats(mask_otsu_result, image, "MÉTHODE OTSU + MORPHOLOGIE")

    # 7. Test d'échantillonnage aléatoire
    print(f"\n{'=' * 60}")
    print(f"  TEST D'ÉCHANTILLONNAGE ALÉATOIRE (1000 tirages)")
    print(f"{'=' * 60}")

    sampling_naive = test_sampling(mask_naive_result, image, 1000)
    sampling_otsu = test_sampling(mask_otsu_result, image, 1000)

    print(f"\n  MÉTHODE NAÏVE (image > 0) :")
    print(f"     Intensité moyenne des pixels tirés : {sampling_naive['mean_intensity']:.2f}")
    print(f"     Intensité min/max                  : {sampling_naive['min_intensity']} / {sampling_naive['max_intensity']}")
    print(f"     % de pixels sombres (< 10)         : {sampling_naive['pct_dark']:.2f} %")

    print(f"\n  MÉTHODE OTSU :")
    print(f"     Intensité moyenne des pixels tirés : {sampling_otsu['mean_intensity']:.2f}")
    print(f"     Intensité min/max                  : {sampling_otsu['min_intensity']} / {sampling_otsu['max_intensity']}")
    print(f"     % de pixels sombres (< 10)         : {sampling_otsu['pct_dark']:.2f} %")

    # 8. Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Ligne 1 : image originale, masque naïf, masque Otsu
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Image originale (IMG002.png)", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mask_naive_result, cmap="gray")
    axes[0, 1].set_title(
        f"Méthode naïve (image > 0)\n{stats_naive['pourcentage']:.1f} % de l'image",
        fontsize=14
    )
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mask_otsu_result, cmap="gray")
    axes[0, 2].set_title(
        f"Méthode Otsu + Morphologie\n{stats_otsu['pourcentage']:.1f} % de l'image",
        fontsize=14
    )
    axes[0, 2].axis("off")

    # Ligne 2 : overlay (image + masque rouge), histogramme, patches échantillonnés
    # 2.1 Overlay naïf
    overlay_naive = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay_naive[mask_naive_result > 0] = [0, 0, 255]  # Rouge
    axes[1, 0].imshow(cv2.cvtColor(overlay_naive, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(
        f"Overlay NAÏF : zones incluses en rouge\n"
        f"→ Inclut le fond bruité (intensité {stats_naive['vmin_ex']}-{stats_naive['vmax_ex']})",
        fontsize=12
    )
    axes[1, 0].axis("off")

    # 2.2 Overlay Otsu
    overlay_otsu = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay_otsu[mask_otsu_result > 0] = [0, 255, 0]  # Vert
    axes[1, 1].imshow(cv2.cvtColor(overlay_otsu, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(
        f"Overlay OTSU : zones incluses en vert\n"
        f"→ Sein uniquement (fond exclu)",
        fontsize=12
    )
    axes[1, 1].axis("off")

    # 2.3 Histogramme avec seuils
    axes[1, 2].hist(image.flatten(), bins=256, range=(0, 255), color="gray", log=True)
    axes[1, 2].axvline(x=0, color="red", linestyle="--", linewidth=2, label="Seuil naïf (0)")
    otsu_thresh, _ = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    axes[1, 2].axvline(x=otsu_thresh, color="green", linestyle="--", linewidth=2,
                       label=f"Seuil Otsu ({otsu_thresh:.0f})")
    axes[1, 2].set_title("Histogramme des intensités", fontsize=14)
    axes[1, 2].set_xlabel("Intensité (0-255)")
    axes[1, 2].set_ylabel("Nombre de pixels (log)")
    axes[1, 2].legend()
    axes[1, 2].set_xlim(0, 60)

    plt.tight_layout()

    # Sauvegarder la figure
    output_path = OUTPUT_DIR / "comparison_IMG002.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\n💾 Figure sauvegardée : {output_path}")

    # 9. Conclusion
    print(f"\n{'=' * 60}")
    print(f"  CONCLUSION")
    print(f"{'=' * 60}")

    diff_pct = stats_naive["pourcentage"] - stats_otsu["pourcentage"]
    print(f"\n  La méthode naïve inclut {diff_pct:.2f} % de pixels EN PLUS")
    print(f"  par rapport à Otsu → ce sont du fond bruité et des artefacts.\n")

    if sampling_naive["pct_dark"] > 5:
        print(f"  ⚠️  Avec la méthode naïve, {sampling_naive['pct_dark']:.1f} % des tirages")
        print(f"      tombent sur des pixels sombres (< 10) → patches inutiles.\n")

    print(f"  ✅ Avec Otsu, seulement {sampling_otsu['pct_dark']:.1f} % des tirages")
    print(f"      tombent sur des pixels sombres → échantillonnage propre.\n")

    plt.show()


if __name__ == "__main__":
    main()