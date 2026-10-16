"""
Test : Comparaison entre Otsu SEUL et Otsu + Morphologie
sur une mammographie.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_PATH = Path(
    r"C:\Users\msi\OneDrive\breast-cancer-ai-framework"
    r"\backend\app\data\DMID_PNG\1024\TIFF\IMG002.png"
)

OUTPUT_DIR = Path(
    r"C:\Users\msi\OneDrive\breast-cancer-ai-framework"
    r"\backend\app\data\mask_comparison_test"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FONCTION 1 : Otsu SEUL
# ============================================================
def mask_otsu_only(image_gray):
    """Seuillage d'Otsu SEUL, sans nettoyage morphologique."""
    _, mask = cv2.threshold(
        image_gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask


# ============================================================
# FONCTION 2 : Otsu + Morphologie
# ============================================================
def mask_otsu_morpho(image_gray):
    """Seuillage d'Otsu + nettoyage morphologique (OPEN → CLOSE)."""
    _, mask = cv2.threshold(
        image_gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


# ============================================================
# FONCTION 3 : Statistiques d'un masque
# ============================================================
def mask_stats(mask, label):
    """Calcule les statistiques d'un masque binaire."""
    total = mask.size
    pixels = np.sum(mask > 0)
    pct = 100 * pixels / total

    num_labels, _ = cv2.connectedComponents(mask)
    num_components = num_labels - 1

    if num_components > 0:
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = areas.max()
        smallest_area = areas.min()
        mean_area = areas.mean()
        median_area = np.median(areas)
    else:
        largest_area = smallest_area = mean_area = median_area = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)

    return {
        "label": label,
        "pixels": pixels,
        "pct": pct,
        "num_components": num_components,
        "largest_area": largest_area,
        "smallest_area": smallest_area,
        "mean_area": mean_area,
        "median_area": median_area,
        "perimeter": perimeter,
    }


# ============================================================
# FONCTION 4 : Analyse détaillée des composants
# ============================================================
def analyze_components(mask, label, image_gray):
    """Analyse chaque composant : taille et intensité moyenne."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    print(f"\n{'-' * 60}")
    print(f"  COMPOSANTS : {label}")
    print(f"{'-' * 60}")

    components_info = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        mean_intensity = image_gray[labels == i].mean()

        components_info.append({
            "id": i,
            "area": area,
            "centroid": (round(cx, 1), round(cy, 1)),
            "mean_intensity": round(mean_intensity, 2),
        })

    components_info.sort(key=lambda x: x["area"], reverse=True)

    print(f"\n  Top 10 des plus grands composants :")
    print(f"  {'Rang':<5} {'Aire':<10} {'Centroide':<20} {'Intensite moy.':<15}")
    for rank, c in enumerate(components_info[:10], 1):
        print(f"  {rank:<5} {c['area']:<10} {str(c['centroid']):<20} {c['mean_intensity']:<15}")

    if len(components_info) > 10:
        small = components_info[10:]
        small_areas = [c["area"] for c in small]
        print(f"\n  ... et {len(small)} autres petits composants")
        print(f"      Aire min : {min(small_areas)}")
        print(f"      Aire max : {max(small_areas)}")
        print(f"      Aire moy : {np.mean(small_areas):.1f}")

    return components_info


# ============================================================
# FONCTION 5 : Test d'impact sur l'échantillonnage
# ============================================================
def test_sampling_impact(mask_only, mask_morpho, image, num_samples=1000):
    """Simule N tirages dans chaque masque et mesure le % sombre."""
    results = {}

    for name, mask in [("Otsu SEUL", mask_only), ("Otsu + MORPHO", mask_morpho)]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            results[name] = {"pct_dark": 100, "pct_very_dark": 100,
                             "mean_intensity": 0, "min_intensity": 0}
            continue

        idx = np.random.randint(0, len(xs), size=num_samples)
        values = image[ys[idx], xs[idx]]

        results[name] = {
            "pct_dark": 100 * np.sum(values < 10) / num_samples,
            "pct_very_dark": 100 * np.sum(values < 5) / num_samples,
            "mean_intensity": values.mean(),
            "min_intensity": values.min(),
        }

    return results


# ============================================================
# SCRIPT PRINCIPAL
# ============================================================
def main():
    print(f"\n{'#' * 65}")
    print(f"#  TEST : Otsu SEUL  vs  Otsu + MORPHOLOGIE")
    print(f"#  Image : {IMAGE_PATH.name}")
    print(f"{'#' * 65}")

    # 1. Vérifier l'image
    if not IMAGE_PATH.exists():
        print(f"\nImage introuvable : {IMAGE_PATH}")
        return

    # 2. Charger l'image
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"\nImpossible de charger l'image")
        return

    print(f"\nImage chargee : {image.shape}")

    # 3. Appliquer les deux méthodes
    mask_only = mask_otsu_only(image)
    mask_morpho = mask_otsu_morpho(image)

    # 4. Statistiques
    stats_only = mask_stats(mask_only, "Otsu SEUL")
    stats_morpho = mask_stats(mask_morpho, "Otsu + MORPHOLOGIE")

    # --- Tableau comparatif (CORRIGE : plus d'apostrophe dans f-string) ---
    print(f"\n{'=' * 65}")
    print(f"  STATISTIQUES COMPAREES")
    print(f"{'=' * 65}")
    print(f"\n  {'Metrique':<30} {'Otsu SEUL':>15} {'Otsu + MORPHO':>15}")
    print(f"  {'-' * 62}")
    print(f"  {'Pixels inclus':<30} {stats_only['pixels']:>15,} {stats_morpho['pixels']:>15,}")
    print(f"  {'Pourcentage image':<30} {stats_only['pct']:>14.2f}% {stats_morpho['pct']:>14.2f}%")
    print(f"  {'Nombre de composants':<30} {stats_only['num_components']:>15} {stats_morpho['num_components']:>15}")
    print(f"  {'Plus grand composant (px)':<30} {stats_only['largest_area']:>15,} {stats_morpho['largest_area']:>15,}")
    print(f"  {'Plus petit composant (px)':<30} {stats_only['smallest_area']:>15} {stats_morpho['smallest_area']:>15}")
    print(f"  {'Aire moyenne (px)':<30} {stats_only['mean_area']:>15.1f} {stats_morpho['mean_area']:>15.1f}")
    print(f"  {'Aire mediane (px)':<30} {stats_only['median_area']:>15.1f} {stats_morpho['median_area']:>15.1f}")
    print(f"  {'Perimetre total (px)':<30} {stats_only['perimeter']:>15.1f} {stats_morpho['perimeter']:>15.1f}")

    # 5. Analyse détaillée des composants
    analyze_components(mask_only, "Otsu SEUL", image)
    analyze_components(mask_morpho, "Otsu + MORPHOLOGIE", image)

    # 6. Différence entre les deux masques
    diff = cv2.absdiff(mask_only, mask_morpho)
    pixels_diff = np.sum(diff > 0)
    pct_diff = 100 * pixels_diff / diff.size

    print(f"\n{'=' * 65}")
    print(f"  DIFFERENCE ENTRE LES DEUX MASQUES")
    print(f"{'=' * 65}")
    print(f"\n  Pixels differents : {pixels_diff:,}  ({pct_diff:.3f} % de l'image)")
    print(f"  -> Ces pixels sont ceux modifies par la morphologie")

    supprimes = cv2.bitwise_and(mask_only, cv2.bitwise_not(mask_morpho))
    nb_supprimes = np.sum(supprimes > 0)
    print(f"\n  Pixels SUPPRIMES par la morphologie : {nb_supprimes:,}")
    if nb_supprimes > 0:
        intensites_suppr = image[supprimes > 0]
        print(f"  Intensite moyenne des pixels supprimes : {intensites_suppr.mean():.2f}")
        print(f"  Intensite min/max : {intensites_suppr.min()} / {intensites_suppr.max()}")

    ajoutes = cv2.bitwise_and(mask_morpho, cv2.bitwise_not(mask_only))
    nb_ajoutes = np.sum(ajoutes > 0)
    print(f"\n  Pixels AJOUTES par la morphologie : {nb_ajoutes:,}")
    if nb_ajoutes > 0:
        intensites_aj = image[ajoutes > 0]
        print(f"  Intensite moyenne des pixels ajoutes : {intensites_aj.mean():.2f}")

    # 7. Test d'impact sur l'échantillonnage
    sampling = test_sampling_impact(mask_only, mask_morpho, image, 1000)

    print(f"\n{'=' * 65}")
    print(f"  IMPACT SUR L'ECHANTILLONNAGE (1000 tirages)")
    print(f"{'=' * 65}")
    print(f"\n  {'Metrique':<35} {'Otsu SEUL':>12} {'Otsu+MORPHO':>12}")
    print(f"  {'-' * 60}")
    print(f"  {'% pixels sombres (< 10)':<35} "
          f"{sampling['Otsu SEUL']['pct_dark']:>11.2f}% "
          f"{sampling['Otsu + MORPHO']['pct_dark']:>11.2f}%")
    print(f"  {'% pixels tres sombres (< 5)':<35} "
          f"{sampling['Otsu SEUL']['pct_very_dark']:>11.2f}% "
          f"{sampling['Otsu + MORPHO']['pct_very_dark']:>11.2f}%")
    print(f"  {'Intensite moyenne':<35} "
          f"{sampling['Otsu SEUL']['mean_intensity']:>11.2f} "
          f"{sampling['Otsu + MORPHO']['mean_intensity']:>11.2f}")
    print(f"  {'Intensite min':<35} "
          f"{sampling['Otsu SEUL']['min_intensity']:>11} "
          f"{sampling['Otsu + MORPHO']['min_intensity']:>11}")

    # 8. Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Image originale", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mask_only, cmap="gray")
    axes[0, 1].set_title(
        f"Otsu SEUL\n"
        f"{stats_only['pct']:.2f}% inclus | "
        f"{stats_only['num_components']} composants",
        fontsize=14
    )
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mask_morpho, cmap="gray")
    axes[0, 2].set_title(
        f"Otsu + MORPHOLOGIE\n"
        f"{stats_morpho['pct']:.2f}% inclus | "
        f"{stats_morpho['num_components']} composants",
        fontsize=14
    )
    axes[0, 2].axis("off")

    # Overlay : Rouge = Otsu seul, Vert = Morpho, Jaune = commun
    overlay = np.zeros((*image.shape, 3), dtype=np.uint8)
    overlay[mask_only > 0] = [255, 0, 0]
    overlay[mask_morpho > 0] = [0, 255, 0]
    overlay[(mask_only > 0) & (mask_morpho > 0)] = [255, 255, 0]

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(
        "Comparaison :\n"
        "Jaune = commun | Rouge = Otsu seul | Vert = Morpho",
        fontsize=12
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(diff, cmap="hot")
    axes[1, 1].set_title(
        f"Difference binaire\n{pixels_diff:,} pixels ({pct_diff:.3f}%)",
        fontsize=14
    )
    axes[1, 1].axis("off")

    # Zoom sur une zone
    M = cv2.moments(mask_morpho)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = image.shape[1] // 2, image.shape[0] // 2

    x_start = max(0, cx - 400)
    y_start = max(0, cy - 150)
    zoom_size = 300

    zoom_only = mask_only[y_start:y_start + zoom_size, x_start:x_start + zoom_size]
    zoom_morpho = mask_morpho[y_start:y_start + zoom_size, x_start:x_start + zoom_size]

    zoom_combined = np.hstack([
        zoom_only,
        np.ones((zoom_size, 10), dtype=np.uint8) * 128,
        zoom_morpho
    ])
    axes[1, 2].imshow(zoom_combined, cmap="gray")
    axes[1, 2].set_title(
        f"Zoom sur une zone (300x300)\n"
        f"Gauche : Otsu seul | Droite : Otsu + Morpho",
        fontsize=12
    )
    axes[1, 2].axis("off")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "otsu_vs_morpho_IMG002.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\nFigure sauvegardee : {output_path}")

    # 9. Conclusion
    print(f"\n{'=' * 65}")
    print(f"  CONCLUSION")
    print(f"{'=' * 65}")

    reduction_composants = stats_only["num_components"] - stats_morpho["num_components"]

    print(f"\n  Composants : {stats_only['num_components']} -> {stats_morpho['num_components']}")
    print(f"     -> La morphologie a supprime {reduction_composants} composants parasites")

    print(f"\n  Pixels : {stats_only['pixels']:,} -> {stats_morpho['pixels']:,}")
    print(f"     -> Difference : {stats_only['pixels'] - stats_morpho['pixels']:+,} pixels")

    print(f"\n  Plus petit composant :")
    print(f"     -> Otsu SEUL  : {stats_only['smallest_area']} pixels")
    print(f"     -> Otsu+Morpho: {stats_morpho['smallest_area']} pixels")

    if pixels_diff > 0:
        print(f"\n  La morphologie a nettoye {pixels_diff:,} pixels parasites")
        print(f"     Ces pixels etaient du BRUIT ou des ARTEFACTS isoles")
    else:
        print(f"\n  Aucune difference (image deja propre)")

    plt.show()


if __name__ == "__main__":
    main()