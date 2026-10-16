# backend/export_matrix.py
from pathlib import Path
import numpy as np
from PIL import Image

# ✅ Chemin absolu robuste
BASE_DIR = Path(__file__).resolve().parent
MASK_PATH = BASE_DIR / "app" / "data" / "DMID_PNG" / "1024" / "Masks" / "IMG001.png"

if not MASK_PATH.exists():
    for p in BASE_DIR.rglob("IMG001.png"):
        MASK_PATH = p
        break

print(f"📁 Fichier source : {MASK_PATH}")
print(f"Existe ?         : {MASK_PATH.exists()}\n")

# Charger la matrice (garder les valeurs originales)
img = Image.open(MASK_PATH)
matrix = np.array(img)

print(f"📐 Shape   : {matrix.shape}")
print(f"🔢 dtype   : {matrix.dtype}")
print(f"📉 Min/Max : {matrix.min()} / {matrix.max()}")

# Si RGB, prendre le premier canal
if matrix.ndim == 3:
    print(f"⚠️  Image {matrix.shape[2]} canaux → extraction canal 0")
    matrix = matrix[:, :, 0]

# ============================================================
# FORMAT 1 : Matrice pleine 1024x1024 (valeurs brutes)
# ============================================================
out1 = BASE_DIR / "mask_matrix_full.txt"
np.savetxt(out1, matrix, fmt="%d", delimiter=" ")
print(f"\n💾 Format 1 (matrice complète) : {out1}")
print(f"   {out1.stat().st_size / 1024 / 1024:.2f} Mo")

# ============================================================
# FORMAT 2 : Compact (une ligne par ligne de la matrice, collée)
#   → lisible, valeurs séparées par rien (0-9 seulement) ou virgule
# ============================================================
out2 = BASE_DIR / "mask_matrix_compact.txt"
with open(out2, "w") as f:
    for row in matrix:
        f.write(",".join(str(int(v)) for v in row) + "\n")
print(f"💾 Format 2 (compact CSV)      : {out2}")
print(f"   {out2.stat().st_size / 1024 / 1024:.2f} Mo")

# ============================================================
# FORMAT 3 : Visuel ASCII (pour "voir" le masque dans le txt)
# ============================================================
out3 = BASE_DIR / "mask_matrix_ascii.txt"
with open(out3, "w", encoding="utf-8") as f:
    f.write(f"# Masque {MASK_PATH.name} — shape {matrix.shape}\n")
    f.write(f"# Min={matrix.min()} Max={matrix.max()}\n")
    f.write("# " + "=" * 1024 + "\n")
    for row in matrix:
        # █ pour valeur max, espace pour 0, nuances intermédiaires
        line = "".join(
            "█" if v > 200 else
            "▓" if v > 128 else
            "▒" if v > 64 else
            "░" if v > 0 else
            " "
            for v in row
        )
        f.write(line + "\n")
print(f"💾 Format 3 (ASCII visuel)     : {out3}")
print(f"   {out3.stat().st_size / 1024:.2f} Ko")

# ============================================================
# FORMAT 4 : Binaire compact 0/1 (le plus petit)
# ============================================================
out4 = BASE_DIR / "mask_matrix_binary.txt"
binary = (matrix > 127).astype(np.uint8)
with open(out4, "w") as f:
    for row in binary:
        f.write("".join(str(v) for v in row) + "\n")
print(f"💾 Format 4 (binaire 0/1)      : {out4}")
print(f"   {out4.stat().st_size / 1024:.2f} Ko")

# ============================================================
# FORMAT 5 : Coordonnées des pixels positifs (utile pour parcimonie)
# ============================================================
out5 = BASE_DIR / "mask_pixels_positive.txt"
with open(out5, "w") as f:
    f.write("# row col value\n")
    coords = np.argwhere(matrix > 0)
    for r, c in coords:
        f.write(f"{r} {c} {matrix[r, c]}\n")
print(f"💾 Format 5 (coordonnées >0)   : {out5}")
print(f"   {out5.stat().st_size / 1024 / 1024:.2f} Mo ({len(coords):,} pixels)")

# ============================================================
# FORMAT 6 : Valeurs uniques + comptage
# ============================================================
out6 = BASE_DIR / "mask_unique_values.txt"
unique, counts = np.unique(matrix, return_counts=True)
with open(out6, "w") as f:
    f.write(f"# Valeurs uniques du masque {MASK_PATH.name}\n")
    f.write(f"# Shape={matrix.shape} dtype={matrix.dtype}\n")
    f.write(f"# {'Valeur':>8} {'Occurrences':>14} {'Pourcentage':>14}\n")
    for v, c in zip(unique, counts):
        f.write(f"  {v:>8} {c:>14,} {c/matrix.size*100:>13.4f}%\n")
print(f"💾 Format 6 (valeurs uniques)  : {out6}")

print("\n" + "=" * 60)
print("✅ Export terminé ! Tous les fichiers sont dans :")
print(f"   {BASE_DIR}")
print("=" * 60)