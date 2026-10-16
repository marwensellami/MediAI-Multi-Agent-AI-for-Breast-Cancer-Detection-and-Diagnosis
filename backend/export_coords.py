# backend/export_coords.py
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

print(f"📁 Fichier source : {MASK_PATH}\n")

# Charger le masque
img = Image.open(MASK_PATH)
mask = np.array(img)

# Si RGB, prendre le 1er canal
if mask.ndim == 3:
    mask = mask[:, :, 0]

print(f"📐 Shape   : {mask.shape}")
print(f"🔢 dtype   : {mask.dtype}")
print(f"📉 Min/Max : {mask.min()} / {mask.max()}")

# ============================================================
# EXTRACTION DES COORDONNÉES
# ============================================================
coords = np.where(mask > 0)
rows, cols = coords

print(f"\n{'=' * 60}")
print(f"🎯 PIXELS AVEC VALEUR > 0")
print(f"{'=' * 60}")
print(f"Nombre de pixels    : {len(rows):,}")
print(f"Ratio sur l'image   : {len(rows) / mask.size * 100:.4f}%")
print(f"Type de coords      : {type(coords)} (tuple de {len(coords)} arrays)")
print(f"Shape de rows       : {rows.shape}")
print(f"Shape de cols       : {cols.shape}")

# ============================================================
# APERÇU CONSOLE (100 premiers pixels)
# ============================================================
print(f"\n{'=' * 60}")
print(f"👀 APERÇU (100 premiers pixels)")
print(f"{'=' * 60}")
print(f"{'#':>5} | {'row':>6} | {'col':>6} | {'value':>6}")
print("-" * 40)
for i in range(min(100, len(rows))):
    r, c = rows[i], cols[i]
    print(f"{i:>5} | {r:>6} | {c:>6} | {mask[r, c]:>6}")

if len(rows) > 100:
    print(f"... ({len(rows) - 100:,} autres pixels)")

# ============================================================
# STATISTIQUES DES COORDONNÉES
# ============================================================
print(f"\n{'=' * 60}")
print(f"📊 STATISTIQUES SPATIALES")
print(f"{'=' * 60}")
print(f"Bounding box :")
print(f"  row min/max : {rows.min()} / {rows.max()}")
print(f"  col min/max : {cols.min()} / {cols.max()}")
print(f"  Hauteur     : {rows.max() - rows.min() + 1}")
print(f"  Largeur     : {cols.max() - cols.min() + 1}")
print(f"Centre masse : (row={rows.mean():.2f}, col={cols.mean():.2f})")
print(f"Écart-type   : (row={rows.std():.2f}, col={cols.std():.2f})")

# ============================================================
# SAUVEGARDE FICHIER .TXT
# ============================================================
out_txt = BASE_DIR / "mask_coords_positive.txt"
with open(out_txt, "w") as f:
    f.write(f"# Coordonnées pixels > 0 du masque {MASK_PATH.name}\n")
    f.write(f"# Shape image : {mask.shape}\n")
    f.write(f"# Nombre de pixels > 0 : {len(rows):,}\n")
    f.write(f"# Format : row col value\n")
    f.write("#" + "=" * 40 + "\n")
    for r, c in zip(rows, cols):
        f.write(f"{r} {c} {mask[r, c]}\n")

print(f"\n💾 Écrit : {out_txt}")
print(f"   Taille : {out_txt.stat().st_size / 1024:.1f} Ko ({len(rows):,} lignes)")

# ============================================================
# SAUVEGARDE CSV (pour pandas/Excel)
# ============================================================
out_csv = BASE_DIR / "mask_coords_positive.csv"
np.savetxt(
    out_csv,
    np.column_stack([rows, cols, mask[rows, cols]]),
    fmt="%d",
    delimiter=",",
    header="row,col,value",
    comments="",
)
print(f"💾 Écrit : {out_csv}")
print(f"   Taille : {out_csv.stat().st_size / 1024:.1f} Ko")

# ============================================================
# SAUVEGARDE NPY (rechargeable rapidement)
# ============================================================
out_npy = BASE_DIR / "mask_coords_positive.npy"
np.save(out_npy, np.column_stack([rows, cols]))
print(f"💾 Écrit : {out_npy}")
print(f"   Taille : {out_npy.stat().st_size / 1024:.1f} Ko")

# ============================================================
# VERSION COMPACTE : (row, col) seulement
# ============================================================
out_compact = BASE_DIR / "mask_coords_compact.txt"
with open(out_compact, "w") as f:
    for r, c in zip(rows, cols):
        f.write(f"({r},{c})\n")
print(f"💾 Écrit : {out_compact}")
print(f"   Taille : {out_compact.stat().st_size / 1024:.1f} Ko")