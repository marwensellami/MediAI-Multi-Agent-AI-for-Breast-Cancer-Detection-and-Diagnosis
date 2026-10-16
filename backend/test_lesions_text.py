# backend/test_lesions_text.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.utils.birads_utils import get_birads_info

# ============================================================
# DONNÉES DE TEST (identiques à celles de votre mammographie)
# ============================================================
lesions = [
    {
        "size_mm": 13.2,
        "malignancy_label": "Malin",
        "malignancy_score": 0.762,
        "area_mm2": 121.44,
        "perimeter_mm": 44.40,
        "circularity": 0.774,
    },
    {
        "size_mm": 4.6,
        "malignancy_label": "Bénin",
        "malignancy_score": 0.205,
        "area_mm2": 13.80,
        "perimeter_mm": 14.80,
        "circularity": 0.792,
    }
]

# ============================================================
# FONCTION DE FORMATAGE (copiée depuis medical_reasoning.py)
# ============================================================
def format_lesion_detail(det, index):
    lines = [f"Lésion {index}:"]
    
    size_mm = det.get("size_mm")
    if size_mm is not None and isinstance(size_mm, (int, float)):
        lines.append(f"  - Taille : {size_mm:.1f} mm")
    else:
        lines.append("  - Taille : Non mesurable")
    
    label = det.get("malignancy_label", "Indéterminé")
    lines.append(f"  - Classification : {label}")
    
    prob = det.get("malignancy_score", 0.5)
    lines.append(f"  - Probabilité de malignité : {prob*100:.1f} %")
    
    # BI‑RADS complet via get_birads_info
    birads_info = get_birads_info(prob)
    lines.append(f"  - BI-RADS : {birads_info['level']}")
    
    area = det.get("area_mm2")
    perimeter = det.get("perimeter_mm")
    if area is not None and perimeter is not None:
        lines.append(f"  - Aire : {area:.2f} mm²")
        lines.append(f"  - Périmètre : {perimeter:.2f} mm")
    
    circularity = det.get("circularity")
    if circularity is not None:
        lines.append(f"  - Circularité : {circularity:.3f}")
    
    return "\n".join(lines)

# ============================================================
# CONSTRUCTION DE LESIONS_TEXT
# ============================================================
lesions_details = []
for i, det in enumerate(lesions, 1):
    lesions_details.append(format_lesion_detail(det, i))

lesions_text = "\n".join(lesions_details)

# ============================================================
# AFFICHAGE
# ============================================================
print("=" * 80)
print("📋 CONTENU DE lesions_text :")
print("=" * 80)
print(lesions_text)
print("=" * 80)
print(f"✅ Longueur : {len(lesions_text)} caractères, {len(lesions_text.split())} mots")