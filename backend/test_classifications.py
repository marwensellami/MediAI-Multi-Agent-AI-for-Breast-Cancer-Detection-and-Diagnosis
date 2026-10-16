# backend/test_classifications.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.utils.birads_utils import get_birads_info

# ============================================================
# DONNÉES DE TEST (exactement celles de IMG001.png)
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
# FONCTION format_classifications_only (copiée depuis medical_reasoning.py)
# ============================================================
def format_classifications_only(lesions):
    lines = []
    for i, det in enumerate(lesions, 1):
        label = det.get("malignancy_label", "Inconnu")
        prob = det.get("malignancy_score", 0.5)
        birads = get_birads_info(prob)["level"]  # ex: "BI-RADS 4C"
        lines.append(f"Lésion {i} : {label} ({birads})")
    return "\n".join(lines)

# ============================================================
# AFFICHAGE
# ============================================================
if __name__ == "__main__":
    classifications_only = format_classifications_only(lesions)
    print("=" * 80)
    print("📝 CONTENU DE {classifications_only} :")
    print("=" * 80)
    print(classifications_only)
    print("=" * 80)
    print(f"✅ Longueur : {len(classifications_only)} caractères")