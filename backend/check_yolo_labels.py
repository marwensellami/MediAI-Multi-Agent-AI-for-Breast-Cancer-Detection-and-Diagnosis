# backend/check_yolo_labels.py
from pathlib import Path

def check_labels(label_dir):
    label_dir = Path(label_dir)
    for file in label_dir.glob("*.txt"):
        with open(file, 'r') as f:
            lines = f.readlines()
        if not lines:
            continue
        for i, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"❌ {file.name}: ligne {i} a {len(parts)} éléments")
                continue
            try:
                cls, x, y, w, h = map(float, parts)
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    print(f"❌ {file.name}: coordonnées hors limites : {line.strip()}")
                if w <= 0 or h <= 0:
                    print(f"❌ {file.name}: largeur/hauteur nulle : {line.strip()}")
            except ValueError:
                print(f"❌ {file.name}: ligne non numérique : {line.strip()}")
    print("✅ Vérification terminée.")

if __name__ == "__main__":
    check_labels("yolo_dataset/labels/train")