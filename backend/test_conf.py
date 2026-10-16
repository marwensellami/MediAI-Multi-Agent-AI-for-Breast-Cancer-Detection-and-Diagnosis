# backend/test_conf.py
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
import sys
from datetime import datetime

# Rediriger la sortie vers un fichier et le terminal
class Tee:
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

# Créer le dossier de logs
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Créer le fichier de log avec timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"test_conf_results_{timestamp}.txt"

# Ouvrir le fichier en écriture
f = open(log_file, 'w', encoding='utf-8')

# Rediriger stdout vers le fichier ET le terminal
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, f)

print("=" * 80)
print(f"TEST CONF - Début : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Charger le modèle YOLO fine‑tuné
MODEL_PATH = Path("app/ml/models/yolo_breast.pt")
if not MODEL_PATH.exists():
    print(f"❌ Modèle non trouvé : {MODEL_PATH}")
    exit()

model = YOLO(str(MODEL_PATH))
print(f"✅ Modèle chargé : {MODEL_PATH}")

# Chemin de l'image 412
IMG_PATH = Path("app/data/DMID_PNG/1024/TIFF/IMG412.png")
if not IMG_PATH.exists():
    print(f"❌ Image non trouvée : {IMG_PATH}")
    exit()

print(f"✅ Image trouvée : {IMG_PATH}")

# Dossier de sortie des images
OUT_DIR = Path("conf_test_results")
OUT_DIR.mkdir(exist_ok=True)

# Charger l'image
image = cv2.imread(str(IMG_PATH))
if image is None:
    print("❌ Impossible de charger l'image")
    exit()

print(f"📐 Dimensions de l'image : {image.shape}")
print()

# Seuils de confiance à tester (de 0.01 à 1.00 par pas de 0.01)
conf_thresholds = np.arange(0.01, 1.01, 0.01)

# Seuils IoU à tester (de 0.01 à 1.00 par pas de 0.01)
iou_thresholds = np.arange(0.01, 1.01, 0.01)

print(f"🔍 Configuration du test :")
print(f"   - Seuils Confiance : de {conf_thresholds[0]:.2f} à {conf_thresholds[-1]:.2f} (pas de 0.01)")
print(f"   - Seuils IoU : de {iou_thresholds[0]:.2f} à {iou_thresholds[-1]:.2f} (pas de 0.01)")
print(f"   - Total combinaisons : {len(conf_thresholds) * len(iou_thresholds)}")
print("=" * 80)
print()

results_summary = []
total_combinations = len(conf_thresholds) * len(iou_thresholds)
current_combination = 0

# Pour chaque seuil de confiance
for conf in conf_thresholds:
    print(f"\n{'='*80}")
    print(f"📊 Confiance = {conf:.2f} ({(list(conf_thresholds).index(conf) + 1)}/{len(conf_thresholds)})")
    print(f"{'='*80}")
    
    # Créer un sous-dossier pour chaque seuil de confiance
    conf_dir = OUT_DIR / f"conf_{conf:.2f}"
    conf_dir.mkdir(exist_ok=True)
    
    # Pour chaque seuil IoU
    for iou in iou_thresholds:
        current_combination += 1
        
        # Lancer l'inférence avec ce seuil de confiance et ce seuil IoU
        results = model(image, conf=conf, iou=iou, verbose=False)
        
        # Extraire les boîtes
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            num_boxes = 0
            cv2.imwrite(str(conf_dir / f"iou_{iou:.2f}.png"), image)
            results_summary.append((conf, iou, num_boxes))
            print(f"  [{current_combination:4d}/{total_combinations:4d}] IoU = {iou:.2f} → {num_boxes} détection(s)")
            continue
        
        # Dessiner les boîtes sur l'image
        img_copy = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {conf_score:.2f}"
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        num_boxes = len(boxes)
        print(f"  [{current_combination:4d}/{total_combinations:4d}] IoU = {iou:.2f} → {num_boxes} détection(s)")
        
        # Sauvegarder l'image annotée
        cv2.imwrite(str(conf_dir / f"iou_{iou:.2f}.png"), img_copy)
        results_summary.append((conf, iou, num_boxes))

print("\n" + "=" * 80)
print("✅ TEST TERMINÉ !")
print("=" * 80)
print(f"📁 Images sauvegardées dans : {OUT_DIR}")
print(f"📁 Résultats sauvegardés dans : {log_file}")
print()

# Statistiques générales
total_detections = sum(nb for _, _, nb in results_summary)
avg_detections = total_detections / len(results_summary) if results_summary else 0
max_detections = max(nb for _, _, nb in results_summary) if results_summary else 0
min_detections = min(nb for _, _, nb in results_summary) if results_summary else 0

print("📊 STATISTIQUES GÉNÉRALES :")
print(f"   - Nombre total de combinaisons testées : {len(results_summary)}")
print(f"   - Nombre total de détections : {total_detections}")
print(f"   - Moyenne de détections par combinaison : {avg_detections:.2f}")
print(f"   - Maximum de détections : {max_detections}")
print(f"   - Minimum de détections : {min_detections}")
print()

print("📊 RÉSUMÉ DES DÉTECTIONS PAR CONF ET IoU :")
print("=" * 80)
print(f"{'Conf':^6} | {'IoU':^6} | {'Nb':^8} | {'Progression':^30}")
print("-" * 80)

for i, (conf, iou, nb) in enumerate(results_summary):
    progress = "█" * int((nb / max_detections) * 20) if max_detections > 0 else ""
    print(f"{conf:.2f}   | {iou:.2f}  | {nb:^8} | {progress}")

print("=" * 80)

# Trouver les meilleures combinaisons pour différents nombres de détections
print("\n🎯 MEILLEURES COMBINAISONS POUR DIFFÉRENTS NOMBRES DE DÉTECTIONS :")
print("-" * 80)

for target in [1, 2, 3, 4, 5]:
    best_comb = None
    best_diff = float('inf')
    for conf, iou, nb in results_summary:
        diff = abs(nb - target)
        if diff < best_diff:
            best_diff = diff
            best_comb = (conf, iou, nb)
    
    if best_comb is not None:
        conf, iou, nb = best_comb
        print(f"  Pour {target} détection(s) : Conf = {conf:.2f}, IoU = {iou:.2f} → {nb} détection(s) (écart: {abs(nb-target)})")

print("\n" + "=" * 80)
print(f"FIN DU TEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Restaurer stdout
sys.stdout = original_stdout
f.close()

print(f"\n✅ Résultats enregistrés dans : {log_file}")