# backend/evaluate_yolo_thresholds.py
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# ============================================
# CHEMINS
# ============================================
BASE_DIR = Path(__file__).parent  # backend/
DMID_DIR = BASE_DIR / "app" / "data" / "DMID_PNG"
METADATA_PATH = DMID_DIR / "Metadata.xlsx"
TIFF_DIR = DMID_DIR / "1024" / "TIFF"
MODEL_PATH = BASE_DIR / "app" / "ml" / "models" / "yolo_breast.pt"
OUTPUT_CSV = BASE_DIR / "threshold_evaluation_results.csv"

# ============================================
# FONCTIONS
# ============================================

def clean_class(value):
    if pd.isna(value):
        return ''
    return str(value).strip().upper()

def get_ground_truth(image_rows):
    """
    Détermine la vérité terrain pour une image à partir de ses lignes.
    Retourne True si l'image a au moins une lésion B ou M, False si normale.
    Ignore les images avec uniquement des N (exclues).
    """
    has_lesion = any(clean_class(r[4]) in ['B', 'M'] for r in image_rows)
    has_only_n = all(clean_class(r[4]) == 'N' for r in image_rows)
    if has_only_n:
        return None  # ignoré
    return has_lesion

def compute_metrics(threshold, predictions, ground_truth):
    """
    Calcule TP, FP, FN, TN pour un seuil donné.
    predictions : dict {img_name: list of (conf, class_id)}
    ground_truth : dict {img_name: bool (True=positive)}
    """
    TP = FP = FN = TN = 0
    for img_name, truth in ground_truth.items():
        # Si l'image n'est pas dans les prédictions (pas de détection), on considère prédiction négative
        preds = predictions.get(img_name, [])
        # Filtrer par seuil
        detections = [p for p in preds if p[0] >= threshold]
        pred_positive = len(detections) > 0
        if truth:
            if pred_positive:
                TP += 1
            else:
                FN += 1
        else:
            if pred_positive:
                FP += 1
            else:
                TN += 1
    return TP, FP, FN, TN

# ============================================
# SCRIPT PRINCIPAL
# ============================================

def main():
    print("📊 Évaluation des seuils YOLO sur le dataset DMID")
    print("="*60)

    # 1. Lire les métadonnées
    print("📂 Lecture du fichier Metadata.xlsx...")
    df = pd.read_excel(METADATA_PATH, header=None)
    df_data = df.iloc[31:].copy()  # à partir de la ligne 32

    # Grouper les lignes par image
    image_rows = {}
    for idx, row in df_data.iterrows():
        img_name = str(row[0]).strip()
        if img_name not in image_rows:
            image_rows[img_name] = []
        image_rows[img_name].append(row)

    # Déterminer la vérité terrain pour chaque image
    ground_truth = {}
    ignored = []
    for img_name, rows in image_rows.items():
        truth = get_ground_truth(rows)
        if truth is None:
            ignored.append(img_name)
        else:
            ground_truth[img_name] = truth

    print(f"✅ Images incluses : {len(ground_truth)} (ignorées : {len(ignored)} images avec uniquement N)")

    # 2. Charger le modèle YOLO
    print(f"📥 Chargement du modèle YOLO depuis {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        print(f"❌ Modèle introuvable : {MODEL_PATH}")
        return
    model = YOLO(str(MODEL_PATH))
    print("✅ Modèle chargé.")

    # 3. Prédire sur toutes les images (une seule fois)
    print("🔄 Prédiction sur toutes les images (conf=0.1)...")
    predictions = {}
    total = len(ground_truth)
    for i, img_name in enumerate(ground_truth.keys()):
        img_file = TIFF_DIR / f"{img_name}.png"
        if not img_file.exists():
            img_file = TIFF_DIR / f"{img_name}.tiff"
            if not img_file.exists():
                continue
        # Inférence
        results = model(img_file, conf=0.1, verbose=False)
        # Récupérer les détections
        dets = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                dets.append((conf, cls_id))
        predictions[img_name] = dets
        if (i+1) % 50 == 0:
            print(f"   Progress: {i+1}/{total}")

    print("✅ Prédictions terminées.")

    # 4. Évaluer pour chaque seuil de 0.20 à 0.30 (pas 0.01)
    thresholds = [round(t/100, 2) for t in range(20, 31)]  # 0.20, 0.21, ..., 0.30
    results = []
    for thresh in thresholds:
        TP, FP, FN, TN = compute_metrics(thresh, predictions, ground_truth)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
        results.append({
            'threshold': thresh,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })
        print(f"Seuil {thresh:.2f}: F1={f1:.4f}, Précision={precision:.4f}, Rappel={recall:.4f}, Acc={accuracy:.4f}")

    # 5. Meilleur seuil
    best = max(results, key=lambda x: x['f1'])
    print("\n" + "="*60)
    print(f"🏆 MEILLEUR SEUIL : {best['threshold']:.2f} avec F1 = {best['f1']:.4f}")
    print(f"   Précision = {best['precision']:.4f}, Rappel = {best['recall']:.4f}")
    print("="*60)

    # 6. Sauvegarde CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"📄 Résultats sauvegardés dans {OUTPUT_CSV}")

    # 7. Optionnel : afficher un résumé graphique (si matplotlib disponible)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(df_results['threshold'], df_results['precision'], label='Précision', marker='o')
        plt.plot(df_results['threshold'], df_results['recall'], label='Rappel', marker='s')
        plt.plot(df_results['threshold'], df_results['f1'], label='F1-score', marker='^')
        plt.xlabel('Seuil de confiance')
        plt.ylabel('Score')
        plt.title('Évolution des métriques en fonction du seuil')
        plt.legend()
        plt.grid(True)
        plt.savefig(BASE_DIR / 'threshold_evaluation_plot.png')
        print("📈 Graphique sauvegardé sous threshold_evaluation_plot.png")
    except ImportError:
        print("⚠️ Matplotlib non installé, pas de graphique.")

if __name__ == "__main__":
    main()