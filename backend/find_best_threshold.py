# backend/find_best_threshold.py
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Cette protection est OBLIGATOIRE sur Windows
if __name__ == "__main__":
    
    # Charger votre modèle entraîné
    model = YOLO("app/ml/models/yolo_breast.pt")
    
    # Seuils à tester : de 0.20 à 0.30 inclus, par pas de 0.01
    thresholds = np.arange(0.20, 0.31, 0.01)
    
    # Liste pour stocker les résultats
    results = []
    
    print("🔍 Évaluation des seuils de confiance sur le jeu de validation...")
    print("-" * 60)
    
    for thresh in thresholds:
        # Valider avec ce seuil, avec workers=0 pour éviter le multiprocessing
        metrics = model.val(
            data="yolo_dataset/dataset.yaml",
            conf=thresh,
            iou=0.5,
            device=0,
            workers=0,  # <-- IMPORTANT : désactive le multiprocessing
            verbose=False
        )
        
        # Récupérer les métriques
        mAP50 = metrics.box.map50
        mAP50_95 = metrics.box.map
        precision = metrics.box.mp
        recall = metrics.box.mr
        
        results.append({
            'seuil': round(thresh, 2),
            'mAP50': round(mAP50, 4),
            'mAP50-95': round(mAP50_95, 4),
            'Précision': round(precision, 4),
            'Rappel': round(recall, 4)
        })
        
        print(f"Seuil {thresh:.2f} → mAP50: {mAP50:.4f}, mAP50-95: {mAP50_95:.4f}")
    
    # Afficher le tableau complet
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("📊 RÉSULTATS COMPLETS")
    print("="*70)
    print(df.to_string(index=False))
    
    # Déterminer le meilleur seuil selon mAP50-95
    best_idx = df['mAP50-95'].idxmax()
    best = df.iloc[best_idx]
    print("\n" + "="*70)
    print(f"✅ MEILLEUR SEUIL : {best['seuil']:.2f}")
    print(f"   - mAP50-95 : {best['mAP50-95']:.4f}")
    print(f"   - mAP50    : {best['mAP50']:.4f}")
    print(f"   - Précision: {best['Précision']:.4f}")
    print(f"   - Rappel   : {best['Rappel']:.4f}")
    print("="*70)
    
    # Enregistrer les résultats dans un fichier CSV
    df.to_csv("threshold_evaluation.csv", index=False)
    print("\n💾 Résultats sauvegardés dans 'threshold_evaluation.csv'")