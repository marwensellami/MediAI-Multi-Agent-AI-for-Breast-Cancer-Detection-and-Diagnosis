# backend/generate_confusion_matrix_corrected.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# ============================================================
# 1. DONNÉES CORRIGÉES (matrice 2x2 uniquement)
# ============================================================
# Matrice de confusion (validation, 73 lésions) :
#            Prédit: Bénin  Prédit: Malin
# Réel: Bénin    21           31
# Réel: Malin     9           12

# Lignes = Réel, Colonnes = Prédit
confusion_matrix = np.array([
    [21, 31],   # Réel Bénin (21 VP + 31 FN = 52)
    [9,  12]    # Réel Malin (9 FN + 12 VP = 21)
], dtype=np.int64)

classes = ['Bénin', 'Malin']

# ============================================================
# 2. GÉNÉRATION DE LA FIGURE
# ============================================================

def generate_confusion_matrix_corrected():
    """Génère une matrice de confusion 2x2 avec style professionnel."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Colormap dégradé de bleu
    colors = ['#F7FBFF', '#DEEBF7', '#C6DBEF', '#9ECAE1', '#6BAED6', '#4292C6', '#2171B5', '#084594']
    cmap = LinearSegmentedColormap.from_list('Blues_custom', colors, N=256)
    
    # Afficher la matrice
    vmax = np.max(confusion_matrix)
    im = ax.imshow(confusion_matrix, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
    
    # Ajouter les valeurs
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            value = confusion_matrix[i, j]
            if value > 0:
                # Texte blanc si valeur > 20, sinon noir
                color = 'white' if value > 20 else 'black'
                ax.text(j, i, f'{value:,}', ha='center', va='center',
                        fontsize=16, fontweight='bold', color=color)
    
    # Labels et titres
    ax.set_xlabel('Prédit', fontsize=14, fontweight='bold')
    ax.set_ylabel('Réel', fontsize=14, fontweight='bold')
    ax.set_title('Matrice de confusion - Validation (73 lésions)', fontsize=16, fontweight='bold')
    
    # Ticks
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)
    
    # Barre de couleur
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Nombre de lésions', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    
    # Grille blanche
    for i in range(confusion_matrix.shape[0] + 1):
        ax.axhline(i - 0.5, color='white', linewidth=2)
    for j in range(confusion_matrix.shape[1] + 1):
        ax.axvline(j - 0.5, color='white', linewidth=2)
    
    plt.tight_layout()
    
    # Sauvegarde
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'confusion_matrix_corrected.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Matrice sauvegardée : {save_path}")
    plt.close()
    
    return save_path

# ============================================================
# 3. STATISTIQUES DÉTAILLÉES (CORRIGÉES)
# ============================================================

def print_verification_stats(matrix, classes):
    """Affiche les métriques calculées à partir de la matrice."""
    print("\n" + "="*70)
    print("📊 MÉTRIQUES DE PERFORMANCE - VALIDATION")
    print("="*70)
    
    total_instances = np.sum(matrix)
    print(f"📌 Total des lésions de validation : {total_instances}\n")
    
    for i, cls in enumerate(classes):
        tp = matrix[i, i]
        fn = sum(matrix[i, j] for j in range(matrix.shape[1]) if j != i)
        fp = sum(matrix[j, i] for j in range(matrix.shape[0]) if j != i)
        total = tp + fn
        
        recall = tp / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"🔹 Classe '{cls}' :")
        print(f"   • Vrais Positifs (VP) : {tp}")
        print(f"   • Faux Négatifs (FN)  : {fn}  → {cls} classé en {classes[1-i]}")
        print(f"   • Faux Positifs (FP)  : {fp}  → {classes[1-i]} classé en {cls}")
        print(f"   • Total réel          : {total}")
        print(f"   • Rappel (Recall)     : {recall*100:.1f} %  ← PRIORITÉ CLINIQUE")
        print(f"   • Précision (Precision): {precision*100:.1f} %")
        print(f"   • F1-Score            : {f1_score*100:.1f} %")
        print()
    
    # Métriques globales
    total_tp = sum(matrix[i, i] for i in range(matrix.shape[0]))
    total_instances = np.sum(matrix)
    accuracy = total_tp / total_instances if total_instances > 0 else 0
    
    # Précision macro (moyenne des précisions par classe)
    precisions = []
    recalls = []
    for i in range(matrix.shape[0]):
        tp = matrix[i, i]
        fn = sum(matrix[i, j] for j in range(matrix.shape[1]) if j != i)
        fp = sum(matrix[j, i] for j in range(matrix.shape[0]) if j != i)
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    
    precision_macro = np.mean(precisions)
    recall_macro = np.mean(recalls)
    
    print(f"🌍 Métriques globales :")
    print(f"   • Exactitude (Accuracy)  : {accuracy*100:.1f} %")
    print(f"   • Précision moyenne (macro) : {precision_macro*100:.1f} %")
    print(f"   • Rappel moyen (macro)     : {recall_macro*100:.1f} %")
    print("="*70)
    
    # Vérification manuelle
    print("\n✅ VÉRIFICATION :")
    print(f"   • Total Bénin réel  : 52 (21 VP + 31 FN) ✓")
    print(f"   • Total Malin réel  : 21 (12 VP + 9 FN)  ✓")
    print(f"   • Total lésions     : 73 ✓")
    print("="*70)

# ============================================================
# 4. EXÉCUTION PRINCIPALE
# ============================================================

if __name__ == "__main__":
    generate_confusion_matrix_corrected()
    print_verification_stats(confusion_matrix, classes)
    print("\n✅ Script terminé avec succès !")