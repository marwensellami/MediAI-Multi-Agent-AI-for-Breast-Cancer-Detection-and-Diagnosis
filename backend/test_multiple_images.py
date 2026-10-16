# backend/test_multiple_images.py
import requests
import time
import json
import re
from pathlib import Path
import pandas as pd

BASE_URL = "http://localhost:8001"

# Liste des images à tester (chemin relatif depuis backend/)
IMAGE_PATHS = [
    "app/data/DMID_PNG/1024/TIFF/IMG014.png",
    "app/data/DMID_PNG/1024/TIFF/IMG030.png",
    "app/data/DMID_PNG/1024/TIFF/IMG011.png",
]

# Lecture de Metadata.xlsx pour obtenir les informations de référence
def get_metadata(image_id: str) -> dict:
    """Retourne les métadonnées (classe, type, etc.) pour une image donnée."""
    meta_path = Path("app/data/DMID_PNG/Metadata.xlsx")
    if not meta_path.exists():
        return None
    df = pd.read_excel(meta_path, header=2)
    df.columns = ['image', 'view', 'tissue', 'type', 'class', 'x', 'y', 'radius']
    df['image'] = df['image'].astype(str).str.strip()
    row = df[df['image'] == image_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def test_image(file_path: str) -> dict:
    """Envoie une image, suit la progression, récupère et analyse le rapport."""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ Fichier introuvable : {path}")
        return None

    image_id = path.stem
    metadata = get_metadata(image_id)
    print(f"\n📂 Test de {path.name} (ID: {image_id})")
    if metadata:
        print(f"   Classe réelle : {metadata.get('class')} (B=Bénin, M=Malin)")
        print(f"   Type d'anomalie : {metadata.get('type')}")
        # On peut essayer de déduire un BI-RADS approximatif (non présent dans ce fichier, mais on l'ajoute manuellement)
        # Pour l'instant on ne l'affiche pas car pas de colonne BI-RADS.
    else:
        print("   (Pas de métadonnées trouvées)")

    # Préparer l'upload
    url = f"{BASE_URL}/api/v1/diagnosis"
    ext = path.suffix.lower()
    if ext == ".dcm":
        mime = "application/dicom"
    elif ext in [".png"]:
        mime = "image/png"
    else:
        mime = "application/octet-stream"

    files = {"mammo_file": (path.name, open(path, "rb"), mime)}
    data = {
        "patient_id": "P-12345",
        "age": "55",
        "history": "Fumeuse, antécédents familiaux",
        "risk_factors": "Tabac",
    }

    try:
        response = requests.post(url, data=data, files=files)
        if response.status_code != 200:
            print(f"❌ Erreur upload : {response.status_code} - {response.text}")
            return None
        result = response.json()
        task_id = result.get("task_id")
        print(f"   Task ID : {task_id}")
    except Exception as e:
        print(f"❌ Erreur requête : {e}")
        return None
    finally:
        files["mammo_file"][1].close()

    # Suivi de la progression
    print("   Progression : ", end="", flush=True)
    for _ in range(60):  # maximum 60 * 2s = 2 minutes
        time.sleep(2)
        status_url = f"{BASE_URL}/api/v1/diagnosis/{task_id}/status"
        try:
            status_response = requests.get(status_url)
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"{status.get('percent', 0)}% ", end="", flush=True)
                if status.get("status") in ["COMPLETED", "FAILED"]:
                    print("")
                    break
            else:
                print("Erreur statut")
                break
        except Exception as e:
            print(f"Erreur : {e}")
            break

    # Récupération du résultat
    result_url = f"{BASE_URL}/api/v1/diagnosis/{task_id}/result"
    result_response = requests.get(result_url)
    if result_response.status_code != 200:
        print(f"❌ Erreur récupération résultat : {result_response.status_code}")
        return None

    report = result_response.json()
    report_text = report.get("report", "")

    # Extraction des informations clés du rapport
    birads_match = re.search(r"BI-RADS\s*(\d)", report_text)
    birads = birads_match.group(1) if birads_match else "N/A"

    prob_match = re.search(r"Probabilité de malignité\s*:\s*(\d+\.?\d*)%", report_text)
    prob = prob_match.group(1) if prob_match else "N/A"

    # Compter les lésions détectées
    lesion_count = report_text.count("Lésion")
    print(f"   Lésions détectées : {lesion_count}")
    print(f"   BI-RADS prédit : {birads}")
    print(f"   Probabilité maximale : {prob}%")

    # Vérification de cohérence avec la métadonnée
    if metadata:
        real_class = metadata.get('class')
        if real_class == 'M' and int(birads) >= 4:
            print("   ✅ Cohérent (malin → BI-RADS ≥4)")
        elif real_class == 'B' and int(birads) <= 3:
            print("   ✅ Cohérent (bénin → BI-RADS ≤3)")
        else:
            print("   ⚠️ Incohérence possible (vérifier manuellement)")

    # Sauvegarde du rapport complet
    report_file = f"report_{image_id}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"   📄 Rapport sauvegardé dans {report_file}")

    return {
        "image_id": image_id,
        "task_id": task_id,
        "birads": birads,
        "prob": prob,
        "lesions": lesion_count,
        "metadata": metadata,
    }


if __name__ == "__main__":
    print("🧪 TEST DE MULTIPLES IMAGES")
    print("=" * 50)
    results = []
    for img_path in IMAGE_PATHS:
        res = test_image(img_path)
        if res:
            results.append(res)
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    for r in results:
        meta = r.get("metadata")
        class_str = meta.get('class') if meta else "?"
        print(f"{r['image_id']}: Classe réelle={class_str}, BI-RADS prédit={r['birads']}, Prob={r['prob']}%, Lésions={r['lesions']}")
    print("\n✅ Tests terminés.")