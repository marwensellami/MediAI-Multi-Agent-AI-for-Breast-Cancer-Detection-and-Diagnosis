# backend/test_api_full.py
"""
Script de test complet pour l'API de diagnostic.
Permet de tester avec différents fichiers et paramètres.
"""

import requests
import time
import json
import sys
import argparse
from pathlib import Path

BASE_URL = "http://localhost:8001"

# ============================================================
# 1. PARAMÈTRES EN LIGNE DE COMMANDE
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Test l'API de diagnostic mammographique")
    parser.add_argument(
        "image",
        nargs="?",
        help="Chemin vers l'image (DICOM, PNG, JPG, TIFF)"
    )
    parser.add_argument(
        "--pixel-spacing",
        type=float,
        default=None,
        help="Pixel spacing (mm/pixel) à forcer pour la conversion (ex: 0.1)"
    )
    parser.add_argument(
        "--patient-id",
        default="P-12345",
        help="ID du patient"
    )
    parser.add_argument(
        "--age",
        type=int,
        default=55,
        help="Âge du patient"
    )
    parser.add_argument(
        "--history",
        default="Fumeuse, antécédents familiaux",
        help="Antécédents médicaux"
    )
    parser.add_argument(
        "--risk-factors",
        default="Tabac",
        help="Facteurs de risque"
    )
    return parser.parse_args()

# ============================================================
# 2. RECHERCHE AUTOMATIQUE DE FICHIER
# ============================================================
def find_file(base_name, extensions, dirs):
    for dir_path in dirs:
        if not dir_path.exists():
            continue
        for ext in extensions:
            file_path = dir_path / f"{base_name}{ext}"
            if file_path.exists():
                return file_path
    return None

# ============================================================
# 3. EXÉCUTION PRINCIPALE
# ============================================================
def main():
    args = parse_args()

    # ---- 3.1 Déterminer le chemin de l'image ----
    image_path = None
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Fichier spécifié introuvable : {image_path}")
            sys.exit(1)
        print(f"📂 Utilisation du fichier : {image_path}")
    else:
        # Recherche automatique
        possible_dirs = [
            Path("app/data/dicom_samples"),
            Path("app/data/uploads"),
            Path("app/data/DMID_PNG/1024/TIFF"),
            Path("app/data/DMID_PNG/512/TIFF"),
        ]
        extensions = [".dcm", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]

        mammo_path = find_file("mammo", extensions, possible_dirs)
        if mammo_path:
            image_path = mammo_path
            print(f"📂 Mammographie trouvée : {image_path}")
        else:
            print("❌ Aucun fichier d'image trouvé.")
            print("   Dossiers recherchés :")
            for d in possible_dirs:
                print(f"     - {d}")
            print("   Vous pouvez aussi passer le chemin en argument :")
            print("     python test_api_full.py chemin/vers/image.png")
            sys.exit(1)

    # ---- 3.2 Test de santé ----
    print("\n🔍 Test de santé...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=60)
        print(f"✅ {response.json()}")
    except Exception as e:
        print(f"❌ Erreur de connexion : {e}")
        print("   Assurez-vous que le serveur est lancé sur le port 8001.")
        sys.exit(1)

    # ---- 3.3 Lancement du diagnostic ----
    print("\n📤 Lancement du diagnostic...")
    url = f"{BASE_URL}/api/v1/diagnosis"

    # Déterminer le type MIME
    ext = image_path.suffix.lower()
    if ext == ".dcm":
        mime = "application/dicom"
    elif ext in [".png"]:
        mime = "image/png"
    elif ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext in [".tif", ".tiff"]:
        mime = "image/tiff"
    else:
        mime = "application/octet-stream"

    # Préparer les fichiers et données
    files = {
        "mammo_file": (image_path.name, open(image_path, "rb"), mime)
    }

    data = {
        "patient_id": args.patient_id,
        "age": str(args.age),
        "history": args.history,
        "risk_factors": args.risk_factors,
    }

    # Ajouter le pixel spacing si fourni
    if args.pixel_spacing is not None:
        data["pixel_spacing"] = str(args.pixel_spacing)
        print(f"📏 Pixel spacing forcé : {args.pixel_spacing} mm/pixel")

    try:
        response = requests.post(url, data=data, files=files, timeout=60)
        if response.status_code == 200:
            result = response.json()
            task_id = result.get("task_id")
            print(f"✅ Task ID: {task_id}")
        else:
            print(f"❌ Erreur: {response.status_code} - {response.text}")
            files["mammo_file"][1].close()
            sys.exit(1)
    finally:
        # Fermer le fichier
        files["mammo_file"][1].close()

    # ---- 3.4 Suivi de la progression ----
    print("\n📊 Suivi de la progression...")
    max_attempts = 60
    for i in range(max_attempts):
        time.sleep(2)
        status_url = f"{BASE_URL}/api/v1/diagnosis/{task_id}/status"
        try:
            status_response = requests.get(status_url, timeout=60)
            if status_response.status_code == 200:
                status = status_response.json()
                percent = status.get('percent', 0)
                status_text = status.get('status', 'UNKNOWN')
                print(f"  [{i+1}/{max_attempts}] {percent}% - {status_text}")
                if status_text in ["COMPLETED", "FAILED"]:
                    break
            else:
                print(f"⚠️ Erreur statut: {status_response.status_code}")
        except Exception as e:
            print(f"⚠️ Erreur: {e}")
    else:
        print("⏰ Délai d'attente dépassé. Vérifiez le statut manuellement.")

    # ---- 3.5 Récupération du résultat ----
    print("\n📄 Récupération du résultat...")
    result_url = f"{BASE_URL}/api/v1/diagnosis/{task_id}/result"
    try:
        result_response = requests.get(result_url, timeout=60)
        if result_response.status_code == 200:
            result = result_response.json()
            print("\n" + "="*60)
            print("RÉSULTAT FINAL:")
            print("="*60)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if result.get("report"):
                print("\n📋 RAPPORT:")
                print(result["report"])
        elif result_response.status_code == 202:
            print("⏳ La tâche est encore en cours de traitement. Réessayez plus tard.")
        else:
            print(f"❌ Erreur: {result_response.status_code} - {result_response.text}")
    except Exception as e:
        print(f"❌ Erreur de récupération : {e}")

    print("\n✅ Test terminé !")

if __name__ == "__main__":
    main()