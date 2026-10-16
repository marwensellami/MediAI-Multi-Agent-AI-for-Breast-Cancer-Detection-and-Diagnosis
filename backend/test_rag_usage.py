# backend/test_rag_integration.py
"""
Test d'intégration du RAG et du Knowledge Graph via l'API REST.
Vérifie que les documents RAG et les connaissances KG sont bien utilisés.
"""

import requests
import time
import json
from pathlib import Path

BASE_URL = "http://localhost:8001"
API_DIAGNOSIS = f"{BASE_URL}/api/v1/diagnosis"
IMAGE_PATH = Path("app/data/DMID_PNG/1024/TIFF/IMG001.png")

def test_rag_and_kg():
    print("🧪 Test d'intégration RAG & KG via l'API")

    # ---- 1. Vérifier que le serveur est accessible ----
    try:
        health = requests.get(f"{BASE_URL}/health")
        if health.status_code != 200:
            print("❌ Le serveur n'est pas accessible.")
            return
        print("✅ Serveur accessible.")
    except requests.ConnectionError:
        print("❌ Impossible de se connecter au serveur. Assurez-vous qu'il tourne sur le port 8001.")
        return

    # ---- 2. Préparer le fichier image ----
    if not IMAGE_PATH.exists():
        print(f"❌ Image de test non trouvée : {IMAGE_PATH}")
        return

    # ---- 3. Lancer un diagnostic ----
    with open(IMAGE_PATH, "rb") as f:
        files = {"mammo_file": (IMAGE_PATH.name, f, "image/png")}
        data = {
            "patient_id": "RAG_TEST",
            "age": 55,
            "history": "Test RAG",
            "risk_factors": "Fumeuse"
        }
        print("📤 Envoi de la requête de diagnostic...")
        response = requests.post(API_DIAGNOSIS, data=data, files=files)

    if response.status_code != 200:
        print(f"❌ Erreur lors de l'appel API : {response.status_code} - {response.text}")
        return

    task_id = response.json().get("task_id")
    print(f"✅ Task ID reçu : {task_id}")

    # ---- 4. Suivre la progression ----
    print("⏳ Attente de la fin du traitement...")
    status_url = f"{BASE_URL}/api/v1/diagnosis/{task_id}/status"
    while True:
        status_resp = requests.get(status_url)
        if status_resp.status_code != 200:
            print("❌ Erreur lors du suivi de la progression.")
            return
        status_data = status_resp.json()
        percent = status_data.get("percent", 0)
        status = status_data.get("status", "UNKNOWN")
        print(f"   Progression : {percent}% - {status}")
        if status in ["COMPLETED", "FAILED"]:
            break
        time.sleep(2)

    if status == "FAILED":
        print("❌ Le diagnostic a échoué.")
        return

    # ---- 5. Récupérer le résultat final ----
    result_url = f"{BASE_URL}/api/v1/diagnosis/{task_id}/result"
    result_resp = requests.get(result_url)
    if result_resp.status_code != 200:
        print(f"❌ Erreur lors de la récupération du résultat : {result_resp.status_code}")
        return

    result = result_resp.json()
    report = result.get("report", "")

    print("\n📋 RAPPORT FINAL (extrait) :")
    print(report[:800] + "..." if len(report) > 800 else report)

    # ---- 6. Vérifier la présence des références RAG ----
    rag_keywords = ["RÉFÉRENCES", "AGO_2025", "breastcancerscreening", "NCCN", "guidelines"]
    found_rag = [kw for kw in rag_keywords if kw.lower() in report.lower()]
    if found_rag:
        print(f"\n✅ RAG utilisé : références trouvées -> {found_rag}")
    else:
        print("\n⚠️ Aucune référence RAG trouvée dans le rapport.")

    # ---- 7. Vérifier la présence des diagnostics KG ----
    kg_keywords = ["diagnostic", "pathologie", "différentiel", "BI-RADS"]
    found_kg = [kw for kw in kg_keywords if kw.lower() in report.lower()]
    if found_kg:
        print(f"✅ KG utilisé : mentions médicales trouvées -> {found_kg}")
    else:
        print("⚠️ Aucune mention du KG trouvée dans le rapport.")

    # ---- 8. Vérifier que le rapport contient des recommandations ----
    if "biopsie" in report.lower() or "surveillance" in report.lower() or "suivi" in report.lower():
        print("✅ Recommandations cliniques présentes.")
    else:
        print("⚠️ Aucune recommandation clinique explicite trouvée.")

    print("\n🎯 Test terminé.")

if __name__ == "__main__":
    test_rag_and_kg()