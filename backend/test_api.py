# backend/test_api.py
import requests
import time
import json

# 1. Lancer un diagnostic
url = "http://localhost:8001/api/v1/diagnosis"
files = {
    "mammo_file": open("app/data/dicom_samples/mammo.dcm", "rb"),
    "us_file": open("app/data/dicom_samples/echo.dcm", "rb"),
}
data = {
    "patient_id": "P-12345",
    "age": "55",
    "history": "Fumeuse, antécédents familiaux",
}

response = requests.post(url, data=data, files=files)
if response.status_code != 200:
    print(f"❌ Erreur: {response.text}")
    exit()

task = response.json()
task_id = task["task_id"]
print(f"✅ Task ID: {task_id}")

# 2. Suivre la progression
for i in range(30):  # Attendre max 30 secondes
    status_response = requests.get(f"http://localhost:8001/api/v1/diagnosis/{task_id}/status")
    if status_response.status_code == 200:
        status = status_response.json()
        print(f"📊 Progression: {status.get('percent', 0)}% - {status.get('status', 'UNKNOWN')}")
        if status.get("status") in ["COMPLETED", "FAILED"]:
            break
    time.sleep(2)

# 3. Récupérer le résultat
result_response = requests.get(f"http://localhost:8001/api/v1/diagnosis/{task_id}/result")
if result_response.status_code == 200:
    result = result_response.json()
    print(f"\n📄 RÉSULTAT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
else:
    print(f"❌ Erreur: {result_response.text}")