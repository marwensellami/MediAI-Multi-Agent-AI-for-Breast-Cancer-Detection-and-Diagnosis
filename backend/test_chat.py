# backend/test_chat.py
import requests

# ✅ URL CORRECTE AVEC LE PRÉFIXE /chat
url = "http://localhost:8001/api/v1/chat/"

payload = {
    "task_id": "60b37d17-12fb-46a5-99ac-404c8a30d3ff",
    "question": "Pourquoi la lésion 1 est-elle classée BI-RADS 4B ?"
}

try:
    response = requests.post(url, json=payload, timeout=120)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Réponse du chatbot :")
        print(data["answer"])
    else:
        print(f"Erreur: {response.text}")
except Exception as e:
    print(f"Erreur de connexion : {e}")