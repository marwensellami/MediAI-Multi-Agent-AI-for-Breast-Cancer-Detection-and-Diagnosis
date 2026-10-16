# backend/run_all_tests.py
import subprocess
import sys
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("LANCEMENT DE TOUS LES TESTS")
print(f"Début : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Exécuter le test IoU
print("\n" + "=" * 40)
print("EXÉCUTION DU TEST IoU...")
print("=" * 40)
try:
    subprocess.run([sys.executable, "test_iou.py"], check=True)
    print("✅ Test IoU terminé avec succès")
except subprocess.CalledProcessError as e:
    print(f"❌ Erreur lors du test IoU : {e}")

# Exécuter le test Confiance
print("\n" + "=" * 40)
print("EXÉCUTION DU TEST CONF...")
print("=" * 40)
try:
    subprocess.run([sys.executable, "test_conf.py"], check=True)
    print("✅ Test Conf terminé avec succès")
except subprocess.CalledProcessError as e:
    print(f"❌ Erreur lors du test Conf : {e}")

print("\n" + "=" * 80)
print("TOUS LES TESTS SONT TERMINÉS")
print(f"Fin : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Afficher les fichiers de log créés
log_dir = Path("logs")
if log_dir.exists():
    log_files = sorted(log_dir.glob("*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)
    print("\n📁 Fichiers de log créés :")
    for log_file in log_files[:4]:  # Afficher les 4 derniers
        print(f"   - {log_file}")