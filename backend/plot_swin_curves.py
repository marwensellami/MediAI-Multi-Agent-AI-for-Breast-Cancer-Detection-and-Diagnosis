# backend/plot_swin_curves.py
import matplotlib.pyplot as plt

# Données extraites des logs de l'entraînement Swin
epochs = list(range(1, 16))
train_acc = [94.63, 96.67, 97.49, 98.16, 98.00, 99.29, 99.41, 99.33, 99.37, 99.84, 99.84, 99.65, 99.14, 99.53, 99.53]
val_acc   = [94.83, 96.24, 95.14, 96.87, 98.43, 96.87, 97.18, 97.96, 98.43, 96.87, 97.49, 97.96, 98.12, 97.02, 97.81]
train_loss = [0.1510, 0.0934, 0.0632, 0.0506, 0.0442, 0.0185, 0.0183, 0.0203, 0.0225, 0.0040, 0.0036, 0.0145, 0.0191, 0.0193, 0.0139]

# Tracer l'accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'b-o', label='Train Accuracy')
plt.plot(epochs, val_acc, 'r-o', label='Validation Accuracy')
plt.xlabel('Époque')
plt.ylabel('Accuracy (%)')
plt.title('Évolution de l\'accuracy (Swin)')
plt.legend()
plt.grid(True)

# Tracer la loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'g-o', label='Train Loss')
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.title('Évolution de la perte (Swin)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('swin_training_curves.png', dpi=150)
plt.show()
print("✅ Courbes Swin sauvegardées sous 'swin_training_curves.png'")