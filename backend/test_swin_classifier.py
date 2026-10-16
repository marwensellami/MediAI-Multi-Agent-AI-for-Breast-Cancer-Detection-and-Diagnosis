# backend/test_swin_classifier.py
import sys
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import timm

from app.ml.training.train_swin_classifier import SwinClassifier, LesionPatchDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path: Path, num_classes: int = 3, device: torch.device = None):
    """Charge le modèle Swin fine‑tuné."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"🔄 Chargement du modèle depuis {model_path}...")
    backbone = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=0)
    model = SwinClassifier(backbone, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("✅ Modèle chargé avec succès.")
    return model


def evaluate_on_dataset(model, data_dir: Path, batch_size: int = 32, device: torch.device = None):
    """Évalue le modèle sur l'ensemble du dataset de patches."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = LesionPatchDataset(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Métriques
    class_names = ["Normal", "Benign", "Malignant"]
    accuracy = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    logger.info(f"📊 Précision globale : {accuracy:.2f}%")

    # Rapport de classification
    print("\n📋 Rapport de classification :")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title('Matrice de confusion')
    plt.tight_layout()
    plt.savefig('confusion_matrix_swin.png', dpi=150)
    logger.info("💾 Matrice de confusion sauvegardée sous 'confusion_matrix_swin.png'")
    plt.show()

    return all_preds, all_labels


def predict_single_image(model, image_path: Path, device: torch.device = None):
    """Test le modèle sur une seule image (patch)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(1).item()

    class_names = ["Normal", "Benign", "Malignant"]
    logger.info(f"🖼️ {image_path.name} -> {class_names[pred_class]} (prob: {probs[0, pred_class].item():.4f})")
    return class_names[pred_class], probs[0, pred_class].item()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test du classifieur Swin (3 classes)")
    parser.add_argument("--model", type=str, default="app/ml/models/swin_breast_3class.pth",
                        help="Chemin du modèle .pth")
    parser.add_argument("--data_dir", type=str, default="app/data/lesion_patches",
                        help="Dossier des patches (normal/benign/malignant)")
    parser.add_argument("--image", type=str, default=None,
                        help="Tester sur une image spécifique (chemin)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Taille de lot pour l'évaluation sur dataset")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"❌ Modèle introuvable : {model_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, num_classes=3, device=device)

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error(f"❌ Image introuvable : {image_path}")
            sys.exit(1)
        predict_single_image(model, image_path, device)
    else:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logger.error(f"❌ Dossier data introuvable : {data_dir}")
            sys.exit(1)
        evaluate_on_dataset(model, data_dir, batch_size=args.batch_size, device=device)