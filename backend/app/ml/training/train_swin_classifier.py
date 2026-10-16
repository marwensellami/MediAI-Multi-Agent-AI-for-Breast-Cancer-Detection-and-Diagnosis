# backend/app/ml/training/train_swin_classifier.py
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import timm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LesionPatchDataset(Dataset):
    """
    Dataset pour les patches de lésions (3 classes).
    Structure attendue :
        lesions_patches/
            normal/    -> classe 0
            benign/    -> classe 1
            malignant/ -> classe 2
    """
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        # Mapping des noms de dossiers vers les indices de classe
        class_mapping = {
            "normal": 0,
            "benign": 1,
            "malignant": 2,
        }

        for folder_name, label in class_mapping.items():
            folder_path = self.data_dir / folder_name
            if not folder_path.exists():
                logger.warning(f"⚠️ Dossier {folder_path} introuvable, ignoré.")
                continue
            for img_path in folder_path.glob("*.png"):
                self.samples.append((str(img_path), label))

        logger.info(f"📊 {len(self.samples)} échantillons chargés.")
        for folder_name, label in class_mapping.items():
            count = sum(1 for _, lbl in self.samples if lbl == label)
            logger.info(f"   {folder_name.capitalize()} : {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class SwinClassifier(nn.Module):
    """
    Swin Transformer avec une tête de classification personnalisée.
    Le backbone est chargé sans tête (num_classes=0).
    """
    def __init__(self, backbone, num_classes=3):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features
        self.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)  # sortie du backbone (B, C) ou (B, C, H, W)
        # Gérer les différents formats de sortie
        if x.dim() == 2:          # (B, C) déjà un vecteur
            pass
        elif x.dim() == 3:        # (B, L, C) : on moyenne sur les patches
            x = x.mean(dim=1)
        elif x.dim() == 4:        # (B, C, H, W) : on moyenne sur H et W
            x = x.mean(dim=[2, 3])
        else:
            raise ValueError(f"Shape inattendue : {x.shape}")
        x = self.fc(x)
        return x


def train_swin_classifier(
    patches_dir: Path,
    model_save_path: Path,
    num_epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_classes: int = 3,
):
    """
    Entraîne un classifieur Swin Transformer sur les patches de lésions.

    Args:
        patches_dir : Dossier contenant les sous-dossiers normal/, benign/, malignant/
        model_save_path : Chemin où sauvegarder les poids du modèle (.pth)
        num_epochs : Nombre d'époques
        batch_size : Taille des lots
        learning_rate : Taux d'apprentissage
        num_classes : Nombre de classes (par défaut 3)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ Utilisation de : {device}")

    # Transformations pour l'entraînement (avec augmentation)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Transformations pour la validation (sans augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Chargement du dataset complet avec transformations d'entraînement
    dataset = LesionPatchDataset(patches_dir, transform=train_transform)

    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    # On remplace la transform du dataset de validation par val_transform
    # (attention : le dataset de validation est un Subset, on doit accéder à son dataset parent)
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    logger.info(f"📊 Train : {len(train_dataset)} échantillons, Val : {len(val_dataset)}")

    # Chargement du backbone Swin (pré-entraîné)
    logger.info("🔄 Chargement du backbone Swin...")
    backbone = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0)
    # On freeze le backbone pour accélérer l'entraînement (optionnel)
    # for param in backbone.parameters():
    #     param.requires_grad = False

    model = SwinClassifier(backbone, num_classes=num_classes)
    model.to(device)

    # Optimiseur et fonction de perte
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        # --- Phase d'entraînement ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Mise à jour de la barre de progression
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100.*correct/total:.2f}%"
            })

        train_acc = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)

        # --- Phase de validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        logger.info(
            f"📊 Epoch {epoch}: Train Loss {avg_loss:.4f}, Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%"
        )

        # Sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, str(model_save_path))
            logger.info(f"💾 Meilleur modèle sauvegardé à {model_save_path} (Val Acc {val_acc:.2f}%)")

    logger.info(f"✅ Entraînement terminé. Meilleure précision en validation : {best_val_acc:.2f}%")
    logger.info(f"📁 Modèle sauvegardé sous : {model_save_path}")


if __name__ == "__main__":
    # Chemins
    BASE_DIR = Path(__file__).parent.parent.parent.parent  # backend/
    PATCHES_DIR = BASE_DIR / "app" / "data" / "lesion_patches"
    MODELS_DIR = BASE_DIR / "app" / "ml" / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = MODELS_DIR / "swin_breast_3class.pth"

    if not PATCHES_DIR.exists():
        logger.error(f"❌ Dossier des patches introuvable : {PATCHES_DIR}")
        logger.error("Veuillez d'abord exécuter extract_patches_from_dmid.py")
        sys.exit(1)

    logger.info(f"📁 Patches : {PATCHES_DIR}")
    logger.info(f"📁 Sauvegarde modèle : {MODEL_SAVE_PATH}")

    # Lancer l'entraînement
    train_swin_classifier(
        patches_dir=PATCHES_DIR,
        model_save_path=MODEL_SAVE_PATH,
        num_epochs=15,          # Vous pouvez augmenter à 20-30 si le dataset est grand
        batch_size=32,          # Ajustez selon votre GPU (RTX 4060 8GB -> 32 convient)
        learning_rate=1e-4,
        num_classes=3,          # Normal, Bénin, Malin
    )