import logging
import numpy as np
import pydicom
import torch
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class DICOMProcessor:
    """Processeur pour les fichiers DICOM (mammographie, écho, IRM)."""

    # Taille d'entrée standard pour Swin Transformer (224x224)
    TARGET_SIZE = (224, 224)

    @staticmethod
    def read_dicom(file_path: Path) -> Dict[str, Any]:
        """
        Lit un fichier DICOM et extrait les informations essentielles.
        Retourne un dictionnaire contenant :
        - 'pixels': tableau numpy 2D (hauteur, largeur)
        - 'metadata': dictionnaire des métadonnées (modality, pixel_spacing, etc.)
        - 'is_valid': bool
        """
        try:
            ds = pydicom.dcmread(file_path, force=True)
            if not hasattr(ds, 'pixel_array'):
                raise ValueError("Le fichier DICOM ne contient pas de pixel_array.")

            pixels = ds.pixel_array

            # Gérer les cas où les pixels sont en 16 bits
            if pixels.dtype == np.uint16:
                # Normalisation simple en uint8 pour l'affichage (mais on garde les valeurs brutes pour le traitement)
                pass

            # ============================================================
            # Extraction robuste du PixelSpacing
            # ============================================================
            pixel_spacing = None

            # Ordre de priorité pour la récupération du PixelSpacing
            # 1. PixelSpacing (tag DICOM standard)
            if hasattr(ds, 'PixelSpacing') and ds.PixelSpacing:
                pixel_spacing = ds.PixelSpacing
                logger.info(f"📏 PixelSpacing trouvé: {pixel_spacing}")
            
            # 2. NominalScannedPixelSpacing (pour les images scannées)
            elif hasattr(ds, 'NominalScannedPixelSpacing') and ds.NominalScannedPixelSpacing:
                pixel_spacing = ds.NominalScannedPixelSpacing
                logger.info(f"📏 NominalScannedPixelSpacing trouvé: {pixel_spacing}")
            
            # 3. ImagerPixelSpacing (pour les images issues d'un imageur)
            elif hasattr(ds, 'ImagerPixelSpacing') and ds.ImagerPixelSpacing:
                pixel_spacing = ds.ImagerPixelSpacing
                logger.info(f"📏 ImagerPixelSpacing trouvé: {pixel_spacing}")
            
            # 4. PixelSpacingCalibration (autre tag possible)
            elif hasattr(ds, 'PixelSpacingCalibration') and ds.PixelSpacingCalibration:
                pixel_spacing = ds.PixelSpacingCalibration
                logger.info(f"📏 PixelSpacingCalibration trouvé: {pixel_spacing}")
            
            # 5. Si aucune valeur n'est trouvée, on utilisera [1.0, 1.0] par défaut
            else:
                logger.warning("⚠️ Aucun PixelSpacing trouvé, utilisation de [1.0, 1.0] par défaut")
                pixel_spacing = [1.0, 1.0]

            # Normalisation du PixelSpacing pour s'assurer que c'est une liste
            if isinstance(pixel_spacing, (int, float)):
                pixel_spacing = [float(pixel_spacing), float(pixel_spacing)]
            elif isinstance(pixel_spacing, pydicom.multival.MultiValue):
                pixel_spacing = list(pixel_spacing)
            elif isinstance(pixel_spacing, (list, tuple)) and len(pixel_spacing) == 1:
                pixel_spacing = [float(pixel_spacing[0]), float(pixel_spacing[0])]
            elif not isinstance(pixel_spacing, (list, tuple)) or len(pixel_spacing) < 2:
                logger.warning(f"⚠️ Format de PixelSpacing inattendu: {pixel_spacing}, utilisation de [1.0, 1.0]")
                pixel_spacing = [1.0, 1.0]
            else:
                # S'assurer que les valeurs sont des flottants
                try:
                    pixel_spacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
                except (ValueError, TypeError):
                    logger.warning(f"⚠️ Impossible de convertir PixelSpacing en flottants: {pixel_spacing}")
                    pixel_spacing = [1.0, 1.0]

            # Extraction des métadonnées importantes
            metadata = {
                "modality": getattr(ds, "Modality", "UNKNOWN"),
                "patient_id": getattr(ds, "PatientID", "UNKNOWN"),
                "patient_name": getattr(ds, "PatientName", "UNKNOWN"),
                "patient_sex": getattr(ds, "PatientSex", "UNKNOWN"),
                "patient_age": getattr(ds, "PatientAge", "UNKNOWN"),
                "study_date": getattr(ds, "StudyDate", "UNKNOWN"),
                "study_time": getattr(ds, "StudyTime", "UNKNOWN"),
                "study_description": getattr(ds, "StudyDescription", "UNKNOWN"),
                "series_description": getattr(ds, "SeriesDescription", "UNKNOWN"),
                "pixel_spacing": pixel_spacing,  # PixelSpacing robuste
                "rows": getattr(ds, "Rows", pixels.shape[0]),
                "columns": getattr(ds, "Columns", pixels.shape[1]),
                "photometric_interpretation": getattr(ds, "PhotometricInterpretation", "MONOCHROME2"),
                "patient_position": getattr(ds, "PatientPosition", None),
                "image_orientation": getattr(ds, "ImageOrientationPatient", None),
                "image_position": getattr(ds, "ImagePositionPatient", None),
                "slice_thickness": getattr(ds, "SliceThickness", None),
                "spacing_between_slices": getattr(ds, "SpacingBetweenSlices", None),
                "manufacturer": getattr(ds, "Manufacturer", "UNKNOWN"),
                "model_name": getattr(ds, "ManufacturerModelName", "UNKNOWN"),
                "institution_name": getattr(ds, "InstitutionName", "UNKNOWN"),
                "laterality": getattr(ds, "Laterality", None),  # R/L pour mammographie
                "view_position": getattr(ds, "ViewPosition", None),  # CC/MLO
            }

            # Ajouter les informations spécifiques à la mammographie (MG)
            if metadata["modality"] == "MG":
                # Essayer d'extraire le quadrant depuis le tag ViewPosition ou une autre info
                view = getattr(ds, "ViewPosition", None)
                if view:
                    metadata["view"] = view
                
                # Essayer d'extraire la latéralité
                laterality = getattr(ds, "Laterality", None)
                if laterality:
                    metadata["laterality"] = laterality

            # Vérification de la modalité pour adapter le prétraitement plus tard
            if metadata["modality"] not in ["MG", "US", "MRI", "OT", "CT", "PET", "NM", "XA", "CR", "DX"]:
                logger.warning(f"⚠️ Modalité non standard détectée : {metadata['modality']}")

            # Si l'image est en couleur (RGB), on la convertit en niveaux de gris
            if len(pixels.shape) == 3 and pixels.shape[2] == 3:
                pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
                logger.info("🔄 Image RGB convertie en niveaux de gris")

            # Si l'image a plusieurs couches (ex: images fonctionnelles), on prend la première
            if len(pixels.shape) == 3 and pixels.shape[2] != 3:
                logger.info(f"📊 Image multi-couches détectée, prise de la première couche")
                pixels = pixels[:, :, 0]

            logger.info(f"✅ DICOM chargé: {file_path.name}")
            logger.info(f"   - Modalité: {metadata['modality']}")
            logger.info(f"   - Dimensions: {pixels.shape}")
            logger.info(f"   - PixelSpacing: {pixel_spacing}")

            return {
                "pixels": pixels,
                "metadata": metadata,
                "is_valid": True,
            }

        except Exception as e:
            logger.error(f"❌ Erreur lors de la lecture du DICOM {file_path}: {e}")
            return {"is_valid": False, "error": str(e)}

    @staticmethod
    def preprocess_for_model(
        pixels: np.ndarray,
        target_size: Tuple[int, int] = TARGET_SIZE,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Prend un tableau numpy de pixels, le redimensionne et le normalise en tenseur PyTorch.
        - Redimensionne à target_size (ex: 224x224).
        - Convertit en float32.
        - Normalise entre 0 et 1 (ou standardisation selon les besoins).
        - Ajoute la dimension de batch (1, C, H, W).
        """
        # 1. Redimensionnement
        if pixels.shape[0] != target_size[0] or pixels.shape[1] != target_size[1]:
            # Utilisation d'OpenCV pour un redimensionnement de qualité
            pixels_resized = cv2.resize(pixels, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            pixels_resized = pixels

        # 2. Normalisation en float32 [0, 1]
        if pixels_resized.dtype != np.float32:
            pixels_resized = pixels_resized.astype(np.float32)

        if normalize:
            # Si les valeurs sont en uint16 (0-65535), on divise par 65535
            if pixels_resized.max() > 1.0:
                # Utiliser le max pour éviter les valeurs extrêmes
                max_val = pixels_resized.max()
                if max_val > 0:
                    pixels_resized = pixels_resized / max_val
                # Alternative: utiliser min-max global (0-255 pour uint8)
                # else: pixels_resized = pixels_resized / 255.0

        # 3. Convertir en tenseur PyTorch (1, H, W) puis ajouter le canal (1, 1, H, W)
        tensor = torch.from_numpy(pixels_resized).unsqueeze(0)  # (1, H, W)

        # 4. Normalisation standard pour les modèles pré-entraînés (optionnelle)
        # Swin attend un normalize avec mean=[0.485, 0.456, 0.406] et std=[0.229, 0.224, 0.225]
        # Mais comme nos images sont en niveaux de gris, on va répéter le canal 3 fois.
        tensor = tensor.repeat(3, 1, 1)  # (3, H, W)

        # 5. Normaliser avec les statistiques ImageNet (pour Swin/YOLO)
        # Cependant, pour les images médicales, on préfère parfois une normalisation différente.
        # On applique une normalisation simple en standardisant avec la moyenne et l'écart-type de l'image.
        # Mais les modèles pré-entraînés s'attendent à des images RGB normalisées par ImageNet.
        # On va donc utiliser une normalisation par min-max.
        # (Nous ajusterons cela dans le fine-tuning ultérieur)

        # Retourne un tenseur prêt à être passé dans le modèle (batch_size=1)
        return tensor.unsqueeze(0)  # (1, 3, H, W)

    @staticmethod
    def load_and_prepare(file_path: Path, target_size: Tuple[int, int] = TARGET_SIZE) -> Optional[torch.Tensor]:
        """
        Fonction utilitaire qui combine read_dicom et preprocess_for_model.
        Retourne le tenseur prêt à l'emploi, ou None en cas d'erreur.
        """
        result = DICOMProcessor.read_dicom(file_path)
        if not result["is_valid"]:
            return None

        pixels = result["pixels"]
        tensor = DICOMProcessor.preprocess_for_model(pixels, target_size)
        return tensor

    @staticmethod
    def prepare_for_yolo(pixels: np.ndarray) -> np.ndarray:
        """
        Prépare l'image pour YOLO (qui attend des images RGB en uint8 0-255).
        """
        if pixels.dtype != np.uint8:
            # Normaliser en 0-255
            if pixels.max() > 0:
                pixels = (pixels / pixels.max() * 255).astype(np.uint8)
            else:
                pixels = pixels.astype(np.uint8)

        # Convertir en RGB (YOLO attend 3 canaux)
        if len(pixels.shape) == 2:
            pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
        return pixels

    @staticmethod
    def extract_pixel_spacing(metadata: Dict[str, Any]) -> Tuple[float, float]:
        """
        Extrait le PixelSpacing des métadonnées de manière robuste.
        
        Args:
            metadata: Dictionnaire des métadonnées DICOM
        
        Returns:
            Tuple[float, float]: (spacing_x, spacing_y) en mm/pixel
        """
        pixel_spacing = metadata.get("pixel_spacing", [1.0, 1.0])
        
        if isinstance(pixel_spacing, (int, float)):
            return float(pixel_spacing), float(pixel_spacing)
        elif isinstance(pixel_spacing, (list, tuple)) and len(pixel_spacing) >= 2:
            try:
                return float(pixel_spacing[0]), float(pixel_spacing[1])
            except (ValueError, TypeError):
                return 1.0, 1.0
        else:
            return 1.0, 1.0

    @staticmethod
    def get_physical_size(pixels: np.ndarray, metadata: Dict[str, Any]) -> Tuple[float, float]:
        """
        Calcule la taille physique de l'image en mm.
        
        Args:
            pixels: Tableau numpy des pixels
            metadata: Métadonnées DICOM
        
        Returns:
            Tuple[float, float]: (width_mm, height_mm)
        """
        spacing_x, spacing_y = DICOMProcessor.extract_pixel_spacing(metadata)
        height, width = pixels.shape[:2]
        
        width_mm = width * spacing_x
        height_mm = height * spacing_y
        
        return width_mm, height_mm