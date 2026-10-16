# backend/app/ml/processors/image_reader.py
import logging
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)

class ImageProcessor:
    @staticmethod
    def read_file(file_path: Path, pixel_spacing_mm: float = None) -> dict:
        """
        Lit une image standard (PNG, JPEG, TIFF, BMP, etc.).
        Calcule la distance physique entre deux pixels adjacents (mm/pixel)
        à partir des métadonnées de résolution (DPI) ou utilise une valeur forcée.

        Args:
            file_path: chemin vers le fichier image.
            pixel_spacing_mm: valeur manuelle en mm/pixel (prioritaire si fournie).

        Returns:
            dict: {
                'pixels': np.ndarray (2D, niveaux de gris, uint8),
                'metadata': {
                    'rows': int,
                    'columns': int,
                    'dtype': str,
                    'pixel_spacing_mm': float,   # distance physique en mm/pixel
                    'dpi': tuple (x_dpi, y_dpi) ou None
                },
                'is_valid': bool,
                'error': str (optionnel)
            }
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return {"is_valid": False, "error": f"File not found: {file_path}"}

        ext = file_path.suffix.lower()
        if ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            return ImageProcessor._read_image(file_path, pixel_spacing_mm)
        else:
            return {"is_valid": False, "error": f"Unsupported file type: {ext}"}

    @staticmethod
    def _read_image(file_path: Path, pixel_spacing_mm: float = None) -> dict:
        try:
            # Lire l'image avec OpenCV (pour les pixels)
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return {"is_valid": False, "error": "Failed to read image"}

            # Convertir en niveaux de gris si nécessaire
            if len(img.shape) == 3:
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            # Normaliser uint16 -> uint8
            if img.dtype == np.uint16:
                img = (img / 65535.0 * 255).astype(np.uint8)

            # ---------- Calcul du pixel spacing (mm/pixel) ----------
            dpi = None
            if pixel_spacing_mm is None:
                # Tenter de lire la résolution DPI via PIL
                try:
                    with Image.open(file_path) as pil_img:
                        # Récupérer les métadonnées EXIF
                        exif = pil_img._getexif()
                        if exif is not None:
                            for tag_id, value in exif.items():
                                tag = TAGS.get(tag_id, tag_id)
                                if tag == 'XResolution':
                                    dpi_x = value[0] / value[1] if isinstance(value, tuple) else value
                                elif tag == 'YResolution':
                                    dpi_y = value[0] / value[1] if isinstance(value, tuple) else value
                            if dpi_x and dpi_y:
                                dpi = (dpi_x, dpi_y)
                        else:
                            # Si pas d'EXIF, essayer d'utiliser info (ex: TIFF)
                            info = pil_img.info
                            if 'dpi' in info:
                                dpi = info['dpi']
                            elif 'resolution' in info:
                                dpi = info['resolution']
                except Exception as e:
                    logger.warning(f"Impossible de lire les métadonnées DPI: {e}")

                if dpi is not None:
                    # Convertir DPI (points par pouce) en mm/pixel
                    # 1 pouce = 25.4 mm
                    # mm/pixel = 25.4 / DPI
                    # Prendre la moyenne si x et y différents (ou prendre le x)
                    dpi_avg = (dpi[0] + dpi[1]) / 2.0
                    pixel_spacing_mm = 25.4 / dpi_avg
                    logger.info(f"✅ Résolution DPI détectée: {dpi_avg:.2f} → {pixel_spacing_mm:.4f} mm/pixel")
                else:
                    # Valeur par défaut (0.1 mm/pixel = 254 DPI, correspond à environ 0.1 mm)
                    pixel_spacing_mm = 0.1
                    logger.warning("distance physique réelle entre deux pixels adjacents de l'image = 0.1 mm/pixel .")
            else:
                logger.info(f"✅ Pixel spacing forcé: {pixel_spacing_mm:.4f} mm/pixel")

            # Métadonnées enrichies
            metadata = {
                "rows": img.shape[0],
                "columns": img.shape[1],
                "dtype": str(img.dtype),
                "pixel_spacing_mm": pixel_spacing_mm,
                "dpi": dpi,
            }

            logger.info(f"✅ Image chargée: {file_path.name}")
            logger.info(f"   - Dimensions: {img.shape}")
            logger.info(f"   - Pixel spacing: {pixel_spacing_mm:.4f} mm/pixel")

            return {
                "pixels": img,
                "metadata": metadata,
                "is_valid": True,
            }

        except Exception as e:
            logger.error(f"❌ Erreur lecture image: {e}")
            return {"is_valid": False, "error": str(e)}

    @staticmethod
    def prepare_for_yolo(pixels: np.ndarray) -> np.ndarray:
        """
        Prépare l'image pour YOLO: RGB uint8.
        (utilisé par les agents de vision)
        """
        if pixels.dtype != np.uint8:
            if pixels.max() > 0:
                pixels = (pixels / pixels.max() * 255).astype(np.uint8)
            else:
                pixels = pixels.astype(np.uint8)

        if len(pixels.shape) == 2:
            pixels = np.stack([pixels] * 3, axis=-1)

        return pixels
    