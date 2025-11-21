"""Utilidades de colorización para la app Streamlit."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Intentar enlazar el backend oficial FalseColor-Python
FALSECOLOR_AVAILABLE = False
FALSECOLOR_ERROR: Optional[str] = None
FALSECOLOR_ROOT = Path(__file__).resolve().parent.parent / "falsecolor"
if FALSECOLOR_ROOT.exists():
    if str(FALSECOLOR_ROOT) not in sys.path:
        sys.path.insert(0, str(FALSECOLOR_ROOT))

try:  # pragma: no cover - dependencias externas
    import falsecolor.coloring as fc  # type: ignore

    FALSECOLOR_AVAILABLE = True
except Exception as exc:  # pragma: no cover - solo informativo
    fc = None  # type: ignore
    FALSECOLOR_ERROR = repr(exc)

class HistologyColorizer:
    """Aplicar colorización simple o FalseColor real."""

    FALSECOLOR_AVAILABLE = FALSECOLOR_AVAILABLE

    COLORMAPS = {
        'H&E': 'custom_he',
        'Viridis': cv2.COLORMAP_VIRIDIS,
        'Hot': cv2.COLORMAP_HOT,
        'Jet': cv2.COLORMAP_JET,
        'Turbo': cv2.COLORMAP_TURBO,
        'Parula': cv2.COLORMAP_PARULA,
    }

    @staticmethod
    def has_falsecolor_backend() -> bool:
        return HistologyColorizer.FALSECOLOR_AVAILABLE

    @staticmethod
    def falsecolor_error_message() -> Optional[str]:
        return FALSECOLOR_ERROR
    
    @staticmethod
    def apply_clahe(img: np.ndarray, 
                    clip_limit: float = 2.0,
                    tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            img: Imagen en escala de grises (uint8)
            clip_limit: Límite de contraste (1.0-4.0 recomendado)
            tile_size: Tamaño de tiles para ecualización local
            
        Returns:
            Imagen con contraste mejorado
        """
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(img)
    
    @staticmethod
    def normalize_channel(img: np.ndarray, target_dtype=np.uint8) -> np.ndarray:
        """Normaliza cualquier imagen a 0-1 y la devuelve como uint."""

        img = img.astype(np.float32)
        img -= np.nanmin(img)
        max_val = np.nanmax(img)
        if max_val > 0:
            img /= max_val
        scale = 255.0 if target_dtype == np.uint8 else 65535.0
        return (img * scale).clip(0, scale).astype(target_dtype)

    @staticmethod
    def intensity_leveling(img: np.ndarray,
                          percentile_low: float = 1.0,
                          percentile_high: float = 99.0) -> np.ndarray:
        """
        Nivelación de intensidad (método FalseColor-Python)
        
        Args:
            img: Imagen en escala de grises
            percentile_low: Percentil inferior
            percentile_high: Percentil superior
            
        Returns:
            Imagen con intensidades niveladas
        """
        p_low = np.percentile(img, percentile_low)
        p_high = np.percentile(img, percentile_high)
        
        if p_high - p_low < 1:
            return img
        
        img_leveled = np.clip(
            (img.astype(float) - p_low) / (p_high - p_low) * 255,
            0, 255
        )
        return img_leveled.astype(np.uint8)
    
    @staticmethod
    def simulate_he_staining(img: np.ndarray) -> np.ndarray:
        """
        Simula tinción Hematoxilina & Eosina
        
        Args:
            img: Imagen en escala de grises
            
        Returns:
            Imagen RGB simulando tinción H&E
        """
        img_inv = 255 - img.astype(float)
        norm = img_inv / 255.0
        
        h, w = img.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        # R: citoplasma (eosina)
        colored[:, :, 0] = (255 * (0.3 + 0.7 * (1 - norm))).astype(np.uint8)
        # G: nivel medio
        colored[:, :, 1] = (160 * (0.7 - 0.4 * norm)).astype(np.uint8)
        # B: núcleos (hematoxilina)
        colored[:, :, 2] = (200 * (0.4 + 0.6 * norm)).astype(np.uint8)
        
        return colored
    
    @staticmethod
    def apply_false_color(img: np.ndarray,
                         colormap: str = 'Viridis',
                         apply_clahe: bool = True,
                         apply_leveling: bool = True,
                         clahe_clip: float = 2.0) -> np.ndarray:
        """
        Pipeline completo de colorización
        
        Args:
            img: Imagen en escala de grises
            colormap: Nombre del colormap
            apply_clahe: Aplicar CLAHE
            apply_leveling: Aplicar nivelación
            clahe_clip: Parámetro de CLAHE
            
        Returns:
            Imagen RGB coloreada
        """
        img_processed = img.copy()
        
        if apply_leveling:
            img_processed = HistologyColorizer.intensity_leveling(img_processed)
        
        if apply_clahe:
            img_processed = HistologyColorizer.apply_clahe(img_processed, clip_limit=clahe_clip)
        
        if colormap == 'H&E':
            return HistologyColorizer.simulate_he_staining(img_processed)
        else:
            colormap_id = HistologyColorizer.COLORMAPS.get(colormap, cv2.COLORMAP_VIRIDIS)
            colored = cv2.applyColorMap(img_processed, colormap_id)
            return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # ================= FALSECOLOR-PYTHON BACKEND =================
    @staticmethod
    def estimate_background(img: np.ndarray, threshold: float = 50.0) -> float:
        if HistologyColorizer.FALSECOLOR_AVAILABLE:
            _, background = fc.getBackgroundLevels(img, threshold=threshold)
            return float(background)
        hi = np.percentile(img, 95)
        return float(hi * 0.2)

    @staticmethod
    def _prepare_for_falsecolor(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0)

    @staticmethod
    def _prepare_for_clahe(img: np.ndarray) -> np.ndarray:
        return HistologyColorizer.normalize_channel(img, target_dtype=np.uint16)

    @staticmethod
    def cpu_sharpen(img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Basic CPU-only sharpening fallback using Laplacian."""

        img_uint8 = HistologyColorizer.normalize_channel(img, target_dtype=np.uint8)
        lap = cv2.Laplacian(img_uint8, cv2.CV_16S, ksize=3)
        lap = cv2.convertScaleAbs(lap)
        sharp = cv2.addWeighted(img_uint8, 1 + alpha, lap, -alpha, 0)
        return sharp.astype(np.float32)

    @staticmethod
    def apply_falsecolor_dual(
        nuclei: np.ndarray,
        cyto: np.ndarray,
        *,
        method: str = 'cpu',
        nuc_threshold: Optional[float] = None,
        cyto_threshold: Optional[float] = None,
        nuc_normfactor: Optional[float] = 5000,
        cyto_normfactor: Optional[float] = 2000,
        color_key: str = 'HE',
        apply_sharpen: bool = False,
        sharpen_alpha: float = 0.5,
        apply_clahe: bool = False,
        clahe_clip: float = 0.05,
    ) -> np.ndarray:
        """Colorización real usando FalseColor-Python."""

        if not HistologyColorizer.FALSECOLOR_AVAILABLE:
            raise RuntimeError("El backend FalseColor no está disponible en este entorno")

        nuclei_f = HistologyColorizer._prepare_for_falsecolor(nuclei)
        cyto_f = HistologyColorizer._prepare_for_falsecolor(cyto)

        if apply_clahe:
            nuclei_f = fc.applyCLAHE(
                HistologyColorizer._prepare_for_clahe(nuclei_f),
                clipLimit=clahe_clip,
            ).astype(np.float32)
            cyto_f = fc.applyCLAHE(
                HistologyColorizer._prepare_for_clahe(cyto_f),
                clipLimit=clahe_clip,
            ).astype(np.float32)

        if apply_sharpen:
            try:
                nuclei_f = fc.sharpenImage(nuclei_f, alpha=sharpen_alpha)
                cyto_f = fc.sharpenImage(cyto_f, alpha=sharpen_alpha)
            except Exception as err:
                if "CUDA" in str(err).upper():
                    nuclei_f = HistologyColorizer.cpu_sharpen(nuclei_f, sharpen_alpha)
                    cyto_f = HistologyColorizer.cpu_sharpen(cyto_f, sharpen_alpha)
                else:
                    raise

        if nuc_threshold is None:
            nuc_threshold = HistologyColorizer.estimate_background(nuclei_f)
        if cyto_threshold is None:
            cyto_threshold = HistologyColorizer.estimate_background(cyto_f)

        color_settings = fc.getColorSettings(key=color_key)

        method = method.lower()
        if method.startswith('gpu'):
            pseudo = fc.rapidFalseColor(
                nuclei_f,
                cyto_f,
                color_settings['nuclei'],
                color_settings['cyto'],
                nuc_normfactor=nuc_normfactor or 8500,
                cyto_normfactor=cyto_normfactor or 2000,
            )
        else:
            pseudo = fc.falseColor(
                nuclei_f,
                cyto_f,
                nuc_threshold=nuc_threshold,
                cyto_threshold=cyto_threshold,
                nuc_normfactor=nuc_normfactor,
                cyto_normfactor=cyto_normfactor,
                color_key=color_key,
                color_settings=color_settings,
            )

        return np.asarray(pseudo, dtype=np.uint8)