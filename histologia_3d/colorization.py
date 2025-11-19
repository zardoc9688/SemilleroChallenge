"""
Módulo de Colorización para Imágenes Histológicas
Implementación basada en FalseColor-Python (Giacomelli et al., 2020)

Autores: Daniel Yaruro, Juan Mantilla
Fecha: Enero 2025
"""

import numpy as np
import cv2
from typing import Tuple

class HistologyColorizer:
    """
    Aplica pseudo-colorización a imágenes histológicas en escala de grises
    """
    
    COLORMAPS = {
        'H&E': 'custom_he',
        'Viridis': cv2.COLORMAP_VIRIDIS,
        'Hot': cv2.COLORMAP_HOT,
        'Jet': cv2.COLORMAP_JET,
        'Turbo': cv2.COLORMAP_TURBO,
        'Parula': cv2.COLORMAP_PARULA,
    }
    
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