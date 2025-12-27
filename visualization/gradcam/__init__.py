"""
GradCAM implementation for PoundNet deepfake detection model.

This module provides GradCAM visualization capabilities specifically adapted
for Vision Transformer architectures used in PoundNet.
"""

from .core import ViTGradCAM
from .poundnet_gradcam import PoundNetGradCAM
from .utils import (
    visualize_gradcam,
    save_gradcam_results,
    create_heatmap_overlay,
    normalize_heatmap
)

__version__ = "1.0.0"
__all__ = [
    "ViTGradCAM",
    "PoundNetGradCAM", 
    "visualize_gradcam",
    "save_gradcam_results",
    "create_heatmap_overlay",
    "normalize_heatmap"
]