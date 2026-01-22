"""
Modality Feature Extractors for Multimodal RetinexFormer.

This module provides a pluggable architecture for adding different modalities
(depth, segmentation, saliency, dinov3, etc.) to RetinexFormer in a modular way.

Usage:
    # Register a new modality extractor
    from basicsr.models.archs.modalities import register_modality, get_modality

    @register_modality("my_modality")
    class MyModalityExtractor(ModalityFeatureExtractor):
        ...

    # Get a modality extractor by name
    extractor = get_modality("depth", config)
    extractor = get_modality("dinov3", config)
"""

from .base import ModalityFeatureExtractor, ModalityFusionModule
from .registry import register_modality, get_modality, get_available_modalities
from .depth import DepthFeatureExtractor
from .dinov3 import DINOv3FeatureExtractor
from .luminance import LuminanceFeatureExtractor

__all__ = [
    # Base classes
    "ModalityFeatureExtractor",
    "ModalityFusionModule",
    # Registry functions
    "register_modality",
    "get_modality",
    "get_available_modalities",
    # Concrete implementations
    "DepthFeatureExtractor",
    "DINOv3FeatureExtractor",
    "LuminanceFeatureExtractor",
]
