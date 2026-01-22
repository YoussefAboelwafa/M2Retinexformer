"""
Modality Registry for dynamic modality loading.

This module implements a registry pattern that allows modality extractors
to be registered and instantiated by name from configuration.
"""

from typing import Dict, Type, Optional, Any, List
from .base import ModalityFeatureExtractor


# Global registry for modality extractors
_MODALITY_REGISTRY: Dict[str, Type[ModalityFeatureExtractor]] = {}


def register_modality(name: str):
    """
    Decorator to register a modality extractor class.

    Usage:
        @register_modality("depth")
        class DepthFeatureExtractor(ModalityFeatureExtractor):
            ...

    Args:
        name: Unique name for the modality

    Returns:
        Decorator function
    """

    def decorator(cls: Type[ModalityFeatureExtractor]):
        if not issubclass(cls, ModalityFeatureExtractor):
            raise TypeError(
                f"Registered modality {name} must inherit from ModalityFeatureExtractor"
            )
        if name in _MODALITY_REGISTRY:
            raise ValueError(f"Modality '{name}' is already registered")
        _MODALITY_REGISTRY[name] = cls
        return cls

    return decorator


def get_modality(
    name: str,
    target_dim: int = 40,
    config: Optional[Dict[str, Any]] = None,
) -> ModalityFeatureExtractor:
    """
    Get a modality extractor by name.

    Args:
        name: Name of the registered modality
        target_dim: Target feature dimension to match RetinexFormer
        config: Modality-specific configuration dict

    Returns:
        Instantiated modality extractor

    Raises:
        ValueError: If modality is not registered
    """
    if name not in _MODALITY_REGISTRY:
        available = list(_MODALITY_REGISTRY.keys())
        raise ValueError(
            f"Modality '{name}' not found. Available modalities: {available}"
        )

    cls = _MODALITY_REGISTRY[name]
    return cls(target_dim=target_dim, config=config)


def get_available_modalities() -> List[str]:
    """
    Get list of all registered modality names.

    Returns:
        List of modality names
    """
    return list(_MODALITY_REGISTRY.keys())


def is_modality_registered(name: str) -> bool:
    """
    Check if a modality is registered.

    Args:
        name: Modality name

    Returns:
        True if registered, False otherwise
    """
    return name in _MODALITY_REGISTRY


def unregister_modality(name: str) -> None:
    """
    Unregister a modality (mainly for testing).

    Args:
        name: Modality name to unregister
    """
    if name in _MODALITY_REGISTRY:
        del _MODALITY_REGISTRY[name]
