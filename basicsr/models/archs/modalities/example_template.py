"""
Example Modality Template.

This file provides a template for creating new modality feature extractors.
Copy this file and modify it to add a new modality to MultiModalRetinexFormer.

Steps to add a new modality:
1. Copy this file to a new file (e.g., segmentation.py)
2. Rename the class and update the @register_modality decorator
3. Implement the __init__, forward, and optionally get_visualization methods
4. The modality will be automatically available to use in config yml files

Example usage in yml:
    network_g:
      type: MultiModalRetinexFormer
      modalities:
        my_new_modality:
          enabled: true
          # your config options here...
"""

from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModalityFeatureExtractor
from .registry import register_modality


# Uncomment and modify the decorator to register your modality
# @register_modality("my_modality_name")
class ExampleModalityExtractor(ModalityFeatureExtractor):
    """
    Example modality feature extractor template.

    Replace this with your actual implementation.

    Config options (passed via yml):
        model_type: str - Type of model to use
        checkpoint: str - Path to pretrained weights
        freeze: bool - Whether to freeze the pretrained model
        # Add your own config options...
    """

    def __init__(
        self,
        target_dim: int = 40,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(target_dim, config)

        # Parse your config options
        self.model_type = self.config.get("model_type", "default")
        self.checkpoint_path = self.config.get("checkpoint", None)
        self.freeze_encoder = self.config.get("freeze", True)

        # Initialize your pretrained model (if any)
        self.encoder = None
        self._load_pretrained_model()

        # Create projection layers to match RetinexFormer dimensions
        # These project your model's features to the required dimensions:
        # - level1: target_dim channels at full resolution
        # - level2: target_dim * 2 channels at half resolution
        # - bottleneck: target_dim * 4 channels at quarter resolution

        encoder_dim = 256  # Replace with your encoder's output dimension

        self.proj_level1 = nn.Sequential(
            nn.Conv2d(encoder_dim, target_dim, 1, bias=False),
            nn.BatchNorm2d(target_dim),
            nn.GELU(),
        )
        self.proj_level2 = nn.Sequential(
            nn.Conv2d(encoder_dim, target_dim * 2, 1, bias=False),
            nn.BatchNorm2d(target_dim * 2),
            nn.GELU(),
        )
        self.proj_bottleneck = nn.Sequential(
            nn.Conv2d(encoder_dim, target_dim * 4, 1, bias=False),
            nn.BatchNorm2d(target_dim * 4),
            nn.GELU(),
        )

        # Freeze encoder if specified
        if self.freeze_encoder and self.encoder is not None:
            self.freeze()

    @property
    def name(self) -> str:
        """Return the unique name of this modality."""
        return "example"  # Change to your modality name

    @property
    def output_scales(self) -> Dict[str, int]:
        """
        Return the output scales and their channel multipliers.

        The keys should match what you return in forward().
        The values are multipliers of target_dim.
        """
        return {"level1": 1, "level2": 2, "bottleneck": 4}

    def _load_pretrained_model(self):
        """Load your pretrained model here."""
        # Example:
        # try:
        #     from some_library import SomeModel
        #     self.encoder = SomeModel()
        #     if self.checkpoint_path:
        #         self.encoder.load_state_dict(torch.load(self.checkpoint_path))
        # except ImportError:
        #     print("Warning: Could not load model, using fallback")
        #     self.encoder = None
        pass

    def _fallback_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fallback feature extraction when pretrained model is unavailable.

        This can be simple gradient-based features or other heuristics.
        """
        b, c, h, w = x.shape

        # Example: use grayscale + gradients as fallback
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Simple edge detection
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device
        ).view(1, 1, 3, 3)

        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

        # Combine features and expand to expected dimension
        features = torch.cat([gray, grad_x, grad_y, grad_mag], dim=1)

        # Expand to match expected encoder dimension (256 in this example)
        repeat_factor = 256 // 4
        features = features.repeat(1, repeat_factor, 1, 1)

        return features

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from input.

        Args:
            x: Input RGB image [B, 3, H, W]

        Returns:
            Dict of features at each scale:
                - 'level1': [B, target_dim, H, W]
                - 'level2': [B, target_dim*2, H/2, W/2]
                - 'bottleneck': [B, target_dim*4, H/4, W/4]
        """
        b, c, h, w = x.shape

        # Extract features using your model or fallback
        if self.encoder is not None:
            # Use your pretrained encoder
            with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                # Example: features = self.encoder.get_features(x)
                features = self._fallback_features(x)  # Replace with actual
        else:
            features = self._fallback_features(x)

        # Create multi-scale features by resizing and projecting
        # Level 1: Full resolution
        feat_level1 = F.interpolate(
            features, size=(h, w), mode="bilinear", align_corners=False
        )
        level1 = self.proj_level1(feat_level1)

        # Level 2: Half resolution
        feat_level2 = F.interpolate(
            features, size=(h // 2, w // 2), mode="bilinear", align_corners=False
        )
        level2 = self.proj_level2(feat_level2)

        # Bottleneck: Quarter resolution
        feat_bottleneck = F.interpolate(
            features, size=(h // 4, w // 4), mode="bilinear", align_corners=False
        )
        bottleneck = self.proj_bottleneck(feat_bottleneck)

        return {
            "level1": level1,
            "level2": level2,
            "bottleneck": bottleneck,
        }

    def get_visualization(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get a visualization of the modality output (optional).

        This is useful for debugging and visualization during inference.

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Visualization tensor [B, 1, H, W] or [B, 3, H, W], or None
        """
        # Example: return a heatmap of your modality's prediction
        # if self.encoder is not None:
        #     with torch.no_grad():
        #         output = self.encoder.predict(x)
        #         return output.unsqueeze(1)  # Make it [B, 1, H, W]
        return None
