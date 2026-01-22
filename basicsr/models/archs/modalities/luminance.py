"""
Luminance Feature Extractor using NTSC Luminance Conversion.

This module provides luminance-based features for multimodal enhancement
using the standard NTSC luminance equation: Y = 0.299*R + 0.587*G + 0.114*B

Unlike depth or DINOv3, this modality doesn't require external pre-trained models,
making it lightweight and fast while still providing useful illumination information.
"""

from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModalityFeatureExtractor
from .registry import register_modality


@register_modality("luminance")
class LuminanceFeatureExtractor(ModalityFeatureExtractor):
    """
    Extract multi-scale luminance features using NTSC conversion.

    This module computes luminance from RGB using the NTSC formula and
    extracts multi-scale features that capture illumination patterns,
    gradients, and local contrast information.

    The NTSC Luminance Conversion Equation:
        Y = 0.299 * R + 0.587 * G + 0.114 * B

    Config options:
        use_gradients: bool - Include gradient features. Default: True
        use_local_contrast: bool - Include local contrast features. Default: True
        use_multiscale: bool - Use multi-scale luminance analysis. Default: True
        num_feature_layers: int - Number of conv layers for feature extraction. Default: 3
    """

    # NTSC Luminance weights
    NTSC_R = 0.299
    NTSC_G = 0.587
    NTSC_B = 0.114

    def __init__(
        self,
        target_dim: int = 40,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(target_dim, config)

        # Parse config
        self.use_gradients = self.config.get("use_gradients", True)
        self.use_local_contrast = self.config.get("use_local_contrast", True)
        self.use_multiscale = self.config.get("use_multiscale", True)
        self.num_feature_layers = self.config.get("num_feature_layers", 3)

        # Calculate input channels for feature extraction
        # Base: luminance (1 channel)
        # + gradients (2 channels: grad_x, grad_y) if enabled
        # + local contrast (1 channel) if enabled
        # + multiscale luminance (2 channels: 2x and 4x downscaled) if enabled
        self.base_channels = 1
        if self.use_gradients:
            self.base_channels += 2  # grad_x, grad_y
        if self.use_local_contrast:
            self.base_channels += 1  # local contrast
        if self.use_multiscale:
            self.base_channels += 2  # 2 additional scales

        # Register NTSC weights as buffers (non-trainable but move with device)
        self.register_buffer(
            "ntsc_weights",
            torch.tensor([self.NTSC_R, self.NTSC_G, self.NTSC_B]).view(1, 3, 1, 1),
        )

        # Sobel kernels for gradient computation
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # Feature extraction networks for each scale
        # Level 1: Full resolution
        self.level1_net = self._make_feature_net(self.base_channels, target_dim)

        # Level 2: Half resolution
        self.level2_net = self._make_feature_net(self.base_channels, target_dim * 2)

        # Bottleneck: Quarter resolution
        self.bottleneck_net = self._make_feature_net(self.base_channels, target_dim * 4)

        print(f"[Luminance] Initialized with {self.base_channels} base channels")
        print(
            f"[Luminance] Features: gradients={self.use_gradients}, "
            f"local_contrast={self.use_local_contrast}, multiscale={self.use_multiscale}"
        )

    def _make_feature_net(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a feature extraction network."""
        layers = []
        current_channels = in_channels

        for i in range(self.num_feature_layers):
            # Gradually increase channels
            if i == self.num_feature_layers - 1:
                next_channels = out_channels
            else:
                next_channels = min(current_channels * 2, out_channels)

            layers.extend(
                [
                    nn.Conv2d(
                        current_channels, next_channels, 3, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(next_channels),
                    nn.GELU(),
                ]
            )
            current_channels = next_channels

        return nn.Sequential(*layers)

    @property
    def name(self) -> str:
        return "luminance"

    @property
    def output_scales(self) -> Dict[str, int]:
        return {"level1": 1, "level2": 2, "bottleneck": 4}

    def _compute_luminance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute NTSC luminance from RGB image.

        Args:
            x: Input RGB image [B, 3, H, W] in range [0, 1]

        Returns:
            Luminance map [B, 1, H, W]
        """
        # Y = 0.299*R + 0.587*G + 0.114*B
        luminance = (x * self.ntsc_weights).sum(dim=1, keepdim=True)
        return luminance

    def _compute_gradients(self, luminance: torch.Tensor) -> tuple:
        """
        Compute spatial gradients of luminance using Sobel operators.

        Args:
            luminance: Luminance map [B, 1, H, W]

        Returns:
            Tuple of (grad_x, grad_y) each [B, 1, H, W]
        """
        grad_x = F.conv2d(luminance, self.sobel_x, padding=1)
        grad_y = F.conv2d(luminance, self.sobel_y, padding=1)
        return grad_x, grad_y

    def _compute_local_contrast(
        self, luminance: torch.Tensor, kernel_size: int = 7
    ) -> torch.Tensor:
        """
        Compute local contrast as the ratio of local std to local mean.

        This helps capture local illumination variations which are important
        for low-light image enhancement.

        Args:
            luminance: Luminance map [B, 1, H, W]
            kernel_size: Size of local neighborhood

        Returns:
            Local contrast map [B, 1, H, W]
        """
        padding = kernel_size // 2

        # Create averaging kernel
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=luminance.device)
        kernel = kernel / (kernel_size * kernel_size)

        # Local mean
        local_mean = F.conv2d(luminance, kernel, padding=padding)

        # Local variance: E[X^2] - E[X]^2
        local_sq_mean = F.conv2d(luminance**2, kernel, padding=padding)
        local_var = local_sq_mean - local_mean**2
        local_var = torch.clamp(local_var, min=0)  # Ensure non-negative

        # Local standard deviation
        local_std = torch.sqrt(local_var + 1e-6)

        # Local contrast: std / (mean + eps)
        local_contrast = local_std / (local_mean + 1e-6)

        return local_contrast

    def _extract_base_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract base luminance features from RGB input.

        Args:
            x: Input RGB image [B, 3, H, W]

        Returns:
            Concatenated luminance features [B, base_channels, H, W]
        """
        # Compute luminance
        luminance = self._compute_luminance(x)
        features = [luminance]

        # Add gradients
        if self.use_gradients:
            grad_x, grad_y = self._compute_gradients(luminance)
            features.extend([grad_x, grad_y])

        # Add local contrast
        if self.use_local_contrast:
            local_contrast = self._compute_local_contrast(luminance)
            features.append(local_contrast)

        # Add multiscale luminance
        if self.use_multiscale:
            b, c, h, w = luminance.shape
            # 2x downscaled luminance (then upscaled back)
            lum_2x = F.interpolate(
                luminance, size=(h // 2, w // 2), mode="bilinear", align_corners=False
            )
            lum_2x = F.interpolate(
                lum_2x, size=(h, w), mode="bilinear", align_corners=False
            )

            # 4x downscaled luminance (then upscaled back)
            lum_4x = F.interpolate(
                luminance, size=(h // 4, w // 4), mode="bilinear", align_corners=False
            )
            lum_4x = F.interpolate(
                lum_4x, size=(h, w), mode="bilinear", align_corners=False
            )

            features.extend([lum_2x, lum_4x])

        return torch.cat(features, dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale luminance features.

        Args:
            x: Input RGB image [B, 3, H, W]

        Returns:
            Dict of luminance features at each scale:
                - 'level1': [B, target_dim, H, W]
                - 'level2': [B, target_dim*2, H/2, W/2]
                - 'bottleneck': [B, target_dim*4, H/4, W/4]
        """
        b, c, h, w = x.shape

        # Extract base features at full resolution
        base_features = self._extract_base_features(x)

        # Level 1: Full resolution
        level1_features = self.level1_net(base_features)

        # Level 2: Half resolution
        base_level2 = F.interpolate(
            base_features, size=(h // 2, w // 2), mode="bilinear", align_corners=False
        )
        level2_features = self.level2_net(base_level2)

        # Bottleneck: Quarter resolution
        base_bottleneck = F.interpolate(
            base_features, size=(h // 4, w // 4), mode="bilinear", align_corners=False
        )
        bottleneck_features = self.bottleneck_net(base_bottleneck)

        return {
            "level1": level1_features,
            "level2": level2_features,
            "bottleneck": bottleneck_features,
        }

    def get_visualization(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get luminance map for visualization.

        Args:
            x: Input RGB image [B, 3, H, W]

        Returns:
            Luminance map [B, 1, H, W] in range [0, 1]
        """
        return self._compute_luminance(x)

    def get_detailed_visualization(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get detailed visualizations of all luminance features.

        Useful for debugging and understanding what the modality captures.

        Args:
            x: Input RGB image [B, 3, H, W]

        Returns:
            Dict containing:
                - 'luminance': [B, 1, H, W]
                - 'grad_x': [B, 1, H, W] (if use_gradients)
                - 'grad_y': [B, 1, H, W] (if use_gradients)
                - 'grad_magnitude': [B, 1, H, W] (if use_gradients)
                - 'local_contrast': [B, 1, H, W] (if use_local_contrast)
        """
        visualizations = {}

        luminance = self._compute_luminance(x)
        visualizations["luminance"] = luminance

        if self.use_gradients:
            grad_x, grad_y = self._compute_gradients(luminance)
            visualizations["grad_x"] = grad_x
            visualizations["grad_y"] = grad_y
            visualizations["grad_magnitude"] = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

        if self.use_local_contrast:
            visualizations["local_contrast"] = self._compute_local_contrast(luminance)

        return visualizations
