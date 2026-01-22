"""
Depth Feature Extractor using Depth-Anything-V2.

This module provides depth-based features for multimodal enhancement.
"""

import os
import sys
import importlib
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModalityFeatureExtractor
from .registry import register_modality


# Add Depth-Anything-V2 to path dynamically
def _get_depth_anything_path():
    """Get the path to Depth-Anything-V2 module."""
    # Try relative to this file
    base_path = os.path.dirname(os.path.abspath(__file__))
    depth_path = os.path.join(base_path, "..", "..", "..", "..", "Depth-Anything-V2")
    if os.path.exists(depth_path):
        return os.path.abspath(depth_path)

    # Try relative to workspace root
    depth_path = os.path.join(
        base_path, "..", "..", "..", "..", "..", "Depth-Anything-V2"
    )
    if os.path.exists(depth_path):
        return os.path.abspath(depth_path)

    return None


@register_modality("depth")
class DepthFeatureExtractor(ModalityFeatureExtractor):
    """
    Extract multi-scale depth features using Depth-Anything-V2.

    This module uses the pre-trained DINOv2 encoder from Depth-Anything-V2
    to extract rich depth features at multiple scales, which are then
    projected to match RetinexFormer's feature dimensions.

    Config options:
        encoder: str - Encoder type ('vits', 'vitb', 'vitl'). Default: 'vits'
        checkpoint: str - Path to Depth-Anything-V2 checkpoint. Default: None
        freeze: bool - Whether to freeze encoder weights. Default: True
        fusion_type: str - Fusion type for attention. Default: 'gate'
    """

    # Embedding dimensions for different encoders
    ENCODER_DIMS = {
        "vits": 384,
        "vitb": 768,
        "vitl": 1024,
    }

    # Intermediate layer indices for multi-scale features
    INTERMEDIATE_LAYERS = {
        "vits": [2, 5, 8, 11],
        "vitb": [2, 5, 8, 11],
        "vitl": [4, 11, 17, 23],
    }

    def __init__(
        self,
        target_dim: int = 40,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(target_dim, config)

        # Parse config
        self.encoder_type = self.config.get("encoder", "vits")
        self.checkpoint_path = self.config.get("checkpoint", None)
        self.freeze_encoder = self.config.get("freeze", True)

        if self.encoder_type not in self.ENCODER_DIMS:
            raise ValueError(f"encoder must be one of {list(self.ENCODER_DIMS.keys())}")

        self.embed_dim = self.ENCODER_DIMS[self.encoder_type]

        # Initialize depth model
        self.depth_model = None
        self._load_depth_model()

        # Projection layers to match RetinexFormer dimensions
        self.depth_proj_level1 = nn.Sequential(
            nn.Conv2d(self.embed_dim, target_dim, 1, bias=False),
            nn.BatchNorm2d(target_dim),
            nn.GELU(),
        )
        self.depth_proj_level2 = nn.Sequential(
            nn.Conv2d(self.embed_dim, target_dim * 2, 1, bias=False),
            nn.BatchNorm2d(target_dim * 2),
            nn.GELU(),
        )
        self.depth_proj_bottleneck = nn.Sequential(
            nn.Conv2d(self.embed_dim, target_dim * 4, 1, bias=False),
            nn.BatchNorm2d(target_dim * 4),
            nn.GELU(),
        )

        # Freeze if specified
        if self.freeze_encoder and self.depth_model is not None:
            self._freeze_depth_model()

    @property
    def name(self) -> str:
        return "depth"

    @property
    def output_scales(self) -> Dict[str, int]:
        return {"level1": 1, "level2": 2, "bottleneck": 4}

    def _load_depth_model(self):
        """Load the Depth-Anything-V2 model."""
        depth_path = _get_depth_anything_path()
        if depth_path and depth_path not in sys.path:
            sys.path.insert(0, depth_path)
        try:
            depth_module = importlib.import_module("depth_anything_v2.dpt")
            DepthAnythingV2 = getattr(depth_module, "DepthAnythingV2")

            # Model configurations
            model_configs = {
                "vits": {
                    "encoder": "vits",
                    "features": 64,
                    "out_channels": [48, 96, 192, 384],
                },
                "vitb": {
                    "encoder": "vitb",
                    "features": 128,
                    "out_channels": [96, 192, 384, 768],
                },
                "vitl": {
                    "encoder": "vitl",
                    "features": 256,
                    "out_channels": [256, 512, 1024, 1024],
                },
            }

            self.depth_model = DepthAnythingV2(**model_configs[self.encoder_type])

            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                state_dict = torch.load(self.checkpoint_path, map_location="cpu")
                self.depth_model.load_state_dict(state_dict)
                print(f"[Depth] Loaded checkpoint from {self.checkpoint_path}")
            else:
                print(
                    f"[Depth] Warning: checkpoint not found at {self.checkpoint_path}"
                )
                print("[Depth] Using randomly initialized encoder (not recommended)")

        except Exception as e:
            print(f"[Depth] Warning: Could not import Depth-Anything-V2: {e}")
            print("[Depth] Falling back to simple gradient-based features")
            self.depth_model = None
            self.depth_model = None

    def _freeze_depth_model(self):
        """Freeze the depth encoder weights."""
        if self.depth_model is not None:
            for param in self.depth_model.parameters():
                param.requires_grad = False
            self.depth_model.eval()
            print("[Depth] Encoder weights frozen")

    def _simple_depth_features(
        self, x: torch.Tensor, target_h: int, target_w: int
    ) -> torch.Tensor:
        """
        Fallback: Generate simple depth-like features using gradient information.
        """
        b, c, h, w = x.shape

        # Convert to grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Compute gradients (proxy for depth edges)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device
        ).view(1, 1, 3, 3)

        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

        # Create multi-scale features
        features = torch.cat([gray, grad_x, grad_y, grad_mag], dim=1)  # [b, 4, h, w]

        # Expand to embed_dim by repeating
        repeat_factor = self.embed_dim // 4
        features = features.repeat(1, repeat_factor, 1, 1)

        # Resize to target resolution
        if features.shape[-2] != target_h or features.shape[-1] != target_w:
            features = F.interpolate(
                features,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        return features

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale depth features.

        Args:
            x: Input RGB image [B, 3, H, W]

        Returns:
            Dict of depth features at each scale
        """
        b, c, h, w = x.shape
        patch_size = 14
        new_h = ((h + patch_size - 1) // patch_size) * patch_size
        new_w = ((w + patch_size - 1) // patch_size) * patch_size
        patch_h, patch_w = new_h // patch_size, new_w // patch_size

        if self.depth_model is not None:
            with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                if self.freeze_encoder:
                    self.depth_model.eval()

                # Resize if needed
                if new_h != h or new_w != w:
                    x_resized = F.interpolate(
                        x, size=(new_h, new_w), mode="bilinear", align_corners=False
                    )
                else:
                    x_resized = x

                # Get intermediate features
                features = self.depth_model.pretrained.get_intermediate_layers(
                    x_resized,
                    self.INTERMEDIATE_LAYERS[self.encoder_type],
                    return_class_token=False,
                )

                feat_level1 = features[0]
                feat_level2 = features[1]
                feat_bottleneck = features[-1]
        else:
            # Fallback
            base_feat = self._simple_depth_features(x, patch_h, patch_w)
            feat_level1 = base_feat.flatten(2).transpose(1, 2)
            feat_level2 = base_feat.flatten(2).transpose(1, 2)
            feat_bottleneck = base_feat.flatten(2).transpose(1, 2)

        # Reshape features from [B, N, C] to [B, C, H, W]
        def reshape_feat(feat, ph, pw):
            return feat.permute(0, 2, 1).reshape(b, self.embed_dim, ph, pw)

        feat_level1 = reshape_feat(feat_level1, patch_h, patch_w)
        feat_level2 = reshape_feat(feat_level2, patch_h, patch_w)
        feat_bottleneck = reshape_feat(feat_bottleneck, patch_h, patch_w)

        # Resize and project features
        depth_level1 = F.interpolate(
            feat_level1, size=(h, w), mode="bilinear", align_corners=False
        )
        depth_level1 = self.depth_proj_level1(depth_level1)

        depth_level2 = F.interpolate(
            feat_level2, size=(h // 2, w // 2), mode="bilinear", align_corners=False
        )
        depth_level2 = self.depth_proj_level2(depth_level2)

        depth_bottleneck = F.interpolate(
            feat_bottleneck, size=(h // 4, w // 4), mode="bilinear", align_corners=False
        )
        depth_bottleneck = self.depth_proj_bottleneck(depth_bottleneck)

        return {
            "level1": depth_level1,
            "level2": depth_level2,
            "bottleneck": depth_bottleneck,
        }

    def get_visualization(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get depth map for visualization."""
        if self.depth_model is not None:
            with torch.no_grad():
                b, c, h, w = x.shape
                patch_size = 14

                # Resize to be multiple of patch size
                new_h = ((h + patch_size - 1) // patch_size) * patch_size
                new_w = ((w + patch_size - 1) // patch_size) * patch_size

                if new_h != h or new_w != w:
                    x_resized = F.interpolate(
                        x, size=(new_h, new_w), mode="bilinear", align_corners=False
                    )
                else:
                    x_resized = x

                depth = self.depth_model(x_resized)

                # Resize back to original size
                if new_h != h or new_w != w:
                    depth = F.interpolate(
                        depth.unsqueeze(1),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    depth = depth.unsqueeze(1)

                return depth
        return None
