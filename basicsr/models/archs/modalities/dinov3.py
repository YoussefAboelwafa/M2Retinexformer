"""
DINOv3 Feature Extractor for Multimodal RetinexFormer.

This module provides semantic features from DINOv3 for multimodal enhancement.
DINOv3 is a self-supervised vision transformer that provides rich semantic features.
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


def _get_dinov3_path():
    """Get the path to DINOv3 module."""
    # Try relative to this file
    base_path = os.path.dirname(os.path.abspath(__file__))
    dinov3_path = os.path.join(base_path, "..", "..", "..", "..", "dinov3")
    if os.path.exists(dinov3_path):
        return os.path.abspath(dinov3_path)

    # Try relative to workspace root
    dinov3_path = os.path.join(base_path, "..", "..", "..", "..", "..", "dinov3")
    if os.path.exists(dinov3_path):
        return os.path.abspath(dinov3_path)

    return None


@register_modality("dinov3")
class DINOv3FeatureExtractor(ModalityFeatureExtractor):
    """
    Extract multi-scale semantic features using DINOv3.

    This module uses the pre-trained DINOv3 Vision Transformer to extract
    rich semantic features at multiple scales, which are then projected
    to match RetinexFormer's feature dimensions.

    DINOv3 provides powerful self-supervised features that can help
    with understanding scene semantics, object boundaries, and textures.

    Config options:
        encoder: str - Encoder type ('vits16', 'vitb16', 'vitl16'). Default: 'vits16'
        checkpoint: str - Path to DINOv3 checkpoint. Default: None (uses pretrained)
        freeze: bool - Whether to freeze encoder weights. Default: True
        fusion_type: str - Fusion type for attention. Default: 'gate'
        pretrained: bool - Whether to use pretrained weights. Default: True
    """

    # Embedding dimensions for different encoders
    ENCODER_DIMS = {
        "vits16": 384,
        "vits16plus": 384,
        "vitb16": 768,
        "vitl16": 1024,
    }

    # Number of blocks for each encoder (for intermediate layer selection)
    ENCODER_DEPTHS = {
        "vits16": 12,
        "vits16plus": 12,
        "vitb16": 12,
        "vitl16": 24,
    }

    # Intermediate layer indices for multi-scale features
    INTERMEDIATE_LAYERS = {
        "vits16": [2, 5, 8, 11],
        "vits16plus": [2, 5, 8, 11],
        "vitb16": [2, 5, 8, 11],
        "vitl16": [5, 11, 17, 23],
    }

    def __init__(
        self,
        target_dim: int = 40,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(target_dim, config)

        # Parse config
        self.encoder_type = self.config.get("encoder", "vits16")
        self.checkpoint_path = self.config.get("checkpoint", None)
        self.freeze_encoder = self.config.get("freeze", True)
        self.use_pretrained = self.config.get("pretrained", True)

        if self.encoder_type not in self.ENCODER_DIMS:
            raise ValueError(f"encoder must be one of {list(self.ENCODER_DIMS.keys())}")

        self.embed_dim = self.ENCODER_DIMS[self.encoder_type]
        self.patch_size = 16  # DINOv3 uses 16x16 patches

        # Initialize DINOv3 model
        self.dinov3_model = None
        self._load_dinov3_model()

        # Projection layers to match RetinexFormer dimensions
        self.dino_proj_level1 = nn.Sequential(
            nn.Conv2d(self.embed_dim, target_dim, 1, bias=False),
            nn.BatchNorm2d(target_dim),
            nn.GELU(),
        )
        self.dino_proj_level2 = nn.Sequential(
            nn.Conv2d(self.embed_dim, target_dim * 2, 1, bias=False),
            nn.BatchNorm2d(target_dim * 2),
            nn.GELU(),
        )
        self.dino_proj_bottleneck = nn.Sequential(
            nn.Conv2d(self.embed_dim, target_dim * 4, 1, bias=False),
            nn.BatchNorm2d(target_dim * 4),
            nn.GELU(),
        )

        # Freeze if specified
        if self.freeze_encoder and self.dinov3_model is not None:
            self._freeze_dinov3_model()

    @property
    def name(self) -> str:
        return "dinov3"

    @property
    def output_scales(self) -> Dict[str, int]:
        return {"level1": 1, "level2": 2, "bottleneck": 4}

    def _load_dinov3_model(self):
        """Load the DINOv3 model."""
        dinov3_path = _get_dinov3_path()
        if dinov3_path and dinov3_path not in sys.path:
            sys.path.insert(0, dinov3_path)

        try:
            # Import DINOv3 hub module
            hub_module = importlib.import_module("dinov3.hub.backbones")

            # Select the appropriate model builder based on encoder type
            model_builders = {
                "vits16": hub_module.dinov3_vits16,
                "vits16plus": hub_module.dinov3_vits16plus,
                "vitb16": hub_module.dinov3_vitb16,
                "vitl16": hub_module.dinov3_vitl16,
            }

            builder = model_builders[self.encoder_type]

            # Check if using local checkpoint
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                # Load model with pretrained=False and then load local weights
                self.dinov3_model = builder(pretrained=False)
                state_dict = torch.load(self.checkpoint_path, map_location="cpu")
                self.dinov3_model.load_state_dict(state_dict, strict=True)
                print(f"[DINOv3] Loaded checkpoint from {self.checkpoint_path}")
            elif self.use_pretrained:
                # Use pretrained weights from the model
                self.dinov3_model = builder(pretrained=True)
                print(f"[DINOv3] Loaded pretrained {self.encoder_type} model")
            else:
                # Initialize without pretrained weights
                self.dinov3_model = builder(pretrained=False)
                print(
                    f"[DINOv3] Initialized {self.encoder_type} model without pretrained weights"
                )

        except Exception as e:
            print(f"[DINOv3] Warning: Could not import DINOv3: {e}")
            print("[DINOv3] Falling back to simple gradient-based features")
            self.dinov3_model = None

    def _freeze_dinov3_model(self):
        """Freeze the DINOv3 encoder weights."""
        if self.dinov3_model is not None:
            for param in self.dinov3_model.parameters():
                param.requires_grad = False
            self.dinov3_model.eval()
            print("[DINOv3] Encoder weights frozen")

    def _simple_semantic_features(
        self, x: torch.Tensor, target_h: int, target_w: int
    ) -> torch.Tensor:
        """
        Fallback: Generate simple semantic-like features using gradient and texture info.
        """
        b, c, h, w = x.shape

        # Convert to grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Compute gradients (proxy for edge/semantic features)
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
        Extract multi-scale semantic features using DINOv3.

        Args:
            x: Input RGB image [B, 3, H, W]

        Returns:
            Dict of semantic features at each scale
        """
        b, c, h, w = x.shape
        patch_size = self.patch_size
        new_h = ((h + patch_size - 1) // patch_size) * patch_size
        new_w = ((w + patch_size - 1) // patch_size) * patch_size
        patch_h, patch_w = new_h // patch_size, new_w // patch_size

        if self.dinov3_model is not None:
            with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                if self.freeze_encoder:
                    self.dinov3_model.eval()

                # Resize if needed to be multiple of patch size
                if new_h != h or new_w != w:
                    x_resized = F.interpolate(
                        x, size=(new_h, new_w), mode="bilinear", align_corners=False
                    )
                else:
                    x_resized = x

                # Get intermediate features using DINOv3's get_intermediate_layers
                # This returns features from multiple layers
                intermediate_layers = self.INTERMEDIATE_LAYERS[self.encoder_type]
                features = self.dinov3_model.get_intermediate_layers(
                    x_resized,
                    n=intermediate_layers,
                    reshape=False,
                    return_class_token=False,
                    norm=True,
                )

                # features is a tuple of tensors, each [B, N, C]
                # N = patch_h * patch_w (number of patches)
                feat_level1 = features[0]  # Early layer
                feat_level2 = features[1]  # Mid layer
                feat_bottleneck = features[-1]  # Last layer
        else:
            # Fallback to simple features
            base_feat = self._simple_semantic_features(x, patch_h, patch_w)
            feat_level1 = base_feat.flatten(2).transpose(1, 2)
            feat_level2 = base_feat.flatten(2).transpose(1, 2)
            feat_bottleneck = base_feat.flatten(2).transpose(1, 2)

        # Reshape features from [B, N, C] to [B, C, H, W]
        def reshape_feat(feat, ph, pw):
            return feat.permute(0, 2, 1).reshape(b, self.embed_dim, ph, pw)

        feat_level1 = reshape_feat(feat_level1, patch_h, patch_w)
        feat_level2 = reshape_feat(feat_level2, patch_h, patch_w)
        feat_bottleneck = reshape_feat(feat_bottleneck, patch_h, patch_w)

        # Resize and project features to match RetinexFormer dimensions
        dino_level1 = F.interpolate(
            feat_level1, size=(h, w), mode="bilinear", align_corners=False
        )
        dino_level1 = self.dino_proj_level1(dino_level1)

        dino_level2 = F.interpolate(
            feat_level2, size=(h // 2, w // 2), mode="bilinear", align_corners=False
        )
        dino_level2 = self.dino_proj_level2(dino_level2)

        dino_bottleneck = F.interpolate(
            feat_bottleneck, size=(h // 4, w // 4), mode="bilinear", align_corners=False
        )
        dino_bottleneck = self.dino_proj_bottleneck(dino_bottleneck)

        return {
            "level1": dino_level1,
            "level2": dino_level2,
            "bottleneck": dino_bottleneck,
        }

    def get_visualization(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get a visualization of DINOv3 features using PCA.

        Returns the first principal component of the patch features.
        """
        if self.dinov3_model is not None:
            with torch.no_grad():
                b, c, h, w = x.shape
                patch_size = self.patch_size

                # Resize to be multiple of patch size
                new_h = ((h + patch_size - 1) // patch_size) * patch_size
                new_w = ((w + patch_size - 1) // patch_size) * patch_size
                patch_h, patch_w = new_h // patch_size, new_w // patch_size

                if new_h != h or new_w != w:
                    x_resized = F.interpolate(
                        x, size=(new_h, new_w), mode="bilinear", align_corners=False
                    )
                else:
                    x_resized = x

                # Get final layer features
                features = self.dinov3_model.get_intermediate_layers(
                    x_resized,
                    n=1,  # Just the last layer
                    reshape=False,
                    return_class_token=False,
                    norm=True,
                )

                # features[0] shape: [B, N, C]
                feat = features[0]  # [B, N, C]

                # Simple visualization: take mean across channels
                # This gives a rough "attention" or "saliency" map
                vis = feat.mean(dim=-1)  # [B, N]
                vis = vis.reshape(b, patch_h, patch_w)  # [B, H', W']
                vis = vis.unsqueeze(1)  # [B, 1, H', W']

                # Normalize for visualization
                vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-6)

                # Resize back to original size
                vis = F.interpolate(
                    vis,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )

                return vis
        return None
