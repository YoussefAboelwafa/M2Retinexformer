"""
Base classes for modality feature extractors.

This module defines the abstract interfaces that all modality extractors
must implement for integration with MultiModalRetinexFormer.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import torch
import torch.nn as nn


class ModalityFeatureExtractor(ABC, nn.Module):
    """
    Abstract base class for modality feature extractors.

    All modality extractors (depth, segmentation, saliency, etc.) should
    inherit from this class and implement the required methods.

    The extractor is responsible for:
    1. Loading any pre-trained models (e.g., Depth-Anything-V2)
    2. Extracting multi-scale features from the input
    3. Projecting features to match RetinexFormer dimensions

    Args:
        target_dim: Target feature dimension to match RetinexFormer
        config: Modality-specific configuration dict
    """

    def __init__(self, target_dim: int = 40, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.target_dim = target_dim
        self.config = config or {}
        self._is_frozen = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this modality (e.g., 'depth', 'segmentation')."""
        pass

    @property
    @abstractmethod
    def output_scales(self) -> Dict[str, int]:
        """
        Return the output scales and their channel multipliers.

        Example:
            {"level1": 1, "level2": 2, "bottleneck": 4}
            means level1 has target_dim channels, level2 has target_dim*2, etc.
        """
        pass

    @abstractmethod
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
        pass

    def freeze(self) -> None:
        """Freeze the feature extractor parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self._is_frozen = True
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze the feature extractor parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self._is_frozen = False
        self.train()

    @property
    def is_frozen(self) -> bool:
        """Check if the extractor is frozen."""
        return self._is_frozen

    def get_visualization(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get a visualization of the modality output (optional).

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Visualization tensor [B, 1, H, W] or [B, 3, H, W], or None
        """
        return None


class ModalityFusionModule(nn.Module):
    """
    Module for fusing features from multiple modalities.

    This module combines RGB features with features from one or more
    auxiliary modalities using various fusion strategies.

    Args:
        dim: Feature dimension
        num_modalities: Number of modalities to fuse (excluding RGB)
        fusion_type: Fusion strategy ('gate', 'add', 'concat', 'attention')
    """

    FUSION_TYPES = ["gate", "add", "concat", "attention"]

    def __init__(
        self,
        dim: int,
        num_modalities: int = 1,
        fusion_type: str = "gate",
    ):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        self.fusion_type = fusion_type

        if fusion_type not in self.FUSION_TYPES:
            raise ValueError(f"fusion_type must be one of {self.FUSION_TYPES}")

        if fusion_type == "gate":
            # Learnable gate for each modality
            self.gates = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dim * 2, dim),
                        nn.Sigmoid(),
                    )
                    for _ in range(num_modalities)
                ]
            )
        elif fusion_type == "concat":
            # Projection to reduce concatenated features
            self.proj = nn.Linear(dim * (1 + num_modalities), dim)
        elif fusion_type == "attention":
            # Cross-attention based fusion
            self.attn_proj_q = nn.Linear(dim, dim)
            self.attn_proj_kv = nn.ModuleList(
                [nn.Linear(dim, dim * 2) for _ in range(num_modalities)]
            )
            self.attn_out = nn.Linear(dim, dim)
            self.scale = dim**-0.5

    def forward(
        self,
        rgb_feat: torch.Tensor,
        modality_feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse RGB features with modality features.

        Args:
            rgb_feat: RGB features [B, H, W, C] or [B, N, C]
            modality_feats: Dict of modality features, each [B, H, W, C] or [B, N, C]

        Returns:
            Fused features with same shape as rgb_feat
        """
        if not modality_feats:
            return rgb_feat

        # Convert dict values to list for consistent ordering
        mod_feats = list(modality_feats.values())

        if self.fusion_type == "gate":
            out = rgb_feat
            for i, mod_feat in enumerate(mod_feats):
                gate_input = torch.cat([out, mod_feat], dim=-1)
                gate = self.gates[i](gate_input)
                out = gate * out + (1 - gate) * mod_feat
            return out

        elif self.fusion_type == "add":
            out = rgb_feat
            for mod_feat in mod_feats:
                out = out + mod_feat
            return out

        elif self.fusion_type == "concat":
            all_feats = [rgb_feat] + mod_feats
            concat = torch.cat(all_feats, dim=-1)
            return self.proj(concat)

        elif self.fusion_type == "attention":
            # Cross-attention: RGB attends to all modalities
            q = self.attn_proj_q(rgb_feat)

            attended = rgb_feat
            for i, mod_feat in enumerate(mod_feats):
                kv = self.attn_proj_kv[i](mod_feat)
                k, v = kv.chunk(2, dim=-1)

                # Compute attention
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attended = attended + self.attn_out(attn @ v)

            return attended

        return rgb_feat
