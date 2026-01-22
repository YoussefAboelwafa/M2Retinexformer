"""
MultiModal RetinexFormer Architecture.

This module provides a modular, pluggable architecture for adding multiple
modalities (depth, segmentation, saliency, etc.) to RetinexFormer.

The design follows these principles:
1. Each modality is independent and can be enabled/disabled via config
2. New modalities can be added without modifying the core architecture
3. Multiple modalities can be fused together
4. Backward compatible with original RetinexFormer

Usage in config (yml):
    network_g:
      type: MultiModalRetinexFormer
      in_channels: 3
      out_channels: 3
      n_feat: 40
      stage: 3
      num_blocks: [1, 2, 2]

      # Modalities are defined here - add/remove as needed
      modalities:
        depth:
          enabled: true
          encoder: vits
          checkpoint: path/to/checkpoint.pth
          freeze: true
        segmentation:
          enabled: true
          model: deeplabv3
          checkpoint: path/to/seg_checkpoint.pth
        # Add more modalities here...

      # Global fusion settings
      fusion_type: gate  # or 'add', 'concat', 'attention'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from typing import Dict, List, Optional, Any

from .modalities import get_modality, get_available_modalities, ModalityFeatureExtractor


# =============================================================================
# Utility Functions
# =============================================================================


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


# =============================================================================
# Multimodal Feature Manager
# =============================================================================


class ModalityManager(nn.Module):
    """
    Manages loading and feature extraction for multiple modalities.

    This class handles:
    1. Dynamic loading of modality extractors based on config
    2. Coordinating feature extraction across all modalities
    3. Providing features at each scale level
    """

    def __init__(
        self,
        modalities_config: Optional[Dict[str, Dict[str, Any]]],
        target_dim: int = 40,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.modality_names: List[str] = []
        self.extractors = nn.ModuleDict()

        if modalities_config is None:
            return

        # Load each enabled modality
        for mod_name, mod_config in modalities_config.items():
            if not mod_config.get("enabled", True):
                continue

            try:
                extractor = get_modality(
                    name=mod_name,
                    target_dim=target_dim,
                    config=mod_config,
                )
                self.extractors[mod_name] = extractor
                self.modality_names.append(mod_name)
                print(f"[ModalityManager] Loaded modality: {mod_name}")
            except ValueError as e:
                print(
                    f"[ModalityManager] Warning: Could not load modality '{mod_name}': {e}"
                )

    @property
    def num_modalities(self) -> int:
        return len(self.modality_names)

    @property
    def has_modalities(self) -> bool:
        return self.num_modalities > 0

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract features from all modalities.

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Dict[modality_name, Dict[scale_name, tensor]]
        """
        all_features = {}

        for mod_name in self.modality_names:
            extractor = self.extractors[mod_name]
            all_features[mod_name] = extractor(x)

        return all_features

    def get_features_at_scale(
        self,
        all_features: Dict[str, Dict[str, torch.Tensor]],
        scale: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Get features from all modalities at a specific scale.

        Args:
            all_features: Features from forward()
            scale: Scale name ('level1', 'level2', 'bottleneck')

        Returns:
            Dict[modality_name, tensor]
        """
        scale_features = {}
        for mod_name, features in all_features.items():
            if scale in features:
                scale_features[mod_name] = features[scale]
        return scale_features


# =============================================================================
# Cross-Attention for Multimodal Fusion
# =============================================================================


class MultiModalCrossAttention(nn.Module):
    """
    Cross-attention module that fuses RGB features with multiple modality features.

    RGB features serve as queries, while modality features provide keys and values.
    Multiple modalities are fused using the specified fusion strategy.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        num_modalities: int = 1,
        fusion_type: str = "gate",
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.num_modalities = num_modalities
        self.fusion_type = fusion_type
        self.scale = dim_head**-0.5

        # Query projection (from RGB features)
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        # Key and Value projections for each modality
        self.to_k = nn.ModuleList(
            [
                nn.Linear(dim, dim_head * heads, bias=False)
                for _ in range(max(1, num_modalities))
            ]
        )
        self.to_v = nn.ModuleList(
            [
                nn.Linear(dim, dim_head * heads, bias=False)
                for _ in range(max(1, num_modalities))
            ]
        )

        # Learnable temperature
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))

        # Output projection
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)

        # Fusion mechanism
        if fusion_type == "gate" and num_modalities > 0:
            self.fusion_gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid(),
            )

    def forward(
        self,
        x: torch.Tensor,
        modality_feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x: RGB features [B, H, W, C]
            modality_feats: Dict of modality features, each [B, H, W, C]

        Returns:
            Fused features [B, H, W, C]
        """
        if not modality_feats:
            return x

        b, h, w, c = x.shape
        x_flat = x.reshape(b, h * w, c)

        # Query from RGB
        q = self.to_q(x_flat)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        q = q.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)

        # Process each modality
        out_accumulated = None
        mod_list = list(modality_feats.values())

        for i, mod_feat in enumerate(mod_list):
            if i >= len(self.to_k):
                break

            mod_flat = mod_feat.reshape(b, h * w, c)

            k = self.to_k[i](mod_flat)
            v = self.to_v[i](mod_flat)

            k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
            v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

            k = k.transpose(-2, -1)
            v = v.transpose(-2, -1)
            k = F.normalize(k, dim=-1, p=2)

            # Attention
            attn = (k @ q.transpose(-2, -1)) * self.rescale
            attn = attn.softmax(dim=-1)
            out = attn @ v

            out = out.permute(0, 3, 1, 2).reshape(
                b, h * w, self.num_heads * self.dim_head
            )
            out = self.proj(out).view(b, h, w, c)

            # Accumulate based on fusion type
            if out_accumulated is None:
                out_accumulated = out
            elif self.fusion_type == "add":
                out_accumulated = out_accumulated + out
            elif self.fusion_type == "gate":
                gate_input = torch.cat([out_accumulated, out], dim=-1)
                gate = self.fusion_gate(gate_input)
                out_accumulated = gate * out_accumulated + (1 - gate) * out
            else:
                out_accumulated = out_accumulated + out

        return out_accumulated if out_accumulated is not None else x


# =============================================================================
# Multimodal Illumination-Guided Multi-head Self-Attention
# =============================================================================


class MultiModal_IG_MSA(nn.Module):
    """
    Illumination-Guided MSA with optional multimodal cross-attention.

    Extends IG_MSA by adding a parallel cross-attention branch for
    any number of auxiliary modalities.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        num_modalities: int = 0,
        fusion_type: str = "gate",
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.num_modalities = num_modalities
        self.fusion_type = fusion_type

        # Self-attention projections
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))

        # Cross-attention for modalities
        if num_modalities > 0:
            self.cross_attn = MultiModalCrossAttention(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                num_modalities=num_modalities,
                fusion_type=fusion_type,
            )

            # Final fusion gate
            if fusion_type == "gate":
                self.final_gate = nn.Sequential(
                    nn.Linear(dim * 2, dim),
                    nn.Sigmoid(),
                )

        # Output projection
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)

        # Positional embedding
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(
        self,
        x_in: torch.Tensor,
        illu_fea_trans: torch.Tensor,
        modality_feats: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_in: Input features [B, H, W, C]
            illu_fea_trans: Illumination features [B, H, W, C]
            modality_feats: Dict of modality features [B, H, W, C] (optional)
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)

        # Self-attention with illumination guidance
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        illu_attn = illu_fea_trans
        q, k, v, illu_attn = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)),
        )

        v = v * illu_attn

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        x_self = attn @ v

        x_self = x_self.permute(0, 3, 1, 2).reshape(
            b, h * w, self.num_heads * self.dim_head
        )
        out_self = self.proj(x_self).view(b, h, w, c)

        # Add positional embedding
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )
        out_self = out_self + out_p

        # Cross-attention with modalities
        if self.num_modalities > 0 and modality_feats:
            out_cross = self.cross_attn(x_in, modality_feats)

            if self.fusion_type == "gate":
                gate_input = torch.cat([out_self, out_cross], dim=-1)
                gate = self.final_gate(gate_input)
                out = gate * out_self + (1 - gate) * out_cross
            elif self.fusion_type == "add":
                out = out_self + out_cross
            else:
                out = out_self + out_cross
        else:
            out = out_self

        return out


# =============================================================================
# Feed Forward Network
# =============================================================================


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


# =============================================================================
# Multimodal Illumination-Guided Attention Block
# =============================================================================


class MultiModal_IGAB(nn.Module):
    """
    Multimodal Illumination-Guided Attention Block.

    Extends IGAB with optional multimodal cross-attention.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        num_blocks: int = 2,
        num_modalities: int = 0,
        fusion_type: str = "gate",
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.num_modalities = num_modalities

        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        MultiModal_IG_MSA(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            num_modalities=num_modalities,
                            fusion_type=fusion_type,
                        ),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        illu_fea: torch.Tensor,
        modality_feats: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]
            illu_fea: Illumination features [B, C, H, W]
            modality_feats: Dict of modality features [B, C, H, W] (optional)
        """
        x = x.permute(0, 2, 3, 1)
        illu_fea_trans = illu_fea.permute(0, 2, 3, 1)

        # Transpose modality features
        mod_feats_trans = None
        if modality_feats:
            mod_feats_trans = {
                k: v.permute(0, 2, 3, 1) for k, v in modality_feats.items()
            }

        for attn, ff in self.blocks:
            x = attn(x, illu_fea_trans, mod_feats_trans) + x
            x = ff(x) + x

        return x.permute(0, 3, 1, 2)


# =============================================================================
# Illumination Estimator
# =============================================================================


class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle,
            n_fea_middle,
            kernel_size=5,
            padding=2,
            bias=True,
            groups=n_fea_in,
        )
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img, mean_c], dim=1)
        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


# =============================================================================
# Multimodal Denoiser
# =============================================================================


class MultiModal_Denoiser(nn.Module):
    """
    Multimodal Denoiser with U-Net structure.

    Extends the original Denoiser with multimodal cross-attention at each level.
    """

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 3,
        dim: int = 31,
        level: int = 2,
        num_blocks: List[int] = [2, 4, 4],
        num_modalities: int = 0,
        fusion_type: str = "gate",
    ):
        super().__init__()
        self.dim = dim
        self.level = level
        self.num_modalities = num_modalities

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        MultiModal_IGAB(
                            dim=dim_level,
                            num_blocks=num_blocks[i],
                            dim_head=dim,
                            heads=dim_level // dim,
                            num_modalities=num_modalities,
                            fusion_type=fusion_type,
                        ),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                    ]
                )
            )
            dim_level *= 2

        # Bottleneck
        self.bottleneck = MultiModal_IGAB(
            dim=dim_level,
            dim_head=dim,
            heads=dim_level // dim,
            num_blocks=num_blocks[-1],
            num_modalities=num_modalities,
            fusion_type=fusion_type,
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            dim_level,
                            dim_level // 2,
                            stride=2,
                            kernel_size=2,
                            padding=0,
                            output_padding=0,
                        ),
                        nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                        MultiModal_IGAB(
                            dim=dim_level // 2,
                            num_blocks=num_blocks[level - 1 - i],
                            dim_head=dim,
                            heads=(dim_level // 2) // dim,
                            num_modalities=num_modalities,
                            fusion_type=fusion_type,
                        ),
                    ]
                )
            )
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        illu_fea: torch.Tensor,
        all_modality_features: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]
            illu_fea: Illumination features [B, C, H, W]
            all_modality_features: Dict[modality_name, Dict[scale_name, tensor]]
        """

        # Helper to get features at scale
        def get_features_at_scale(scale: str) -> Optional[Dict[str, torch.Tensor]]:
            if not all_modality_features:
                return None
            return {
                mod_name: features[scale]
                for mod_name, features in all_modality_features.items()
                if scale in features
            }

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        scale_names = ["level1", "level2"]

        for i, (IGAB_block, FeaDownSample, IlluFeaDownsample) in enumerate(
            self.encoder_layers
        ):
            scale = scale_names[i] if i < len(scale_names) else "level2"
            mod_feats = get_features_at_scale(scale)

            fea = IGAB_block(fea, illu_fea, mod_feats)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        mod_feats = get_features_at_scale("bottleneck")
        fea = self.bottleneck(fea, illu_fea, mod_feats)

        # Decoder
        for i, (FeaUpSample, Fusion, IGAB_block) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]

            scale = (
                scale_names[self.level - 1 - i]
                if (self.level - 1 - i) < len(scale_names)
                else "level1"
            )
            mod_feats = get_features_at_scale(scale)

            fea = IGAB_block(fea, illu_fea, mod_feats)

        # Output
        out = self.mapping(fea) + x
        return out


# =============================================================================
# Single Stage
# =============================================================================


class MultiModalRetinexFormer_SingleStage(nn.Module):
    """Single stage of MultiModal RetinexFormer."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_feat: int = 31,
        level: int = 2,
        num_blocks: List[int] = [1, 1, 1],
        num_modalities: int = 0,
        fusion_type: str = "gate",
    ):
        super().__init__()
        self.num_modalities = num_modalities

        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = MultiModal_Denoiser(
            in_dim=in_channels,
            out_dim=out_channels,
            dim=n_feat,
            level=level,
            num_blocks=num_blocks,
            num_modalities=num_modalities,
            fusion_type=fusion_type,
        )

    def forward(
        self,
        img: torch.Tensor,
        all_modality_features: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img, illu_fea, all_modality_features)
        return output_img


# =============================================================================
# Main Model: MultiModal RetinexFormer
# =============================================================================


class MultiModalRetinexFormer(nn.Module):
    """
    MultiModal RetinexFormer for Low-Light Image Enhancement.

    This model extends RetinexFormer with a modular, pluggable system for
    adding multiple modalities (depth, segmentation, saliency, etc.).

    Args:
        in_channels: Input image channels (default: 3)
        out_channels: Output image channels (default: 3)
        n_feat: Base feature dimension (default: 40)
        stage: Number of refinement stages (default: 3)
        num_blocks: Number of attention blocks per level (default: [1, 2, 2])
        modalities: Dict of modality configurations (optional)
        fusion_type: Global fusion type (default: 'gate')

    Example config in yml:
        network_g:
          type: MultiModalRetinexFormer
          modalities:
            depth:
              enabled: true
              encoder: vits
              checkpoint: path/to/checkpoint.pth
            # Add more modalities here
          fusion_type: gate
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_feat: int = 40,
        stage: int = 3,
        num_blocks: List[int] = [1, 2, 2],
        modalities: Optional[Dict[str, Dict[str, Any]]] = None,
        fusion_type: str = "gate",
    ):
        super().__init__()
        self.stage = stage
        self.n_feat = n_feat
        self.fusion_type = fusion_type

        # Initialize modality manager
        self.modality_manager = ModalityManager(
            modalities_config=modalities,
            target_dim=n_feat,
        )

        num_modalities = self.modality_manager.num_modalities
        print(f"[MultiModalRetinexFormer] Initialized with {num_modalities} modalities")

        # Build multi-stage pipeline
        self.stages = nn.ModuleList(
            [
                MultiModalRetinexFormer_SingleStage(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_feat=n_feat,
                    level=2,
                    num_blocks=num_blocks,
                    num_modalities=num_modalities,
                    fusion_type=fusion_type,
                )
                for _ in range(stage)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input low-light image [B, 3, H, W]

        Returns:
            Enhanced image [B, 3, H, W]
        """
        # Extract modality features once (shared across stages)
        all_modality_features = None
        if self.modality_manager.has_modalities:
            all_modality_features = self.modality_manager(x)

        # Progressive refinement
        out = x
        for stage in self.stages:
            out = stage(out, all_modality_features)

        return out

    def get_modality_visualizations(
        self,
        x: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Get visualizations from all modalities."""
        visualizations = {}
        for mod_name in self.modality_manager.modality_names:
            extractor = self.modality_manager.extractors[mod_name]
            visualizations[mod_name] = extractor.get_visualization(x)
        return visualizations


# =============================================================================
# Backward Compatible Alias
# =============================================================================


# For backward compatibility with existing configs
DepthGuidedRetinexFormer = MultiModalRetinexFormer


# =============================================================================
# Testing
# =============================================================================


if __name__ == "__main__":
    print("Testing MultiModalRetinexFormer...")

    # Test without modalities
    print("\n1. Testing without modalities...")
    model = MultiModalRetinexFormer(
        stage=1,
        n_feat=40,
        num_blocks=[1, 2, 2],
        modalities=None,
    ).cuda()

    inputs = torch.randn((1, 3, 256, 256)).cuda()
    with torch.no_grad():
        output = model(inputs)
    print(f"   Input: {inputs.shape}, Output: {output.shape}")
    print(f"   Params: {sum(p.nelement() for p in model.parameters()):,}")

    # Test with depth modality
    print("\n2. Testing with depth modality...")
    model_depth = MultiModalRetinexFormer(
        stage=1,
        n_feat=40,
        num_blocks=[1, 2, 2],
        modalities={
            "depth": {
                "enabled": True,
                "encoder": "vits",
                "checkpoint": None,
                "freeze": True,
            }
        },
        fusion_type="gate",
    ).cuda()

    with torch.no_grad():
        output_depth = model_depth(inputs)
    print(f"   Input: {inputs.shape}, Output: {output_depth.shape}")

    trainable = sum(p.nelement() for p in model_depth.parameters() if p.requires_grad)
    total = sum(p.nelement() for p in model_depth.parameters())
    print(f"   Trainable: {trainable:,}, Total: {total:,}")

    print("\nAll tests passed!")
