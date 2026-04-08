"""
Pooling layers and helpers used by classification models.

This module provides various spatial and sequence pooling strategies for both 
Vision Transformers (ViT) and Convolutional Neural Networks (CNNs).
"""

import logging
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class ViTMeanPooling(nn.Module):
    """
    Simple Mean Pooling for Vision Transformer output of shape [B, N, D].
    Averages all token embeddings to force the model to look at the whole image.
    """
    def __init__(self):
        super().__init__()
        logger.debug("Initialized ViTMeanPooling layer.")

    def forward(
        self, 
        x: torch.Tensor, 
        object_mask: Optional[torch.Tensor] = None, 
        return_attention_map: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # Average over the N dimension (number of patches)
        # x has shape [B, N, D], pooled will be [B, D]
        pooled = x.mean(dim=1) 
        
        # Mean Pooling doesn't have a learnable attention map, 
        # but we can return uniform weights if required for logging/visualization
        if return_attention_map:
            B, N, _ = x.shape
            # Assign an equal weight of 1/N to each patch
            uniform_weights = torch.ones((B, N, 1), device=x.device) / N
            return pooled, uniform_weights
            
        return pooled, None


class ViTAttentionPooling_single_head(nn.Module):
    """
    Attention Pooling for Vision Transformer output of shape [B, N, D].
    Computes a weighted sum of patch embeddings based on learned attention.
    
    Args:
        in_features: Input feature dimension.
        hidden_features: Hidden dimension for the attention network.
        num_heads: Number of attention heads (kept for interface compatibility).
    """
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, num_heads: int = 1):
        super().__init__()
        if hidden_features is None:
            hidden_features = max(in_features // 4, 128)

        logger.debug(f"Initialized ViTAttentionPooling_single_head (in={in_features}, hidden={hidden_features})")
        self.attention_net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, 1)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        object_mask: Optional[torch.Tensor] = None, 
        return_attention_map: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: ViT output of shape [B, N, D].
            object_mask: Not used for ViT, kept for interface compatibility.
            return_attention_map: Whether to return attention weights.
        
        Returns:
            pooled: Pooled features of shape [B, D].
            weights: Optional attention weights of shape [B, N, 1].
        """
        attention_scores = self.attention_net(x)  # [B, N, 1]
        weights = F.softmax(attention_scores, dim=1)  # [B, N, 1]
        pooled = (x * weights).sum(dim=1)  # [B, D]

        if return_attention_map:
            return pooled, weights
        return pooled, None


class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling between average (p=1) and max (p→∞).
    Supports both CNN feature maps [B, C, H, W] and ViT token sequences [B, N, D].
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer('p', torch.ones(1) * p)
        self.eps = eps
        self.learnable = learnable
        logger.debug(f"Initialized GeMPooling (p={p}, learnable={learnable})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            # ViT tokens: [B, N, D] — pool over token dimension N
            return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)
        # CNN feature maps: [B, C, H, W]
        return (
            F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1)
            .pow(1.0 / self.p)
            .squeeze(-1)
            .squeeze(-1)
        )

    def __repr__(self):
        return f"GeMPooling(p={self.p.item():.2f}, learnable={self.learnable})"


class ViTAttentionPooling(nn.Module):
    """Multi-head attention pooling for ViT tokens."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        num_heads: int = 4,
        temperature: float = 2.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.tensor(temperature))

        if hidden_features is None:
            hidden_features = max(in_features // 4, 128)

        # Attention heads (with GELU – great for DINOv2)
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.GELU(),                    
                nn.Linear(hidden_features, 1),
            )
            for _ in range(num_heads)
        ])

        # Fusion layer (smartly combining heads)
        self.head_fusion = (
            nn.Sequential(
                nn.Linear(in_features * num_heads, in_features),
                nn.LayerNorm(in_features),
            )
            if num_heads > 1 else None
        )

    @property
    def attention_net(self):
        return self.attention_heads[0]

    def forward(
        self,
        x: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None, # Kept for API compatibility, not used internally
        return_attention_map: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:

        """
        x: [B, N, C]
        """
        B, N, C = x.shape

        head_outputs = []
        all_weights = []
        all_raw_scores = []

        for head in self.attention_heads:
            scores = head(x)              # [B, N, 1]
            all_raw_scores.append(scores)

            # REMOVED: scores.masked_fill(... -1e9).
            # The model should learn to ignore background through the Guidance Loss.

            weights = F.softmax(scores / self.temperature.abs().clamp(min=0.1), dim=1)
            pooled = (x * weights).sum(dim=1)  # [B, C]

            head_outputs.append(pooled)
            all_weights.append(weights)

        # --- Fuse heads ---
        if self.head_fusion is not None:
            # Concatenate along the channel dimension [B, C * num_heads] and project back to [B, C]
            pooled = self.head_fusion(torch.cat(head_outputs, dim=1))
        else:
            pooled = head_outputs[0]

        # --- Optional attention map for visualization ---
        attn_out = torch.cat(all_weights, dim=-1) if return_attention_map else None  # [B, N, heads]

        raw_scores = torch.cat(all_raw_scores, dim=-1)  # [B, N, H]

        return pooled, attn_out, raw_scores


def _resize_mask(
    mask: torch.Tensor,
    target_h: int,
    target_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Helper function to resize object masks to match feature map dimensions."""
    mask = mask.float().to(device)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[2] != target_h or mask.shape[3] != target_w:
        mask = F.interpolate(mask, size=(target_h, target_w), mode='nearest')
    return mask


class AttentionPooling(nn.Module):
    """Attention pooling strictly for CNN feature maps."""

    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(in_channels // 4, 32)

        logger.debug(f"Initialized CNN AttentionPooling (in={in_channels}, hidden={hidden_channels})")
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
        return_attention_map: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, _, H_feat, W_feat = x.shape
        x_for_attn = x

        if object_mask is not None:
            x_for_attn = x * _resize_mask(object_mask, H_feat, W_feat, x.device)

        weights = torch.sigmoid(self.attention_conv(x_for_attn))

        if object_mask is not None:
            _, _, H_attn, W_attn = weights.shape
            weights = weights * _resize_mask(object_mask, H_attn, W_attn, weights.device)

        sum_weighted = (x * weights).sum(dim=(2, 3))
        sum_weights = weights.sum(dim=(2, 3)).clamp(min=1e-6)
        pooled = sum_weighted / sum_weights

        return pooled, (weights if return_attention_map else None)


class HybridPooling(nn.Module):
    """Hybrid pooling combining GeM and spatial attention for CNNs."""

    def __init__(
        self,
        in_channels: int,
        gem_p: float = 3.0,
        attention_hidden: Optional[int] = None,
        output_mode: Literal['concat', 'add'] = 'concat',
    ):
        super().__init__()
        logger.debug(f"Initialized HybridPooling (mode={output_mode})")
        self.gem = GeMPooling(p=gem_p, learnable=True)
        self.attention = AttentionPooling(in_channels, attention_hidden)
        self.output_mode = output_mode

        if output_mode == 'add':
            self.gem_weight = nn.Parameter(torch.tensor(0.5))
            self.attn_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
        return_attention_map: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        gem_out = self.gem(x)
        attn_out, attn_map = self.attention(x, object_mask, return_attention_map=True)

        if self.output_mode == 'concat':
            pooled = torch.cat([gem_out, attn_out], dim=1)
        else:
            w_gem = torch.sigmoid(self.gem_weight)
            w_attn = torch.sigmoid(self.attn_weight)
            pooled = w_gem * gem_out + w_attn * attn_out

        return pooled, (attn_map if return_attention_map else None)

    @property
    def output_features(self) -> int:
        return 2 if self.output_mode == 'concat' else 1