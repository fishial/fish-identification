# -*- coding: utf-8 -*-
"""
Model architectures for Fish Classification with Metric Learning.

This module provides flexible model architectures supporting:
- Multiple backbone types (ViT, ConvNeXt, EfficientNet, etc.)
- Multiple pooling strategies (Attention, GeM, Average)
- ArcFace classification head
- Configurable embedding layers

Key Classes:
- StableEmbeddingModelViT: For Vision Transformer backbones
- StableEmbeddingModel: For CNN backbones (ConvNeXt, EfficientNet, etc.)
"""

# Standard library
import math
from typing import Optional, Literal, Tuple

# PyTorch and ecosystem
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.vision_transformer import VisionTransformer


# =============================================================================
# Pooling Layers
# =============================================================================

class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    
    Popular in image retrieval tasks. Provides a learnable pooling between
    average pooling (p=1) and max pooling (p→∞).
    
    Reference: "Fine-tuning CNN Image Retrieval with No Human Annotation" (Radenović et al.)
    
    Args:
        p: Initial pooling parameter (default: 3.0)
        eps: Small value for numerical stability
        learnable: Whether p should be learnable
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer('p', torch.ones(1) * p)
        self.eps = eps
        self.learnable = learnable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, H, W]
        Returns:
            Pooled features [B, C]
        """
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            1
        ).pow(1.0 / self.p).squeeze(-1).squeeze(-1)
    
    def __repr__(self):
        return f"GeMPooling(p={self.p.item():.2f}, learnable={self.learnable})"


class ViTAttentionPooling(nn.Module):
    """
    Attention Pooling for Vision Transformer output of shape [B, N, D].
    Computes a weighted sum of patch embeddings based on learned attention.
    
    Args:
        in_features: Input feature dimension
        hidden_features: Hidden dimension for attention network
    """
    def __init__(self, in_features: int, hidden_features: Optional[int] = None):
        super().__init__()
        if hidden_features is None:
            hidden_features = max(in_features // 4, 128)

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
            x: ViT output [B, N, D]
            object_mask: Not used for ViT, kept for interface compatibility
            return_attention_map: Whether to return attention weights
        
        Returns:
            pooled: Pooled features [B, D]
            weights: Optional attention weights [B, N, 1]
        """
        attention_scores = self.attention_net(x)  # [B, N, 1]
        weights = F.softmax(attention_scores, dim=1)  # [B, N, 1]
        pooled = (x * weights).sum(dim=1)  # [B, D]

        if return_attention_map:
            return pooled, weights
        return pooled, None

        
class AttentionPooling(nn.Module):
    """
    Attention-based pooling for CNN feature maps.
    Weighs spatial features based on learned attention, optionally focusing
    on regions within a provided object mask.
    
    Args:
        in_channels: Number of input channels
        hidden_channels: Hidden channels for attention conv
    """
    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(in_channels // 4, 32)

        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        object_mask: Optional[torch.Tensor] = None, 
        return_attention_map: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Feature map [B, C, H, W]
            object_mask: Optional binary mask [B, 1, H', W'] or [B, H', W']
            return_attention_map: Whether to return attention weights
        
        Returns:
            pooled: Pooled features [B, C]
            weights: Optional attention map [B, 1, H, W]
        """
        x_for_attn = x

        if object_mask is not None:
            B, _, H_feat, W_feat = x.shape
            object_mask_for_x = object_mask.float().to(x.device)
            if object_mask_for_x.ndim == 3:
                object_mask_for_x = object_mask_for_x.unsqueeze(1)

            if object_mask_for_x.shape[2] != H_feat or object_mask_for_x.shape[3] != W_feat:
                object_mask_for_x_resized = F.interpolate(
                    object_mask_for_x, size=(H_feat, W_feat), mode='nearest'
                )
            else:
                object_mask_for_x_resized = object_mask_for_x

            x_for_attn = x * object_mask_for_x_resized

        attention_scores = self.attention_conv(x_for_attn)
        weights = torch.sigmoid(attention_scores)

        final_weights_for_pooling = weights
        if object_mask is not None:
            B_w, _, H_attn, W_attn = weights.shape
            object_mask_for_weights = object_mask.float().to(weights.device)
            if object_mask_for_weights.ndim == 3:
                object_mask_for_weights = object_mask_for_weights.unsqueeze(1)
            mask_downsampled_for_weights = F.interpolate(
                object_mask_for_weights, size=(H_attn, W_attn), mode='nearest'
            )
            final_weights_for_pooling = weights * mask_downsampled_for_weights

        weighted_features = x * final_weights_for_pooling
        sum_weighted_features = weighted_features.sum(dim=(2, 3))
        sum_weights = final_weights_for_pooling.sum(dim=(2, 3)).clamp(min=1e-6)
        pooled = sum_weighted_features / sum_weights

        if return_attention_map:
            return pooled, final_weights_for_pooling
        return pooled, None


class HybridPooling(nn.Module):
    """
    Hybrid pooling combining GeM and Attention pooling.
    
    Concatenates GeM-pooled features with attention-pooled features.
    
    Args:
        in_channels: Number of input channels
        gem_p: Initial p for GeM pooling
        attention_hidden: Hidden channels for attention
        output_mode: 'concat' (2*C output) or 'add' (C output)
    """
    def __init__(
        self,
        in_channels: int,
        gem_p: float = 3.0,
        attention_hidden: Optional[int] = None,
        output_mode: Literal['concat', 'add'] = 'concat',
    ):
        super().__init__()
        self.gem = GeMPooling(p=gem_p, learnable=True)
        self.attention = AttentionPooling(in_channels, attention_hidden)
        self.output_mode = output_mode
        
        if output_mode == 'add':
            # Learnable weights for combining
            self.gem_weight = nn.Parameter(torch.tensor(0.5))
            self.attn_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self, 
        x: torch.Tensor, 
        object_mask: Optional[torch.Tensor] = None,
        return_attention_map: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        gem_out = self.gem(x)
        attn_out, attn_map = self.attention(x, object_mask, return_attention_map=True)
        
        if self.output_mode == 'concat':
            pooled = torch.cat([gem_out, attn_out], dim=1)
        else:
            w_gem = torch.sigmoid(self.gem_weight)
            w_attn = torch.sigmoid(self.attn_weight)
            pooled = w_gem * gem_out + w_attn * attn_out
        
        if return_attention_map:
            return pooled, attn_map
        return pooled, None
    
    @property
    def output_features(self) -> int:
        """Returns output feature dimension multiplier."""
        return 2 if self.output_mode == 'concat' else 1


# =============================================================================
# ArcFace Head
# =============================================================================

class ArcFaceHead(nn.Module):
    """
    ArcFace loss head for metric learning.
    Implements the additive angular margin penalty.
    
    Reference: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    
    Args:
        embedding_dim: Dimension of input embeddings
        num_classes: Number of classes
        s: Scale factor (default: 32.0)
        m: Angular margin (default: 0.10)
    """
    def __init__(
        self, 
        embedding_dim: int, 
        num_classes: int, 
        s: float = 32.0, 
        m: float = 0.10
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Buffers for constants
        self.register_buffer('cos_m', torch.tensor(math.cos(m)))
        self.register_buffer('sin_m', torch.tensor(math.sin(m)))
        self.register_buffer('th', torch.tensor(math.cos(math.pi - m)))
        self.register_buffer('mm', torch.tensor(math.sin(math.pi - m) * m))
        self.register_buffer('eps', torch.tensor(1e-7))

    def set_margin(self, new_m: float):
        """Dynamically update the margin 'm' and its related constants."""
        self.m = new_m
        self.cos_m.data = torch.tensor(math.cos(new_m), device=self.cos_m.device)
        self.sin_m.data = torch.tensor(math.sin(new_m), device=self.sin_m.device)
        self.th.data = torch.tensor(math.cos(math.pi - new_m), device=self.th.device)
        self.mm.data = torch.tensor(math.sin(math.pi - new_m) * new_m, device=self.mm.device)

    def forward(
        self, 
        normalized_emb: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            normalized_emb: L2-normalized embeddings [B, D]
            labels: Optional class labels [B] (required during training)
        
        Returns:
            Scaled logits [B, num_classes]
        """
        normalized_w = F.normalize(self.weight, dim=1)
        cosine = F.linear(normalized_emb, normalized_w)

        if labels is not None:
            cosine_sq = cosine ** 2
            sine = torch.sqrt((1.0 - cosine_sq).clamp(min=self.eps.item()))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            output = cosine.clone()
            idx = labels.to(dtype=torch.long, device=cosine.device).view(-1, 1)
            src = phi.gather(1, idx).to(dtype=output.dtype)
            output.scatter_(1, idx, src)
            output *= self.s
        else:
            output = cosine * self.s

        return output


# =============================================================================
# Main Model Classes
# =============================================================================

class StableEmbeddingModelViT(nn.Module):
    """
    Embedding model for Vision Transformer backbones.
    
    Supports various ViT architectures from timm including:
    - BEiT v2, DeiT, ViT
    - MaxViT, MaxxViT
    - EVA, DINOv2
    - Swin Transformer
    
    Args:
        embedding_dim: Output embedding dimension
        num_classes: Number of classes for ArcFace head
        pretrained_backbone: Whether to use pretrained weights
        freeze_backbone_initially: Freeze backbone at init
        backbone_model_name: timm model name
        custom_backbone: Optional pre-configured backbone
        attention_hidden_channels: Hidden dim for attention pooling
        arcface_s: ArcFace scale parameter
        arcface_m: ArcFace margin parameter
        add_bn_to_embedding: Add BatchNorm after embedding FC
        embedding_dropout_rate: Dropout rate for embedding
        pooling_type: 'attention', 'gem', or 'avg'
    """
    def __init__(
        self,
        embedding_dim: int = 128,
        num_classes: int = 1000,
        pretrained_backbone: bool = True,
        freeze_backbone_initially: bool = False,
        backbone_model_name: str = 'beitv2_base_patch16_224.in1k_ft_in22k_in1k',
        custom_backbone: Optional[VisionTransformer] = None,
        attention_hidden_channels: Optional[int] = None,
        arcface_s: float = 64.0,
        arcface_m: float = 0.5,
        add_bn_to_embedding: bool = False,
        embedding_dropout_rate: float = 0.11,
        pooling_type: str = 'attention',
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.pooling_type = pooling_type

        if custom_backbone:
            self.backbone = custom_backbone
            print("Using custom ViT backbone.")
        else:
            print(f"Loading ViT backbone: {backbone_model_name}")
            self.backbone: VisionTransformer = timm.create_model(
                backbone_model_name,
                pretrained=pretrained_backbone,
                num_classes=0
            )

        self.backbone_out_features = self._infer_backbone_embedding_dim()
        self.backbone_feature_extractor = self.backbone.forward_features
        
        if freeze_backbone_initially:
            self.freeze_backbone()
        
        # Pooling layer
        if pooling_type == 'attention':
            self.pooling = ViTAttentionPooling(
                in_features=self.backbone_out_features,
                hidden_features=attention_hidden_channels
            )
        else:
            # For ViT, we'll reshape tokens back to spatial and use GeM
            self.pooling = None  # Will use global average pooling
        
        # Embedding layers
        embedding_layers = [nn.Linear(self.backbone_out_features, embedding_dim)]
        if add_bn_to_embedding:
            embedding_layers.append(nn.BatchNorm1d(embedding_dim))
        if embedding_dropout_rate > 0.0:
            embedding_layers.append(nn.Dropout(embedding_dropout_rate))
            
        self.embedding_fc = nn.Sequential(*embedding_layers)
        self.arcface_head = ArcFaceHead(embedding_dim, num_classes, s=arcface_s, m=arcface_m)
        
        print(f"StableEmbeddingModel initialized with ViT backbone: "
              f"{backbone_model_name if not custom_backbone else 'custom'}")
        print(f"  Embedding Dim: {embedding_dim}, Num Classes: {num_classes}")
        print(f"  ArcFace s: {arcface_s}, m: {arcface_m}")
        print(f"  Backbone out features: {self.backbone_out_features}")
        print(f"  Pooling type: {pooling_type}")

    def _tokens_and_grid_from_features(self, features: torch.Tensor):
        """Normalize backbone features into token tensor [B, N, D] + optional grid."""
        if features.ndim == 4:
            B, C, H, W = features.shape
            tokens = features.flatten(2).transpose(1, 2).contiguous()
            return tokens, (H, W)

        if features.ndim == 3:
            tokens = features
            if hasattr(self.backbone, "cls_token") and tokens.shape[1] > 1:
                tokens = tokens[:, 1:, :]

            if hasattr(self.backbone, "patch_embed") and hasattr(self.backbone.patch_embed, "grid_size"):
                gs = self.backbone.patch_embed.grid_size
                if isinstance(gs, (tuple, list)) and len(gs) == 2 and int(gs[0]) * int(gs[1]) == tokens.shape[1]:
                    return tokens, (int(gs[0]), int(gs[1]))

            N = tokens.shape[1]
            s = int(round(math.sqrt(N)))
            if s * s == N:
                return tokens, (s, s)

            return tokens, None

        raise ValueError(f"Unsupported backbone output shape: {tuple(features.shape)}")

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        print("Freezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, specific_layer_keywords=None, verbose=False):
        """Unfreeze backbone parameters, optionally filtering by keywords."""
        print(f"Unfreezing backbone parameters... (Keywords: {specific_layer_keywords})")
        unfrozen_count = 0
        for name, param in self.backbone.named_parameters():
            if specific_layer_keywords is None or any(kw in name for kw in specific_layer_keywords):
                param.requires_grad = True
                unfrozen_count += 1
                if verbose:
                    print(f"  Unfroze: {name}")
        print(f"Total parameters unfrozen: {unfrozen_count}")

    def _infer_backbone_embedding_dim(self) -> int:
        """Infer backbone output dimension."""
        for attr in ("num_features", "embed_dim"):
            v = getattr(self.backbone, attr, None)
            if isinstance(v, int) and v > 0:
                return int(v)

        def _infer_input_hw() -> int:
            cfg = getattr(self.backbone, "default_cfg", None) or {}
            inp = cfg.get("input_size", None)
            if isinstance(inp, (tuple, list)) and len(inp) == 3:
                return int(inp[1])
            name = str(getattr(self.backbone, "name", "") or "")
            for s in (512, 384, 256, 224):
                if name.endswith(f"_{s}"):
                    return s
            return 224

        self.backbone.eval()
        orig_device = next(self.backbone.parameters()).device
        self.backbone.to("cpu")
        with torch.no_grad():
            hw = _infer_input_hw()
            dummy = torch.randn(1, 3, hw, hw)
            features = self.backbone.forward_features(dummy)
        self.backbone.to(orig_device)

        if features.ndim == 4:
            return int(features.shape[1])
        if features.ndim == 3:
            return int(features.shape[-1])
        raise ValueError(f"Unsupported output shape: {tuple(features.shape)}")

    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None, 
        object_mask: Optional[torch.Tensor] = None, 
        return_softmax: bool = False, 
        return_attention_map: bool = True
    ):
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            labels: Optional class labels [B]
            object_mask: Optional object mask (ignored for ViT)
            return_softmax: Return softmax probabilities instead of logits
            return_attention_map: Return attention visualization map
        
        Returns:
            emb_norm: L2-normalized embeddings [B, D]
            logits/probs: Class logits or probabilities [B, num_classes]
            attn_map: Optional attention map for visualization
        """
        features = self.backbone_feature_extractor(x)
        tokens, grid = self._tokens_and_grid_from_features(features)
        
        if self.pooling is not None:
            pooled, attn_weights = self.pooling(tokens, object_mask=object_mask, return_attention_map=True)
        else:
            # Global average pooling
            pooled = tokens.mean(dim=1)
            attn_weights = None

        emb_raw = self.embedding_fc(pooled)
        emb_norm = F.normalize(emb_raw, p=2, dim=1)
        logits = self.arcface_head(emb_norm, labels)

        vis_attn_map = None
        if return_attention_map and attn_weights is not None and grid is not None:
            try:
                B, N, _ = attn_weights.shape
                H, W = grid
                if H * W == N:
                    vis_attn_map = attn_weights.permute(0, 2, 1).reshape(B, 1, H, W)
            except Exception:
                vis_attn_map = None

        if return_softmax:
            probabilities = F.softmax(logits, dim=1)
            return emb_norm, probabilities, vis_attn_map if return_attention_map else None
        return emb_norm, logits, vis_attn_map if return_attention_map else None

        
class StableEmbeddingModel(nn.Module):
    """
    Embedding model for CNN backbones (ConvNeXt, EfficientNet, ResNet, etc.).
    
    Args:
        embedding_dim: Output embedding dimension
        num_classes: Number of classes for ArcFace head
        pretrained_backbone: Whether to use pretrained weights
        freeze_backbone_initially: Freeze backbone at init
        backbone_model_name: timm model name
        custom_backbone: Optional pre-configured backbone
        backbone_out_features: Output channels (auto-detected if None)
        attention_hidden_channels: Hidden channels for attention
        arcface_s: ArcFace scale parameter
        arcface_m: ArcFace margin parameter
        add_bn_to_embedding: Add BatchNorm after embedding FC
        embedding_dropout_rate: Dropout rate for embedding
        pooling_type: 'attention', 'gem', 'hybrid', or 'avg'
    """
    def __init__(
        self,
        embedding_dim: int = 256,
        num_classes: int = 1000,
        pretrained_backbone: bool = True,
        freeze_backbone_initially: bool = False,
        backbone_model_name: str = 'convnext_tiny',
        custom_backbone=None,
        backbone_out_features: int = 768,
        attention_hidden_channels: Optional[int] = None,
        arcface_s: float = 32.0,
        arcface_m: float = 0.11,
        add_bn_to_embedding: bool = True,
        embedding_dropout_rate: float = 0.0,
        pooling_type: str = 'attention',
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.backbone_out_features = backbone_out_features
        self.pooling_type = pooling_type

        if custom_backbone:
            self.backbone = custom_backbone
            self.backbone_feature_extractor = self.backbone
            print("Using custom backbone.")
        elif 'convnext' in backbone_model_name:
            print(f"Loading backbone from timm: {backbone_model_name}")
            self.backbone = timm.create_model(
                backbone_model_name,
                pretrained=pretrained_backbone,
                features_only=True,
                out_indices=(-1,)
            )
            self.backbone_feature_extractor = lambda x: self.backbone(x)[-1]
            
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                out = self.backbone_feature_extractor(dummy_input)
            self.backbone_out_features = out.shape[1]
            print(f"  Detected backbone output channels: {self.backbone_out_features}")
            
        else:
            try:
                print(f"Attempting to load generic backbone from timm: {backbone_model_name}")
                self.backbone = timm.create_model(
                    backbone_model_name,
                    pretrained=pretrained_backbone,
                    num_classes=0,
                    global_pool=''
                )
                self.backbone_feature_extractor = self.backbone.forward_features
                
                dummy_input = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    out = self.backbone_feature_extractor(dummy_input)
                self.backbone_out_features = out.shape[1]
                print(f"  Detected backbone output channels: {self.backbone_out_features}")
            except Exception as e:
                raise ValueError(f"Unsupported backbone: {backbone_model_name}. Error: {e}")

        if freeze_backbone_initially:
            self.freeze_backbone()

        # Pooling layer
        if pooling_type == 'attention':
            self.pooling = AttentionPooling(
                in_channels=self.backbone_out_features,
                hidden_channels=attention_hidden_channels
            )
            pooling_out_features = self.backbone_out_features
        elif pooling_type == 'gem':
            self.pooling = GeMPooling(p=3.0, learnable=True)
            pooling_out_features = self.backbone_out_features
        elif pooling_type == 'hybrid':
            self.pooling = HybridPooling(
                in_channels=self.backbone_out_features,
                attention_hidden=attention_hidden_channels,
                output_mode='concat'
            )
            pooling_out_features = self.backbone_out_features * 2
        else:  # 'avg'
            self.pooling = nn.AdaptiveAvgPool2d(1)
            pooling_out_features = self.backbone_out_features

        # Embedding layers
        embedding_layers = [nn.Linear(pooling_out_features, embedding_dim)]
        if add_bn_to_embedding:
            embedding_layers.append(nn.BatchNorm1d(embedding_dim))
        if embedding_dropout_rate > 0.0:
            embedding_layers.append(nn.Dropout(embedding_dropout_rate))

        self.embedding_fc = nn.Sequential(*embedding_layers)
        self.arcface_head = ArcFaceHead(embedding_dim, num_classes, s=arcface_s, m=arcface_m)

        print(f"StableEmbeddingModel initialized with backbone: "
              f"{backbone_model_name if not custom_backbone else 'custom'}")
        print(f"  Embedding Dim: {embedding_dim}, Num Classes: {num_classes}")
        print(f"  ArcFace s: {arcface_s}, m: {arcface_m}")
        print(f"  Backbone out features: {self.backbone_out_features}")
        print(f"  Pooling type: {pooling_type}")

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        print("Freezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, specific_layer_keywords=None, verbose=False):
        """Unfreeze backbone parameters."""
        print(f"Unfreezing backbone parameters... (Keywords: {specific_layer_keywords})")
        unfrozen_count = 0
        for name, param in self.backbone.named_parameters():
            if specific_layer_keywords is None or any(kw in name for kw in specific_layer_keywords):
                param.requires_grad = True
                unfrozen_count += 1
                if verbose:
                    print(f"  Unfroze: {name}")
        print(f"Total parameters unfrozen: {unfrozen_count}")

    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None, 
        object_mask: Optional[torch.Tensor] = None, 
        return_softmax: bool = False, 
        return_attention_map: bool = True
    ):
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            labels: Optional class labels [B]
            object_mask: Optional object mask for attention guidance
            return_softmax: Return softmax probabilities instead of logits
            return_attention_map: Return attention visualization map
        
        Returns:
            emb_norm: L2-normalized embeddings [B, D]
            logits/probs: Class logits or probabilities [B, num_classes]
            attn_map: Optional attention map for visualization
        """
        features = self.backbone_feature_extractor(x)
        
        attn_map = None
        if self.pooling_type == 'attention':
            pooled, attn_map = self.pooling(features, object_mask=object_mask, return_attention_map=return_attention_map)
        elif self.pooling_type == 'hybrid':
            pooled, attn_map = self.pooling(features, object_mask=object_mask, return_attention_map=return_attention_map)
        elif self.pooling_type == 'gem':
            pooled = self.pooling(features)
        else:  # avg
            pooled = self.pooling(features).squeeze(-1).squeeze(-1)
        
        emb_raw = self.embedding_fc(pooled)
        emb_norm = F.normalize(emb_raw, p=2, dim=1)
        logits = self.arcface_head(emb_norm, labels)

        attn_map_to_return = attn_map if return_attention_map else None

        if return_softmax:
            probabilities = F.softmax(logits, dim=1)
            return emb_norm, probabilities, attn_map_to_return
        return emb_norm, logits, attn_map_to_return
