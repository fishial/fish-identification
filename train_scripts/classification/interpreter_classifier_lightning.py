# -*- coding: utf-8 -*-
"""
Standalone Interpreter for Lightning-trained Fish Classification Models.

This module provides a self-contained classifier for loading and using models
trained with lightning_train.py. All necessary classes are included in this file
to enable standalone deployment without additional module dependencies.

Features:
- Load PyTorch Lightning checkpoints
- Support for both ViT and CNN backbones
- Multiple pooling strategies (Attention, GeM, Hybrid)
- FAISS-based nearest neighbor search (can be disabled)
- Centroid-based class filtering
- Automatic input size detection
- Robust error handling and validation
- Configurable kNN classifier (enable/disable)

Usage:
    config = {
        'log_level': 'INFO',
        'dataset': {'path': 'path/to/embeddings.pt'},
        'model': {
            'checkpoint_path': 'path/to/model.ckpt',
            'backbone_model_name': 'maxvit_base_tf_224',
            'embedding_dim': 512,
            'num_classes': 639,
            'arcface_s': 64.0,
            'arcface_m': 0.2,
            'pooling_type': 'attention',
            'input_size': 224,  # Optional, auto-detected if not provided
            'device': 'cuda'
        },
        # Optional inference parameters
        'use_knn': True,  # Enable/disable kNN classifier (default: True)
        'use_albumentations': False,  # Use albumentations transforms (default: False, uses torchvision)
        'arcface_min_score': 0.1,
        'centroid_fallback_score': 0.1,
        'topk_centroid': 5,
        'topk_neighbors': 10,
        'topk_arcface': 5,
        'centroid_threshold': 0.7,
        'neighbor_threshold': 0.8
    }
    
    # Initialize classifier
    classifier = EmbeddingClassifier(config)
    
    # Optional: warmup for stable performance
    classifier.warmup(num_iterations=5)
    
    # Single image inference
    results = classifier(image_array)  # np.ndarray [H, W, 3]
    
    # Batch inference
    results = classifier([img1, img2, img3])  # List[np.ndarray]
    
    # Get model information
    info = classifier.get_model_info()
    
    # Context manager usage (recommended)
    with EmbeddingClassifier(config) as classifier:
        results = classifier(image_array)
    # Auto cleanup on exit

Security Warning:
    This module uses torch.load() which relies on pickle and can execute arbitrary code.
    Only load checkpoints from trusted sources. The module attempts to use weights_only=True
    first for safety, but falls back to weights_only=False if needed. Always verify checksums
    and only load files from trusted sources in production environments.

Performance Notes:
    - Memory usage scales with number of classes and database size
    - Expected inference time: ~10-50ms per image (depending on backbone and device)
    - FAISS indices are pre-built for faster search but require memory
    - Large batches are automatically split into chunks (MAX_BATCH_SIZE) to prevent OOM errors
    - For optimal performance, keep batch sizes <= 32 images
"""

import logging
import time
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Literal

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
from torchvision import transforms
import timm
from timm.models.vision_transformer import VisionTransformer

# Optional: Albumentations support (install with: pip install albumentations)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None
    ToTensorV2 = None


# Constants
SUPPORTED_VIT_BACKBONES = ['vit', 'beit', 'deit', 'maxvit', 'maxxvit', 'eva', 'dino', 'swin']
DEFAULT_IMAGE_SIZE = 224
GEM_POOLING_DEFAULT_P = 3.0
ATTENTION_HIDDEN_DIVISOR = 4
ATTENTION_HIDDEN_MIN = 128
NUMERICAL_EPSILON = 1e-6
WEIGHT_NORMALIZATION_EPSILON = 1e-10
MAX_BATCH_SIZE = 32  # Maximum batch size to prevent OOM
DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_ARCFACE_MIN_SCORE = 0.1
DEFAULT_CENTROID_FALLBACK_SCORE = 0.1
DEFAULT_TOPK_CENTROID = 5
DEFAULT_TOPK_NEIGHBORS = 10
DEFAULT_TOPK_ARCFACE = 5
DEFAULT_CENTROID_THRESHOLD = 0.7
DEFAULT_NEIGHBOR_THRESHOLD = 0.8
DEFAULT_USE_KNN = True
DEFAULT_RERANK_MODE = 'hybrid'  # 'hybrid', 'weighted_fusion', or 'rrf'
DEFAULT_ARCFACE_WEIGHT = 0.6  # Weight for ArcFace in weighted fusion
DEFAULT_KNN_WEIGHT = 0.4  # Weight for kNN in weighted fusion
DEFAULT_RRF_K = 60  # Constant for Reciprocal Rank Fusion
DEFAULT_USE_ALBUMENTATIONS = False  # Use albumentations for transforms (if available)


# Setup Logger
logger = logging.getLogger("EmbeddingClassifier")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    name: str
    species_id: int
    distance: float
    accuracy: float  # Average similarity score (kept for backward compatibility)
    image_id: Optional[str]
    annotation_id: Optional[str]
    drawn_fish_id: Optional[str]
    
    @property
    def average_similarity(self) -> float:
        """Alias for accuracy field (which is actually average similarity)."""
        return self.accuracy


# =============================================================================
# Pooling Layers
# =============================================================================

class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    
    Popular in image retrieval tasks. Provides a learnable pooling between
    average pooling (p=1) and max pooling (p→∞).
    
    Reference: "Fine-tuning CNN Image Retrieval with No Human Annotation" (Radenović et al.)
    """
    def __init__(self, p: float = GEM_POOLING_DEFAULT_P, eps: float = NUMERICAL_EPSILON, learnable: bool = True):
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
        # Clamp both min and max for numerical stability
        x_clamped = x.clamp(min=self.eps, max=1e4)
        return F.adaptive_avg_pool2d(
            x_clamped.pow(self.p),
            1
        ).pow(1.0 / self.p.clamp(min=1e-2)).squeeze(-1).squeeze(-1)
    
    def __repr__(self):
        return f"GeMPooling(p={self.p.item():.2f}, learnable={self.learnable})"


class ViTAttentionPooling(nn.Module):
    """
    Attention Pooling for Vision Transformer output of shape [B, N, D].
    Computes a weighted sum of patch embeddings based on learned attention.
    """
    def __init__(self, in_features: int, hidden_features: Optional[int] = None):
        super().__init__()
        if hidden_features is None:
            hidden_features = max(in_features // ATTENTION_HIDDEN_DIVISOR, ATTENTION_HIDDEN_MIN)

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
    """
    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(in_channels // ATTENTION_HIDDEN_DIVISOR, 32)

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
        sum_weights = final_weights_for_pooling.sum(dim=(2, 3)).clamp(min=NUMERICAL_EPSILON)
        pooled = sum_weighted_features / sum_weights

        if return_attention_map:
            return pooled, final_weights_for_pooling
        return pooled, None


class HybridPooling(nn.Module):
    """
    Hybrid pooling combining GeM and Attention pooling.
    Concatenates GeM-pooled features with attention-pooled features.
    """
    def __init__(
        self,
        in_channels: int,
        gem_p: float = GEM_POOLING_DEFAULT_P,
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
        self.register_buffer('eps', torch.tensor(NUMERICAL_EPSILON))

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
# Model Classes
# =============================================================================

class StableEmbeddingModelViT(nn.Module):
    """
    Embedding model for Vision Transformer backbones.
    
    Supports various ViT architectures from timm including:
    - BEiT v2, DeiT, ViT
    - MaxViT, MaxxViT
    - EVA, DINOv2
    - Swin Transformer
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
            logger.info("Using custom ViT backbone.")
        else:
            logger.info(f"Loading ViT backbone: {backbone_model_name}")
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
            # For ViT, we'll use global average pooling
            self.pooling = None
        
        # Embedding layers
        embedding_layers = [nn.Linear(self.backbone_out_features, embedding_dim)]
        if add_bn_to_embedding:
            embedding_layers.append(nn.BatchNorm1d(embedding_dim))
        if embedding_dropout_rate > 0.0:
            embedding_layers.append(nn.Dropout(embedding_dropout_rate))
            
        self.embedding_fc = nn.Sequential(*embedding_layers)
        self.arcface_head = ArcFaceHead(embedding_dim, num_classes, s=arcface_s, m=arcface_m)
        
        logger.info(f"StableEmbeddingModelViT initialized")
        logger.info(f"  Embedding Dim: {embedding_dim}, Num Classes: {num_classes}")
        logger.info(f"  ArcFace s: {arcface_s}, m: {arcface_m}")
        logger.info(f"  Backbone out features: {self.backbone_out_features}")
        logger.info(f"  Pooling type: {pooling_type}")

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
        logger.info("Freezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, specific_layer_keywords=None, verbose=False):
        """Unfreeze backbone parameters, optionally filtering by keywords."""
        logger.info(f"Unfreezing backbone parameters... (Keywords: {specific_layer_keywords})")
        unfrozen_count = 0
        for name, param in self.backbone.named_parameters():
            if specific_layer_keywords is None or any(kw in name for kw in specific_layer_keywords):
                param.requires_grad = True
                unfrozen_count += 1
                if verbose:
                    logger.info(f"  Unfroze: {name}")
        logger.info(f"Total parameters unfrozen: {unfrozen_count}")

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

        output_attn = vis_attn_map if return_attention_map else None
        
        if return_softmax:
            probabilities = F.softmax(logits, dim=1)
            return emb_norm, probabilities, output_attn
        return emb_norm, logits, output_attn

        
class StableEmbeddingModel(nn.Module):
    """
    Embedding model for CNN backbones (ConvNeXt, EfficientNet, ResNet, etc.).
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
            logger.info("Using custom backbone.")
        elif 'convnext' in backbone_model_name:
            logger.info(f"Loading backbone from timm: {backbone_model_name}")
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
            logger.info(f"  Detected backbone output channels: {self.backbone_out_features}")
            
        else:
            try:
                logger.info(f"Attempting to load generic backbone from timm: {backbone_model_name}")
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
                logger.info(f"  Detected backbone output channels: {self.backbone_out_features}")
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

        logger.info(f"StableEmbeddingModel initialized")
        logger.info(f"  Embedding Dim: {embedding_dim}, Num Classes: {num_classes}")
        logger.info(f"  ArcFace s: {arcface_s}, m: {arcface_m}")
        logger.info(f"  Backbone out features: {self.backbone_out_features}")
        logger.info(f"  Pooling type: {pooling_type}")

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        logger.info("Freezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, specific_layer_keywords=None, verbose=False):
        """Unfreeze backbone parameters."""
        logger.info(f"Unfreezing backbone parameters... (Keywords: {specific_layer_keywords})")
        unfrozen_count = 0
        for name, param in self.backbone.named_parameters():
            if specific_layer_keywords is None or any(kw in name for kw in specific_layer_keywords):
                param.requires_grad = True
                unfrozen_count += 1
                if verbose:
                    logger.info(f"  Unfroze: {name}")
        logger.info(f"Total parameters unfrozen: {unfrozen_count}")

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

        output_attn = attn_map if return_attention_map else None

        if return_softmax:
            probabilities = F.softmax(logits, dim=1)
            return emb_norm, probabilities, output_attn
        return emb_norm, logits, output_attn


# =============================================================================
# Embedding Classifier
# =============================================================================

class EmbeddingClassifier:
    """
    Main classifier for inference using embedding-based approach.
    
    This classifier loads a trained model and uses FAISS for fast nearest neighbor search
    combined with centroid-based filtering for efficient classification.
    
    Configuration example:
        config = {
            'log_level': 'INFO',
            'dataset': {'path': 'embeddings.pt'},
            'model': {
                'checkpoint_path': 'model.ckpt',
                'backbone_model_name': 'maxvit_base_tf_224',
                'embedding_dim': 512,
                'num_classes': 639,
                'arcface_s': 64.0,
                'arcface_m': 0.2,
                'pooling_type': 'attention',
                'device': 'cuda'
            },
            'use_knn': True  # Enable/disable kNN classifier (default: True)
        }
    """
    
    def __init__(self, config: Dict):
        # Validate configuration
        self._validate_config(config)
        
        logger.setLevel(getattr(logging, config.get('log_level', 'INFO').upper()))
        
        # Load dataset
        self._load_data(config["dataset"]["path"])
        self.dim = self.db_embeddings.shape[1]
        self._prepare_centroids()

        logger.info("Initializing EmbeddingClassifier...")

        # Setup device
        self.device = config["model"].get("device", "cpu")
        
        # Load inference configuration
        self.use_knn = config.get('use_knn', DEFAULT_USE_KNN)
        self.arcface_min_score = config.get('arcface_min_score', DEFAULT_ARCFACE_MIN_SCORE)
        self.centroid_fallback_score = config.get('centroid_fallback_score', DEFAULT_CENTROID_FALLBACK_SCORE)
        self.default_topk_centroid = config.get('topk_centroid', DEFAULT_TOPK_CENTROID)
        self.default_topk_neighbors = config.get('topk_neighbors', DEFAULT_TOPK_NEIGHBORS)
        self.default_centroid_threshold = config.get('centroid_threshold', DEFAULT_CENTROID_THRESHOLD)
        self.default_neighbor_threshold = config.get('neighbor_threshold', DEFAULT_NEIGHBOR_THRESHOLD)
        self.default_topk_arcface = config.get('topk_arcface', DEFAULT_TOPK_ARCFACE)
        
        # Reranking configuration
        self.rerank_mode = config.get('rerank_mode', DEFAULT_RERANK_MODE)
        self.arcface_weight = config.get('arcface_weight', DEFAULT_ARCFACE_WEIGHT)
        self.knn_weight = config.get('knn_weight', DEFAULT_KNN_WEIGHT)
        self.rrf_k = config.get('rrf_k', DEFAULT_RRF_K)
        
        # Transform configuration
        self.use_albumentations = config.get('use_albumentations', DEFAULT_USE_ALBUMENTATIONS)
        
        logger.info(f"Inference config: use_knn={self.use_knn}, "
                   f"arcface_min={self.arcface_min_score}, "
                   f"centroid_fallback={self.centroid_fallback_score}, "
                   f"topk_centroid={self.default_topk_centroid}, "
                   f"topk_neighbors={self.default_topk_neighbors}, "
                   f"topk_arcface={self.default_topk_arcface}")
        logger.info(f"Reranking config: mode={self.rerank_mode}, "
                   f"arcface_weight={self.arcface_weight}, "
                   f"knn_weight={self.knn_weight}, "
                   f"rrf_k={self.rrf_k}")
        
        # Load model
        self._load_model(config["model"])
        
        # Validate embedding dimensions match
        model_embedding_dim = config["model"]["embedding_dim"]
        if self.dim != model_embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: dataset has {self.dim}, "
                f"but model expects {model_embedding_dim}"
            )
        
        # Infer input size from model or use config/default
        self.input_size = self._get_input_size(config["model"])

        # Setup transforms based on configuration
        self.transform = self._create_transforms()
        
        logger.info(f"Using {'Albumentations' if self.use_albumentations else 'torchvision'} transforms")

        # Create ID to label mapping
        self.id_to_label = {internal_id: self.keys[internal_id]['label'] for internal_id in self.keys}
        
        # Pre-build FAISS indices for better performance (only if kNN is enabled)
        if self.use_knn:
            self._prepare_faiss_indices()
        else:
            logger.info("kNN classifier is disabled - skipping FAISS index creation")
        
        logger.info("EmbeddingClassifier initialized successfully.")

    def _create_transforms(self):
        """Create image transforms based on configuration.
        
        Returns:
            Transform pipeline (Albumentations or torchvision)
        """
        if self.use_albumentations:
            if not ALBUMENTATIONS_AVAILABLE:
                logger.warning("Albumentations requested but not installed. Falling back to torchvision.")
                logger.warning("Install with: pip install albumentations")
                self.use_albumentations = False
            else:
                logger.info("Creating Albumentations transform pipeline")
                return A.Compose([
                    A.Resize(self.input_size, self.input_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
        
        # Default: torchvision transforms
        logger.info("Creating torchvision transform pipeline")
        return transforms.Compose([
            transforms.Resize((self.input_size, self.input_size), Image.Resampling.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    @staticmethod
    def _safe_int_to_str(value) -> str:
        """Safely convert value to string, handling tensors, numpy arrays, UUIDs, etc.
        
        Args:
            value: Any value (tensor, numpy array, int, float, string/UUID, etc.)
            
        Returns:
            String representation of the value
        """
        # Handle torch tensors
        if hasattr(value, 'item'):
            value = value.item()
        # Handle numpy arrays
        elif hasattr(value, 'tolist'):
            value = value.tolist()
        
        # If already a string, return as is
        if isinstance(value, str):
            return value
        
        # Try to convert to int, fallback to str if it fails (e.g., UUIDs)
        try:
            return str(int(value))
        except (ValueError, TypeError):
            return str(value)
    
    def _validate_config(self, config: Dict) -> None:
        """Validate configuration structure and required fields."""
        if not isinstance(config, dict):
            raise TypeError(f"Config must be a dictionary, got {type(config)}")
        
        # Check required keys
        if "dataset" not in config:
            raise ValueError("Config must contain 'dataset' key")
        if "path" not in config["dataset"]:
            raise ValueError("Config['dataset'] must contain 'path' key")
        if "model" not in config:
            raise ValueError("Config must contain 'model' key")
        
        required_model_keys = ["checkpoint_path", "backbone_model_name", "embedding_dim", "num_classes"]
        for key in required_model_keys:
            if key not in config["model"]:
                raise ValueError(f"Config['model'] must contain '{key}' key")
        
        # Validate numeric parameters
        if config["model"]["embedding_dim"] <= 0:
            raise ValueError(f"embedding_dim must be positive, got {config['model']['embedding_dim']}")
        if config["model"]["num_classes"] <= 0:
            raise ValueError(f"num_classes must be positive, got {config['model']['num_classes']}")
        
        # Validate optional thresholds if present
        for param in ["arcface_min_score", "centroid_fallback_score", "centroid_threshold", "neighbor_threshold"]:
            if param in config and (config[param] < 0 or config[param] > 1):
                raise ValueError(f"{param} must be between 0 and 1, got {config[param]}")
        
        logger.info("Configuration validated successfully")

    def _get_input_size(self, model_config: Dict) -> int:
        """Infer input size from model config or backbone."""
        # Check if explicitly provided in config
        if "input_size" in model_config:
            return model_config["input_size"]
        
        # Try to infer from backbone name
        backbone_name = model_config.get("backbone_model_name", "")
        
        # Check for common size patterns in backbone name
        for size in [512, 384, 256, 224]:
            if f"_{size}" in backbone_name or f"{size}" in backbone_name:
                logger.info(f"Inferred input size {size} from backbone name")
                return size
        
        # Try to get from model's default config
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'default_cfg'):
            cfg = self.model.backbone.default_cfg
            if 'input_size' in cfg:
                input_size = cfg['input_size']
                if isinstance(input_size, (tuple, list)) and len(input_size) == 3:
                    size = input_size[1]  # Get height
                    logger.info(f"Using input size {size} from model config")
                    return size
        
        # Default fallback
        logger.info(f"Using default input size {DEFAULT_IMAGE_SIZE}")
        return DEFAULT_IMAGE_SIZE

    def _load_model(self, model_config: Dict):
        """Load model from Lightning checkpoint or regular PyTorch checkpoint."""
        checkpoint_path = model_config["checkpoint_path"]
        
        # Validate checkpoint exists
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        backbone_name = model_config.get("backbone_model_name", "maxvit_base_tf_224")
        embedding_dim = model_config.get("embedding_dim", 512)
        num_classes = model_config.get("num_classes", 639)
        arcface_s = model_config.get("arcface_s", 64.0)
        arcface_m = model_config.get("arcface_m", 0.2)
        pooling_type = model_config.get("pooling_type", "attention")
        
        # Determine model class based on backbone
        is_vit = any(x in backbone_name.lower() for x in SUPPORTED_VIT_BACKBONES)
        
        model_cls = StableEmbeddingModelViT if is_vit else StableEmbeddingModel
        
        # Create model
        if is_vit:
            self.model = model_cls(
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                backbone_model_name=backbone_name,
                arcface_s=arcface_s,
                arcface_m=arcface_m,
                pooling_type=pooling_type,
                pretrained_backbone=False,  # We'll load from checkpoint
            )
        else:
            self.model = model_cls(
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                backbone_model_name=backbone_name,
                arcface_s=arcface_s,
                arcface_m=arcface_m,
                pooling_type=pooling_type,
                pretrained_backbone=False,  # We'll load from checkpoint
            )
        
        # Load checkpoint
        # WARNING: torch.load uses pickle which can execute arbitrary code.
        # Only load checkpoints from trusted sources!
        # TODO: Add checksum verification for production use
        logger.warning(f"Loading checkpoint with weights_only=False (security risk). "
                      f"Only load from trusted sources: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
        except Exception as e:
            logger.warning(f"Failed to load with weights_only=True: {e}. Falling back to weights_only=False")
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Handle Lightning checkpoint format
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present (from Lightning)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        
        # Load state dict with error handling
        try:
            self.model.load_state_dict(state_dict, strict=True)
            logger.info(f"Model loaded successfully from {checkpoint_path}")
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {str(e)[:200]}")
            result = self.model.load_state_dict(state_dict, strict=False)
            if result.missing_keys:
                logger.warning(f"Missing keys in checkpoint: {result.missing_keys[:5]}")
            if result.unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {result.unexpected_keys[:5]}")
            logger.info(f"Model loaded with strict=False from {checkpoint_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model loaded and moved to {self.device}")
        logger.info(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
        
        return self.model

    def _load_data(self, dataset_path: str) -> None:
        """Load embeddings database."""
        # Validate dataset file exists
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        try:
            logger.info(f"Loading dataset from {dataset_path}")
            try:
                data = torch.load(dataset_path, weights_only=True)
            except Exception as e:
                logger.warning(f"Failed to load dataset with weights_only=True: {e}. Using weights_only=False")
                data = torch.load(dataset_path, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {dataset_path}: {e}")
        
        # Validate required keys
        required_keys = ['embeddings', 'labels', 'image_ids', 'annotation_ids', 'drawn_fish_ids', 'labels_keys']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Dataset missing required key: '{key}'")
        
        # Optimize: direct conversion to float32 numpy array
        self.db_embeddings = np.asarray(data['embeddings'], dtype=np.float32)
        
        self.db_labels = np.array(data['labels'])
        self.image_ids = data['image_ids']
        self.annotation_ids = data['annotation_ids']
        self.drawn_fish_ids = data['drawn_fish_ids']
        self.keys = data['labels_keys']
        
        # Validate array lengths match
        n_embeddings = len(self.db_embeddings)
        if not (len(self.db_labels) == len(self.image_ids) == len(self.annotation_ids) == len(self.drawn_fish_ids) == n_embeddings):
            raise ValueError(
                f"Array length mismatch: embeddings={n_embeddings}, labels={len(self.db_labels)}, "
                f"image_ids={len(self.image_ids)}, annotation_ids={len(self.annotation_ids)}, "
                f"drawn_fish_ids={len(self.drawn_fish_ids)}"
            )
        
        self.label_to_species_id = {
            v['label']: v['species_id'] for v in self.keys.values()
        }
        
        # Calculate memory usage
        embeddings_size_mb = self.db_embeddings.nbytes / (1024 * 1024)
        
        logger.info(f"Dataset loaded from {dataset_path}")
        logger.info(f"  Embeddings shape: {self.db_embeddings.shape}")
        logger.info(f"  Embeddings memory: {embeddings_size_mb:.2f} MB")
        logger.info(f"  Unique labels: {len(np.unique(self.db_labels))}")

    def __call__(self, img: Union[np.ndarray, List[np.ndarray]]):
        """
        Perform inference on image(s).
        
        Args:
            img: Single image as np.ndarray or list of images
        
        Returns:
            List of prediction results for each image
        """
        if isinstance(img, np.ndarray):
            return self.inference_numpy(img)
        elif isinstance(img, list) and all(isinstance(i, np.ndarray) for i in img):
            return self.inference_numpy_batch(img)
        else:
            raise TypeError("Input must be np.ndarray or List[np.ndarray].")

    def _preprocess_image(self, img: np.ndarray, img_index: int = 0) -> np.ndarray:
        """Preprocess a single image to RGB uint8 format.
        
        Args:
            img: Input image array
            img_index: Index of image in batch (for error messages)
            
        Returns:
            Preprocessed RGB image as uint8 array
        """
        # Validate input
        if img.ndim not in [2, 3]:
            raise ValueError(f"Image {img_index} must be 2D or 3D array, got shape {img.shape}")
        if img.ndim == 3 and img.shape[2] not in [1, 3, 4]:
            raise ValueError(f"Image {img_index} must have 1, 3, or 4 channels, got {img.shape[2]}")
        
        # Check for empty/invalid images
        if img.size == 0 or min(img.shape[:2]) == 0:
            raise ValueError(f"Image {img_index} has invalid dimensions: {img.shape}")
        
        # Convert grayscale to RGB if needed
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            img = np.stack([img.squeeze()] * 3, axis=-1)
        elif img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]
        
        # Ensure correct dtype and range
        if img.dtype != np.uint8:
            max_val = img.max()
            if max_val == 0:
                logger.warning(f"Image {img_index} is completely black (all zeros)")
                img = np.zeros(img.shape, dtype=np.uint8)
            elif max_val <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        return img
    
    def inference_numpy(self, img: np.ndarray):
        """Inference on a single numpy image."""
        try:
            img = self._preprocess_image(img, img_index=0)
            
            # Apply transforms based on type
            if self.use_albumentations and ALBUMENTATIONS_AVAILABLE:
                # Albumentations expects numpy array in HWC format
                transformed = self.transform(image=img)
                tensor = transformed['image'].unsqueeze(0).to(self.device)
            else:
                # torchvision expects PIL Image
                pil_img = Image.fromarray(img)
                tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            return self._inference_batch_tensor(tensor)[0]
        except Exception as e:
            logger.error(f"Failed to process image: {e}", exc_info=True)
            raise RuntimeError(f"Image processing failed: {e}")

    def inference_numpy_batch(self, imgs: List[np.ndarray]):
        """Inference on a batch of numpy images."""
        if not imgs:
            raise ValueError("Empty image list provided")
        
        if len(imgs) > MAX_BATCH_SIZE:
            logger.info(f"Large batch detected ({len(imgs)} images). "
                       f"Will be processed in chunks of {MAX_BATCH_SIZE}.")
        
        try:
            processed_tensors = []
            for i, img in enumerate(imgs):
                img = self._preprocess_image(img, img_index=i)
                
                # Apply transforms based on type
                if self.use_albumentations and ALBUMENTATIONS_AVAILABLE:
                    # Albumentations expects numpy array
                    transformed = self.transform(image=img)
                    processed_tensors.append(transformed['image'])
                else:
                    # torchvision expects PIL Image
                    pil_img = Image.fromarray(img)
                    processed_tensors.append(self.transform(pil_img))
            
            tensors = torch.stack(processed_tensors).to(self.device)
            return self._inference_batch_tensor(tensors)
        except Exception as e:
            logger.error(f"Failed to process image batch: {e}", exc_info=True)
            raise RuntimeError(f"Batch image processing failed: {e}")

    def _inference_batch_tensor(self, tensors: torch.Tensor):
        """Internal inference on tensor batch."""
        batch_size = tensors.shape[0]
        
        # Validate batch size to prevent OOM
        if batch_size > MAX_BATCH_SIZE:
            logger.warning(f"Batch size {batch_size} exceeds MAX_BATCH_SIZE={MAX_BATCH_SIZE}. "
                         f"Processing in chunks to prevent OOM.")
            # Process in chunks
            all_results = []
            for i in range(0, batch_size, MAX_BATCH_SIZE):
                chunk = tensors[i:i + MAX_BATCH_SIZE]
                chunk_results = self._inference_batch_tensor(chunk)
                all_results.extend(chunk_results)
            return all_results
        
        with torch.no_grad():
            embeddings, archead_logits, _ = self.model(tensors, return_softmax=False)
            
        # Get top-5 ArcFace predictions
        k_arcface = min(5, archead_logits.shape[1])
        top_probabilities, top_indices = torch.topk(archead_logits, k_arcface)
        
        # Store top-5 ArcFace predictions with their scores
        topk_arcface = []
        for i in range(len(top_indices)):
            batch_top5 = []
            for rank in range(k_arcface):
                pred_id = top_indices[i][rank].item()
                pred_score = top_probabilities[i][rank].item()
                batch_top5.append((pred_id, pred_score, rank))
            topk_arcface.append(batch_top5)
        
        # Use kNN search if enabled
        if self.use_knn:
            knn_output = self.get_top_neighbors_from_embeddings(embeddings)
            
            # Log summary instead of full output (only if debug enabled)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Inference: {len(knn_output)} predictions generated (kNN enabled)")
        else:
            # kNN disabled - use empty results
            knn_output = [{} for _ in range(len(top_indices))]
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Inference: kNN disabled, using only ArcFace predictions")

        return self._postprocess_hybrid(knn_output, topk_arcface)

    def _rerank_predictions(
        self, 
        arcface_predictions: List[Tuple[int, float, int]], 
        knn_predictions: Dict,
        mode: str = 'weighted_fusion'
    ) -> List[Tuple[int, float, str]]:
        """
        Rerank predictions using different fusion strategies.
        
        Args:
            arcface_predictions: List of (label_id, score, rank) from ArcFace
            knn_predictions: Dict of {label_id: data} from kNN
            mode: Reranking mode ('weighted_fusion', 'rrf', 'hybrid')
            
        Returns:
            List of (label_id, final_score, source) tuples, sorted by final_score
        """
        combined_scores = {}
        
        if mode == 'weighted_fusion':
            # Weighted Fusion: combine normalized scores with weights
            # ArcFace scores are already softmax probabilities [0, 1]
            for label_id, prob, rank in arcface_predictions:
                combined_scores[label_id] = {
                    'arcface_score': prob,
                    'arcface_rank': rank,
                    'knn_score': 0.0,
                    'knn_rank': None
                }
            
            # Add kNN scores (already normalized similarities [0, 1])
            for idx, (label_id, data) in enumerate(
                sorted(knn_predictions.items(), 
                       key=lambda x: x[1]['similarity'] / x[1]['times'], 
                       reverse=True)
            ):
                knn_score = data['similarity'] / data['times']
                knn_score = max(0.0, min(1.0, knn_score))  # Clamp to [0, 1]
                
                if isinstance(label_id, (int, np.integer)):
                    label_id_int = int(label_id)
                else:
                    # Find corresponding ID for string label
                    label_id_int = None
                    for k, v in self.id_to_label.items():
                        if v == str(label_id):
                            label_id_int = k
                            break
                    if label_id_int is None:
                        continue
                
                if label_id_int not in combined_scores:
                    combined_scores[label_id_int] = {
                        'arcface_score': 0.0,
                        'arcface_rank': None,
                        'knn_score': knn_score,
                        'knn_rank': idx
                    }
                else:
                    combined_scores[label_id_int]['knn_score'] = knn_score
                    combined_scores[label_id_int]['knn_rank'] = idx
            
            # Calculate weighted final scores
            final_scores = []
            for label_id, scores in combined_scores.items():
                final_score = (
                    self.arcface_weight * scores['arcface_score'] +
                    self.knn_weight * scores['knn_score']
                )
                
                # Determine source
                if scores['arcface_rank'] is not None and scores['knn_rank'] is not None:
                    source = 'both'
                elif scores['arcface_rank'] is not None:
                    source = 'arcface'
                else:
                    source = 'knn'
                
                final_scores.append((label_id, final_score, source))
        
        elif mode == 'rrf':
            # Reciprocal Rank Fusion
            for label_id, prob, rank in arcface_predictions:
                rrf_score = 1.0 / (self.rrf_k + rank)
                combined_scores[label_id] = {
                    'rrf_score': rrf_score,
                    'arcface_rank': rank
                }
            
            # Add kNN RRF scores
            for idx, (label_id, data) in enumerate(
                sorted(knn_predictions.items(), 
                       key=lambda x: x[1]['similarity'] / x[1]['times'], 
                       reverse=True)
            ):
                if isinstance(label_id, (int, np.integer)):
                    label_id_int = int(label_id)
                else:
                    label_id_int = None
                    for k, v in self.id_to_label.items():
                        if v == str(label_id):
                            label_id_int = k
                            break
                    if label_id_int is None:
                        continue
                
                knn_rrf = 1.0 / (self.rrf_k + idx)
                
                if label_id_int not in combined_scores:
                    combined_scores[label_id_int] = {
                        'rrf_score': knn_rrf,
                        'knn_rank': idx
                    }
                else:
                    combined_scores[label_id_int]['rrf_score'] += knn_rrf
            
            final_scores = [
                (label_id, scores['rrf_score'], 
                 'both' if 'arcface_rank' in scores and 'knn_rank' in scores else 
                 'arcface' if 'arcface_rank' in scores else 'knn')
                for label_id, scores in combined_scores.items()
            ]
        
        else:  # 'hybrid' - original behavior
            # Top-5 ArcFace first, then top-5 unique kNN
            return None  # Will be handled separately
        
        # Sort by final score (descending)
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores
    
    def _postprocess_hybrid(self, knn_results, topk_arcface) -> List[PredictionResult]:
        """Combine top-5 ArcFace and top-5 unique kNN predictions.
        
        Args:
            knn_results: kNN prediction results (list of dicts)
            topk_arcface: List of lists with (label_id, score, rank) tuples for top-5 ArcFace
            
        Returns:
            List of PredictionResult objects:
            - Positions 1-5: Top-5 ArcFace predictions (with softmax probabilities)
            - Positions 6-10: Top-5 unique kNN predictions (not in ArcFace top-5)
        """
        results = []
        
        for batch_idx in range(len(knn_results)):
            arcface_top5 = topk_arcface[batch_idx]
            knn_dict = knn_results[batch_idx]
            
            # Step 1: Apply softmax to ArcFace logits to get probabilities
            arcface_scores = torch.tensor([score for _, score, _ in arcface_top5])
            arcface_probs = F.softmax(arcface_scores, dim=0).cpu().numpy()
            
            # Update arcface_top5 with probabilities
            arcface_top5_with_probs = [
                (label_id, float(arcface_probs[idx]), rank)
                for idx, (label_id, score, rank) in enumerate(arcface_top5)
            ]
            
            # Step 2: Rerank predictions based on mode
            if self.rerank_mode in ['weighted_fusion', 'rrf']:
                reranked = self._rerank_predictions(
                    arcface_top5_with_probs, 
                    knn_dict, 
                    mode=self.rerank_mode
                )
                
                # Convert reranked results to PredictionResult objects
                final_predictions = []
                for label_id, final_score, source in reranked[:10]:  # Top-10
                    label = self.id_to_label.get(label_id, str(label_id))
                    species_id = self.label_to_species_id.get(label, -1)
                    
                    # Get additional info from kNN if available
                    image_id = None
                    annotation_id = None
                    drawn_fish_id = None
                    
                    if label_id in [int(k) if isinstance(k, (int, np.integer)) else None 
                                    for k in knn_dict.keys()]:
                        for k, data in knn_dict.items():
                            k_int = int(k) if isinstance(k, (int, np.integer)) else None
                            if k_int == label_id and data.get('index') is not None:
                                idx = data['index']
                                try:
                                    if 0 <= idx < len(self.image_ids):
                                        # Convert to string, handling tensors/numpy
                                        image_id = self._safe_int_to_str(self.image_ids[idx])
                                        annotation_id = self._safe_int_to_str(self.annotation_ids[idx])
                                        drawn_fish_id = self._safe_int_to_str(self.drawn_fish_ids[idx])
                                except (IndexError, KeyError):
                                    pass
                                break
                    
                    final_predictions.append(PredictionResult(
                        name=label,
                        species_id=species_id,
                        distance=final_score,
                        accuracy=final_score,
                        image_id=image_id,
                        annotation_id=annotation_id,
                        drawn_fish_id=drawn_fish_id,
                    ))
                
                results.append(final_predictions)
                continue
            
            # Step 3: Hybrid mode - original behavior (top-5 ArcFace + top-5 unique kNN)
            arcface_predictions = []
            arcface_label_ids = set()
            
            for idx, (label_id, score, rank) in enumerate(arcface_top5):
                label = self.id_to_label.get(label_id, str(label_id))
                arcface_label_ids.add(label_id)
                
                species_id = self.label_to_species_id.get(label)
                if species_id is None:
                    species_id = -1
                
                probability = float(arcface_probs[idx])  # Softmax probability [0, 1]
                
                arcface_predictions.append(PredictionResult(
                    name=label,
                    species_id=species_id,
                    distance=score,              # Keep raw logit for reference
                    accuracy=probability,        # Use softmax probability
                    image_id=None,
                    annotation_id=None,
                    drawn_fish_id=None,
                ))
            
            # Step 3: Create kNN predictions (exclude those already in ArcFace top-5)
            knn_predictions = []
            
            for label_id, data in knn_dict.items():
                # Handle label conversion
                if isinstance(label_id, (int, np.integer)):
                    label = self.id_to_label.get(int(label_id), str(label_id))
                    label_id_int = int(label_id)
                else:
                    # Already a string label name
                    label = str(label_id)
                    # Try to find corresponding ID
                    label_id_int = None
                    for k, v in self.id_to_label.items():
                        if v == label:
                            label_id_int = k
                            break
                
                # Skip if this label is already in ArcFace top-5
                if label_id_int in arcface_label_ids:
                    continue
                
                index = data.get("index")
                
                # Safely access arrays with bounds checking
                image_id = None
                annotation_id = None
                drawn_fish_id = None
                
                if index is not None:
                    try:
                        if 0 <= index < len(self.image_ids):
                            # Convert to string, handling tensors/numpy
                            image_id = self._safe_int_to_str(self.image_ids[index])
                            annotation_id = self._safe_int_to_str(self.annotation_ids[index])
                            drawn_fish_id = self._safe_int_to_str(self.drawn_fish_ids[index])
                    except (IndexError, KeyError) as e:
                        logger.warning(f"Error accessing index {index}: {e}")
                
                species_id = self.label_to_species_id.get(label)
                if species_id is None:
                    species_id = -1
                
                # Calculate average similarity score (already normalized in [0, 1] from cosine similarity)
                avg_similarity = data['similarity'] / data['times']
                # Clamp to [0, 1] for safety
                avg_similarity = max(0.0, min(1.0, avg_similarity))
                
                knn_predictions.append(PredictionResult(
                    name=label,
                    species_id=species_id,
                    distance=data['similarity'],
                    accuracy=avg_similarity,  # Normalized similarity score
                    image_id=image_id,
                    annotation_id=annotation_id,
                    drawn_fish_id=drawn_fish_id,
                ))
            
            # Step 4: Sort kNN predictions by average similarity (descending) and take top-5
            knn_predictions.sort(key=lambda x: x.accuracy, reverse=True)
            top5_knn = knn_predictions[:5]
            
            # Step 5: Combine: ArcFace top-5 first, then unique kNN top-5
            final_predictions = arcface_predictions + top5_knn
            
            results.append(final_predictions)
        
        return results
    
    def _postprocess(self, class_results, top1_arcface) -> List[PredictionResult]:
        """Convert raw results to PredictionResult objects with custom sorting.
        
        Args:
            class_results: Raw prediction results
            top1_arcface: List of (label_id, score) tuples for top-1 ArcFace predictions
            
        Returns:
            List of sorted PredictionResult objects
        """
        results = []
        for batch_idx, single_fish in enumerate(class_results):
            fish_results = []
            top1_result = None
            top1_label_id = top1_arcface[batch_idx][0]
            
            for label_id, data in single_fish.items():
                # Handle label conversion - label_id can be int or string
                if isinstance(label_id, (int, np.integer)):
                    label = self.id_to_label.get(int(label_id), str(label_id))
                    label_id_int = int(label_id)
                else:
                    # Already a string label name
                    label = str(label_id)
                    # Try to find corresponding ID for comparison
                    label_id_int = None
                    for k, v in self.id_to_label.items():
                        if v == label:
                            label_id_int = k
                            break
                
                index = data["index"]
                
                # Safely access arrays with bounds checking
                image_id = None
                annotation_id = None
                drawn_fish_id = None
                
                if index is not None:
                    try:
                        if 0 <= index < len(self.image_ids):
                            # Convert to string, handling tensors/numpy
                            image_id = self._safe_int_to_str(self.image_ids[index])
                            annotation_id = self._safe_int_to_str(self.annotation_ids[index])
                            drawn_fish_id = self._safe_int_to_str(self.drawn_fish_ids[index])
                        else:
                            logger.warning(f"Index {index} out of bounds for arrays of length {len(self.image_ids)}")
                    except (IndexError, KeyError) as e:
                        logger.warning(f"Error accessing index {index}: {e}")
                
                species_id = self.label_to_species_id.get(label)
                if species_id is None:
                    logger.warning(f"Unknown label '{label}' not found in label_to_species_id mapping")
                    species_id = -1  # Fallback for backward compatibility
                
                # Calculate average similarity score
                avg_similarity = data['similarity'] / data['times']
                
                result = PredictionResult(
                    name=label,
                    species_id=species_id,
                    distance=data['similarity'],
                    accuracy=avg_similarity,  # Average similarity score
                    image_id=image_id,
                    annotation_id=annotation_id,
                    drawn_fish_id=drawn_fish_id,
                )
                
                # Check if this is the top-1 ArcFace prediction
                is_arcface_top1 = (
                    (label_id_int is not None and label_id_int == top1_label_id) or
                    (data.get('source') == 'arcface' and data.get('arcface_rank') == 0)
                )
                
                if is_arcface_top1:
                    top1_result = result
                else:
                    fish_results.append(result)
            
            # Sort remaining results by average similarity (descending)
            fish_results.sort(key=lambda x: x.accuracy, reverse=True)
            
            # Place top-1 ArcFace prediction first, then kNN results
            if top1_result is not None:
                final_results = [top1_result] + fish_results
            else:
                final_results = fish_results
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"Top-1 ArcFace prediction not found in results for batch {batch_idx}")
            
            results.append(final_results)
        return results

    def _prepare_centroids(self) -> None:
        """Compute class centroids for efficient filtering."""
        unique_labels = np.unique(self.db_labels)
        self.label_to_centroid = {}
        skipped_labels = []
        
        for label in unique_labels:
            class_embs = self.db_embeddings[self.db_labels == label]
            if len(class_embs) == 0:
                logger.warning(f"Label {label} has no embeddings, skipping")
                skipped_labels.append(label)
                continue
                
            centroid = np.mean(class_embs, axis=0)
            norm = np.linalg.norm(centroid)
            
            if norm < NUMERICAL_EPSILON:
                logger.warning(f"Label {label} has zero-norm centroid, using unnormalized")
                self.label_to_centroid[label] = centroid
            else:
                self.label_to_centroid[label] = centroid / norm

        self.centroid_matrix = np.stack([self.label_to_centroid[label] for label in self.label_to_centroid])
        self.centroid_labels = list(self.label_to_centroid.keys())
        
        if skipped_labels:
            logger.warning(f"Skipped {len(skipped_labels)} labels with no embeddings")
        logger.info(f"Prepared {len(self.centroid_labels)} class centroids")
    
    def _prepare_faiss_indices(self) -> None:
        """Pre-build FAISS indices for each class for faster search."""
        logger.info("Building FAISS indices for each class...")
        self.class_indices = {}
        unique_labels = np.unique(self.db_labels)
        
        for label in unique_labels:
            # Use np.where directly to get indices (more memory efficient)
            global_indices = np.where(self.db_labels == label)[0]
            class_embs = self.db_embeddings[global_indices]
            
            if len(class_embs) > 0:
                # Create FAISS index for this class
                index = faiss.IndexFlatIP(self.dim)
                index.add(class_embs)
                
                self.class_indices[label] = {
                    'index': index,
                    'global_indices': global_indices,
                    'size': len(class_embs)
                }
        
        logger.info(f"Built FAISS indices for {len(self.class_indices)} classes")

    def get_top_neighbors_from_embeddings(
        self,
        query_embeddings: Union[np.ndarray, torch.Tensor],
        topk_centroid: Optional[int] = None,
        topk_neighbors: Optional[int] = None,
        centroid_threshold: Optional[float] = None,
        neighbor_threshold: Optional[float] = None
    ) -> List[Dict[str, Dict[str, Union[float, int, None]]]]:
        """
        Find top neighbors using centroid filtering + FAISS search.
        
        Args:
            query_embeddings: Query embeddings [B, D]
            topk_centroid: Number of top centroids to consider (None = use default)
            topk_neighbors: Number of neighbors to retrieve (None = use default)
            centroid_threshold: Minimum similarity to centroid (None = use default)
            neighbor_threshold: Minimum similarity to neighbor (None = use default)
        
        Returns:
            List of dictionaries mapping labels to similarity scores
        """
        # Use default values if not specified
        topk_centroid = self.default_topk_centroid if topk_centroid is None else topk_centroid
        topk_neighbors = self.default_topk_neighbors if topk_neighbors is None else topk_neighbors
        centroid_threshold = self.default_centroid_threshold if centroid_threshold is None else centroid_threshold
        neighbor_threshold = self.default_neighbor_threshold if neighbor_threshold is None else neighbor_threshold
        
        # Validate parameters
        if topk_centroid <= 0:
            raise ValueError(f"topk_centroid must be positive, got {topk_centroid}")
        if topk_neighbors <= 0:
            raise ValueError(f"topk_neighbors must be positive, got {topk_neighbors}")
        if not 0 <= centroid_threshold <= 1:
            raise ValueError(f"centroid_threshold must be in [0, 1], got {centroid_threshold}")
        if not 0 <= neighbor_threshold <= 1:
            raise ValueError(f"neighbor_threshold must be in [0, 1], got {neighbor_threshold}")
        
        start_time = time.time()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Starting search over {len(query_embeddings)} embeddings")

        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.cpu().numpy().astype("float32")

        # Timing breakdown
        timing = {'centroid': 0, 'faiss': 0, 'aggregation': 0}
        
        # Step 1: Vectorized centroid similarity computation for all queries
        t0 = time.time()
        # Embeddings are already L2-normalized, use matrix multiplication for cosine similarity
        all_centroid_sims = np.dot(query_embeddings, self.centroid_matrix.T)  # [B, num_centroids]
        timing['centroid'] = time.time() - t0
        
        results = []
        for query_idx, query_emb in enumerate(query_embeddings):
            centroid_sims = all_centroid_sims[query_idx]
            top_centroid_indices = np.argsort(-centroid_sims)[:topk_centroid]

            centroid_scores = {
                self.centroid_labels[idx]: centroid_sims[idx]
                for idx in top_centroid_indices if centroid_sims[idx] >= centroid_threshold
            }
            selected_classes = set(centroid_scores.keys())

            if not selected_classes:
                if logger.isEnabledFor(logging.DEBUG):
                    max_sim = centroid_sims[top_centroid_indices[0]] if len(top_centroid_indices) > 0 else 0
                    logger.debug(f"Query {query_idx}: No classes passed centroid threshold "
                               f"(max similarity: {max_sim:.3f}, threshold: {centroid_threshold})")
                results.append({})
                continue

            # Step 2: FAISS search using pre-built indices
            t0 = time.time()
            score_map = defaultdict(lambda: {'index': None, 'similarity': 0.0, 'times': 0, 'source': 'knn'})
            
            for label in selected_classes:
                if label not in self.class_indices:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Label {label} not found in class_indices, skipping")
                    continue
                
                class_data = self.class_indices[label]
                class_index = class_data['index']
                global_indices = class_data['global_indices']
                
                # Search within this class
                k = min(topk_neighbors, class_data['size'])
                distances, indices = class_index.search(query_emb.reshape(1, -1), k)
                
                # Aggregate results for this class
                for rank, idx in enumerate(indices[0]):
                    sim = float(distances[0][rank])
                    if sim >= neighbor_threshold:
                        original_idx = int(global_indices[idx])
                        score_map[label]['similarity'] += sim
                        score_map[label]['times'] += 1
                        score_map[label]['source'] = 'knn'
                        if score_map[label]['index'] is None:
                            score_map[label]['index'] = original_idx
            
            timing['faiss'] += time.time() - t0

            # Step 3: Add centroid-only predictions for classes without neighbors
            t0 = time.time()
            for label, sim in centroid_scores.items():
                if label not in score_map:
                    # Use actual centroid similarity instead of fixed fallback score
                    centroid_sim = max(float(sim), self.centroid_fallback_score)
                    score_map[label] = {
                        'index': None, 
                        'similarity': centroid_sim, 
                        'times': 1,
                        'source': 'knn'
                    }
            timing['aggregation'] += time.time() - t0

            results.append(dict(score_map))

        total_time = time.time() - start_time
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Search completed in {total_time:.3f}s "
                        f"(centroid: {timing['centroid']:.3f}s, "
                        f"faiss: {timing['faiss']:.3f}s, "
                        f"aggregation: {timing['aggregation']:.3f}s)")
        
        # Log performance metrics for production monitoring (only for larger batches)
        if len(query_embeddings) > 5:
            throughput = len(query_embeddings) / total_time if total_time > 0 else 0
            logger.info(f"Batch search: {len(query_embeddings)} queries in {total_time:.3f}s "
                       f"({throughput:.1f} queries/s)")
        
        return results
    
    def get_model_info(self) -> Dict:
        """Return model configuration and statistics.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'embedding_dim': self.dim,
            'num_classes': len(self.keys),
            'num_embeddings': len(self.db_embeddings),
            'device': str(self.device),
            'input_size': self.input_size,
            'num_centroid_classes': len(self.centroid_labels) if self.use_knn else 0,
            'inference_config': {
                'use_knn': self.use_knn,
                'arcface_min_score': self.arcface_min_score,
                'centroid_fallback_score': self.centroid_fallback_score,
                'topk_centroid': self.default_topk_centroid,
                'topk_neighbors': self.default_topk_neighbors,
                'topk_arcface': self.default_topk_arcface,
                'centroid_threshold': self.default_centroid_threshold,
                'neighbor_threshold': self.default_neighbor_threshold,
            }
        }
        
        if hasattr(self, 'model') and hasattr(self.model, 'backbone'):
            info['backbone'] = self.model.backbone.__class__.__name__
        
        return info
    
    def warmup(self, num_iterations: int = DEFAULT_WARMUP_ITERATIONS) -> float:
        """Warmup model with dummy data for stable performance.
        
        Args:
            num_iterations: Number of warmup iterations
            
        Returns:
            Average warmup time per iteration in seconds
        """
        logger.info(f"Warming up model with {num_iterations} iterations...")
        dummy = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
        
        # Warmup iterations
        times = []
        for i in range(num_iterations):
            start = time.time()
            with torch.no_grad():
                self.model(dummy, return_softmax=False)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        logger.info(f"Warmup completed: avg={avg_time*1000:.2f}ms, "
                   f"min={min(times)*1000:.2f}ms, max={max(times)*1000:.2f}ms")
        return avg_time
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False  # Don't suppress exceptions
    
    def cleanup(self) -> None:
        """Release resources and cleanup."""
        logger.info("Cleaning up resources...")
        
        # Clear FAISS indices with error handling (only if kNN was enabled)
        if self.use_knn and hasattr(self, 'class_indices'):
            for label, data in self.class_indices.items():
                try:
                    if 'index' in data and data['index'] is not None:
                        data['index'].reset()
                except Exception as e:
                    logger.warning(f"Failed to reset FAISS index for label {label}: {e}")
            try:
                self.class_indices.clear()
            except Exception as e:
                logger.warning(f"Failed to clear class_indices: {e}")
        
        # Move model to CPU and clear cache
        if hasattr(self, 'model'):
            try:
                self.model.cpu()
            except Exception as e:
                logger.warning(f"Failed to move model to CPU: {e}")
            
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Failed to empty CUDA cache: {e}")
        
        logger.info("Cleanup completed")
    
    def __del__(self):
        """Destructor - logs warning if cleanup wasn't called.
        
        Note: Do not rely on __del__ for cleanup. Always use context manager
        or explicitly call cleanup().
        """
        try:
            # Check if resources are still allocated (only relevant if kNN was enabled)
            if hasattr(self, 'use_knn') and self.use_knn:
                if hasattr(self, 'class_indices') and self.class_indices:
                    logger.warning("EmbeddingClassifier destroyed without cleanup(). "
                                 "Use context manager or call cleanup() explicitly.")
        except Exception:
            # Silently ignore errors in destructor during interpreter shutdown
            pass
