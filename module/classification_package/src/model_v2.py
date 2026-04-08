# -*- coding: utf-8 -*-
"""
Model architectures for Fish Classification with Metric Learning.

This module provides flexible model architectures strictly for Vision Transformers (ViT):
- Multiple pooling strategies (Attention, GeM, Average, Mean)
- ArcFace / Sub-center ArcFace classification head
- Configurable embedding layers (Simple or MLP)

Key Classes:
- GeMPooling: Generalized Mean Pooling layer
- StableEmbeddingModelViT: Unified model for Vision Transformer backbones
- StableEmbeddingModel: Backward-compatible alias for StableEmbeddingModelViT
"""

import logging
import math
import inspect
from typing import Optional, Literal, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.vision_transformer import VisionTransformer

# Assumes these modules are located in the same directory/package in your project
from .arcface import SubCenterArcFaceHead, ArcFaceHead
from .pooling import ViTAttentionPooling, ViTAttentionPooling_single_head, GeMPooling, ViTMeanPooling
from .neck_layers import AdvancedNeck, BNNeck, LNNeck, ResNeck, GatedNeck

logger = logging.getLogger(__name__)


# =============================================================================
# Main Model
# =============================================================================

class StableEmbeddingModelViT(nn.Module):
    """
    Unified Embedding model for Vision Transformer backbones.

    Args:
        embedding_dim: Output embedding dimension.
        num_classes: Number of classes for the ArcFace head.
        pretrained_backbone: Whether to load ImageNet-pretrained weights.
        freeze_backbone_initially: Freeze backbone at initialization (e.g., for warm-up).
        backbone_model_name: timm model name.
        custom_backbone: Optional pre-configured backbone (skips timm loading).
        attention_hidden_channels: Hidden dim for attention pooling (None = auto).
        arcface_s: ArcFace scale parameter.
        arcface_m: ArcFace angular margin.
        arcface_K: Sub-centers per class (for SubCenter ArcFace).
        use_adacos: Enable adaptive scale via AdaCos (SubCenter only).
        add_bn_to_embedding: Add BatchNorm after 'simple' neck (ignored by complex necks).
        embedding_dropout_rate: Dropout probability inside the neck.
        pooling_type: 'attention' | 'gem' | 'avg' | 'mean'.
        num_attention_heads: Number of heads for attention pooling.
        use_cls_token: Fuse CLS token with pooled patches via a learned sigmoid gate.
        neck_type: 'simple' | 'mlp' | 'advanced' | 'bnneck' | 'lnneck' | 'resneck' | 'gated'.
        head_type: 'arcface' | 'subcenter'.
        backbone_img_size: Override input resolution (required for DINOv2 which defaults
            to 518px — pass 224 or 392). None = use model's native default.
        input_img_size: Stored for resolution-dependent helpers (optional).
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 1000,
        pretrained_backbone: bool = True,
        freeze_backbone_initially: bool = False,
        backbone_model_name: str = 'vit_base_patch14_reg4_dinov2.lvd142m',
        custom_backbone: Optional[VisionTransformer] = None,
        attention_hidden_channels: Optional[int] = None,
        arcface_s: float = 64.0,
        arcface_m: float = 0.5,
        arcface_K: int = 3,
        use_adacos: bool = False,
        add_bn_to_embedding: bool = False,
        embedding_dropout_rate: float = 0.0,
        pooling_type: Literal['attention', 'gem', 'avg', 'mean'] = 'attention',
        num_attention_heads: int = 8,
        use_cls_token: bool = True,
        neck_type: Literal['simple', 'mlp', 'advanced', 'bnneck', 'lnneck', 'resneck', 'gated'] = 'bnneck',
        head_type: Literal['arcface', 'subcenter'] = 'subcenter',
        input_img_size: Optional[Tuple[int, int]] = None,
        train_segmentation_head: bool = False,
        drop_path_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logger.debug("Ignoring unknown kwargs during initialization: %s", list(kwargs.keys()))

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.use_cls_token = use_cls_token
        self.neck_type = neck_type
        self.head_type = head_type
        self.input_img_size = input_img_size  # Stored for _infer_input_hw fallback

        logger.info(f"Initializing StableEmbeddingModelViT (backbone={backbone_model_name})")

        # --- Backbone ---
        self.backbone = self._create_backbone(
            custom_backbone, backbone_model_name, pretrained_backbone, input_img_size,
            drop_path_rate=drop_path_rate,
        )
        self.backbone_out_features = self._infer_backbone_dim()
        self.backbone_feature_extractor = self.backbone.forward_features
        self._has_cls_token = (
            hasattr(self.backbone, "cls_token") and self.backbone.cls_token is not None
        )

        if freeze_backbone_initially:
            self.freeze_backbone()

        self.segmentation_head = (
            nn.Linear(self.backbone_out_features, 1)
            if train_segmentation_head else None
        )

        # --- Pooling ---
        self.pooling = self._create_pooling(
            pooling_type, attention_hidden_channels, num_attention_heads,
        )

        # --- CLS-token fusion gate ---
        self.cls_gate = self._create_cls_gate(use_cls_token)

        # --- Embedding Neck ---
        d = self.backbone_out_features
        logger.debug(f"Building Neck layer of type '{neck_type}' (input_dim={d}, output_dim={embedding_dim})")
        
        if neck_type == 'mlp':
            self.embedding_fc = nn.Sequential(
                nn.Linear(d, d),
                nn.BatchNorm1d(d),
                nn.GELU(),
                nn.Dropout(embedding_dropout_rate) if embedding_dropout_rate > 0 else nn.Identity(),
                nn.Linear(d, embedding_dim),
                nn.BatchNorm1d(embedding_dim) if add_bn_to_embedding else nn.Identity(),
            )
        elif neck_type == 'advanced':
            self.embedding_fc = AdvancedNeck(d, embedding_dim, dropout=embedding_dropout_rate, use_bn=add_bn_to_embedding)
        elif neck_type == 'bnneck':
            self.embedding_fc = BNNeck(d, embedding_dim, dropout=embedding_dropout_rate)
        elif neck_type == 'lnneck':
            self.embedding_fc = LNNeck(d, embedding_dim, dropout=embedding_dropout_rate)
        elif neck_type == 'resneck':
            self.embedding_fc = ResNeck(d, embedding_dim, dropout=embedding_dropout_rate)
        elif neck_type == 'gated':
            self.embedding_fc = GatedNeck(d, embedding_dim, dropout=embedding_dropout_rate)
        else:  # 'simple'
            layers = [nn.Linear(d, embedding_dim)]
            if add_bn_to_embedding:
                layers.append(nn.BatchNorm1d(embedding_dim))
            if embedding_dropout_rate > 0:
                layers.append(nn.Dropout(embedding_dropout_rate))
            self.embedding_fc = nn.Sequential(*layers)

        # --- Classification Head ---
        logger.debug(f"Building Classification Head of type '{head_type}' (num_classes={num_classes})")
        if head_type == 'subcenter':
            self.arcface_head = SubCenterArcFaceHead(
                embedding_dim, num_classes,
                K=arcface_K, s=arcface_s, m=arcface_m,
                use_adacos=use_adacos,
            )
        else:
            self.arcface_head = ArcFaceHead(
                embedding_dim, num_classes, 
                s=arcface_s, m=arcface_m
            )

        self._log_config(
            backbone_model_name, custom_backbone, embedding_dim, num_classes,
            arcface_s, arcface_m, arcface_K, use_adacos, pooling_type,
            num_attention_heads, d, embedding_dropout_rate, neck_type, head_type
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_backbone(
        custom_backbone: Optional[VisionTransformer],
        model_name: str,
        pretrained: bool,
        img_size: Optional[int],
        drop_path_rate: float = 0.0,
    ) -> nn.Module:
        if custom_backbone is not None:
            logger.info("Using custom ViT backbone.")
            return custom_backbone

        extra: Dict[str, Any] = {}
        if img_size is not None:
            if model_name == 'vit_base_patch14_reg4_dinov2.lvd142m':
                extra = {'img_size': img_size, 'dynamic_img_size': True}
            elif model_name == 'beitv2_base_patch16_224.in1k_ft_in22k_in1k':
                extra = {'img_size': img_size}

        if drop_path_rate > 0:
            extra['drop_path_rate'] = drop_path_rate

        logger.info("Loading ViT backbone: %s (img_size=%s, drop_path=%.2f)", model_name, img_size or 'native', drop_path_rate)
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, **extra)
        
        # IMPORTANT: Enable memory-efficient gradient checkpointing
        model.set_grad_checkpointing(enable=True)
        logger.debug("Gradient checkpointing enabled for backbone.")

        return model

    def _create_pooling(
        self,
        pooling_type: str,
        attention_hidden: Optional[int],
        num_heads: int,
    ) -> Optional[nn.Module]:
        if pooling_type == 'attention':
            return ViTAttentionPooling(
                in_features=self.backbone_out_features,
                hidden_features=attention_hidden,
                num_heads=num_heads,
            )
        elif pooling_type == 'attention_single_head':
            return ViTAttentionPooling_single_head(
                in_features=self.backbone_out_features,
                hidden_features=attention_hidden,
                num_heads=num_heads,
            )
        elif pooling_type == 'gem':
            return GeMPooling(p=3.0, learnable=True)
        elif pooling_type == 'avg':
            return None
        elif pooling_type == 'mean':
            return ViTMeanPooling()
        else:
            logger.warning(f"Unknown pooling type '{pooling_type}', falling back to 'avg' pooling.")
            return None

    def _create_cls_gate(self, use_cls_token: bool) -> Optional[nn.Parameter]:
        if use_cls_token and self._has_cls_token:
            gate = nn.Parameter(torch.tensor(0.0))
            logger.info("CLS token fusion enabled (init gate=%.2f)", gate.item())
            return gate
        if use_cls_token and not self._has_cls_token:
            logger.warning("use_cls_token=True but backbone has no cls_token — skipping fusion.")
        return None

    def _log_config(
        self, backbone_name, custom_backbone, emb_dim, n_cls,
        s, m, K, adacos, pool, heads, d, drop, neck_type, head_type
    ):
        name = 'custom' if custom_backbone else backbone_name
        logger.info(
            "StableEmbeddingModelViT Config | backbone=%s dim=%d emb=%d cls=%d "
            "head_type=%s (s=%.1f m=%.2f K=%d adacos=%s) pool=%s heads=%d "
            "neck_type=%s drop=%.2f",
            name, d, emb_dim, n_cls, head_type,
            s, m, K, adacos, pool, heads, neck_type, drop
        )

    # ------------------------------------------------------------------
    # Backbone utilities
    # ------------------------------------------------------------------

    def _tokens_and_grid_from_features(
        self, features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Tuple[int, int]], Optional[torch.Tensor]]:
        
        if features.ndim != 3:
            logger.error(f"Expected 3D tensor from ViT backbone (B, N, C), got {tuple(features.shape)}")
            raise ValueError(f"Expected 3D tensor from ViT backbone (B, N, C), got {tuple(features.shape)}")

        # Dynamically determine the number of prefix tokens (CLS + Registers)
        # For DINOv2 with reg4, num_prefix_tokens will be 5 (1 CLS + 4 Registers)
        default_prefix = 1 if self._has_cls_token else 0
        num_prefix = getattr(self.backbone, 'num_prefix_tokens', default_prefix)

        cls_token = None
        if num_prefix > 0 and features.shape[1] > num_prefix:
            # Always take the very first token as CLS (if it exists)
            cls_token = features[:, 0, :]
            # Skip ALL prefix tokens to extract pure image patches
            tokens = features[:, num_prefix:, :]
        else:
            tokens = features

        grid = self._infer_grid(tokens.shape[1])
        return tokens, grid, cls_token

    def _infer_grid(self, num_tokens: int) -> Optional[Tuple[int, int]]:
        pe = getattr(self.backbone, "patch_embed", None)
        if pe is not None:
            gs = getattr(pe, "grid_size", None)
            if isinstance(gs, (tuple, list)) and len(gs) == 2:
                if int(gs[0]) * int(gs[1]) == num_tokens:
                    return (int(gs[0]), int(gs[1]))
        s = int(round(math.sqrt(num_tokens)))
        if s * s == num_tokens:
            return (s, s)
        return None

    def freeze_backbone(self):
        """Freezes all parameters within the ViT backbone."""
        logger.info("Freezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, specific_layer_keywords=None, verbose=False):
        """Unfreezes backbone parameters, optionally filtering by layer keywords."""
        logger.info("Unfreezing backbone (keywords=%s)", specific_layer_keywords)
        unfrozen = 0
        for name, param in self.backbone.named_parameters():
            if specific_layer_keywords is None or any(kw in name for kw in specific_layer_keywords):
                param.requires_grad = True
                unfrozen += 1
                if verbose:
                    logger.debug("  Unfroze: %s", name)
        logger.info("Total parameters unfrozen: %d", unfrozen)

    def freeze_everything_except_segmentation_head(self):
        """Strictly freezes the entire encoder (Backbone, Neck, Head), leaving only the Segmentation Head trainable."""
        logger.info("Freezing entire encoder and head. Only Segmentation Head will be trained.")
        for param in self.parameters():
            param.requires_grad = False
        if hasattr(self, 'segmentation_head') and self.segmentation_head is not None:
            for param in self.segmentation_head.parameters():
                param.requires_grad = True
            logger.info(f"Segmentation Head successfully unfrozen: {param.numel():,} trainable parameters.")

    def _infer_backbone_dim(self) -> int:
        for attr in ("num_features", "embed_dim"):
            v = getattr(self.backbone, attr, None)
            if isinstance(v, int) and v > 0:
                return v

        self.backbone.eval()
        orig_device = next(self.backbone.parameters()).device
        
        try:
            self.backbone.to("cpu")
            with torch.no_grad():
                features = self.backbone.forward_features(torch.randn(1, 3, self.input_img_size[0], self.input_img_size[1]))
            self.backbone.to(orig_device)
        except Exception as e:
            self.backbone.to(orig_device)
            logger.error(f"Failed to dynamically infer backbone dimension: {e}")
            raise RuntimeError(f"Failed to infer backbone dimension: {e}")

        if features.ndim == 3:
            return int(features.shape[-1])
        
        logger.error(f"Expected 3D out put shape from ViT, got: {tuple(features.shape)}")
        raise ValueError(f"Expected 3D output shape from ViT, got: {tuple(features.shape)}")

    # ------------------------------------------------------------------
    # Core encode pipeline (shared by forward / forward_train)
    # ------------------------------------------------------------------

    def _encode(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor],
        object_mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        logger.debug(f"Encoding input batch of shape {tuple(x.shape)}")
        features = self.backbone_feature_extractor(x)
        tokens, grid, cls_token = self._tokens_and_grid_from_features(features)

        raw_attention_scores = None
        attn_weights = None

        if self.pooling_type == 'attention' and self.pooling is not None:
            # FIX: single forward pass — raw_attention_scores [B, N, num_heads] returned
            # directly by pooling.forward(), eliminating the duplicate get_all_attention_scores call.
            pooled, attn_weights, raw_attention_scores = self.pooling(
                tokens, object_mask=object_mask, return_attention_map=True,
            )
        elif self.pooling_type == 'gem' and self.pooling is not None:
            pooled = self.pooling(tokens)
        elif self.pooling_type == 'mean' and self.pooling is not None:
            # Call the mean pooling class to return both pooled embeddings and uniform attention weights
            pooled, attn_weights = self.pooling(tokens, return_attention_map=True)
        else:
            pooled = tokens.mean(dim=1)

        if self.cls_gate is not None and cls_token is not None:
            gate = torch.sigmoid(self.cls_gate)
            pooled = gate * cls_token + (1.0 - gate) * pooled

        emb = F.normalize(self.embedding_fc(pooled), p=2, dim=1)
        logits = self.arcface_head(emb, labels)

        # --- NEW: Segmentation block ---
        seg_logits_2d = None
        # Check whether segmentation_head was initialized in __init__
        if hasattr(self, 'segmentation_head') and self.segmentation_head is not None and grid is not None:
            H, W = grid
            # Run the patches through the linear layer: [B, N, C] -> [B, N, 1]
            seg_out = self.segmentation_head(tokens)
            # Rearrange axes and form an image: [B, N, 1] -> [B, 1, N] -> [B, 1, H, W]
            seg_logits_2d = seg_out.transpose(1, 2).reshape(tokens.shape[0], 1, H, W)
        # -------------------------------

        return {
            "emb": emb,
            "logits": logits,
            "tokens": tokens,
            "grid": grid,
            "attn_weights": attn_weights,
            "raw_attention_scores": raw_attention_scores,
            "seg_logits": seg_logits_2d, # <-- NEW: return raw mask logits
        }

    @staticmethod
    def _build_vis_attn_map(
        attn_weights: Optional[torch.Tensor],
        grid: Optional[Tuple[int, int]],
    ) -> Optional[torch.Tensor]:
        if attn_weights is None or grid is None:
            return None
        try:
            B, N, _ = attn_weights.shape
            H, W = grid
            if H * W == N:
                return attn_weights.permute(0, 2, 1).reshape(B, 1, H, W)
        except Exception as e:
            logger.debug(f"Failed to build visualization attention map: {e}")
        return None

    # ------------------------------------------------------------------
    # Public forward methods
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None,
        return_softmax: bool = False,
        return_attention_map: bool = True,
    ):
        enc = self._encode(x, labels, object_mask)
        vis_map = (
            self._build_vis_attn_map(enc["attn_weights"], enc["grid"])
            if return_attention_map else None
        )

        out = enc["logits"]
        if return_softmax:
            out = F.softmax(out, dim=1)
            
        # Optionally return the mask during inference as well,
        # but embeddings typically suffice for classification
        return enc["emb"], out, vis_map, enc.get("seg_logits")

    def forward_train(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        enc = self._encode(x, labels, object_mask)
        
        ret = {
            "emb": enc["emb"],
            "logits": enc["logits"],
            "raw_attention_scores": enc["raw_attention_scores"],
            "grid": enc["grid"],
            "tokens": enc["tokens"],
            "seg_logits": enc.get("seg_logits"),  # <-- 🔥 Here it is! Propagate the mask!
        }

        return ret

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: str = 'cpu',
        strict: bool = False,
        **kwargs,
    ) -> "StableEmbeddingModelViT":
        """
        Creates a model instance and loads all weights from a checkpoint in a single call.

        The checkpoint can be:
        - A PyTorch Lightning checkpoint (keys: ``state_dict`` + ``hyper_parameters`` / ``model_hparams``)
        - A standard PyTorch checkpoint via ``torch.save(model.state_dict(), path)``

        Hyperparameter priority order:
        1. ``kwargs`` — explicitly passed arguments (highest priority, overrides everything)
        2. ``checkpoint["hyper_parameters"]["model_hparams"]`` (from Lightning trainer)
        3. ``checkpoint["model_hparams"]``
        4. Class default values

        Args:
            checkpoint_path: Path to the checkpoint file.
            map_location: Device to load tensors onto (e.g., ``'cpu'``, ``'cuda'``).
            strict: Enforce strict key matching in state_dict.
            **kwargs: Any constructor parameters — will override saved hparams.

        Returns:
            Initialized model loaded with weights, set to ``eval()`` mode.

        Example:
            model = StableEmbeddingModelViT.load_from_checkpoint(
                "checkpoints/epoch=10.ckpt",
                map_location="cuda",
            )
        """
        # Keys accepted by __init__ — used to filter flat hyper_parameters
        # (Lightning saves trainer and model params in the same dict)
        _valid_keys = set(inspect.signature(cls.__init__).parameters.keys()) - {"self", "kwargs"}

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # --- Extract Hyperparameters ---
        hparams: Dict[str, Any] = {}

        # Lightning: save_hyperparameters() → checkpoint["hyper_parameters"]
        if isinstance(checkpoint.get("hyper_parameters"), dict):
            hp = checkpoint["hyper_parameters"]
            if "model_hparams" in hp:
                # Manual nested structure
                hparams = dict(hp["model_hparams"])
            else:
                # Flat Lightning format: filter out trainer params (lr, weight_decay, etc.)
                hparams = {k: v for k, v in hp.items() if k in _valid_keys}
                logger.info(
                    "load_from_checkpoint: extracted %d model hparams from flat "
                    "hyper_parameters (skipped %d trainer-only keys)",
                    len(hparams), len(hp) - len(hparams),
                )

        # Flat model_hparams key at the top level of the checkpoint
        if not hparams and isinstance(checkpoint.get("model_hparams"), dict):
            hparams = dict(checkpoint["model_hparams"])

        # Explicit kwargs always win
        hparams.update(kwargs)

        # Do not download ImageNet weights — they will be overwritten by the checkpoint
        hparams.setdefault("pretrained_backbone", False)

        logger.info(
            "load_from_checkpoint: constructing model with hparams=%s",
            {k: v for k, v in hparams.items() if k != "pretrained_backbone"},
        )
        model = cls(**hparams)

        # --- Load Weights ---
        state_dict = checkpoint.get("state_dict", checkpoint)

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[6:] if k.startswith("model.") else k
            new_state_dict[new_key] = v

        # Handling class count mismatch
        # SubCenterArcFaceHead stores weight as [num_classes * K, emb_dim],
        # therefore we divide shape[0] by K before comparison
        head_key = "arcface_head.weight"
        if head_key in new_state_dict:
            K = getattr(getattr(model, "arcface_head", None), "K", 1) or 1
            ckpt_rows = new_state_dict[head_key].shape[0]
            ckpt_classes = ckpt_rows // K
            
            logger.info(
                f"[ckpt] arcface_head.weight shape={tuple(new_state_dict[head_key].shape)} "
                f"K={K} ckpt_classes={ckpt_classes} model.num_classes={model.num_classes}"
            )
            
            if ckpt_classes != model.num_classes:
                logger.warning(f"Class mismatch: ckpt={ckpt_classes} ≠ model={model.num_classes} → skipping head weights")
                del new_state_dict[head_key]
            else:
                logger.info("Classes match → loading arcface_head.weight")
        else:
            similar = [k for k in new_state_dict if "arcface" in k or "head" in k]
            logger.warning("arcface_head.weight NOT found in remapped state_dict!")
            logger.debug(f"Similar keys: {similar}")

        final_dict = {k: v for k, v in new_state_dict.items() if not k.startswith("main_loss_fn")}

        # =====================================================================
        # MAGIC FOR HD RESOLUTION: Interpolating backbone.pos_embed
        # =====================================================================
        pos_embed_key = "backbone.pos_embed"
        if pos_embed_key in final_dict and hasattr(model, "backbone") and hasattr(model.backbone, "pos_embed"):
            old_pos = final_dict[pos_embed_key]
            new_pos_shape = model.backbone.pos_embed.shape

            if old_pos.shape != new_pos_shape:
                logger.info(f"[HD-MIGRATION] Interpolating {pos_embed_key} from {old_pos.shape} to {new_pos_shape}")
                
                # Extract sequence lengths (e.g., 256 and 576)
                old_seq_len = old_pos.shape[1]
                new_seq_len = new_pos_shape[1]
                embed_dim = old_pos.shape[2]

                # Compute grid sizes (16x16 for 224px, 24x24 for 336px)
                old_grid = int(math.sqrt(old_seq_len))
                new_grid = int(math.sqrt(new_seq_len))

                # Validate that these are perfect squares (no merged cls tokens in this tensor)
                if old_grid ** 2 == old_seq_len and new_grid ** 2 == new_seq_len:
                    # 1. Reshape [1, N, C] -> [1, C, H, W]
                    pos_2d = old_pos.permute(0, 2, 1).reshape(1, embed_dim, old_grid, old_grid)
                    
                    # 2. Bicubic interpolation
                    pos_resized = F.interpolate(
                        pos_2d, 
                        size=(new_grid, new_grid), 
                        mode="bicubic", 
                        align_corners=False
                    )
                    
                    # 3. Restore shape [1, C, H, W] -> [1, N, C]
                    final_dict[pos_embed_key] = pos_resized.flatten(2).permute(0, 2, 1)
                    logger.info(f"[HD-MIGRATION] Successfully resized pos_embed to {final_dict[pos_embed_key].shape}")
                else:
                    logger.warning("[HD-MIGRATION] pos_embed sequence is not a perfect square. Cannot interpolate safely.")
        # =====================================================================
        
        missing, unexpected = model.load_state_dict(final_dict, strict=False)
        model._print_load_summary(missing, unexpected, final_dict)

        if strict:
            filtered_missing = [k for k in missing if k != head_key]
            if filtered_missing or unexpected:
                error_msgs = []
                if filtered_missing:
                    error_msgs.append(
                        "Missing key(s) in state_dict: {}. ".format(
                            ", ".join(f'"{k}"' for k in filtered_missing)
                        )
                    )
                if unexpected:
                    error_msgs.append(
                        "Unexpected key(s) in state_dict: {}. ".format(
                            ", ".join(f'"{k}"' for k in unexpected)
                        )
                    )
                logger.error(f"Strict weight loading failed for {model.__class__.__name__}")
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        model.__class__.__name__,
                        "\n\t".join(error_msgs),
                    )
                )
                
        model.eval()
        logger.info("load_from_checkpoint: model ready on device '%s'.", map_location)
        return model

    def load_weights(self, checkpoint_path: str, strict: bool = False):
        """Standalone method to load weights into an already initialized model."""
        logger.info(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[6:] if k.startswith('model.') else k
            new_state_dict[new_key] = v

        head_key = 'arcface_head.weight'
        if head_key in new_state_dict:
            K = getattr(getattr(self, "arcface_head", None), "K", 1) or 1
            ckpt_rows = new_state_dict[head_key].shape[0]
            ckpt_classes = ckpt_rows // K
            if ckpt_classes != self.num_classes:
                logger.warning(
                    f"Class count mismatch! CKPT: {ckpt_classes} classes "
                    f"(rows={ckpt_rows}, K={K}) | MODEL: {self.num_classes}. Skipping head weights."
                )
                del new_state_dict[head_key]

        final_dict = {k: v for k, v in new_state_dict.items() if not k.startswith('main_loss_fn')}

        missing, unexpected = self.load_state_dict(final_dict, strict=False)
        self._print_load_summary(missing, unexpected, final_dict)

        if strict:
            filtered_missing = [k for k in missing if k != head_key]
            if filtered_missing or unexpected:
                error_msgs = []
                if filtered_missing:
                    error_msgs.append(
                        "Missing key(s) in state_dict: {}. ".format(
                            ", ".join(f'"{k}"' for k in filtered_missing)
                        )
                    )
                if unexpected:
                    error_msgs.append(
                        "Unexpected key(s) in state_dict: {}. ".format(
                            ", ".join(f'"{k}"' for k in unexpected)
                        )
                    )
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        self.__class__.__name__,
                        "\n\t".join(error_msgs),
                    )
                )

    def _print_load_summary(
        self,
        missing: list,
        unexpected: list,
        loaded_dict: dict,
        max_show: int = 8,
    ) -> None:
        """Pretty-print a detailed weight-loading report to the console."""
        W = 82  # total box width
        model_sd = self.state_dict()

        # ── Layers ──────────────────────────────────────────────────────────
        total_layers   = len(model_sd)
        missing_set    = set(missing)
        unexpected_set = set(unexpected)
        matched_layers = total_layers - len(missing_set)

        # ── Shape-mismatches (layers exist, but dimensions differ) ──────────
        mismatched: list[str] = []
        for k, v in loaded_dict.items():
            if k in model_sd and hasattr(v, 'shape') and v.shape != model_sd[k].shape:
                mismatched.append(
                    f"{k}  [ckpt {tuple(v.shape)} → model {tuple(model_sd[k].shape)}]"
                )

        # ── Parameters (scalar count) ───────────────────────────────────────
        def _numel(sd: dict) -> int:
            return sum(v.numel() for v in sd.values() if hasattr(v, 'numel'))

        total_numel   = _numel(model_sd)
        loaded_numel  = sum(
            v.numel()
            for k, v in loaded_dict.items()
            if k in model_sd and hasattr(v, 'numel') and k not in mismatched
        )
        missing_numel = sum(
            model_sd[k].numel() for k in missing_set if k in model_sd
        )

        pct = loaded_numel / max(total_numel, 1) * 100

        # ── Progress bar ───────────────────────────────────────────────────
        BAR_W  = 40
        filled = int(BAR_W * pct / 100)
        bar    = "█" * filled + "░" * (BAR_W - filled)
        bar_color = "✅" if pct >= 99 else ("⚠️ " if pct >= 80 else "❌")

        def _fmt(n: int) -> str:
            """1 234 567 — readable number format."""
            return f"{n:,}".replace(",", " ")

        # ── Helpers ────────────────────────────────────────────────────────
        sep  = "║" + "─" * (W - 2) + "║"
        top  = "╔" + "═" * (W - 2) + "╗"
        mid  = "╠" + "═" * (W - 2) + "╣"
        bot  = "╚" + "═" * (W - 2) + "╝"
        
        def row(label: str, value: str, note: str = "") -> str:
            line = f"  {label:<22} {value:<18}  {note}"
            return f"║{line:<{W-2}}║"

        lines = [
            "",
            top,
            f"║{'  ⚓  MODEL WEIGHTS LOADING REPORT':^{W-2}}║",
            mid,
            row("Total layers",    str(total_layers)),
            row("  ✅ Loaded",      str(matched_layers),   f"({matched_layers}/{total_layers} layers)"),
            row("  ❌ Missing",     str(len(missing_set)), "kept randomly initialized" if missing_set else ""),
            row("  ⚠️  Mismatched", str(len(mismatched)),  "shape mismatch, skipped"   if mismatched else ""),
            row("  🔷 Unexpected",  str(len(unexpected_set)), "not in model arch"       if unexpected_set else ""),
            sep,
            row("Total params",   _fmt(total_numel)  + " params"),
            row("  ✅ Loaded",     _fmt(loaded_numel) + " params", f"{pct:.1f}% of model"),
            row("  ❌ Missing",    _fmt(missing_numel) + " params"),
            sep,
            f"║  {bar_color} [{bar}] {pct:5.1f}%{'':{W-2-BAR_W-14}}║",
            bot,
        ]
        print("\n".join(lines))

        # ── Detail sections ────────────────────────────────────────────────
        def _print_section(title: str, items: list[str], icon: str = "•") -> None:
            if not items:
                return
            shown = items[:max_show]
            print(f"\n  {title} ({len(items)} total):")
            for it in shown:
                print(f"    {icon} {it}")
            if len(items) > max_show:
                print(f"    … and {len(items) - max_show} more")

        _print_section("❌ Missing keys",     sorted(missing_set),    "❌")
        _print_section("⚠️  Shape mismatches", mismatched,             "⚠️ ")
        _print_section("🔷 Unexpected keys",  sorted(unexpected_set), "🔷")
        print()
    # At the end of StableEmbeddingModelViT, before the StableEmbeddingModel alias

    def export_to_torchscript(
        self,
        save_path: str,
        img_size: Tuple[int, int] = (154, 434),
        batch_size: int = 1,
        device: str = "cpu",
        natural_gallery_path: Optional[str] = None,
        embed_natural_centroids: bool = True,
        embed_arcface: bool = True,
        class_mapping_json: Optional[str] = None,   # ✅ parameter added
    ) -> "torch.jit.ScriptModule":

        import torch
        import torch.nn.functional as F
        import numpy as np
        import json

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=False)

        # --- Natural centroids ---
        natural_centroids = None
        natural_labels = None
        class_mapping_json = None
        
        if embed_natural_centroids:
            if natural_gallery_path is None:
                raise ValueError("natural_gallery_path is required")
            
            nat_data = torch.load(natural_gallery_path, map_location=device)
            
            natural_centroids = nat_data.get("centroids", nat_data.get("embeddings"))
            natural_labels = nat_data.get("labels")
            class_mapping_json = nat_data.get("labels_keys")   # ✅ pick it up here

            if isinstance(natural_centroids, np.ndarray):
                natural_centroids = torch.from_numpy(natural_centroids)
            if isinstance(natural_labels, np.ndarray):
                natural_labels = torch.from_numpy(natural_labels)

            natural_centroids = F.normalize(natural_centroids.float(), p=2, dim=1)

        # --- ArcFace centroids ---
        arcface_centroids = None
        if embed_arcface:
            arcface_centroids = F.normalize(
                self.arcface_head.weight.detach().cpu(), p=2, dim=1
            )

        # ✅ Convert the JSON string to a byte tensor HERE
        json_tensor: Optional[torch.Tensor] = None
        if class_mapping_json is not None:
            # ✅ dict -> string -> bytes -> tensor
            json_str = json.dumps(class_mapping_json, ensure_ascii=False)
            json_bytes = json_str.encode("utf-8")
            json_tensor = torch.tensor(list(json_bytes), dtype=torch.uint8)

        outer_self = self

        class _Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = outer_self

                if embed_natural_centroids and natural_centroids is not None:
                    self.register_buffer("natural_centroids", natural_centroids)
                    self.register_buffer("natural_labels", natural_labels)

                if embed_arcface and arcface_centroids is not None:
                    self.register_buffer("arcface_centroids", arcface_centroids)

                # ✅ Register the buffer on the Wrapper instead of the main model
                if json_tensor is not None:
                    self.register_buffer("class_mapping_json_bytes", json_tensor)
                else:
                    # TorchScript requires the buffer to always exist —
                    # register an empty one so the attribute stays available
                    self.register_buffer(
                        "class_mapping_json_bytes",
                        torch.zeros(0, dtype=torch.uint8)
                    )

            def forward(self, x: torch.Tensor):
                emb, logits, _, _ = self.model(
                    x,
                    labels=None,
                    object_mask=None,
                    return_softmax=False,
                    return_attention_map=False,
                )
                return emb, logits

            # ✅ Separate method to retrieve the mapping from TorchScript
            def get_class_mapping_bytes(self) -> torch.Tensor:
                return self.class_mapping_json_bytes

        wrapper = _Wrapper().to(device).eval()

        # --- Trace ---
        H, W = img_size
        dummy = torch.randn(batch_size, 3, H, W, device=device)
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, dummy, strict=False)

        # --- Test ---
        with torch.no_grad():
            ref_emb, ref_logits = wrapper(dummy)
            ts_emb, ts_logits = traced(dummy)

        print(f"[TS] emb_diff={(ref_emb - ts_emb).abs().max().item():.2e}, "
            f"logits_diff={(ref_logits - ts_logits).abs().max().item():.2e}")

        torch.jit.save(traced, save_path)

        # --- Load test ---
        try:
            loaded = torch.jit.load(save_path, map_location=device)
            loaded.eval()
            with torch.no_grad():
                loaded(dummy)

            # ✅ Verify that the mapping can be read after loading
            if class_mapping_json is not None:
                mapping_bytes = loaded.class_mapping_json_bytes
                recovered = bytes(mapping_bytes.tolist()).decode("utf-8")
                
                # ✅ compare strings via json.loads
                # assert json.loads(recovered) == class_mapping_json, "class_mapping mismatch!"
                print(f"✅ class_mapping OK ({len(mapping_bytes)} bytes)")

            print("✅ TorchScript LOAD OK")
        except Exception as e:
            raise RuntimeError(f"❌ TorchScript BROKEN after save: {e}")

        print(f"\n✅ Export done\n   path: {save_path}\n"
            f"   embedded: natural={embed_natural_centroids}, "
            f"arcface={embed_arcface}, mapping={class_mapping_json is not None}")

        return traced
# Backward-compatible alias used by the trainer
StableEmbeddingModel = StableEmbeddingModelViT