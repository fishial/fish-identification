# -*- coding: utf-8 -*-
"""
Loss Functions for Fine-Grained Fish Classification.

This module provides various loss functions for metric learning and classification,
with support for switching between different configurations via parameters.

Supported loss types:
- 'combined': Original CombinedLoss (CrossEntropy + ThresholdConsistentMarginLoss)
- 'combined_v2': Improved loss with Focal + MultiSimilarity + optional SubCenter
- 'focal': Focal Loss for handling class imbalance
- 'subcenter_arcface': SubCenterArcFace for intra-class variation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

from pytorch_metric_learning import losses, miners


# =============================================================================
# Focal Loss - for class imbalance
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in fine-grained classification.
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Args:
        alpha: Weighting factor for rare classes (default: 0.25)
        gamma: Focusing parameter - higher values focus more on hard examples (default: 2.0)
        label_smoothing: Label smoothing factor (default: 0.0)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# =============================================================================
# Original CombinedLoss (for backward compatibility)
# =============================================================================

class CombinedLoss(nn.Module):
    """
    Original combined loss: CrossEntropy on ArcFace logits + metric learning loss.
    
    This is the default loss used in previous versions. Kept for backward compatibility.
    
    Args:
        arcface_weight: Weight for CE loss on ArcFace logits
        metric_weight: Weight for metric learning loss
        label_smoothing: Label smoothing for CE loss
        metric_loss_type: Type of metric loss ('threshold_consistent', 'multi_similarity', 'triplet')
        miner_type: Type of hard negative miner ('batch_hard', 'multi_similarity', 'triplet')
    """
    def __init__(
        self, 
        arcface_weight: float = 0.7, 
        metric_weight: float = 0.3, 
        label_smoothing: float = 0.0,
        metric_loss_type: str = 'threshold_consistent',
        miner_type: str = 'batch_hard',
    ):
        super().__init__()
        self.arcface_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Select metric loss
        if metric_loss_type == 'threshold_consistent':
            self.metric_loss = losses.ThresholdConsistentMarginLoss()
        elif metric_loss_type == 'multi_similarity':
            self.metric_loss = losses.MultiSimilarityLoss(
                alpha=2.0,
                beta=50.0,
                base=0.5,
            )
        elif metric_loss_type == 'triplet':
            self.metric_loss = losses.TripletMarginLoss(margin=0.1)
        elif metric_loss_type == 'circle':
            self.metric_loss = losses.CircleLoss(m=0.25, gamma=80)
        elif metric_loss_type == 'supcon':
            self.metric_loss = losses.SupConLoss(temperature=0.07)
        else:
            raise ValueError(f"Unknown metric_loss_type: {metric_loss_type}")
        
        # Select miner
        if miner_type == 'batch_hard':
            self.miner = miners.BatchHardMiner()
        elif miner_type == 'multi_similarity':
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        elif miner_type == 'triplet':
            self.miner = miners.TripletMarginMiner(margin=0.1, type_of_triplets="semihard")
        elif miner_type == 'none':
            self.miner = None
        else:
            raise ValueError(f"Unknown miner_type: {miner_type}")
            
        self.arcface_weight = arcface_weight
        self.metric_weight = metric_weight

    def forward(self, embeddings: torch.Tensor, arcface_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Classification loss
        arcface_loss = self.arcface_criterion(arcface_logits, labels)

        # Metric learning loss
        if self.miner is not None:
            hard_pairs = self.miner(embeddings, labels)
            metric_loss = self.metric_loss(embeddings, labels, indices_tuple=hard_pairs)
        else:
            metric_loss = self.metric_loss(embeddings, labels)

        # Combine the two losses
        total_loss = self.arcface_weight * arcface_loss + self.metric_weight * metric_loss
        return total_loss


# =============================================================================
# Improved CombinedLoss V2
# =============================================================================

class CombinedLossV2(nn.Module):
    """
    Improved combined loss with Focal Loss and better metric learning.
    
    Components:
    1. Focal Loss on ArcFace logits (handles class imbalance)
    2. Multi-Similarity Loss with MS-Miner (better hard negative mining)
    3. Optional: Cross-batch memory for larger effective batch size
    
    Args:
        arcface_weight: Weight for Focal/CE loss
        metric_weight: Weight for metric learning loss
        focal_weight: Additional weight for pure focal loss component
        label_smoothing: Label smoothing for losses
        focal_gamma: Gamma parameter for focal loss (higher = focus more on hard examples)
        use_focal: Whether to use Focal Loss instead of CE
        metric_loss_type: 'multi_similarity', 'circle', 'supcon'
        use_cross_batch_memory: Enable cross-batch memory for metric loss
        memory_size: Size of cross-batch memory
        embedding_dim: Embedding dimension (required if use_cross_batch_memory=True)
    """
    def __init__(
        self,
        arcface_weight: float = 0.6,
        metric_weight: float = 0.3,
        focal_weight: float = 0.1,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        use_focal: bool = True,
        metric_loss_type: str = 'multi_similarity',
        miner_type: str = 'multi_similarity',
        use_cross_batch_memory: bool = False,
        memory_size: int = 4096,
        embedding_dim: int = 512,
    ):
        super().__init__()
        
        # Classification loss
        if use_focal:
            self.classification_loss = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
        else:
            self.classification_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Additional focal component for tail classes
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            label_smoothing=0.0,  # No smoothing for pure focal
        ) if focal_weight > 0 else None
        
        # Metric learning loss
        if metric_loss_type == 'multi_similarity':
            base_metric_loss = losses.MultiSimilarityLoss(
                alpha=2.0,
                beta=50.0,
                base=0.5,
            )
        elif metric_loss_type == 'circle':
            base_metric_loss = losses.CircleLoss(m=0.25, gamma=80)
        elif metric_loss_type == 'supcon':
            base_metric_loss = losses.SupConLoss(temperature=0.07)
        elif metric_loss_type == 'ntxent':
            base_metric_loss = losses.NTXentLoss(temperature=0.07)
        else:
            base_metric_loss = losses.ThresholdConsistentMarginLoss()
        
        # Optionally wrap with cross-batch memory
        if use_cross_batch_memory:
            self.metric_loss = losses.CrossBatchMemory(
                loss=base_metric_loss,
                embedding_size=embedding_dim,
                memory_size=memory_size,
            )
        else:
            self.metric_loss = base_metric_loss
        
        # Miner
        if miner_type == 'multi_similarity':
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        elif miner_type == 'batch_hard':
            self.miner = miners.BatchHardMiner()
        elif miner_type == 'none':
            self.miner = None
        else:
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        
        self.arcface_weight = arcface_weight
        self.metric_weight = metric_weight
        self.focal_weight = focal_weight
        self.use_cross_batch_memory = use_cross_batch_memory

    def forward(
        self, 
        embeddings: torch.Tensor, 
        arcface_logits: torch.Tensor, 
        labels: torch.Tensor,
        raw_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Normalized embeddings [B, D]
            arcface_logits: Logits with ArcFace margin applied [B, num_classes]
            labels: Ground truth labels [B]
            raw_logits: Optional raw logits without margin (for focal loss)
        """
        # Main classification loss on ArcFace logits
        loss_arc = self.classification_loss(arcface_logits, labels)
        
        # Metric loss
        if self.miner is not None and not self.use_cross_batch_memory:
            hard_pairs = self.miner(embeddings, labels)
            loss_metric = self.metric_loss(embeddings, labels, indices_tuple=hard_pairs)
        else:
            loss_metric = self.metric_loss(embeddings, labels)
        
        total = self.arcface_weight * loss_arc + self.metric_weight * loss_metric
        
        # Additional focal loss component on raw logits
        if self.focal_loss is not None and self.focal_weight > 0:
            logits_for_focal = raw_logits if raw_logits is not None else arcface_logits
            loss_focal = self.focal_loss(logits_for_focal, labels)
            total = total + self.focal_weight * loss_focal
        
        return total


# =============================================================================
# Factory function for creating loss functions
# =============================================================================

def create_loss_function(
    loss_type: str = 'combined',
    num_classes: int = 1000,
    embedding_dim: int = 512,
    arcface_weight: float = 0.7,
    metric_weight: float = 0.3,
    focal_weight: float = 0.0,
    label_smoothing: float = 0.1,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    metric_loss_type: str = 'threshold_consistent',
    miner_type: str = 'batch_hard',
    use_cross_batch_memory: bool = False,
    memory_size: int = 4096,
) -> nn.Module:
    """
    Factory function to create loss functions based on configuration.
    
    Args:
        loss_type: One of 'combined', 'combined_v2', 'focal_only'
        num_classes: Number of classes (for some loss types)
        embedding_dim: Embedding dimension
        arcface_weight: Weight for classification loss
        metric_weight: Weight for metric learning loss
        focal_weight: Weight for additional focal loss
        label_smoothing: Label smoothing factor
        focal_gamma: Gamma for focal loss
        focal_alpha: Alpha for focal loss
        metric_loss_type: Type of metric loss
        miner_type: Type of hard negative miner
        use_cross_batch_memory: Enable cross-batch memory
        memory_size: Size of cross-batch memory
    
    Returns:
        Configured loss function module
    """
    if loss_type == 'combined':
        return CombinedLoss(
            arcface_weight=arcface_weight,
            metric_weight=metric_weight,
            label_smoothing=label_smoothing,
            metric_loss_type=metric_loss_type,
            miner_type=miner_type,
        )
    
    elif loss_type == 'combined_v2':
        return CombinedLossV2(
            arcface_weight=arcface_weight,
            metric_weight=metric_weight,
            focal_weight=focal_weight,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            use_focal=True,
            metric_loss_type=metric_loss_type,
            miner_type=miner_type,
            use_cross_batch_memory=use_cross_batch_memory,
            memory_size=memory_size,
            embedding_dim=embedding_dim,
        )
    
    elif loss_type == 'focal_only':
        return FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. "
                        f"Supported: 'combined', 'combined_v2', 'focal_only'")


# =============================================================================
# Legacy losses (kept for backward compatibility)
# =============================================================================

class MultiSimilarityLoss(nn.Module):
    """Multi-Similarity Loss - manual implementation for reference."""
    def __init__(self):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1
        self.scale_pos = 2.0
        self.scale_neg = 40.0

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True, device=feats.device)

        loss = sum(loss) / batch_size
        return loss


class WrapperOHNM(nn.Module):
    """Triplet Loss with Online Hard Negative Mining."""
    def __init__(self):
        super(WrapperOHNM, self).__init__()
        self.p = 2
        self.margin = 0.1
        self.eps = 1e-7
        self.loss_func = losses.TripletMarginLoss(margin=self.margin)
        self.miner = miners.TripletMarginMiner(margin=self.margin, type_of_triplets="all")
      
    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        hard_triplets = self.miner(feats, labels)
        loss = self.loss_func(feats, labels, hard_triplets)
        return loss
    

class WrapperAngular(nn.Module):
    """Angular Loss wrapper."""
    def __init__(self):
        super(WrapperAngular, self).__init__()
        self.loss_func = losses.AngularLoss()
        self.miner = miners.AngularMiner()
      
    def forward(self, feats, labels):
        hard_triplets = self.miner(feats, labels)
        loss = self.loss_func(feats, labels, hard_triplets)
        return loss
    
    
class WrapperPNPLoss(nn.Module):
    """PNP Loss wrapper."""
    def __init__(self):
        super(WrapperPNPLoss, self).__init__()
        self.loss_func = losses.PNPLoss()
      
    def forward(self, feats, labels):
        loss = self.loss_func(feats, labels)
        return loss


# =============================================================================
# MixUp utilities
# =============================================================================

def mixup_data(
    x: torch.Tensor, 
    y: torch.Tensor, 
    alpha: float = 0.4,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    MixUp data augmentation.
    
    Args:
        x: Input images [B, C, H, W]
        y: Labels [B]
        alpha: Beta distribution parameter (higher = more mixing)
        device: Device for tensors
    
    Returns:
        mixed_x: Mixed images
        y_a: Original labels
        y_b: Shuffled labels
        lam: Mixing coefficient
    """
    import numpy as np
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    if device is None:
        device = x.device
    
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Compute MixUp loss.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: Original labels
        y_b: Shuffled labels
        lam: Mixing coefficient
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
