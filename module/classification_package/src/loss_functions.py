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
# Corrected Focal Loss
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(
        self, 
        alpha = 0.25, 
        gamma: float = 2.0, 
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if isinstance(alpha, torch.Tensor):
            self.register_buffer('alpha', alpha.float())
        elif isinstance(alpha, (list, tuple)):
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(float(alpha)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Compute the TRUE probability (pt) WITHOUT label_smoothing!
        with torch.no_grad():
            log_pt = -F.cross_entropy(logits, targets, reduction='none', label_smoothing=0.0)
            pt = torch.exp(log_pt)

        # 2. Compute the CE loss (with smoothing applied if needed)
        ce_loss = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.label_smoothing)

        # 3. Apply class weights
        if self.alpha.ndim > 0 and self.alpha.numel() > 1:
            alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha

        # 4. Assemble the proper Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
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
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.arcface_criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=class_weights,
        )
        
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

class CombinedLossV3(nn.Module):
    """
    Production version:
    ✔ ArcFace + CrossEntropy (NO smoothing)
    ✔ MultiSimilarity Loss
    ✔ CrossBatchMemory (miner correctly passed through)
    """
    def __init__(
        self,
        num_classes: int,
        arcface_weight: float = 1.0,
        metric_weight: float = 0.15,
        metric_loss_type: str = 'multi_similarity',
        miner_type: str = 'multi_similarity',
        use_cross_batch_memory: bool = True,
        memory_size: int = 4096,
        embedding_dim: int = 512,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.arcface_weight = arcface_weight
        self.metric_weight = metric_weight

        # 1. Classification (ArcFace -> CE without smoothing)
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.0, weight=class_weights)

        # 2. Metric Loss
        if metric_loss_type == 'circle':
            base_metric = losses.CircleLoss(m=0.25, gamma=80)
        elif metric_loss_type == 'proxy_anchor':
            base_metric = losses.ProxyAnchorLoss(num_classes=num_classes, embedding_size=embedding_dim)
            use_cross_batch_memory = False # ProxyAnchor is incompatible with XBM
        else:
            base_metric = losses.MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5)

        # 3. Miner
        if miner_type == 'batch_hard':
            miner = miners.BatchHardMiner()
        elif miner_type == 'none':
            miner = None
        else:
            miner = miners.MultiSimilarityMiner(epsilon=0.1)

        # 4. Cross-Batch Memory
        if use_cross_batch_memory:
            self.metric_loss = losses.CrossBatchMemory(
                loss=base_metric,
                embedding_size=embedding_dim,
                memory_size=memory_size,
                miner=miner # Passing miner INTO the memory!
            )
            self.miner = None # Clearing the external miner
        else:
            self.metric_loss = base_metric
            self.miner = miner

    def forward(self, embeddings, arcface_logits, labels, **kwargs):
        loss_arc = self.classification_loss(arcface_logits, labels)

        if self.miner is not None:
            hard_pairs = self.miner(embeddings, labels)
            loss_metric = self.metric_loss(embeddings, labels, indices_tuple=hard_pairs)
        else:
            loss_metric = self.metric_loss(embeddings, labels)

        total = self.arcface_weight * loss_arc + self.metric_weight * loss_metric
        return total, loss_arc.detach(), loss_metric.detach()


# =============================================================================
# Fixed CombinedLoss V2
# =============================================================================
class CombinedLossV2(nn.Module):
    def __init__(
        self,
        arcface_weight: float = 1.0,
        metric_weight: float = 0.3,
        label_smoothing: float = 0.0,
        use_focal: bool = False, # Recommend False for ArcFace
        focal_gamma: float = 2.0,
        focal_alpha = 0.25,
        metric_loss_type: str = 'multi_similarity',
        miner_type: str = 'multi_similarity',
        use_cross_batch_memory: bool = False,
        memory_size: int = 4096,
        embedding_dim: int = 512,
    ):
        super().__init__()
        
        # 1. Classification loss (ArcFace)
        if use_focal:
            self.classification_loss = FocalLoss(
                alpha=focal_alpha, gamma=focal_gamma, label_smoothing=label_smoothing
            )
        else:
            self.classification_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # 2. Setup Miner
        if miner_type == 'multi_similarity':
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        elif miner_type == 'batch_hard':
            self.miner = miners.BatchHardMiner()
        else:
            self.miner = None

        # 3. Setup Metric Loss
        if metric_loss_type == 'multi_similarity':
            base_metric_loss = losses.MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5)
        elif metric_loss_type == 'circle':
            base_metric_loss = losses.CircleLoss(m=0.25, gamma=80)
        else:
            base_metric_loss = losses.ThresholdConsistentMarginLoss()
        
        # 4. Cross-Batch Memory (proper integration)
        self.use_cross_batch_memory = use_cross_batch_memory
        if use_cross_batch_memory:
            self.metric_loss = losses.CrossBatchMemory(
                loss=base_metric_loss,
                embedding_size=embedding_dim,
                memory_size=memory_size,
                miner=self.miner # <-- Now the miner runs inside the memory!
            )
            self.miner = None # Nullify the external miner to avoid double mining
        else:
            self.metric_loss = base_metric_loss
        
        self.arcface_weight = arcface_weight
        self.metric_weight = metric_weight

    def forward(self, embeddings: torch.Tensor, arcface_logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        # 1. ArcFace Loss
        loss_arc = self.classification_loss(arcface_logits, labels)
        
        # 2. Metric Loss
        if self.miner is not None:
            # When XBM is disabled, use classic mining
            hard_pairs = self.miner(embeddings, labels)
            loss_metric = self.metric_loss(embeddings, labels, indices_tuple=hard_pairs)
        else:
            # When XBM is enabled, it handles mining internally
            loss_metric = self.metric_loss(embeddings, labels)
        
        return self.arcface_weight * loss_arc + self.metric_weight * loss_metric


# =============================================================================
# Factory function for creating loss functions
# =============================================================================

def compute_class_weights(targets, num_classes: int, smoothing: float = 0.1) -> torch.Tensor:
    """
    Compute inverse-frequency per-class weights for focal loss.
    
    Args:
        targets: List or tensor of integer labels
        num_classes: Total number of classes
        smoothing: Laplace smoothing to avoid div-by-zero for missing classes
    
    Returns:
        Tensor of shape [num_classes] with balanced weights (mean = 1.0)
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    
    counts = torch.zeros(num_classes)
    for t in targets:
        counts[int(t)] += 1
    
    counts = counts + smoothing
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return weights


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
    class_weights: Optional[torch.Tensor] = None,
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
        focal_alpha: Alpha or per-class weight tensor for focal loss.
                     When class_weights is provided, it overrides this.
        metric_loss_type: Type of metric loss
        miner_type: Type of hard negative miner
        use_cross_batch_memory: Enable cross-batch memory
        memory_size: Size of cross-batch memory
        class_weights: Optional per-class weight tensor [num_classes] from
                       compute_class_weights(). Replaces scalar focal_alpha.
    
    Returns:
        Configured loss function module
    """
    effective_alpha = class_weights if class_weights is not None else focal_alpha
    
    if loss_type == 'combined':
        return CombinedLoss(
            arcface_weight=arcface_weight,
            metric_weight=metric_weight,
            label_smoothing=label_smoothing,
            metric_loss_type=metric_loss_type,
            miner_type=miner_type,
            class_weights=class_weights,
        )
    
    elif loss_type == 'combined_v2':
        return CombinedLossV2(
            arcface_weight=arcface_weight,
            metric_weight=metric_weight,
            focal_weight=focal_weight,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            focal_alpha=effective_alpha,
            use_focal=True,
            metric_loss_type=metric_loss_type,
            miner_type=miner_type,
            use_cross_batch_memory=use_cross_batch_memory,
            memory_size=memory_size,
            embedding_dim=embedding_dim,
        )
    elif loss_type == 'combined_v3':
        return CombinedLossV3(
            num_classes=num_classes,
            arcface_weight=arcface_weight,
            metric_weight=metric_weight,
            metric_loss_type='multi_similarity',
            miner_type='multi_similarity',
            use_cross_batch_memory=True,
            embedding_dim=embedding_dim,
            class_weights=class_weights,
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
