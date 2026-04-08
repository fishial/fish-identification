"""ArcFace-based classification heads."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ArcFaceBase(nn.Module):
    """Base logic for additive angular margin computation."""

    def _register_margin_buffers(self, m: float):
        self.m = m
        self.register_buffer('cos_m', torch.tensor(math.cos(m)))
        self.register_buffer('sin_m', torch.tensor(math.sin(m)))
        self.register_buffer('th', torch.tensor(math.cos(math.pi - m)))
        self.register_buffer('mm', torch.tensor(math.sin(math.pi - m) * m))
        self.register_buffer('eps', torch.tensor(1e-7))

    def set_margin(self, new_m: float):
        """Safe in-place update for dynamic margin scheduling."""
        self.m = new_m
        self.cos_m.fill_(math.cos(new_m))
        self.sin_m.fill_(math.sin(new_m))
        self.th.fill_(math.cos(math.pi - new_m))
        self.mm.fill_(math.sin(math.pi - new_m) * new_m)

    def _apply_margin(
        self,
        cosine: torch.Tensor,
        labels: torch.Tensor,
        s: float,
    ) -> torch.Tensor:
        idx = labels.to(dtype=torch.long, device=cosine.device).view(-1, 1)

        # 1. Extract cosines of the target classes
        cos_target = cosine.gather(1, idx)
        
        # 2. PROTECTION AGAINST NaN GRADIENTS (The fix!)
        cos_target = cos_target.clamp(-1.0 + self.eps.item(), 1.0 - self.eps.item())

        # 3. Trigonometry application
        sine_target = torch.sqrt(1.0 - cos_target ** 2)
        phi_target = cos_target * self.cos_m - sine_target * self.sin_m
        
        # 4. Protection against obtuse angles (thresholding)
        phi_target = torch.where(cos_target > self.th, phi_target, cos_target - self.mm)

        # 5. Assemble final logits
        output = cosine.clone()
        output.scatter_(1, idx, phi_target.to(dtype=output.dtype))
        
        return output * s


class ArcFaceHead(_ArcFaceBase):
    """Standard ArcFace classification head."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        s: float = 32.0,
        m: float = 0.10,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s
        self._register_margin_buffers(m)

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.normal_(self.weight, std=0.01)

    def forward(
        self,
        normalized_emb: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cosine = F.linear(normalized_emb, F.normalize(self.weight, dim=1))
        if labels is not None:
            return self._apply_margin(cosine, labels, self.s)
        return cosine * self.s


class SubCenterArcFaceHead(_ArcFaceBase):
    """
    Sub-center ArcFace with robust median AdaCos scaling.
    Uses Max-Pooling over sub-centers as defined in the original CVPR 2020 paper.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        K: int = 3,
        s: float = 32.0,
        m: float = 0.20,
        use_adacos: bool = False,
        adacos_momentum: float = 0.9,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.K = K
        self.use_adacos = use_adacos
        self.adacos_momentum = adacos_momentum
        
        # Register margin buffers (inherited from _ArcFaceBase)
        self._register_margin_buffers(m)

        # Weights: [Classes * K, Embedding_Dim]
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * K, embedding_dim))
        
        # For ArcFace, normal distribution often provides a faster start than Xavier
        nn.init.normal_(self.weight, std=0.01)

        if use_adacos:
            # Scale initialization based on the AdaCos paper formula
            init_s = math.sqrt(2.0) * math.log(max(num_classes - 1, 1))
            self.register_buffer('s', torch.tensor(max(init_s, 10.0)))
        else:
            self.register_buffer('s', torch.tensor(float(s)))

    def forward(self, normalized_emb, labels=None):
        # 1. Calculate cosines with ALL sub-centers: [Batch, Classes * K]
        cosine_all = F.linear(normalized_emb, F.normalize(self.weight, dim=1))

        # 2. Sub-Center Max Pooling
        if self.K > 1:
            # Reshape to [Batch, Classes, K]
            cosine_all = cosine_all.view(-1, self.num_classes, self.K)
            
            # PURE MAX POOLING: Select the closest sub-center for each class
            # max() returns a tuple (values, indices); we take [0] (values)
            cosine = cosine_all.max(dim=2)[0]
        else:
            cosine = cosine_all

        # 3. Update dynamic scale (Training phase only)
        if self.use_adacos and labels is not None and self.training:
            self._update_scale(cosine, labels)

        # 4. Apply angular margin to the target class logits
        if labels is not None:
            return self._apply_margin(cosine, labels, self.s)
            
        return cosine * self.s
        
    @torch.no_grad()
    def _update_scale(self, cosine, labels):
        idx = labels.view(-1, 1)
        
        # 1. Extract target class cosines and strictly clamp to prevent NaN
        cos_target = cosine.gather(1, idx).squeeze(1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        
        # 2. Convert to radians
        theta = torch.acos(cos_target)
        
        # 3. ANTI-DOMINANCE HACK: Calculate median based on unique classes in the batch!
        unique_classes = labels.unique()
        theta_per_class = torch.stack([
            theta[labels == c].mean() for c in unique_classes
        ])
        theta_med = theta_per_class.median()
        
        # 4. Protect against division by zero / negative cosines
        safe_theta = theta_med.clamp(0.01, math.pi / 2 - 0.1)
        
        # 5. Calculate new scale (Python float / PyTorch Tensor = Tensor result)
        new_s = (math.log(max(self.num_classes - 1, 1)) / torch.cos(safe_theta)).clamp(10.0, 128.0)
        
        # 6. Exponential Moving Average (EMA) update
        self.s.copy_(
            self.adacos_momentum * self.s +
            (1 - self.adacos_momentum) * new_s
        )