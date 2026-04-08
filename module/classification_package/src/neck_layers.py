"""
Embedding neck architectures for metric learning models.

This module provides various "neck" layers that sit between the backbone's pooling 
output and the classification head (e.g., ArcFace). They handle dimensionality 
reduction, feature normalization, and gating to optimize the embedding space.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AdvancedNeck(nn.Module):
    """
    Bottleneck neck architecture using PReLU activation.
    
    Reduces the input dimensionality by half before projecting to the final 
    output dimension. PReLU is utilized as it generally outperforms GELU/ReLU 
    for dense embedding tasks by preventing dead neurons.

    Args:
        in_dim (int): Input feature dimension from the backbone/pooling layer.
        out_dim (int): Final output embedding dimension.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        use_bn (bool, optional): Whether to use BatchNorm1d (True) or LayerNorm (False) 
            in the intermediate layer. Defaults to True.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, use_bn: bool = True):
        super().__init__()
        logger.debug(f"Initialized AdvancedNeck (in={in_dim}, out={out_dim}, use_bn={use_bn}, dropout={dropout})")
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.BatchNorm1d(in_dim // 2) if use_bn else nn.LayerNorm(in_dim // 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BNNeck(nn.Module):
    """
    BNNeck architecture from 'Bag of Tricks for Person Re-ID' (CVPR 2019).
    
    A simple linear projection followed by Batch Normalization. This is considered 
    the gold standard for ArcFace metric learning. It intentionally omits non-linear 
    activations to avoid competing with ArcFace's angular margin constraints.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output embedding dimension.
        dropout (float, optional): Dropout probability before the projection. Defaults to 0.0.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        logger.debug(f"Initialized BNNeck (in={in_dim}, out={out_dim}, dropout={dropout})")
        
        layers = [nn.Linear(in_dim, out_dim, bias=False)]
        layers.append(nn.BatchNorm1d(out_dim))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        self.net = nn.Sequential(*layers)
        
        # Apply specialized initialization for metric learning stability
        nn.init.kaiming_normal_(self.net[0].weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.net[1].weight, 1.0)
        nn.init.constant_(self.net[1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LNNeck(nn.Module):
    """
    LayerNorm neck architecture.
    
    Applies a linear projection followed by Layer Normalization. This is often 
    more stable than BNNeck when dealing with small batch sizes (< 32) or highly 
    imbalanced class distributions, as LayerNorm does not rely on batch statistics.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output embedding dimension.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        logger.debug(f"Initialized LNNeck (in={in_dim}, out={out_dim}, dropout={dropout})")
        
        layers = [nn.Linear(in_dim, out_dim, bias=False), nn.LayerNorm(out_dim)]
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNeck(nn.Module):
    """
    Residual projection neck with pre-normalization.
    
    Follows the structure: LN -> Linear -> GELU -> Linear + Skip Connection.
    This design aligns with standard Vision Transformer conventions and plays well 
    with frozen or partially fine-tuned backbones. It is highly recommended when 
    `in_dim` equals `out_dim` to prevent information bottlenecks.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Final output embedding dimension.
        dropout (float, optional): Dropout probability applied after the activation. Defaults to 0.1.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        logger.debug(f"Initialized ResNeck (in={in_dim}, out={out_dim}, dropout={dropout})")
        
        mid_dim = max(in_dim, out_dim)
        
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(mid_dim, out_dim)
        
        # Handle dimensionality mismatch for the skip connection
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.out_norm = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))
        return self.out_norm(out + residual)


class GatedNeck(nn.Module):
    """
    Gated feature selection neck.
    
    Learns to dynamically weight the importance of specific backbone output 
    dimensions via a sigmoid gate before projecting them into the embedding space.
    This is particularly effective for fine-grained recognition tasks where only 
    specific feature channels carry discriminative signal.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Final output embedding dimension.
        dropout (float, optional): Dropout probability applied before projection. Defaults to 0.1.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        logger.debug(f"Initialized GatedNeck (in={in_dim}, out={out_dim}, dropout={dropout})")
        
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the learned sigmoid gate to the input features
        gated = x * self.gate(x)
        return self.norm(self.proj(self.drop(gated)))