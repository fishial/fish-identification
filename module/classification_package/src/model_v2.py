# Standard library
import math

# PyTorch and ecosystem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from typing import Optional

import timm
from timm.models.vision_transformer import VisionTransformer # For type hinting


class ViTAttentionPooling(nn.Module):
    """
    Attention Pooling for Vision Transformer output of shape [B, N, D].
    Computes a weighted sum of patch embeddings.
    """
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        if hidden_features is None:
            hidden_features = max(in_features // 4, 128) # A sensible default value

        self.attention_net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(), # Using Tanh for better gradient stabilization
            nn.Linear(hidden_features, 1)
        )

    def forward(self, x, object_mask=None, return_attention_map=False):
        """
        Args:
            x (torch.Tensor): ViT output in the format [Batch, Num_Tokens, Embedding_Dim].
            object_mask: Masking for ViT is not implemented in this version, as it
                         requires complex logic to map pixel masks to patches.
                         The parameter is kept for interface compatibility.
            return_attention_map (bool): Whether to return the attention map.
        """
        # x shape: [B, N, D]
        # Calculate attention weights for each token
        attention_scores = self.attention_net(x) # -> [B, N, 1]
        # Apply softmax across the token sequence so that weights sum to 1
        weights = F.softmax(attention_scores, dim=1) # -> [B, N, 1]

        # Weighted sum of token embeddings
        # (B, N, D) * (B, N, 1) -> (B, N, D) -> sum over N -> (B, D)
        pooled = (x * weights).sum(dim=1)

        if return_attention_map:
            # Return the weights, which can be used for visualization
            return pooled, weights
        else:
            return pooled

        
class AttentionPooling(nn.Module):
    """
    An attention-based pooling layer that weighs features based on an attention map,
    optionally focusing only on features within a provided object mask.
    """
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(in_channels // 4, 32)  # Sensible default for hidden channels

        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False)  # Output one channel for attention scores
        )

    def forward(self, x, object_mask=None, return_attention_map=False):
        x_for_attn = x  # By default, use the original features

        if object_mask is not None:
            B, _, H_feat, W_feat = x.shape  # Get feature map dimensions from x

            # Prepare object_mask for multiplication with x
            object_mask_for_x = object_mask.float().to(x.device)
            if object_mask_for_x.ndim == 3:
                object_mask_for_x = object_mask_for_x.unsqueeze(1)

            # The object_mask_for_x could be [B, 1, H_orig, W_orig]
            # while x could be [B, C, H_feat, W_feat].
            # If the dimensions don't match, we need to interpolate the mask.
            if object_mask_for_x.shape[2] != H_feat or object_mask_for_x.shape[3] != W_feat:
                object_mask_for_x_resized = F.interpolate(object_mask_for_x, size=(H_feat, W_feat), mode='nearest')
            else:
                object_mask_for_x_resized = object_mask_for_x

            x_for_attn = x * object_mask_for_x_resized  # Mask the input features

        attention_scores = self.attention_conv(x_for_attn)  # Attention now only sees the object's features
        weights = torch.sigmoid(attention_scores)

        # Important: If x_for_attn was already masked, attention_conv should learn
        # to generate low activations for the masked (zeroed-out) regions.
        # However, for robustness, we can still apply the mask to the final weights
        # before pooling, especially if the masks used for x and for the weights differ
        # slightly (e.g., due to interpolation).
        final_weights_for_pooling = weights
        if object_mask is not None:
            # Use the original mask (or the same one used for x) to clean up the final weights.
            # Here we ensure the mask is downsampled to the attention map's dimensions.
            B_w, _, H_attn, W_attn = weights.shape
            object_mask_for_weights = object_mask.float().to(weights.device)
            if object_mask_for_weights.ndim == 3:
                object_mask_for_weights = object_mask_for_weights.unsqueeze(1)
            mask_downsampled_for_weights = F.interpolate(object_mask_for_weights, size=(H_attn, W_attn), mode='nearest')
            final_weights_for_pooling = weights * mask_downsampled_for_weights

        # Multiply the *original x* by the final attention weights
        weighted_features = x * final_weights_for_pooling
        sum_weighted_features = weighted_features.sum(dim=(2, 3))
        sum_weights = final_weights_for_pooling.sum(dim=(2, 3)).clamp(min=1e-6)
        pooled = sum_weighted_features / sum_weights

        if return_attention_map:
            return pooled, final_weights_for_pooling  # Return the weights used for pooling
        else:
            return pooled


class ArcFaceHead(nn.Module):
    """
    ArcFace loss head for metric learning. Implements the additive angular margin penalty.
    """
    def __init__(self, embedding_dim, num_classes, s=32.0, m=0.10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s  # Scale factor
        self.m = m  # Margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Buffers for constants to avoid recomputing and ensure they are on the correct device
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

    def forward(self, normalized_emb, labels=None):
        normalized_w = F.normalize(self.weight, dim=1)
        cosine = F.linear(normalized_emb, normalized_w)

        # Apply additive angular margin penalty during training
        if labels is not None:
            cosine_sq = cosine ** 2
            sine = torch.sqrt((1.0 - cosine_sq).clamp(min=self.eps.item()))
            # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
            phi = cosine * self.cos_m - sine * self.sin_m
            # Apply correction to keep phi monotonically decreasing
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            one_hot = F.one_hot(labels, num_classes=self.num_classes).float().to(cosine.device)
            # Combine results: apply margin penalty only to the correct class
            output = one_hot * phi + (1.0 - one_hot) * cosine
            output *= self.s
        else:
            # During inference, just return scaled cosines
            output = cosine * self.s

        return output


class StableEmbeddingModelViT(nn.Module):
    """
    The main model, adapted for Vision Transformer (ViT) backbones.
    """
    def __init__(self,
                 embedding_dim=128,              # Increased default for ViT
                 num_classes=1000,
                 pretrained_backbone=True,
                 freeze_backbone_initially=False,
                 backbone_model_name='beitv2_base_patch16_224.in1k_ft_in22k_in1k', # A popular ViT model
                 custom_backbone: Optional[VisionTransformer] = None, # For passing a custom backbone
                 attention_hidden_channels=None, # For ViT pooling, this is `hidden_features`
                 arcface_s=64.0,
                 arcface_m=0.5,
                 add_bn_to_embedding=False,
                 embedding_dropout_rate=0.11):   # Slightly increase dropout
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        if custom_backbone:
            self.backbone = custom_backbone
            self.backbone_out_features = self.backbone.embed_dim
            print("Using custom ViT backbone.")
        else:
            print(f"Loading ViT backbone: {backbone_model_name}")
            self.backbone: VisionTransformer = timm.create_model(
                backbone_model_name,
                pretrained=pretrained_backbone,
                num_classes=0  # Important: disable the classification head
            )
            # Output feature size for ViT is the embedding dimension
            self.backbone_out_features = self.backbone.embed_dim

        # ViT doesn't have a `.features` attribute; feature extraction is done via `forward_features`
        self.backbone_feature_extractor = self.backbone.forward_features
        
        if freeze_backbone_initially:
              self.freeze_backbone()
                
        # Use the new ViT-compatible pooling layer
        self.pooling = ViTAttentionPooling(in_features=self.backbone_out_features,
                                           hidden_features=attention_hidden_channels)
        
        embedding_layers = [nn.Linear(self.backbone_out_features, embedding_dim)]
        if add_bn_to_embedding:
            embedding_layers.append(nn.BatchNorm1d(embedding_dim))
        if embedding_dropout_rate > 0.0:
            embedding_layers.append(nn.Dropout(embedding_dropout_rate))
            
        self.embedding_fc = nn.Sequential(*embedding_layers)
        self.arcface_head = ArcFaceHead(embedding_dim, num_classes, s=arcface_s, m=arcface_m)
        
        print(f"StableEmbeddingModel initialized with ViT backbone: {backbone_model_name if not custom_backbone else 'custom'}")
        print(f"  Embedding Dim: {embedding_dim}, Num Classes: {num_classes}")
        print(f"  ArcFace s: {arcface_s}, m: {arcface_m}")
        print(f"  Backbone out features (ViT embed_dim): {self.backbone_out_features}")
        print(f"  BN in embedding: {add_bn_to_embedding}, Dropout in embedding: {embedding_dropout_rate}")

    def freeze_backbone(self):
        print("Freezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, specific_layer_keywords=None, verbose=False):
        print(f"Unfreezing backbone parameters... (Specific keywords: {specific_layer_keywords})")
        unfrozen_count = 0
        for name, param in self.backbone.named_parameters():
            if specific_layer_keywords is None or any(keyword in name for keyword in specific_layer_keywords):
                param.requires_grad = True
                unfrozen_count += 1
                if verbose: print(f"  Unfroze: {name}")
            elif verbose and not param.requires_grad:
                print(f"  Kept frozen: {name}")
        print(f"Total parameters unfrozen in backbone: {unfrozen_count}")

    def forward(self, x, labels=None, object_mask=None, return_softmax=False, return_attention_map=True):
        """
        Forward pass of the model.
        `object_mask` is ignored in this ViT version but is kept for compatibility.
        """
        # features shape: [B, Num_Tokens, Embedding_Dim]
        features = self.backbone_feature_extractor(x)

        # Remove the CLS token before pooling, if it exists, as attention should work on image patches.
        # In standard ViT models from timm, the CLS token comes first.
        if hasattr(self.backbone, 'cls_token'):
            patch_tokens = features[:, 1:, :]
        else:
            patch_tokens = features

        pooled, attn_map = self.pooling(patch_tokens, object_mask=object_mask, return_attention_map=True)

        emb_raw = self.embedding_fc(pooled)
        emb_norm = F.normalize(emb_raw, p=2, dim=1)
        logits = self.arcface_head(emb_norm, labels)

        # For visualization, we need to convert the attn_map [B, N, 1] into a 2D map.
        # This requires knowing the patch grid dimensions.
        grid_size = self.backbone.patch_embed.grid_size
        B, N, _ = attn_map.shape
        # Check that the number of tokens matches the grid size
        if N == grid_size[0] * grid_size[1]:
            vis_attn_map = attn_map.permute(0, 2, 1).reshape(B, 1, grid_size[0], grid_size[1])
        else:
            # If something went wrong, return None for the map
            vis_attn_map = None

        if return_softmax:
            probabilities = F.softmax(logits, dim=1)
            return emb_norm, probabilities, vis_attn_map if return_attention_map else None
        else:
            return emb_norm, logits, vis_attn_map if return_attention_map else None
        
class StableEmbeddingModel(nn.Module):
    """
    A complete model for generating stable embeddings, combining a backbone,
    attention pooling, an embedding layer, and an ArcFace head for training.
    """
    def __init__(self,
                 embedding_dim=256,
                 num_classes=1000,
                 pretrained_backbone=True,
                 freeze_backbone_initially=False,
                 backbone_model_name='convnext_tiny',  # Added for backbone selection
                 custom_backbone=None,  # For passing a fully custom backbone
                 backbone_out_features=768,  # For convnext_tiny; adapt for other models
                 attention_hidden_channels=None,
                 arcface_s=32.0,
                 arcface_m=0.11,
                 add_bn_to_embedding=True,  # Enable BN by default
                 embedding_dropout_rate=0.0):  # Optional dropout
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.backbone_out_features = backbone_out_features

        if custom_backbone:
            self.backbone = custom_backbone
            # The user must ensure that custom_backbone.forward() returns a feature tensor
            # and that backbone_out_features matches its output dimension.
            # For simplicity, we assume the custom_backbone itself is the feature extractor.
            self.backbone_feature_extractor = self.backbone
            print("Using custom backbone.")
        elif backbone_model_name == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            self.backbone = convnext_tiny(weights=weights)
            self.backbone_feature_extractor = self.backbone.features  # Shape: [batch, 768, H/32, W/32]
            if not backbone_out_features == 768: # Check for convnext_tiny
                print(f"Warning: backbone_out_features is {backbone_out_features}, but convnext_tiny typically outputs 768 features. Correcting to 768.")
                self.backbone_out_features = 768
        else:
            raise ValueError(f"Unsupported backbone_model_name: {backbone_model_name}. Provide 'convnext_tiny' or a custom_backbone.")

        if freeze_backbone_initially:
            self.freeze_backbone()

        self.pooling = AttentionPooling(in_channels=self.backbone_out_features,
                                        hidden_channels=attention_hidden_channels)

        embedding_layers = [nn.Linear(self.backbone_out_features, embedding_dim)]
        if add_bn_to_embedding:
            embedding_layers.append(nn.BatchNorm1d(embedding_dim))
        # An activation can be added if necessary, e.g.:
        # embedding_layers.append(nn.ReLU(inplace=True))
        if embedding_dropout_rate > 0.1:
            embedding_layers.append(nn.Dropout(embedding_dropout_rate))

        self.embedding_fc = nn.Sequential(*embedding_layers)
        self.arcface_head = ArcFaceHead(embedding_dim, num_classes, s=arcface_s, m=arcface_m)

        print(f"StableEmbeddingModel initialized with backbone: {backbone_model_name if not custom_backbone else 'custom'}")
        print(f"  Embedding Dim: {embedding_dim}, Num Classes: {num_classes}")
        print(f"  ArcFace s: {arcface_s}, m: {arcface_m}")
        print(f"  Backbone out features: {self.backbone_out_features}")
        print(f"  BN in embedding: {add_bn_to_embedding}, Dropout in embedding: {embedding_dropout_rate}")

        # Attempt to print the std of the weights of the backbone's first convolutional layer
        try:
            first_conv_layer = None
            if hasattr(self.backbone, 'features') and isinstance(self.backbone.features, nn.Sequential) and len(self.backbone.features) > 0:
                if isinstance(self.backbone.features[0], nn.ModuleList) and len(self.backbone.features[0]) > 0:  # ConvNeXt structure
                    first_conv_layer = self.backbone.features[0][0]
                elif isinstance(self.backbone.features[0], nn.Conv2d):  # Other common structures
                    first_conv_layer = self.backbone.features[0]
            elif hasattr(self.backbone, 'conv1'):  # ResNet-like structure
                first_conv_layer = self.backbone.conv1
            elif hasattr(self.backbone, 'stem') and hasattr(self.backbone.stem, '0') and isinstance(self.backbone.stem[0], nn.Conv2d): # Timm models often have a 'stem'
                first_conv_layer = self.backbone.stem[0]

            if first_conv_layer and hasattr(first_conv_layer, 'weight'):
                print(f"  Backbone first conv weight std: {first_conv_layer.weight.std().item():.4f}")
            else:
                print("  Could not automatically determine the first conv layer to print its weight std.")
        except Exception as e:
            print(f"  Error printing first conv weight std: {e}")

    def freeze_backbone(self):
        """Sets requires_grad to False for all backbone parameters."""
        print("Freezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, specific_layer_keywords=None, verbose=False):
        """
        Unfreezes backbone parameters.

        Args:
            specific_layer_keywords (list, optional): If provided, only layers whose names
                contain any of these keywords will be unfrozen. Defaults to None, which unfreezes all.
            verbose (bool): If True, prints the name of each unfrozen/kept-frozen parameter.
        """
        print(f"Unfreezing backbone parameters... (Specific keywords: {specific_layer_keywords})")
        unfrozen_count = 0
        for name, param in self.backbone.named_parameters():
            if specific_layer_keywords is None:
                param.requires_grad = True
                unfrozen_count += 1
                if verbose: print(f"  Unfroze: {name}")
            else:
                if any(keyword in name for keyword in specific_layer_keywords):
                    param.requires_grad = True
                    unfrozen_count += 1
                    if verbose: print(f"  Unfroze (matches keyword): {name}")
                else:
                    if verbose and not param.requires_grad:  # Only print if it was and remains frozen
                        print(f"  Kept frozen: {name}")
        print(f"Total parameters unfrozen in backbone: {unfrozen_count}")

    def forward(self, x, labels=None, object_mask=None, return_softmax=False, return_attention_map=True):
        features = self.backbone_feature_extractor(x)
        pooled, attn_map = self.pooling(features, object_mask=object_mask, return_attention_map=return_attention_map)
        emb_raw = self.embedding_fc(pooled)
        emb_norm = F.normalize(emb_raw, p=2, dim=1)
        logits = self.arcface_head(emb_norm, labels)

        attn_map_to_return = attn_map if return_attention_map else None

        if return_softmax:
            probabilities = F.softmax(logits, dim=1)
            return emb_norm, probabilities, attn_map_to_return
        else:
            return emb_norm, logits, attn_map_to_return