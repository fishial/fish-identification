# --- Required Imports ---
import os
import random

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
from torchmetrics.classification import Accuracy
import torchvision.utils as vutils
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from PIL import Image

# Assume these classes are defined in the same file or imported from your project
from module.classification_package.src.model_v2 import StableEmbeddingModelViT, StableEmbeddingModel
from module.classification_package.src.loss_functions import CombinedLoss 
from module.classification_package.src.visualize_utils import save_attention_overlay

# --- PyTorch Lightning Module ---

class ImageEmbeddingTrainerViT(pl.LightningModule):
    """
    A PyTorch Lightning module to train a StableEmbeddingModelViT for image embeddings.

    This module handles the training, validation, optimizer configuration, and logging.
    It uses a combined loss function (classification + metric learning) and an
    auxiliary attention guidance loss to focus the model on relevant object parts.
    """
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        backbone_model_name: str = 'beitv2_base_patch16_224.in1k_ft_in22k_in1k',
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        lr_eta_min: float = 1e-7,
        attention_loss_lambda: float = 0.15,
        load_checkpoint: str = None,
        output_dir: str = "output_dir",
        visualize_attention_map: bool = False,
        seed: int = 42,
        count_random_img: int = 63,
    ):
        """
        Args:
            num_classes (int): The number of classes for the classification head.
            embedding_dim (int): The dimensionality of the output embeddings.
            backbone_model_name (str): The name of the ViT backbone from the `timm` library.
            lr (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay for the AdamW optimizer.
            lr_eta_min (float): The minimum learning rate for the cosine annealing scheduler.
            attention_loss_lambda (float): The weight for the attention guidance loss.
            load_checkpoint (str, optional): Path to checkpoint file to load.
            output_dir (str): Directory for saving outputs and visualizations.
            visualize_attention_map (bool): Whether to visualize attention maps during validation.
            seed (int): Random seed for reproducibility.
            count_random_img (int): Number of random images to visualize per epoch.
        """
        super().__init__()
        # This saves all hyperparameters to self.hparams and makes them accessible
        # like self.hparams.lr. It also allows Lightning to log them automatically.
        self.save_hyperparameters()

        # --- Model Initialization ---
        self.model = StableEmbeddingModelViT(
            embedding_dim=self.hparams.embedding_dim,
            num_classes=self.hparams.num_classes,
            backbone_model_name=self.hparams.backbone_model_name,
        )
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        if load_checkpoint:
            self._load_weights(load_checkpoint)

        # --- Loss Functions ---
        self.main_loss_fn = CombinedLoss()
        # BCEWithLogitsLoss is numerically stable and expects raw, unnormalized scores (logits).
        self.attention_guidance_loss_fn = nn.BCEWithLogitsLoss()

        # --- Metrics ---
        # Using torchmetrics is the standard and recommended way in Lightning.
        # It handles device placement, aggregation, and synchronization automatically.
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        
    def _load_weights(self, path):
        """Load weights from checkpoint with proper error handling."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        try:
            # Load to CPU first, then move to device to avoid device mismatch issues
            state_dict = torch.load(path, map_location='cpu')
            state_dict = state_dict.get("state_dict", state_dict)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")  # Show first 5
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")
        
    def forward(self, x, labels=None, object_mask=None):
        """
        Performs a forward pass through the underlying StableEmbeddingModelViT.
        
        Note: The model returns raw logits during training and softmax probabilities
        during evaluation.
        """
        return self.model(
            x, 
            labels=labels, 
            object_mask=object_mask, 
            return_softmax=(not self.training),
            return_attention_map=True
        )
    
    
    def on_fit_start(self):
        """Initialize seeds and visualization indices at the start of training."""
        # Use pl.seed_everything for comprehensive seed setting
        pl.seed_everything(self.hparams.seed, workers=True)
        
        # Setup visualization indices
        if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            try:
                val_dataloader = self.trainer.datamodule.val_dataloader()
                if val_dataloader is not None and hasattr(val_dataloader, 'dataset'):
                    total_val_size = len(val_dataloader.dataset)
                    count_to_sample = min(self.hparams.count_random_img, total_val_size)
                    if count_to_sample > 0:
                        self.visualization_indices = set(random.sample(range(total_val_size), count_to_sample))
                    else:
                        self.visualization_indices = set()
                else:
                    self.visualization_indices = set()
            except Exception as e:
                print(f"Warning: Could not setup visualization indices: {e}")
                self.visualization_indices = set()
        else:
            self.visualization_indices = set()

    def training_step(self, batch, batch_idx):
        x, y, object_mask = batch
        
        # --- Attention Guidance Logic ---
        # This part calculates an auxiliary loss to guide the model's attention
        # to focus on the area specified by the `object_mask`.
        
        # Step 1: Extract patch features and raw attention scores from the model.
        # We access the internal components of the model to get intermediate values.
        with torch.no_grad():
            features = self.model.backbone_feature_extractor(x)
            patch_tokens = features[:, 1:, :] if hasattr(self.model.backbone, 'cls_token') else features
        
        # Get the raw scores (logits) from the attention pooling layer *before* softmax.
        raw_attention_scores = self.model.pooling.attention_net(patch_tokens)

        # Step 2: Create the target attention mask from the high-resolution object mask.
        patch_size = self.model.backbone.patch_embed.patch_size[0]
        downsampler = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size).to(self.device)
        
        # Ensure the mask is 4D [B, 1, H, W] and float type for downsampling.
        object_mask_float = object_mask.float()
        if object_mask_float.ndim == 3:
            object_mask_float = object_mask_float.unsqueeze(1)
        
        target_patch_mask = downsampler(object_mask_float)
        # Flatten the grid-like mask to a sequence of patches and binarize it.
        target_for_loss = (target_patch_mask.flatten(1) > 0).float()
        
        # Step 3: Calculate the attention guidance loss.
        loss_attention_guidance = self.attention_guidance_loss_fn(
            raw_attention_scores.squeeze(-1),  # Input shape: [B, Num_Patches]
            target_for_loss                    # Target shape: [B, Num_Patches]
        )

        # --- Main Training Logic ---
        # Perform a full forward pass to get embeddings and final classification logits.
        emb, arc_logits, _ = self(x, labels=y, object_mask=object_mask)

        # Calculate the main loss (e.g., ArcFace loss + metric loss).
        loss_main = self.main_loss_fn(emb, arc_logits, y)

        # Combine the losses.
        total_loss = loss_main + self.hparams.attention_loss_lambda * loss_attention_guidance
        
        # --- Logging ---
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_main", loss_main, on_step=False, on_epoch=True)
        self.log("train/loss_attn_guidance", loss_attention_guidance, on_step=False, on_epoch=True)
        
        # Update and log accuracy (use raw logits without margin for accurate metrics)
        with torch.no_grad():
            raw_logits = self.model.arcface_head(emb)  # Get logits without margin
            preds = torch.argmax(raw_logits, dim=1)
            self.train_accuracy.update(preds, y)
        self.log("train/accuracy", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, object_mask = batch
        
        # Get model outputs for the validation batch.
        # During validation, forward returns probabilities (softmax) when not training
        emb, probabilities, attn_map = self(x, object_mask=object_mask)

        # Calculate the main validation loss (need logits with margin for loss)
        arc_logits_with_margin = self.model.arcface_head(emb, labels=y)
        loss = self.main_loss_fn(emb, arc_logits_with_margin, y)
        
        # --- Logging ---
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Update validation accuracy metric using probabilities
        preds = torch.argmax(probabilities, dim=1)
        self.val_accuracy.update(preds, y)

    def on_validation_epoch_start(self):
        if not self.hparams.visualize_attention_map: return
            
        dataset = self.trainer.datamodule.val_dataloader().dataset
        
        for img_id in tqdm(self.visualization_indices):
            
            tensor_img, _, object_mask = dataset[img_id]

            tensor_img = tensor_img.unsqueeze(0)  # [1, 3, H, W]
            object_mask = object_mask.unsqueeze(0)  # [1, 1, H, W]

            with torch.no_grad():
                _, _, attn_map = self.model(tensor_img.to(self.device), object_mask=object_mask.to(self.device))

            attn_map = attn_map[0]  # [H, W] или [1, H, W]

            img_folder_path_to_save = os.path.join(self.hparams.output_dir, "visualize", f"img_{img_id}")
            os.makedirs(img_folder_path_to_save, exist_ok=True)

            # Convert tensor directly to PIL Image without saving to disk
            # Normalize tensor to [0, 1] range and convert to uint8
            img_tensor = tensor_img[0].cpu()
            img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
            img_array = (img_tensor * 255).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
            pil_image = Image.fromarray(img_array)

            img_path = os.path.join(img_folder_path_to_save, f"epoch_{self.current_epoch}.png")
            save_attention_overlay(pil_image, attn_map, save_path=img_path, title=f"Epoch: {self.current_epoch}")
            
    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to compute and log aggregated metrics.
        """
        # Log the final epoch accuracy. The metric object handles the aggregation.
        self.log("val/accuracy_epoch", self.val_accuracy.compute(), prog_bar=True)
        # Explicitly reset the metric for the next epoch
        self.val_accuracy.reset()
        
    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        """
        # Create an optimizer that uses the hyperparameters defined in __init__.
        optimizer = AdamW(
            self.parameters(),  # Use self.parameters() to include all model parameters.
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        # Create a learning rate scheduler.
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs, # Number of epochs for the cosine cycle.
            eta_min=self.hparams.lr_eta_min # Minimum learning rate.
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # Note: CosineAnnealingLR doesn't support 'monitor' parameter
                # It schedules based on epoch count, not metric values
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    
class ImageEmbeddingTrainerConvnext(pl.LightningModule):
    """
    A PyTorch Lightning module to train a StableEmbeddingModel for image embeddings.

    This module handles the training, validation, optimizer configuration, and logging.
    It uses a combined loss function (classification + metric learning) and an
    auxiliary attention guidance loss to focus the model on relevant object parts.
    """
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        backbone_model_name: str = 'convnext_tiny',
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        lr_eta_min: float = 1e-7,
        attention_loss_lambda: float = 0.15,
        load_checkpoint: str = None,
        output_dir: str = "output_dir",
        visualize_attention_map: bool = False,
        seed: int = 42,
        count_random_img: int = 63,
    ):
        """
        Args:
            num_classes (int): The number of classes for the classification head.
            embedding_dim (int): The dimensionality of the output embeddings.
            backbone_model_name (str): The name of the backbone model from the `timm` library (e.g., 'convnext_tiny').
            lr (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay for the AdamW optimizer.
            lr_eta_min (float): The minimum learning rate for the cosine annealing scheduler.
            attention_loss_lambda (float): The weight for the attention guidance loss.
            load_checkpoint (str, optional): Path to checkpoint file to load.
            output_dir (str): Directory for saving outputs and visualizations.
            visualize_attention_map (bool): Whether to visualize attention maps during validation.
            seed (int): Random seed for reproducibility.
            count_random_img (int): Number of random images to visualize per epoch.
        """
        super().__init__()
        # This saves all hyperparameters to self.hparams and makes them accessible
        # like self.hparams.lr. It also allows Lightning to log them automatically.
        self.save_hyperparameters()

        # --- Model Initialization ---
        self.model = StableEmbeddingModel(
            embedding_dim=self.hparams.embedding_dim,
            num_classes=self.hparams.num_classes,
            backbone_model_name=self.hparams.backbone_model_name,
        )
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        if load_checkpoint:
            self._load_weights(load_checkpoint)

        # --- Loss Functions ---
        self.main_loss_fn = CombinedLoss()
        # BCEWithLogitsLoss is numerically stable and expects raw, unnormalized scores (logits).
        self.attention_guidance_loss_fn = nn.BCEWithLogitsLoss()

        # --- Metrics ---
        # Using torchmetrics is the standard and recommended way in Lightning.
        # It handles device placement, aggregation, and synchronization automatically.
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        
    def _load_weights(self, path):
        """Load weights from checkpoint with proper error handling."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        try:
            # Load to CPU first, then move to device to avoid device mismatch issues
            state_dict = torch.load(path, map_location='cpu')
            state_dict = state_dict.get("state_dict", state_dict)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")  # Show first 5
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")
        
    def forward(self, x, labels=None, object_mask=None):
        """
        Performs a forward pass through the underlying StableEmbeddingModel.
        
        Note: The model returns raw logits during training and softmax probabilities
        during evaluation.
        """
        return self.model(
            x, 
            labels=labels, 
            object_mask=object_mask, 
            return_softmax=(not self.training),
            return_attention_map=True
        )
    
    
    def on_fit_start(self):
        """Initialize seeds and visualization indices at the start of training."""
        # Use pl.seed_everything for comprehensive seed setting
        pl.seed_everything(self.hparams.seed, workers=True)
        
        # Setup visualization indices
        if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            try:
                val_dataloader = self.trainer.datamodule.val_dataloader()
                if val_dataloader is not None and hasattr(val_dataloader, 'dataset'):
                    total_val_size = len(val_dataloader.dataset)
                    count_to_sample = min(self.hparams.count_random_img, total_val_size)
                    if count_to_sample > 0:
                        self.visualization_indices = set(random.sample(range(total_val_size), count_to_sample))
                    else:
                        self.visualization_indices = set()
                else:
                    self.visualization_indices = set()
            except Exception as e:
                print(f"Warning: Could not setup visualization indices: {e}")
                self.visualization_indices = set()
        else:
            self.visualization_indices = set()

    def training_step(self, batch, batch_idx):
        x, y, object_mask = batch
        
        # --- Attention Guidance Logic ---
        # This part calculates an auxiliary loss to guide the model's attention
        # to focus on the area specified by the `object_mask`.
        
        # Step 1: Extract patch features and raw attention scores from the model.
        # We access the internal components of the model to get intermediate values.
        with torch.no_grad():
            features = self.model.backbone_feature_extractor(x)
         
        raw_attention_scores = self.model.pooling.attention_conv(features)
        # Note: BCEWithLogitsLoss expects raw logits, not sigmoid outputs
        # We'll use raw_attention_scores directly for loss calculation
        
        # --- Main Training Logic ---
        # Perform a full forward pass to get embeddings and final classification logits.
        emb, arc_logits, _ = self(x, labels=y, object_mask=object_mask)

        # Calculate the main loss (e.g., ArcFace loss + metric loss).
        loss_main = self.main_loss_fn(emb, arc_logits, y)
        
        # Prepare target mask for attention guidance loss
        B, _, H_attn, W_attn = raw_attention_scores.shape
        
        object_mask_gt_for_loss = object_mask.float().to(raw_attention_scores.device)
        if object_mask_gt_for_loss.ndim == 3:
            object_mask_gt_for_loss = object_mask_gt_for_loss.unsqueeze(1)

        target_attention_mask = F.interpolate(object_mask_gt_for_loss, size=(H_attn, W_attn), mode='nearest')
        
        # Use raw logits (not sigmoid) for BCEWithLogitsLoss
        loss_attention_guidance = self.attention_guidance_loss_fn(raw_attention_scores, target_attention_mask)
        
        # Combine the losses.
        total_loss = loss_main + self.hparams.attention_loss_lambda * loss_attention_guidance
        
        # --- Logging ---
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_main", loss_main, on_step=False, on_epoch=True)
        self.log("train/loss_attn_guidance", loss_attention_guidance, on_step=False, on_epoch=True)
        
        # Update and log accuracy (use raw logits without margin for accurate metrics)
        with torch.no_grad():
            raw_logits = self.model.arcface_head(emb)  # Get logits without margin
            preds = torch.argmax(raw_logits, dim=1)
            self.train_accuracy.update(preds, y)
        self.log("train/accuracy", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, object_mask = batch
        
        # Get model outputs for the validation batch.
        # During validation, forward returns probabilities (softmax) when not training
        emb, probabilities, attn_map = self(x, object_mask=object_mask)

        # Calculate the main validation loss (need logits with margin for loss)
        arc_logits_with_margin = self.model.arcface_head(emb, labels=y)
        loss = self.main_loss_fn(emb, arc_logits_with_margin, y)
        
        # --- Logging ---
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Update validation accuracy metric using probabilities
        preds = torch.argmax(probabilities, dim=1)
        self.val_accuracy.update(preds, y)

    def on_validation_epoch_start(self):
        if not self.hparams.visualize_attention_map: return
            
        dataset = self.trainer.datamodule.val_dataloader().dataset
        
        for img_id in tqdm(self.visualization_indices):
            
            tensor_img, _, object_mask = dataset[img_id]

            tensor_img = tensor_img.unsqueeze(0)  # [1, 3, H, W]
            object_mask = object_mask.unsqueeze(0)  # [1, 1, H, W]

            with torch.no_grad():
                _, _, attn_map = self.model(tensor_img.to(self.device), object_mask=object_mask.to(self.device))

            attn_map = attn_map[0]  # [H, W] или [1, H, W]

            img_folder_path_to_save = os.path.join(self.hparams.output_dir, "visualize", f"img_{img_id}")
            os.makedirs(img_folder_path_to_save, exist_ok=True)

            # Convert tensor directly to PIL Image without saving to disk
            # Normalize tensor to [0, 1] range and convert to uint8
            img_tensor = tensor_img[0].cpu()
            img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
            img_array = (img_tensor * 255).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
            pil_image = Image.fromarray(img_array)

            img_path = os.path.join(img_folder_path_to_save, f"epoch_{self.current_epoch}.png")
            save_attention_overlay(pil_image, attn_map, save_path=img_path, title=f"Epoch: {self.current_epoch}")
            
    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to compute and log aggregated metrics.
        """
        # Log the final epoch accuracy. The metric object handles the aggregation.
        self.log("val/accuracy_epoch", self.val_accuracy.compute(), prog_bar=True)
        # Explicitly reset the metric for the next epoch
        self.val_accuracy.reset()
        
    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        """
        # Create an optimizer that uses the hyperparameters defined in __init__.
        optimizer = AdamW(
            self.parameters(),  # Use self.parameters() to include all model parameters.
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        # Create a learning rate scheduler.
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs, # Number of epochs for the cosine cycle.
            eta_min=self.hparams.lr_eta_min # Minimum learning rate.
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # Note: CosineAnnealingLR doesn't support 'monitor' parameter
                # It schedules based on epoch count, not metric values
                "interval": "epoch",
                "frequency": 1
            }
        }