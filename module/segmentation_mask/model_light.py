import os
import torch
import lightning as L
import segmentation_models_pytorch as smp
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader


class FishSeg(L.LightningModule):
    """
    Lightning module for fish segmentation using Segmentation Models PyTorch.
    """

    def __init__(self, arch: str, encoder_name: str, in_channels: int, out_classes: int,
                 load_checkpoint: str = None, learning_rate: float = 0.0001, batch_size: int = 16, **kwargs):
        """
        Initializes the FishSeg model.

        Args:
            arch (str): Model architecture.
            encoder_name (str): Name of the encoder backbone.
            in_channels (int): Number of input channels.
            out_classes (int): Number of output classes.
            load_checkpoint (str, optional): Path to checkpoint for loading weights. Defaults to None.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.0001.
            batch_size (int, optional): Batch size for training. Defaults to 16.
            **kwargs: Additional parameters for the model.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        if load_checkpoint:
            state_dict = torch.load(load_checkpoint, map_location=self.device)
            self.load_state_dict(state_dict['state_dict'], strict=False)

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.learning_rate = learning_rate

        self.step_outputs = {"train": [], "valid": [], "test": []}

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(image)

    def shared_step(self, batch: dict, stage: str):
        """
        Shared step for training, validation, and testing.

        Args:
            batch (dict): Batch containing images and masks.
            stage (str): Stage name ('train', 'valid', 'test').
        """
        image, mask = batch["image"], batch["mask"]
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        self.log(f"{stage}_loss", loss, prog_bar=True)
        
        return loss if stage == 'train' else loss.detach(), {"tp": tp.detach(), "fp": fp.detach(), "fn": fn.detach(), "tn": tn.detach()}

    def shared_epoch_end(self, outputs: list, stage: str):
        """
        Aggregates and logs metrics at the end of each epoch.

        Args:
            outputs (list): List of step outputs.
            stage (str): Stage name ('train', 'valid', 'test').
        """
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        self.log_dict({f"{stage}_img_iou": per_image_iou, f"{stage}_iou": dataset_iou}, prog_bar=True)

    def training_step(self, batch: dict, batch_idx: int):
        """Training step."""
        loss, metrics = self.shared_step(batch, "train")
        self.step_outputs["train"].append(metrics)
        self.log("lr", self.learning_rate, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step."""
        _, metrics = self.shared_step(batch, "valid")
        self.step_outputs["valid"].append(metrics)

    def test_step(self, batch: dict, batch_idx: int):
        """Test step."""
        _, metrics = self.shared_step(batch, "test")
        self.step_outputs["test"].append(metrics)

    def on_epoch_end(self, stage: str):
        """
        Called at the end of each epoch to compute aggregated metrics.

        Args:
            stage (str): Stage name ('train', 'valid', 'test').
        """
        if self.step_outputs[stage]:
            self.shared_epoch_end(self.step_outputs[stage], stage)
            self.step_outputs[stage].clear()

    def on_train_epoch_end(self):
        """Called at the end of a training epoch."""
        self.on_epoch_end("train")

    def on_validation_epoch_end(self):
        """Called at the end of a validation epoch."""
        self.on_epoch_end("valid")

    def on_test_epoch_end(self):
        """Called at the end of a test epoch."""
        self.on_epoch_end("test")

    def configure_optimizers(self):
        """Configures the optimizer."""
        return Adam(self.parameters(), lr=self.learning_rate)

    def to_torchscript(self, path: str):
        """
        Exports the model to TorchScript format.

        Args:
            path (str): File path to save the TorchScript model.
        """
        self.eval()
        scripted_model = torch.jit.script(self.model)
        frozen_model = torch.jit.freeze(scripted_model)
        frozen_model.save(path)


# class SimplyFishClassifier(nn.Module):
#     """Simple CNN classifier for fish classification."""

#     def __init__(self, input_height, input_width):
#         super().__init__()

#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
#         )

#         conv_output_height, conv_output_width = input_height // 8, input_width // 8
#         self.fc1_input_size = 128 * conv_output_height * conv_output_width

#         self.fc_layers = nn.Sequential(
#             nn.Linear(self.fc1_input_size, 128), nn.ReLU(), nn.Dropout(0.5),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = x.view(-1, self.fc1_input_size)
#         x = self.fc_layers(x)
#         return x


# class FishClassifier(L.LightningModule):
#     """Lightning module for binary fish classification."""

#     def __init__(self, image_size, learning_rate=0.001):
#         super().__init__()
#         self.save_hyperparameters()

#         self.image_size = image_size
#         self.model = SimplyFishClassifier(image_size, image_size)
#         self.loss_fn = nn.BCEWithLogitsLoss()
#         self.learning_rate = learning_rate

#         # Store predictions for metric calculations
#         self.training_step_outputs = [[], []]
#         self.validation_step_outputs = [[], []]

#     def forward(self, x):
#         return self.model(x)

#     def configure_optimizers(self):
#         return Adam(self.parameters(), lr=self.learning_rate)

#     def shared_step(self, batch, stage):
#         inputs, labels = batch
#         outputs = self(inputs).squeeze()
#         loss = self.loss_fn(outputs, labels.float())

#         logits = torch.sigmoid(outputs)
#         predicted_classes = (logits > 0.5).int().cpu().numpy()

#         if stage == "train":
#             self.training_step_outputs[0].extend(predicted_classes)
#             self.training_step_outputs[1].extend(labels.cpu().numpy())
#         else:
#             self.validation_step_outputs[0].extend(predicted_classes)
#             self.validation_step_outputs[1].extend(labels.cpu().numpy())

#         self.log(f"{stage}_loss", loss, prog_bar=True)
#         return loss

#     def training_step(self, batch, batch_idx):
#         return self.shared_step(batch, "train")

#     def on_train_epoch_end(self):
#         self.compute_metrics(self.training_step_outputs, "train")

#     def validation_step(self, batch, batch_idx):
#         return self.shared_step(batch, "valid")

#     def on_validation_epoch_end(self):
#         self.compute_metrics(self.validation_step_outputs, "valid")

#     def compute_metrics(self, outputs, stage):
#         if outputs[0]:
#             preds, labels = outputs
#             precision, recall, f1 = precision_score(labels, preds), recall_score(labels, preds), f1_score(labels, preds)
#             self.log_dict({f"{stage}_precision": precision, f"{stage}_recall": recall, f"{stage}_f1": f1}, prog_bar=True)
#             outputs[0].clear()
#             outputs[1].clear()