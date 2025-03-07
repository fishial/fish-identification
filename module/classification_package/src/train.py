import os
import numpy as np
import torch
from apex import amp
from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from module.classification_package.src.eval import evaluate_model, pairwise_distance, compute_retrieval_metrics, evaluate_classification_accuracy
from module.classification_package.src.utils import AverageMeter, save_checkpoint
from module.classification_package.src.dataset import FishialDataset


def simple_accuracy(preds, labels):
    """Compute the simple accuracy given predictions and true labels."""
    return (preds == labels).mean()


def remove_file_if_exists(file_path: str):
    """Remove a file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} has been removed.")
    else:
        print(f"{file_path} does not exist.")


def train(
    scheduler,
    num_epochs: int,
    optimizer: Optimizer,
    model: nn.Module,
    train_loader: DataLoader,
    val_dataset: FishialDataset,
    device: torch.device,
    metrics: list,
    loss_fn: nn.Module,
    logger,
    output_folder="output",
    eval_every_epochs: int = None,
    test_set=None,
):
    """
    Train the model over a number of epochs and evaluate periodically.
    
    Args:
        scheduler: Learning rate scheduler.
        num_epochs (int): Total number of epochs to train.
        optimizer (Optimizer): The optimizer.
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_dataset (FishialDataset): Validation dataset.
        device (torch.device): The device (CPU/GPU).
        metrics (list): List containing metric mode as first element ('accuracy' or 'at_k').
        loss_fn (nn.Module): Loss function.
        logger: Logger for logging information.
        output_folder (str): Folder to save checkpoints.
        eval_every_epochs (int, optional): Frequency (in epochs) for evaluation. If None, evaluation occurs at every epoch.
        test_set: Extra validation data if needed.
    """
    model = model.to(device)
    best_val_acc = -float("inf")
    global_step = 0
    best_model_filepath = None
    metric_mode = metrics[0]
    top_k = [1, 3, 5]
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        # Adjust gradient requirements if in accuracy mode
        if metric_mode == 'accuracy':
            model.fc_parallel.requires_grad_(True)
            model.backbone.requires_grad_(False)
            model.embeddings.requires_grad_(False)

        epoch_loss = AverageMeter()
        all_preds = []  # for accuracy mode
        all_labels = []  # for accuracy mode
        metric_accum = {f"at_{k}": [] for k in top_k}  # for at_k mode
        metric_accum.update({f"mf1_{k}": [] for k in top_k})
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for batch in progress_bar:
            batch = tuple(t.to(device) for t in batch)
            images, labels = batch
            optimizer.zero_grad()
            
            # Forward pass: choose branch based on metric mode
            if metric_mode == 'at_k':
                outputs = model(images)[0]
            else:
                outputs = model(images)[1]
            
            loss = loss_fn(outputs, labels)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm=1.0)
            scheduler.step()
            optimizer.step()
            
            global_step += 1
            epoch_loss.update(loss.item())
            
            # Update metrics for progress display
            if metric_mode == 'accuracy':
                preds = torch.argmax(outputs, dim=-1)
                all_preds.append(preds.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
                # Concatenate predictions and labels to compute running accuracy
                preds_concat = np.concatenate(all_preds)
                labels_concat = np.concatenate(all_labels)
                running_acc = simple_accuracy(preds_concat, labels_concat)
                metric_str = f"{running_acc:.3f}"
            else:
                outputs_cpu = outputs.cpu()
                batch_metrics = compute_retrieval_metrics(
                    pairwise_distance(outputs_cpu, outputs_cpu),
                    labels, 
                    labels,
                    at_k=top_k, start_index=1
                )[0]
                for key, value in batch_metrics.items():
                    metric_accum[key].append(value)
                # Compute average metrics for display
                metric_str = " ".join(
                    f"{key}: {round(sum(vals) / len(vals) if vals else 0, 3)}"
                    for key, vals in metric_accum.items()
                )
            
            progress_bar.set_description(
                f"Epoch {epoch}/{num_epochs} Loss: {epoch_loss.val:.5f} Acc: [{metric_str}] Best Val: {best_val_acc:.4f}"
            )
        
        # Perform evaluation if scheduled or at the final epoch
        if eval_every_epochs is None or epoch % eval_every_epochs == 0 or epoch == num_epochs:
            model.eval()
            if metric_mode != 'accuracy':
                with torch.no_grad():
                    scores = evaluate_model(
                        model=model,
                        database=train_loader.dataset,
                        validation=val_dataset,
                        extra_validation=test_set,
                        save_as=[output_folder, global_step]
                    )
                logger.info(f"Step {global_step} :: {scores}")
                val_acc = scores['validation_on_database']['mf1_1']
            else:
                val_acc = evaluate_classification_accuracy(model, val_dataset)
            
            # Update best model if improvement is found
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_filepath = os.path.join(output_folder, "best_model.ckpt")
                logger.info(f"New best model saved: {best_model_filepath}")
                save_checkpoint(model, best_model_filepath)
            model.train()
            
            if metric_mode == 'accuracy':
                model.fc_parallel.requires_grad_(True)
                model.backbone.requires_grad_(False)
                model.embeddings.requires_grad_(False)
    
    logger.info("Training complete.")