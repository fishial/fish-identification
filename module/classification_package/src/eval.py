import os
import typing as t

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from module.classification_package.src.dataset import FishialDataset
from module.classification_package.src.utils import save_json, read_json

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KDTree


def pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distance between two sets of vectors using torch.cdist.
    
    Args:
        x: Tensor of shape (n, d).
        y: Tensor of shape (m, d).
        
    Returns:
        Tensor of shape (n, m) with the Euclidean distances.
    """
    return torch.cdist(x, y, p=2)


def get_embeddings(
    model: nn.Module,
    dataset,
    device: t.Union[str, torch.device] = "cuda",
    use_at_k: bool = False,
    batch_size: int = 128,
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract embeddings and corresponding labels from a model and dataset.
    
    Args:
        model: The neural network model.
        dataset: A dataset with a 'targets' attribute.
        device: Device to perform computation.
        use_at_k: Flag indicating which output from model(batch) to use. If True, assumes
                  model returns (embeddings, ...). If False, uses the second output.
        batch_size: Batch size for the DataLoader.
        
    Returns:
        A tuple (embeddings, labels). Labels are converted to a tensor if using retrieval mode.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = compute_embeddings(data_loader, model, device, use_at_k)
    labels = torch.tensor(dataset.targets) if use_at_k else dataset.targets
    return embeddings, labels


def compute_embeddings(
    dataloader: DataLoader, model: nn.Module, device: t.Union[str, torch.device], use_at_k: bool
) -> torch.Tensor:
    """
    Compute embeddings for all batches in the dataloader.
    
    Args:
        dataloader: DataLoader providing the dataset.
        model: The neural network model.
        device: Device to perform computation.
        use_at_k: Flag to decide which output index of model(batch) to use.
    
    Returns:
        A tensor containing all embeddings.
    """
    embeddings = []
    for batch in tqdm(dataloader, desc="Embedding Extraction", bar_format="{l_bar}{r_bar}"):
        inputs = batch[0].to(device)
        outputs = model(inputs)
        # Use the first output if in retrieval mode (use_at_k), else use the second output (e.g. logits).
        embedding = outputs[0] if use_at_k else outputs[1]
        embeddings.append(embedding.cpu().detach())
    return torch.vstack(embeddings)


def compute_retrieval_metrics(
    distances: torch.Tensor,
    reference_labels: torch.Tensor,
    query_labels: torch.Tensor,
    at_k: t.List[int] = [1, 3, 5, 10],
    start_index: int = 0,
    save_as: t.Optional[t.Tuple[str, int]] = None,
    title: str = "No",
) -> t.Tuple[dict, torch.Tensor]:
    """
    Compute accuracy and mean F1 score at different k values for retrieval tasks.
    
    Args:
        distances: Pairwise distance matrix (query vs. reference).
        reference_labels: Labels for the reference embeddings.
        query_labels: Labels for the query embeddings.
        at_k: List of k values at which to compute metrics.
        start_index: Index to start from in the sorted indices (useful to skip self).
        save_as: Optional tuple (output_folder, step) to save the metrics as JSON.
        title: Title used for saving the JSON file.
    
    Returns:
        A tuple containing a dictionary of metrics and the final binary mask of correct retrievals.
    """
    # Sort distances along each row.
    _, sorted_indices = torch.sort(distances, dim=1)
    query_labels_expanded = query_labels.view(-1, 1)

    # Load previous results if saving is enabled.
    if save_as is not None:
        output_folder, step = save_as
        filepath = os.path.join(output_folder, f"{title}.json")
        try:
            data = read_json(filepath)
        except Exception:
            data = {}

    metrics = {}
    for k in at_k:
        # Get top-k neighbors, starting from start_index (to optionally skip self).
        neighbor_indices = sorted_indices[:, start_index : k + start_index]
        retrieved_labels = reference_labels[neighbor_indices]
        correct_mask = torch.any(query_labels_expanded == retrieved_labels, dim=1)
        f1_scores = get_f1_score_per_class(query_labels, retrieved_labels)
        mean_f1 = sum(f1_scores.values()) / len(f1_scores) if f1_scores else 0.0

        if save_as is not None:
            data.setdefault(str(step), {})[f"at_{k}"] = {
                "f1_scores": f1_scores,
                "mean_f1": mean_f1,
            }
        metrics[f"at_{k}"] = correct_mask.float().mean().item()
        metrics[f"mf1_{k}"] = mean_f1

    if save_as is not None:
        save_json(data, filepath)
        print(filepath)

    return metrics, correct_mask


def accuracy_at_k(
    y_true: np.ndarray, embeddings: np.ndarray, k: int, sample: t.Optional[int] = None
) -> float:
    """
    Compute accuracy at k using a KDTree for nearest neighbors search.
    
    Args:
        y_true: True labels as a numpy array.
        embeddings: Embeddings as a numpy array.
        k: The number of neighbors to consider.
        sample: Optionally, a number of samples to evaluate.
        
    Returns:
        Accuracy at k.
    """
    tree = KDTree(embeddings)
    if sample is None:
        sample = len(y_true)
    y_true_sample = y_true[:sample]
    # Query k+1 neighbors and skip the first (assumed to be self).
    neighbor_indices = tree.query(embeddings[:sample], k=k + 1, return_distance=False)[:, 1:]
    predicted_labels = y_true[neighbor_indices]
    matching = (y_true_sample.reshape(-1, 1) == predicted_labels)
    correct_count = np.sum(matching.sum(axis=1) > 0)
    return correct_count / len(y_true_sample)


def classification_accuracy(true_labels: torch.Tensor, outputs: torch.Tensor) -> float:
    """
    Compute classification accuracy given true labels and model outputs.
    
    Args:
        true_labels: Ground truth labels.
        outputs: Model outputs (logits).
        
    Returns:
        Classification accuracy.
    """
    predicted = torch.argmax(outputs, dim=1)
    return accuracy_score(true_labels, predicted)


def evaluate_at_k(true_labels: np.ndarray, embeddings: np.ndarray) -> t.List[float]:
    """
    Evaluate retrieval accuracy at k for k in [1, 3, 5].
    
    Args:
        true_labels: True labels as a numpy array.
        embeddings: Embeddings as a numpy array.
        
    Returns:
        A list of accuracies at k.
    """
    return [round(accuracy_at_k(true_labels, embeddings, k), 7) for k in [1, 3, 5]]


def get_f1_score_per_class(true_labels: torch.Tensor, top_k_preds: torch.Tensor) -> dict:
    """
    Compute the F1 score per class for retrieval predictions.
    
    For each class, counts true positives (TP), false positives (FP), and false negatives (FN)
    across all samples and then computes the F1 score.
    
    Args:
        true_labels: Tensor of true labels (shape: [N] or [N, 1]).
        top_k_preds: Tensor of retrieved predictions for each sample (shape: [N, k]).
        
    Returns:
        Dictionary mapping each class to its F1 score.
    """
    true_labels_flat = true_labels.view(-1)
    unique_classes = torch.unique(true_labels_flat).tolist()
    metrics_dict = {cls: {"TP": 0, "FP": 0, "FN": 0} for cls in unique_classes}

    for i in range(len(true_labels_flat)):
        actual = true_labels_flat[i].item()
        preds = top_k_preds[i].tolist()
        for cls in unique_classes:
            if actual == cls:
                if cls in preds:
                    metrics_dict[cls]["TP"] += 1
                else:
                    metrics_dict[cls]["FN"] += 1
            else:
                if cls in preds:
                    metrics_dict[cls]["FP"] += 1

    f1_scores = {}
    for cls, counts in metrics_dict.items():
        TP = counts["TP"]
        FP = counts["FP"]
        FN = counts["FN"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_scores[cls] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_scores


def evaluate_model(
    model: nn.Module,
    database: FishialDataset,
    validation: FishialDataset,
    extra_validation: t.Optional[FishialDataset] = None,
    save_as: t.Optional[t.Tuple[str, int]] = None,
) -> dict:
    """
    Evaluate the model on multiple datasets using retrieval metrics.
    
    The evaluation is performed in retrieval mode (using embeddings) on:
      - The database.
      - Validation data against the database.
      - (Optionally) Extra validation data.
      
    Args:
        model: The neural network model.
        database: Dataset used as the reference (database).
        validation: Dataset used for validation (queries).
        extra_validation: An optional extra validation dataset.
        save_as: Optional tuple (output_folder, step) to save the metrics as JSON.
        
    Returns:
        A dictionary with metrics for each evaluation scenario.
    """
    total_metrics = {}
    model.eval()
    device = next(model.parameters()).device

    # Compute embeddings using retrieval mode.
    db_embeddings, db_labels = get_embeddings(model, database, device=device, use_at_k=True)
    val_embeddings, val_labels = get_embeddings(model, validation, device=device, use_at_k=True)

    if extra_validation is not None:
        extra_embeddings, extra_labels = get_embeddings(model, extra_validation, device=device, use_at_k=True)
        distances_extra = pairwise_distance(extra_embeddings, extra_embeddings)
        metrics_extra, _ = compute_retrieval_metrics(
            distances_extra, extra_labels, extra_labels, start_index=1, save_as=save_as, title="extra_validation"
        )
        total_metrics["extra_validation"] = metrics_extra

    distances_db_val = pairwise_distance(val_embeddings, db_embeddings)
    metrics_db_val, _ = compute_retrieval_metrics(
        distances_db_val, db_labels, val_labels, start_index=0, save_as=save_as, title="validation_on_database"
    )
    total_metrics["validation_on_database"] = metrics_db_val

    distances_val = pairwise_distance(val_embeddings, val_embeddings)
    metrics_val, _ = compute_retrieval_metrics(
        distances_val, val_labels, val_labels, start_index=1, save_as=save_as, title="validation_on_validation"
    )
    total_metrics["validation_on_validation"] = metrics_val

    return total_metrics


def evaluate_classification_accuracy(model: nn.Module, dataset: FishialDataset) -> float:
    """
    Evaluate classification accuracy on the given dataset.
    
    Args:
        model: The neural network model.
        dataset: The dataset to evaluate.
        
    Returns:
        Classification accuracy.
    """
    model.eval()
    outputs, labels = get_embeddings(
        model, dataset, device=next(model.parameters()).device, use_at_k=False
    )
    return classification_accuracy(labels, outputs)