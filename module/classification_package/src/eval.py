import numpy as np
import typing as t
import torch
import os

from torch import nn
from torch.utils.data import DataLoader
from module.classification_package.src.dataset import FishialDataset
from module.classification_package.src.utils import save_json, read_json

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KDTree

from tqdm import tqdm

def pairwise_distance(x, y):

    x_square = torch.sum(x**2, dim=1, keepdim=True)  # (n, 1)
    y_square = torch.sum(y**2, dim=1, keepdim=True).t()  # (1, m)
    
    xy_inner_product = torch.matmul(x, y.t())  # (n, m)
    
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * (a . b)
    distances = x_square + y_square - 2 * xy_inner_product
    
    distances = torch.clamp(distances, min=0.0)
    
    distances = torch.sqrt(distances)
    
    return distances

def get_embeddings(model, dataset, device = 'cuda', metrics = ['acc'] , batch_size = 128):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dump = dump_embeddings(data_loader, model, device = device, metrics = metrics)
    if metrics[0] == 'at_k':
        labels = torch.tensor(dataset.targets)
    else:
        labels = dataset.targets
    return dump, labels

def dump_embeddings(dataloader: DataLoader, model: nn.Module, device: torch.device, metrics: list) -> np.ndarray:
    embeddings = []
    
    validation_iterator = tqdm(dataloader,
                              desc="Validation",
                              bar_format="{l_bar}{r_bar}",
                              disable=False)

    with torch.no_grad():
        for batch in validation_iterator:
            if metrics[0] == 'at_k':
                embeddings_batch = model(batch[0].to(device))[0]
                embeddings.append(embeddings_batch.cpu().detach())
            else:
                embeddings_batch = model(batch[0].to(device))[1]
                embeddings.append(embeddings_batch.cpu().detach())

    return torch.vstack(embeddings)


def get_acc(distances, data_set_labels, val_labels, at_k = [1,3,5,10], start_since = 0, save_as = None, title = 'No', f1_score_treshold = 0.7):
    val, indi = torch.sort(distances, dim = 1)
    expand_labels = torch.unsqueeze(val_labels, 1)

    if save_as is not None:
        output_folder, step = save_as
        try:
            filpath_name = os.path.join(output_folder, f"{title}.json")
            data = read_json(filpath_name)
        except Exception as e:
            data = {}
                
    acc = {}
    for k in at_k:
        slice_k = indi[:,start_since:k+1]
        labels_true_hat = data_set_labels[slice_k]

        mask_2d = expand_labels == labels_true_hat
        mask_1d = torch.any(mask_2d, dim=1)
        
        f1_score = get_f1_score_per_class(expand_labels, labels_true_hat)
        mf1 = sum([f1_score[i] for i in f1_score])/len(f1_score)

        if save_as is not None:
            if str(step) not in data:
                data.update({str(step): {}})
            data[str(step)].update({
                f"at_{k}": {
                     "f1_mean": f1_score,
                     "f1_mean_score_per_class": mf1,
                }
            })
            

        acc.update({
            f"at_{k}": (mask_1d.sum()/mask_1d.shape[0]).item(),
            f"mf1_{k}": mf1
        })
    if save_as is not None:
        save_json(data, filpath_name)
        print(filpath_name)
    return acc, mask_1d

def accuracy_at_k(y_true: np.ndarray, embeddings: np.ndarray, K: int, sample: int = None) -> float:
    kdtree = KDTree(embeddings)
    if sample is None:
        sample = len(y_true)
    y_true_sample = y_true[:sample]

    indices_of_neighbours = kdtree.query(embeddings[:sample], k=K + 1, return_distance=False)[:, 1:]

    y_hat = y_true[indices_of_neighbours]

    matching_category_mask = np.expand_dims(np.array(y_true_sample), -1) == y_hat

    matching_cnt = np.sum(matching_category_mask.sum(-1) > 0)
    accuracy = matching_cnt / len(y_true_sample)

    return accuracy


def accuracy(labels, dump) -> t.List[float]:
    y_pred = torch.argmax(dump, dim=1)
    acc = accuracy_score(labels, y_pred)
    return acc


def evaluate_at_k(labels: np.array, dump: np.ndarray) -> t.List[float]:
    accuracies = []
    for K in [1, 3, 5]:
        acc_k = accuracy_at_k(labels, dump, K)
        accuracies.append(round(acc_k, 7))
    return accuracies


def get_f1_score_per_class(true_labels, top_k_preds):
    unique_classes = torch.unique(true_labels)

    metrics = {label.item(): {'TP': 0, 'FP': 0, 'FN': 0} for label in unique_classes}

    for i in range(len(true_labels)):
        true_label = true_labels[i].item()
        preds = top_k_preds[i].tolist()

        if true_label in preds:
            metrics[true_label]['TP'] += 1
        else:
            metrics[true_label]['FN'] += 1

        for pred in preds:
            if pred != true_label:
                if pred not in true_labels.tolist():
                    metrics[pred]['FP'] += 1

    f1_scores = {}
    for label, counts in metrics.items():
        TP = counts['TP']
        FP = counts['FP']
        FN = counts['FN']

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        f1_scores[label] = f1
    return f1_scores

def evaluate(model: nn.Module, data_base: FishialDataset, validation: FishialDataset, extra_val: FishialDataset = None, save_as = None):
    total_accuracy = {}
    model.eval()

    
    database, database_labels = get_embeddings(model, data_base, metrics = ['at_k'], batch_size = 128)
    fo_validation_emb, fo_validation_emb_labels = get_embeddings(model, validation, metrics = ['at_k'], batch_size = 128)
    
    if extra_val:
        database_val, database_labels_val = get_embeddings(model, extra_val, metrics = ['at_k'], batch_size = 128)
        
        distances = pairwise_distance(database_val, database_val)
        accuracy = get_acc(distances, database_labels_val, database_labels_val, start_since = 1, save_as = save_as, title = 'extra_validation')[0]
        total_accuracy.update({
            'extra_validation':accuracy
        })
        
    
    distances = pairwise_distance(fo_validation_emb, database)
    accuracy = get_acc(distances, database_labels, fo_validation_emb_labels, start_since = 0, save_as = save_as, title = 'validation_on_database')[0]
    
    total_accuracy.update({
        'validation_on_database':accuracy
    })
    
    distances = pairwise_distance(fo_validation_emb, fo_validation_emb)
    accuracy = get_acc(distances, fo_validation_emb_labels, fo_validation_emb_labels, start_since = 1, save_as = save_as, title = 'validation_on_validation')[0]
    
    total_accuracy.update({
        'validation_on_validation':accuracy
    })
    

    return total_accuracy

def evaluate_acc(model: nn.Module, dataset: FishialDataset):
    
    model.eval()
    outputs, labels = get_embeddings(model, dataset, metrics = ['acc'])
    acc_val = accuracy(labels, outputs)

    return acc_val