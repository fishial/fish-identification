import logging
import time
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional
import math

import faiss
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny

from PIL import Image
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
from torchvision import transforms

# Third-party libraries for model building (timm)
import timm
from timm.models.vision_transformer import VisionTransformer # For type hinting


# Setup Logger
logger = logging.getLogger("EmbeddingClassifier")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@dataclass
class PredictionResult:
    name: str
    species_id: int
    distance: float
    accuracy: float
    image_id: int
    annotation_id: int
    drawn_fish_id: int

        
class EmbeddingClassifier:
    def __init__(self, config: Dict):
        logger.setLevel(getattr(logging, config.get('log_level', 'INFO').upper()))
        self._load_data(config["dataset"]["path"])

        self.dim = self.db_embeddings.shape[1]
        self._prepare_centroids()

        logger.info("Initializing EmbeddingClassifier...")

        self.device = config["model"].get("device", "cpu")
        self._load_model(config["model"]["path"])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Create ID to label mapping
        self.id_to_label = {internal_id: self.keys[internal_id]['label'] for internal_id in self.keys}
        
        logger.info("EmbeddingClassifier initialized successfully.")

    def _load_model(self, checkpoint_path: str):
        self.model = StableEmbeddingModel(embedding_dim=512, num_classes=639)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Torch model loaded from {checkpoint_path}")
        return self.model

    def _load_data(self, dataset_path: str):
        data = torch.load(dataset_path)
        self.db_embeddings = data['embeddings'].numpy().astype("float32")
        self.db_labels = np.array(data['labels'])
        self.image_ids = data['image_id']
        self.annotation_ids = data['annotation_id']
        self.drawn_fish_ids = data['drawn_fish_id']
        self.keys = data['labels_keys']
        self.label_to_species_id = {
            v['label']: v['species_id'] for v in self.keys.values()
        }
        logger.info(f"Dataset loaded from {dataset_path}")

    def __call__(self, img: Union[np.ndarray, List[np.ndarray]]):
        if isinstance(img, np.ndarray):
            return self.inference_numpy(img)
        elif isinstance(img, list) and all(isinstance(i, np.ndarray) for i in img):
            return self.inference_numpy_batch(img)
        else:
            raise TypeError("Input must be np.ndarray or List[np.ndarray].")

    def inference_numpy(self, img: np.ndarray):
        tensor = self.transform(Image.fromarray(img)).unsqueeze(0).to(self.device)
        return self._inference_batch_tensor(tensor)[0]

    def inference_numpy_batch(self, imgs: List[np.ndarray]):
        tensors = torch.stack([self.transform(Image.fromarray(img)) for img in imgs]).to(self.device)
        return self._inference_batch_tensor(tensors)

    def _inference_batch_tensor(self, tensors: torch.Tensor):
        with torch.no_grad():
            embeddings, archead_logits, _ = self.model(tensors)
            
        top_probabilities, top_indices = torch.topk(archead_logits, 5)
        
        output = self.get_top_neighbors_from_embeddings(embeddings)
        logger.debug(f"Inference output: {output}")

        for i, (item, arc_idx) in enumerate(zip(output, top_indices)):
            
            for num_pred, pred_by_arc_head_id in enumerate(arc_idx):
                
                arc_label = self.id_to_label[pred_by_arc_head_id.item()]
                score = top_probabilities[i][num_pred]
                
#                 print(f"Label: {arc_label} - {score}")

                if arc_label not in item and score > 0.1:
                    item[arc_label] = {'index': None, 'similarity': round(score.item(), 3), 'times': 1}

        return self._postprocess(output)

    def _postprocess(self, class_results) -> List[PredictionResult]:
        results = []
        for single_fish in class_results:
            fish_results = []
            for label, data in single_fish.items():
                index = data["index"]
                fish_results.append(PredictionResult(
                    name=label,
                    species_id=self.label_to_species_id[label],
                    distance=data['similarity'],
                    accuracy=data['similarity'] / data['times'],
                    image_id=self.image_ids[index] if index is not None else None,
                    annotation_id=self.annotation_ids[index] if index is not None else None,
                    drawn_fish_id=self.drawn_fish_ids[index] if index is not None else None,
                ))
            results.append(fish_results)
        return results

    def _prepare_centroids(self):
        unique_labels = np.unique(self.db_labels)
        self.label_to_centroid = {}
        for label in unique_labels:
            class_embs = self.db_embeddings[self.db_labels == label]
            centroid = np.mean(class_embs, axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-10)
            self.label_to_centroid[label] = centroid

        self.centroid_matrix = np.stack([self.label_to_centroid[label] for label in self.label_to_centroid])
        self.centroid_labels = list(self.label_to_centroid.keys())

    def get_top_neighbors_from_embeddings(
        self,
        query_embeddings: Union[np.ndarray, torch.Tensor],
        topk_centroid: int = 5,
        topk_neighbors: int = 10,
        centroid_threshold: float = 0.7,
        neighbor_threshold: float = 0.8
    ) -> List[Dict[str, Dict[str, Union[float, int, None]]]]:
        start_time = time.time()
        logger.info(f"Starting search over {len(query_embeddings)} embeddings")

        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.cpu().numpy().astype("float32")

        results = []
        for query_emb in query_embeddings:
            centroid_sims = 1.0 - pairwise_distances(query_emb.reshape(1, -1), self.centroid_matrix, metric='cosine')[0]
            top_centroid_indices = np.argsort(-centroid_sims)[:topk_centroid]

            centroid_scores = {
                self.centroid_labels[idx]: centroid_sims[idx]
                for idx in top_centroid_indices if centroid_sims[idx] >= centroid_threshold
            }
            selected_classes = set(centroid_scores.keys())

            if not selected_classes:
                results.append({})
                continue

            class_mask = np.isin(self.db_labels, list(selected_classes))
            selected_embeddings = self.db_embeddings[class_mask]
            selected_labels = self.db_labels[class_mask]
            selected_indices = np.where(class_mask)[0]

            if len(selected_embeddings) == 0:
                results.append({"top_neighbors": [], "centroid_scores": centroid_scores})
                continue

            faiss_index = faiss.IndexFlatIP(self.dim)
            faiss_index.add(selected_embeddings)
            distances, indices = faiss_index.search(query_emb.reshape(1, -1), min(topk_neighbors, len(selected_embeddings)))

            score_map = defaultdict(lambda: {'index': None, 'similarity': 0.0, 'times': 0})
            for rank, idx in enumerate(indices[0]):
                label = selected_labels[idx]
                sim = distances[0][rank]
                original_idx = selected_indices[idx]
                if sim >= neighbor_threshold:
                    score_map[label]['similarity'] += sim
                    score_map[label]['times'] += 1
                    if score_map[label]['index'] is None:
                        score_map[label]['index'] = original_idx

            for label, sim in centroid_scores.items():
                if label not in score_map:
                    score_map[label] = {'index': None, 'similarity': 0.1, 'times': 1}

            results.append(score_map)

        logger.info(f"Completed in {time.time() - start_time:.2f} seconds")
        return results
    

# --- Attention Pooling for Vision Transformer ---
class ViTAttentionPooling(nn.Module):
    """
    Attention Pooling for Vision Transformer output [B, N, D].
    Calculates a weighted sum of patch embeddings.
    """
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        if hidden_features is None:
            hidden_features = max(in_features // 4, 128) # Reasonable default value

        self.attention_net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(), # Using Tanh for better gradient stability
            nn.Linear(hidden_features, 1)
        )

    def forward(self, x, object_mask=None, return_attention_map=False):
        """
        Args:
            x (torch.Tensor): ViT output in [Batch, Num_Tokens, Embedding_Dim] format.
            object_mask: Mask for ViT is not implemented in this version, as it
                         requires complex logic to map pixel masks to patches.
                         The parameter is kept for interface compatibility.
            return_attention_map (bool): Whether to return the attention map.
        """
        # x shape: [B, N, D]
        # Calculate attention weights for each token
        attention_scores = self.attention_net(x) # -> [B, N, 1]
        # Apply softmax over the sequence of tokens so weights sum to 1
        weights = F.softmax(attention_scores, dim=1) # -> [B, N, 1]

        # Weighted sum of token embeddings
        # (B, N, D) * (B, N, 1) -> (B, N, D) -> sum over N -> (B, D)
        pooled = (x * weights).sum(dim=1)

        if return_attention_map:
            # Return weights which can be used for visualization
            return pooled, weights
        else:
            return pooled


# --- ArcFace Head ---
class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, s=32.0, m=0.10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.register_buffer('cos_m', torch.tensor(math.cos(m)))
        self.register_buffer('sin_m', torch.tensor(math.sin(m)))
        self.register_buffer('th', torch.tensor(math.cos(math.pi - m)))
        self.register_buffer('mm', torch.tensor(math.sin(math.pi - m) * m))
        self.register_buffer('eps', torch.tensor(1e-7))

    def set_margin(self, new_m: float):
        self.m = new_m
        self.cos_m.data = torch.tensor(math.cos(new_m), device=self.cos_m.device)
        self.sin_m.data = torch.tensor(math.sin(new_m), device=self.sin_m.device)
        self.th.data = torch.tensor(math.cos(math.pi - new_m), device=self.th.device)
        self.mm.data = torch.tensor(math.sin(math.pi - new_m) * new_m, device=self.mm.device)

    def forward(self, normalized_emb, labels=None):
        normalized_w = F.normalize(self.weight, dim=1)
        cosine = F.linear(normalized_emb, normalized_w)

        if labels is not None:
            cosine_sq = cosine ** 2
            sine = torch.sqrt((1.0 - cosine_sq).clamp(min=self.eps.item()))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            one_hot = F.one_hot(labels, num_classes=self.num_classes).float().to(cosine.device)
            output = one_hot * phi + (1.0 - one_hot) * cosine
            output *= self.s
        else:
            output = cosine * self.s

        return output


class StableEmbeddingModel(nn.Module):
    def __init__(self,
                 embedding_dim=512, # Increased by default for ViT
                 num_classes=639,
                 pretrained_backbone=True,
                 freeze_backbone_initially=False,
                 backbone_model_name='beitv2_base_patch16_224.in1k_ft_in22k_in1k', # Popular ViT model
                 custom_backbone: Optional[VisionTransformer]=None, # To pass a custom backbone
                 attention_hidden_channels=None, # For ViT pooling, this is `hidden_features`
                 arcface_s=64.0,
                 arcface_m=0.5,
                 add_bn_to_embedding=False,
                 embedding_dropout_rate=0.11): # Slightly increased dropout
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        print(f"Loading ViT backbone: {backbone_model_name}")
        self.backbone: VisionTransformer = timm.create_model(
            backbone_model_name,
            pretrained=pretrained_backbone,
            num_classes=0 # Important: disable the classification head
        )
        # Output feature size for ViT
        self.backbone_out_features = self.backbone.embed_dim
            
        # ViT doesn't have `.features`, extraction happens via forward_features
        # We simply call self.backbone as a function, but need to handle the output
        self.backbone_feature_extractor = self.backbone.forward_features
        
        self.pooling = ViTAttentionPooling(in_features=self.backbone_out_features,
                                           hidden_features=attention_hidden_channels)

        embedding_layers = [nn.Linear(self.backbone_out_features, embedding_dim)]
#         embedding_layers.append(nn.BatchNorm1d(embedding_dim))
            
        self.embedding_fc = nn.Sequential(*embedding_layers)
        self.arcface_head = ArcFaceHead(embedding_dim, num_classes, s=arcface_s, m=arcface_m)

        print(f"StableEmbeddingModel initialized with ViT backbone: {backbone_model_name if not custom_backbone else 'custom'}")
        print(f"  Embedding Dim: {embedding_dim}, Num Classes: {num_classes}")
        print(f"  ArcFace s: {arcface_s}, m: {arcface_m}")
        print(f"  Backbone out features (ViT embed_dim): {self.backbone_out_features}")
        print(f"  BN in embedding: {add_bn_to_embedding}, Dropout in embedding: {embedding_dropout_rate}")

    def forward(self, x, labels=None, object_mask=None, return_attention_map=False):
        """
        Forward pass of the model.
        `object_mask` is ignored in this ViT version but kept for compatibility.
        """
        # features shape: [B, Num_Tokens, Embedding_Dim]
        features = self.backbone_feature_extractor(x)
        
        # Remove CLS token before pooling, if it exists, as attention should work on image patches.
        # In standard timm ViTs, the CLS token comes first.
        if hasattr(self.backbone, 'cls_token'):
            patch_tokens = features[:, 1:, :]
        else:
            patch_tokens = features

        pooled, attn_map = self.pooling(patch_tokens, object_mask=object_mask, return_attention_map=True)
        
        emb_raw = self.embedding_fc(pooled)
        emb_norm = F.normalize(emb_raw, p=2, dim=1)
        logits = self.arcface_head(emb_norm, labels)
        
        # For visualization, we need to reshape attn_map [B, N, 1] to a 2D map.
        # This requires knowledge of the patch grid dimensions.
        grid_size = self.backbone.patch_embed.grid_size
        B, N, _ = attn_map.shape
        # Check that the number of tokens matches the grid size
        if N == grid_size[0] * grid_size[1]:
            vis_attn_map = attn_map.permute(0, 2, 1).reshape(B, 1, grid_size[0], grid_size[1])
        else:
            # If something went wrong, return None for the map
            vis_attn_map = None
            
        probabilities = F.softmax(logits, dim=1)
        return emb_norm, probabilities, vis_attn_map if return_attention_map else None
        
