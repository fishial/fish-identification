import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class EfficientNetEmbedding(nn.Module):
    def __init__(self, model_name='efficientnet-b0', embedding_dim=128):
        super(EfficientNetEmbedding, self).__init__()
        
        # Load the pre-trained EfficientNet model
        self.efficientnet = EfficientNet.from_pretrained(model_name)
        
        # Capture the number of input features of the final layer
        in_features = self.efficientnet._fc.in_features
        
        # Remove the classification layer
        self.efficientnet._fc = nn.Identity()
        
        print(f"NUM OF INPUT FEATURES: {in_features} ")
        
        # Add a new fully connected layer for generating embeddings
        self.embedding_layer = nn.Linear(in_features, embedding_dim)
        
    def forward(self, x):
        # Extract features using EfficientNet
        features = self.efficientnet(x)
        
        # Generate embeddings
        embeddings = self.embedding_layer(features)
        
        return embeddings
    

def init_model(num_classes, embeddings = 256, backbone_name='resnet18', checkpoint_path = None, device = 'cpu'):
    if backbone_name == 'resnet18':
        backbone = models.resnet18(pretrained=True)
        
        features = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif backbone_name == 'resnet50':
        backbone = models.resnet50(pretrained=True)
        
        features = resnet.fc.in_features
        backbone.fc = nn.Identity()
    elif backbone_name == 'efficientnet-b0':
        backbone = EfficientNet.from_pretrained(backbone_name)
        features = backbone._fc.in_features
        backbone._fc = nn.Identity()
        
    elif backbone_name == 'efficientnet_v2_s':
        backbone = models.efficientnet_v2_s(pretrained = True)
        features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Identity()
    elif backbone_name == 'convnext_tiny':
        backbone = models.convnext_tiny(pretrained = True)
        features = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Identity()
    elif backbone_name == 'convnext_small':
        backbone = models.convnext_small(pretrained = True)
        features = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Identity()

    embedding_model = EmbeddingModel(backbone, num_classes, features, embeddings)
    if checkpoint_path:
        
        checkpoint = torch.load(checkpoint_path)
        model_dict = embedding_model.state_dict()
        
        for k, v in checkpoint.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                pass
            else:
                if k in model_dict:
                    print(f"Mismatch SHAPE: {k} | { model_dict[k].shape} vs {v.shape}")
                else:
                    print(f"Mismatch LAYER: {k}")

        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}

        model_dict.update(checkpoint)
        embedding_model.load_state_dict(model_dict)
        
    return embedding_model


class Backbone(nn.Module):
    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.resnet = resnet

    def forward(self, x: torch.Tensor):
        return self.resnet(x)


    
class EmbeddingModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes, last_layer, emb_dim=256):
        super().__init__()
        self.backbone = backbone
        self.embeddings = nn.Linear(last_layer, emb_dim)
        self.fc_parallel = nn.Linear(last_layer, num_classes)
        
    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        output_embedding = self.embeddings(features)
        output_fc = self.fc_parallel(features)
        return output_embedding, output_fc


class Model(nn.Module):
    # Build the model

    def __init__(self, full_model: nn.Module, loss):
        super().__init__()
        self.full_model = full_model
        self.loss = loss

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        out = self.model(x)
        loss = self.loss(out, y)
        return loss


class FcNet(nn.Module):
    def __init__(self, backbone: nn.Module, n_classes):
        super().__init__()
        self.backbone = backbone
        self.fc_1 = nn.Linear(512, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.fc_1(x)
        return x