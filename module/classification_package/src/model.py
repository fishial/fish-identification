import torch
import torch.nn as nn
import torchvision.models as models


def init_model(num_classes, embeddings = 256, backbone='resnet18', checkpoint = None, device = 'cpu'):
    if backbone == 'resnet18':
        resnet = models.resnet18(pretrained=True)
    elif backbone == 'resnet50':
        resnet = models.resnet50(pretrained=True)
    else:
        resnet = models.resnet18(pretrained=True)
    features = resnet.fc.in_features

    resnet.fc = nn.Identity()
    embedding_model = EmbeddingModel(resnet, num_classes, features, embeddings)
    if checkpoint:
        embedding_model.load_state_dict(torch.load(checkpoint))
        
    return embedding_model


class Backbone(nn.Module):
    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.resnet = resnet

    def forward(self, x: torch.Tensor):
        return self.resnet(x)


class EmbeddingModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes,  last_layer = 512, emb_dim=256):
        super().__init__()
        self.backbone = backbone
        self.embeddings = nn.Linear(last_layer, emb_dim)
        self.fc_parallel = nn.Linear(last_layer, num_classes)
        
    def forward(self, x: torch.Tensor):
        output_embedding = self.embeddings(self.backbone(x))
        output_fc = self.fc_parallel(self.backbone(x))
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