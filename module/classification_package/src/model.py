import torch
import torch.nn as nn
import torchvision.models as models


def init_model(config):
    if config['model']['backbone'] == 'resnet18':
        resnet = models.resnet18(pretrained=True)
    elif config['model']['backbone'] == 'resnet50':
        resnet = models.resnet50(pretrained=True)
    else:
        resnet = models.resnet18(pretrained=True)
    cnt = resnet.fc.in_features

    resnet.fc = nn.Identity()
    embedding_model = EmbeddingModel(resnet, cnt, config['model']['embeddings'])

    if config['checkpoint']:
        embedding_model.load_state_dict(torch.load(config['checkpoint']))
    return embedding_model


class Backbone(nn.Module):
    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.resnet = resnet

    def forward(self, x: torch.Tensor):
        return self.resnet(x)


class EmbeddingModel(nn.Module):
    def __init__(self, backbone: nn.Module, last_layer=512, emb_dim=128):
        super().__init__()
        self.backbone = backbone
        self.embeddings = nn.Linear(last_layer, emb_dim)
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor):
        return self.embeddings(self.backbone(x))


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