import torch
import torch.nn as nn
from torchvision import models

def build_model(pretrained=True):
    # Si vous faites du finetuning, vous pouvez passer pretrained=True pour charger les poids d'ImageNet
    model = models.resnet50(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    # Pour une classification binaire, on remplace la dernière couche par une couche linéaire suivie d'une Sigmoid
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Modèle sauvegardé dans {path}")

def load_model(path, device, pretrained=False):
    model = build_model(pretrained=pretrained)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Modèle chargé depuis {path}")
    return model
