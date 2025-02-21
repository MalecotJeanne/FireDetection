import torch
import torch.nn as nn   

def get_criterion(name):
    """
    pick the criterion to use for the training.
    args:
        name: str, the name of the criterion to use.    
    """
    name = name.lower()
    supported_criterion = True
    if "mse" in name or name.lower() == "l2":
        criterion = nn.MSELoss()
    elif "crossentropy" in name or name == "ce" or name == "celoss":
        criterion = nn.CrossEntropyLoss()
    elif "binarycrossentropy" in name or name == "bce" or name == "bceloss":
        criterion = nn.BCELoss()
    elif "binarycrossentropywithlogits" in name or name == "bcewithlogits" or name == "bcelogits":
        criterion = nn.BCEWithLogitsLoss()

    else:
        supported_criterion = False
        criterion = nn.MSELoss()
    return criterion, supported_criterion