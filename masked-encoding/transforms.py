import torch
import cv2
import numpy as np
import torchvision.transforms as transforms


def training_transforms(transfo_dict):
    """
    Returns a list of transforms for training data.
    Args:
    transfo_dict: dict, configuration for data augmentation
    """
    transform_list = [transforms.ToTensor()]
    if 'resize' in transfo_dict and transfo_dict['resize']:
        transform_list.append(transforms.Resize(transfo_dict['resize']))

    #normalize (not an option)
    if 'normalize' in transfo_dict:
        mean = transfo_dict['normalize'].get('mean', [0.485, 0.456, 0.406])
        std = transfo_dict['normalize'].get('std', [0.229, 0.224, 0.225])
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    transform_list.append(transforms.Normalize(mean=mean, std=std))

    #data augmentation  
    if 'horizontal_flip' in transfo_dict and transfo_dict['horizontal_flip']:
        transform_list.append(transforms.RandomHorizontalFlip())
    if 'vertical_flip' in transfo_dict and transfo_dict['vertical_flip']:
        transform_list.append(transforms.RandomVerticalFlip())

    if 'rotation' in transfo_dict and transfo_dict['rotation']:
        transform_list.append(transforms.RandomRotation(degrees=transfo_dict['rotation']))
       
    train_transforms = transforms.Compose(transform_list)

    return train_transforms

def test_transforms(transfo_dict):
    """
    Returns a list of transforms for test data.
    Args:
    transfo_dict: dict, configuration for data augmentation
    """
    transform_list = [transforms.ToTensor()]
    if 'resize' in transfo_dict and transfo_dict['resize']:
        transform_list.append(transforms.Resize(transfo_dict['resize']))

    if 'normalize' in transfo_dict:
        mean = transfo_dict['normalize'].get('mean', [0.485, 0.456, 0.406])
        std = transfo_dict['normalize'].get('std', [0.229, 0.224, 0.225])
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    test_transforms = transforms.Compose(transform_list)
    return test_transforms