import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.transforms import Normalize

class FireDataset(Dataset):
    """
    Custom dataset for wildfire detection
    Args:
    root_dir: str, root directory of the dataset
    dataset: str, either "train", "valid" or "test"
    transform: torch transform, data augmentation to apply
    config_mask: dict, configuration for mask creation
    """
    def __init__(self, root_dir, dataset = "valid", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        dataset = "valid" if "val" in dataset else dataset

        self.images = []  
        self.labels = None if dataset == "train" else []

        for label in os.listdir(os.path.join(root_dir, dataset)):
            for file in os.listdir(os.path.join(root_dir, dataset, label)):
                img_path =os.path.join(root_dir, dataset, label, file)
                self.images.append(img_path)
                if dataset == "test" or dataset == "valid": 
                    self.labels.append(1 if label == "wildfire" else 0) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        if self.labels:
            return img, torch.tensor(self.labels[idx])
        
        return img
    
    def get_mini_set(self, n):
        """
        Create a mini dataset of n samples.
        """
        indices = np.random.choice(len(self), n, replace=False)
        mini_set = torch.utils.data.Subset(self, indices)
        return mini_set

        
# Patches dataset

class Patches(Dataset):
    def __init__(self, images_dataset, n_patches):
        """
        Initialize a Patches dataset class with the given number of patches per dimension.
        Args:
        n_patches: Number of patches per dimension.
        """
        self.n_patches = n_patches
        self.images_dataset = images_dataset
        
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images_dataset)

    def __getitem__(self, idx):
        """
        Get the image and its patches at the given index.

        Args:
        idx: Index of the image to retrieve.

        Returns:
        A tuple containing the image and its patches.
        """
        batch = self.images_dataset[idx]
        if len(batch) == 2:
            image, labels = batch
        else:
            image = batch

        patches = self.create_patches(image.unsqueeze(0), self.n_patches)
        if len(batch) == 2:
            return image, patches, labels

        return image, patches

    def show(self, idx):
        """
        Display the image and its patches at the given index.

        Args:
        idx: Index of the image to display.
        """
        image, patches = self[idx]
        mean, std = self.get_norm_params()

        #unnormalize the image and patches if applicable
        if mean is not None and std is not None:
            image = image * std + mean
            image = image.clamp(0, 1)

            patches = patches * std + mean
            patches = patches.clamp(0, 1)

        n_patches = self.n_patches

        fig = plt.figure(figsize=(8, 4))
        subfigs = fig.subfigures(1, 2, wspace=0.07)

        ax1 = subfigs[0].subplots(1, 1)
        ax1.imshow(image.permute(1, 2, 0))
        ax1.axis("off")

        ax2 = subfigs[1].subplots(n_patches, n_patches)
        for x in range(n_patches):
            for y in range(n_patches):
                ax2[x, y].imshow(patches[x * n_patches + y].permute(1, 2, 0))
                ax2[x, y].axis("off")
        plt.show()

    def show_random(self, nb):
        """
        Display nb random images and their patches.
        """
        rd_idx = np.random.choice(self.n_patches, nb, replace=False)
        for i in rd_idx:
            self.show(i)

    @staticmethod
    def create_patches(images, n_patches):
        """
        Create non-overlapping patches from an image (tensor).

        Args:
        images: Tensor of images.
        n_patches: Number of patches per dimension.

        Returns:
        Tensor of patches.
        """
        h, w = images.shape[-2:]
        patch_size = (h // n_patches, w // n_patches)
        patches = []

        for i in range(n_patches):
            for j in range(n_patches):
                patch = images[:, :, i * patch_size[0]:(i + 1) * patch_size[0], j * patch_size[1]:(j + 1) * patch_size[1]]
                patches.append(patch)

        return torch.cat(patches)

    def get_norm_params(self):
        transforms = self.images_dataset.transform.transforms
        for transform in transforms:
            if isinstance(transform, Normalize):
                mean_list = transform.mean
                std_list = transform.std
                mean = torch.tensor(mean_list).view(3, 1, 1)
                std = torch.tensor(std_list).view(3, 1, 1)
                return mean, std
        
        return None, None
    
   