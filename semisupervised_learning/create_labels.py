import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np
from tqdm import tqdm
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "/home/maelys/ensta3A/computer_vision/FireDectection/resnet50_finetuned.pth"
valid_dir = "/home/maelys/ensta3A/computer_vision/valid"
train_dir = "/home/maelys/ensta3A/computer_vision/train"
nonlabeled_size = 4000 # number of images used for label prediction


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) # normalisation of images using the mean and std of the images of ImageNet

# Dataset for validation
class ValidDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(1 if label == 'wildfire' else 0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32).unsqueeze(0)

# Dataset with nonlabeled images :
class NonLabeledDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(1 if label == 'wildfire' else 0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path


# Datasets creation :
full_valid_dataset = ValidDataset(valid_dir, transform=transform)
val_size = int(0.2 * len(full_valid_dataset))
val_size = max(val_size, 1)
val_dataset, _ = random_split(full_valid_dataset, [val_size, len(full_valid_dataset) - val_size])

full_nonlabeled_dataset = NonLabeledDataset(train_dir, transform=transform)
nonlabeled_size = max(nonlabeled_size, 1)
nonlabeled_dataset, _ = random_split(full_nonlabeled_dataset, [nonlabeled_size, len(full_nonlabeled_dataset) - nonlabeled_size])

# DataLoaders
batch_size = 32
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
nonlabeled_loader = DataLoader(nonlabeled_dataset, batch_size=batch_size, shuffle=True)

def evaluate_model(model, val_loader):
    """
    Evaluate the accuracy on the valdation set
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total * 100
    return val_acc

def create_labels(model, nonlabeled_loader, output_csv="test.csv"):
    """
    Create a csv file with two columns : 
    - the image's paths
    - the associated predictions
    This file will be used the refine tune the model
    """
    model.eval()
    results = []
    with torch.no_grad():
        for images, img_paths in tqdm(nonlabeled_loader):
            outputs = model(images)
            preds = (outputs > 0.5).float().cpu().numpy()
            for img_paths, preds in zip(img_paths, preds):
                results.append([img_paths, preds.item()])
    
    # Écriture des résultats dans un fichier CSV
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "prediction"])
        writer.writerows(results)
    

model = models.resnet50()
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

# Save the weights of the model :
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

# val_acc = evaluate_model(model, val_loader)
# print("Accuracy sur l'ensemble de validation :", val_acc)
create_labels(model, nonlabeled_loader, output_csv="predicted_labels_by_teacher_model.csv")
