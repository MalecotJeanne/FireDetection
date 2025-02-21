import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm

# --- 1. Préparation du modèle pour l'extraction des features ---
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        base_model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pooling(x)
        return x.view(x.size(0), -1)

model = FeatureExtractor().eval()
if torch.cuda.is_available():
    model = model.cuda()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(img_path, model):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    if torch.cuda.is_available():
        img = img.cuda()
    with torch.no_grad():
        features = model(img)
    return features.cpu().numpy().flatten()

# --- 2. Sélection des images ---
def get_random_images(directory, nb_processed):
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return files[:nb_processed]

chemin1 = '/home/maelys/ensta3A/computer_vision/train/nowildfire'
chemin2 = '/home/maelys/ensta3A/computer_vision/train/wildfire'

nb_processed = 5000
images_chemin1 = get_random_images(chemin1, nb_processed)
images_chemin2 = get_random_images(chemin2, nb_processed)

all_images = images_chemin1 + images_chemin2
true_labels = np.array([0] * len(images_chemin1) + [1] * len(images_chemin2))

# Mélanger les images
data = np.column_stack((all_images, true_labels))
np.random.seed(42)
np.random.shuffle(data)
all_images = data[:, 0]
true_labels = data[:, -1].astype(int)

# --- Validation set ---
nb_img_val = 100
images_val_chemin1 = get_random_images('/home/maelys/ensta3A/computer_vision/valid/nowildfire', nb_img_val)
images_val_chemin2 = get_random_images('/home/maelys/ensta3A/computer_vision/valid/wildfire', nb_img_val)

all_images_val = images_val_chemin1 + images_val_chemin2
true_labels_val = np.array([0] * len(images_val_chemin1) + [1] * len(images_val_chemin2))
all_images_final = list(all_images) + list(all_images_val)

# --- 3. Extraction des features ---
features = [extract_features(p, model) for p in tqdm(all_images_final)]
features = np.array(features)

# --- 4. Clustering avec KMeans ---
kmeans = KMeans(n_clusters=2, random_state=42)
predicted_clusters = kmeans.fit_predict(features)

# --- 5. Calcul de l'accuracy ---
accuracy_directe = accuracy_score(true_labels_val, predicted_clusters[-2*nb_img_val:])
accuracy_inverse = accuracy_score(true_labels_val, 1 - predicted_clusters[-2*nb_img_val:])
accuracy = max(accuracy_directe, accuracy_inverse)

print(f"Accuracy directe sur valid set : {accuracy_directe*100:.2f}%")
print(f"Accuracy inverse sur valid set : {accuracy_inverse*100:.2f}%")

if accuracy_inverse > accuracy_directe:
    predicted_clusters = 1 - predicted_clusters

predicted_clusters = predicted_clusters[:-2*nb_img_val]

centers = kmeans.cluster_centers_
distances = np.array([np.linalg.norm(feature - centers[label])
                      for feature, label in zip(features, predicted_clusters)])

# --- 5. Sélectionner les prédictions les plus fiables ---
selected_indices = []
for cluster in np.unique(predicted_clusters):
    cluster_idx = np.where(predicted_clusters == cluster)[0]
    threshold = np.quantile(distances[cluster_idx], 0.75)
    selected = cluster_idx[distances[cluster_idx] <= threshold]
    selected_indices.extend(selected)

selected_indices = np.array(selected_indices)
df = pd.DataFrame({
    'filepath': np.array(all_images)[selected_indices],
    'pseudo_label': predicted_clusters[selected_indices],
    'distance': distances[selected_indices]
})
df.to_csv("pseudo_labels_torch.csv", index=False)

print("Nombre d'images sélectionnées avec des autolabels jugés fiables :", len(df))
print("Fichier 'pseudo_labels.csv' généré avec succès.")
