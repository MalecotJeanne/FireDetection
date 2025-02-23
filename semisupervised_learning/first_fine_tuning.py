import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Définition des chemins
data_dir = "/home/maelys/ensta3A/computer_vision/valid"

# Définir des transformations pour les images (re-échelle, normalisation, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation pour ResNet50
])

# Créer une classe Dataset personnalisée pour charger les données
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
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
                    self.labels.append(1 if label == 'wildfire' else 0)  # Modification pour correspondre à 'positive'/'negative'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# Créer des ensembles d'entraînement et de validation
dataset = CustomDataset(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Charger les ensembles de données
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Charger ResNet50 pré-entrainé (sans la partie classification)
model = models.resnet50(pretrained=True)

# Geler les paramètres de ResNet50 (pas d'apprentissage sur les couches de base)
for param in model.parameters():
    param.requires_grad = False

# Modifier la dernière couche pour la classification binaire
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

# Définir l'optimiseur et la fonction de perte
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()  # Binary Cross Entropy pour classification binaire

# Fonction d'entraînement
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()  # Éliminer la dimension supplémentaire
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs >= 0.5).float()  # Seuil pour la classification binaire
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions * 100
        
        # Évaluation sur le set de validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total * 100
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    return model

# Entraîner le modèle
print("Beginning of the training")
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)

# Sauvegarde du modèle fine-tuné
# torch.save(model.state_dict(), "resnet50_finetuned.pth")

# Évaluation finale sur le set de validation
model.eval()
correct_predictions = 0
total_predictions = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).squeeze()
        predicted = (outputs >= 0.5).float()
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

final_accuracy = correct_predictions / total_predictions * 100
print(f"Final Validation Accuracy: {final_accuracy:.2f}%")
