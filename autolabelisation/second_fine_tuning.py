import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# 🔥 Vérification du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Entraînement sur :", device)

# 📂 Définition des chemins
csv_path = "/home/maelys/ensta3A/computer_vision/FireDectection/pseudo_labels.csv"
valid_dir = "/home/maelys/ensta3A/computer_vision/valid"
model_path = "/home/maelys/ensta3A/computer_vision/FireDectection/resnet50_finetuned.pth"

# 📌 Définition des transformations
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# 📌 Dataset pour le train
class FireDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["filepath"]
        label = self.data.iloc[idx]["pseudo_label"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32).unsqueeze(0)

# 📌 Dataset pour la validation
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

# 📂 Charger le CSV pour construire le train set
df = pd.read_csv(csv_path, header=0, delimiter=",", names=["filepath", "pseudo_label", "distance"])
df['pseudo_label'] = df['pseudo_label'].astype(int)
train_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 📌 Création des datasets
train_dataset = FireDataset(train_df, transform=transform['train'])
full_valid_dataset = ValidDataset(valid_dir, transform=transform['val'])
val_size = int(0.2 * len(full_valid_dataset))
val_size = max(val_size, 1)  # S'assure d'avoir au moins 1 échantillon
val_dataset, _ = random_split(full_valid_dataset, [val_size, len(full_valid_dataset) - val_size])

# 📌 DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 📌 Charger le modèle sauvegardé
model = models.resnet50()
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

# 🔥 Charger les poids sauvegardés
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

# 📌 Définir l'optimiseur et la fonction de perte
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 📌 Fonction d'évaluation
def evaluate_model(model, val_loader):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total * 100
    return val_loss / len(val_loader), val_acc

# 📌 Fonction d'entraînement
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=3):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # 🔄 Entraînement avec barre de progression
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        val_loss, val_acc = evaluate_model(model, val_loader)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# 📌 Entraînement du modèle
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=3)

# 📌 Sauvegarde du modèle après fine-tuning
torch.save(model.state_dict(), "./resnet50_refinetuned.pth")
