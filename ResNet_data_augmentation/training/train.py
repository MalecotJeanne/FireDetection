import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from config import EPOCHS_PHASE1, EPOCHS_PHASE2, LEARNING_RATE_PHASE1, LEARNING_RATE_PHASE2
from models.model import build_model, save_model
from utils import load_datasets, get_loaders

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for inputs, labels in tqdm(loader, desc="Training batches", leave=False):
        inputs = inputs.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = (outputs > 0.5).float()
        running_corrects += torch.sum(preds == labels)
        total += inputs.size(0)
    return running_loss / total, running_corrects.double() / total

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation batches", leave=False):
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            running_corrects += torch.sum(preds == labels)
            total += inputs.size(0)
    return running_loss / total, running_corrects.double() / total

def train_model(device):
    # Préparation des données
    train_dataset, val_dataset = load_datasets()
    train_loader, val_loader = get_loaders(train_dataset, val_dataset)
    
    # Phase 1 : Entraînement du classifieur (couches gelées)
    model = build_model(pretrained=True)
    model.to(device)
    
    # Geler toutes les couches sauf la dernière
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE_PHASE1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    print("Phase 1 : Entraînement du classifieur (couches gelées)")
    for epoch in range(EPOCHS_PHASE1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{EPOCHS_PHASE1} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        scheduler.step(val_loss)
    
    # Sauvegarder le modèle après la phase 1
    save_model(model, 'resnet_finetuned.pth')
    
    # Phase 2 : Fine-tuning (débloquer toutes les couches)
    print("Phase 2 : Fine-tuning")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_PHASE2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    for epoch in range(EPOCHS_PHASE2):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{EPOCHS_PHASE2} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        scheduler.step(val_loss)
    
    # Sauvegarder le modèle final
    save_model(model, 'resnet_finetuned_final.pth')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Utilisation de l'appareil :", device)
    train_model(device)
