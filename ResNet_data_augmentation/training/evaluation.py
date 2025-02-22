import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from models.model import load_model
from config import DATASET_PATH, IMG_SIZE, BATCH_SIZE, BASE_DIR
import os

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

def load_datasets():
    dataset = datasets.ImageFolder(DATASET_PATH, transform=get_transforms(train=True))
    num_total = len(dataset)
    num_train = int(0.8 * num_total)
    num_val = num_total - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    val_dataset.dataset.transform = get_transforms(train=False)
    return train_dataset, val_dataset

def evaluate_model(model, device):
    _, val_dataset = load_datasets()
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_pred.extend(outputs.cpu().numpy().ravel())
            y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.show()
    
    # Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(7,5))
    plt.plot(recall, precision, label='Courbe Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Courbe Precision-Recall')
    plt.legend(loc="upper right")
    plt.show()
    
    # Matrice de confusion et rapport de classification
    y_pred_class = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_class)
    print("Matrice de confusion :")
    print(cm)
    print("\nRapport de classification :")
    print(classification_report(y_true, y_pred_class))

if __name__ == '__main__':
    # Définir device avant de charger le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Utilisation de l'appareil :", device)
    
    # Construire le chemin complet vers le modèle et charger le modèle
    model_path = os.path.join(BASE_DIR, 'saved_models', 'resnet_finetuned_final.pth')
    model = load_model(model_path, device, pretrained=False)
    
    evaluate_model(model, device)
