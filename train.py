import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn, optim
from sklearn.metrics import log_loss
import wandb
import matplotlib.pyplot as plt
from dataset import CustomImageDataset
from model import TResNet
from utils import load_config
from seed_utils import seed_everything

def get_transforms(cfg):
    import PIL
    train_transform = A.Compose([
        A.LongestMaxSize(max_size=cfg['img_size'], interpolation=PIL.Image.BILINEAR),
        A.PadIfNeeded(min_height=cfg['img_size'], min_width=cfg['img_size'], border_mode=0, fill=(0,0,0)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=cfg['img_size'], interpolation=PIL.Image.BILINEAR),
        A.PadIfNeeded(min_height=cfg['img_size'], min_width=cfg['img_size'], border_mode=0, fill=(0,0,0)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ])
    return train_transform, val_transform

def main():
    cfg = load_config()
    seed_everything(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_transform, val_transform = get_transforms(cfg)
    full_dataset = CustomImageDataset(cfg['train_root'], transform=None)
    targets = [label for _, label in full_dataset.samples]
    class_names = full_dataset.classes
    num_classes = len(class_names)
    train_idx, val_idx = train_test_split(
        range(len(targets)), test_size=0.2, stratify=targets, random_state=cfg['seed']
    )
    train_dataset = Subset(CustomImageDataset(cfg['train_root'], transform=train_transform), train_idx)
    val_dataset = Subset(CustomImageDataset(cfg['train_root'], transform=val_transform), val_idx)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False)

    # wandb
    wandb.init(entity=cfg['wandb']['entity'], project=cfg['wandb']['project'], name=cfg['wandb']['name'], config=cfg)

    model = TResNet(num_classes=num_classes, weights_path=cfg['pretrained_weights']).to(device)
    best_logloss = float('inf')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    for epoch in range(cfg['epochs']):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{cfg['epochs']}] Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{cfg['epochs']}] Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_logloss = log_loss(all_labels, all_probs, labels=list(range(num_classes)))
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_logloss": val_logloss
        })
        print(f"Train Loss : {avg_train_loss:.4f} || Valid Loss : {avg_val_loss:.4f} | Valid Accuracy : {val_accuracy:.4f}%")
        if val_logloss < best_logloss:
            best_logloss = val_logloss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved at epoch {epoch+1} (logloss: {val_logloss:.4f})")

if __name__ == '__main__':
    main()
