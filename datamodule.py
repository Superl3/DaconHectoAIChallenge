import os
import numpy as np
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from dataset import CustomImageDataset

class CarDataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_transform, val_transform):
        super().__init__()
        self.cfg = cfg
        self.train_transform = train_transform
        self.val_transform = val_transform

    def setup(self, stage=None):
        full_dataset = CustomImageDataset(self.cfg['train_root'], transform=None)
        targets = [label for _, label in full_dataset.samples]
        self.class_names = full_dataset.classes
        train_idx, val_idx = train_test_split(
            range(len(targets)), test_size=0.2, stratify=targets, random_state=self.cfg['seed']
        )
        self.train_dataset = Subset(CustomImageDataset(self.cfg['train_root'], transform=self.train_transform), train_idx)
        self.val_dataset = Subset(CustomImageDataset(self.cfg['train_root'], transform=self.val_transform), val_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
