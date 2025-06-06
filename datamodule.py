import os
import numpy as np
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from dataset import CustomImageDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging

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
        return DataLoader(self.train_dataset, batch_size=self.cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

# --- Trainer 콜백 설정 함수 추가 ---
def get_callbacks_from_config(cfg):
    callbacks = []
    # EarlyStopping
    early_stop_callback = EarlyStopping(
        monitor=cfg.get('early_stopping_monitor', 'val_loss'),
        patience=cfg.get('early_stopping_patience', 3),
        mode=cfg.get('early_stopping_mode', 'min'),
        min_delta=cfg.get('early_stopping_min_delta', 0.0),
        verbose=True
    )
    callbacks.append(early_stop_callback)
    # ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.get('checkpoint_monitor', 'val_loss'),
        save_top_k=cfg.get('checkpoint_save_top_k', 1),
        mode=cfg.get('checkpoint_mode', 'min'),
        filename=cfg.get('checkpoint_filename', 'best_model'),
        save_last=True,
        every_n_epochs=1,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    # LearningRateMonitor
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    # SWA (Stochastic Weight Averaging)
    if cfg.get('use_swa', False):
        callbacks.append(StochasticWeightAveraging(swa_lrs=cfg.get('swa_lr', 1e-4)))
    # 기타 콜백 추가 가능
    return callbacks
