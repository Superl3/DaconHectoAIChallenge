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
        # train/val batch_size, num_workers 분리 지원
        self._train_batch_size = cfg.get('train_batch_size', cfg['batch_size'])
        self._val_batch_size = cfg.get('val_batch_size', cfg['batch_size'])
        self._train_num_workers = cfg.get('train_num_workers', cfg.get('num_workers', 8))
        self._val_num_workers = cfg.get('val_num_workers', cfg.get('num_workers', 8))

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
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            shuffle=True,
            num_workers=self._train_num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self._val_batch_size,
            shuffle=False,
            num_workers=self._val_num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    @property
    def batch_size(self):
        return self._train_batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._train_batch_size = value
        self.cfg['batch_size'] = value

    @property
    def val_batch_size(self):
        return self._val_batch_size

    @val_batch_size.setter
    def val_batch_size(self, value):
        self._val_batch_size = value
        self.cfg['val_batch_size'] = value

    @property
    def train_num_workers(self):
        return self._train_num_workers

    @property
    def val_num_workers(self):
        return self._val_num_workers

# --- Trainer 콜백 설정 함수 추가 ---
def get_callbacks_from_config(cfg):
    callbacks = []
    # EarlyStopping
    early_stop_callback = EarlyStopping(
        monitor=cfg.get('early_stopping_monitor', 'val_logloss'),
        patience=cfg.get('early_stopping_patience', 3),
        mode=cfg.get('early_stopping_mode', 'min'),
        min_delta=cfg.get('early_stopping_min_delta', 0.0),
        verbose=True
    )
    callbacks.append(early_stop_callback)
    # ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.get('checkpoint_dir', 'checkpoints'),
        monitor=cfg.get('checkpoint_monitor', 'val_logloss'),
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
