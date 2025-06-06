import os
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import load_config
from seed_utils import seed_everything
from model import get_lightning_model_from_config
from datamodule import CarDataModule, get_callbacks_from_config

torch.set_float32_matmul_precision('medium')  # Tensor Core 최적화

def get_transforms(cfg):
    import PIL
    train_transform = A.Compose([
        A.LongestMaxSize(max_size=cfg['img_size'], interpolation=PIL.Image.BILINEAR),
        A.PadIfNeeded(min_height=cfg['img_size'], min_width=cfg['img_size'], border_mode=0, fill=(0,0,0)),
        A.RandomCrop(
        height=int(cfg['crop_size']),   # 예: 294 (368*0.8), 더 작게 하고 싶으면 0.6~0.8 추천
        width=int(cfg['crop_size']),    # 전체 면적의 60~100%에서 랜덤 크롭
        p=0.5),
        A.Resize(height=cfg['crop_size'], width=cfg['crop_size'], interpolation=PIL.Image.BILINEAR),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=cfg['crop_size'], interpolation=PIL.Image.BILINEAR),
        A.PadIfNeeded(min_height=cfg['crop_size'], min_width=cfg['crop_size'], border_mode=0, fill=(0,0,0)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ])
    return train_transform, val_transform

def main():
    cfg = load_config()
    seed_everything(cfg['seed'])
    train_transform, val_transform = get_transforms(cfg)
    datamodule = CarDataModule(cfg, train_transform, val_transform)
    datamodule.setup()
    num_classes = len(datamodule.class_names)
   
    lightning_model = get_lightning_model_from_config(cfg, num_classes)
    #lightning_model = torch.compile(lightning_model)
    precision_mode = '16-mixed'

    # --- 콜백 및 부가기능 설정 ---
    callbacks = get_callbacks_from_config(cfg)
    wandb_logger = WandbLogger(entity=cfg['wandb']['entity'], project=cfg['wandb']['project'], name=cfg['wandb']['name'], config=cfg)

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=wandb_logger,
        callbacks=callbacks,
        precision=precision_mode,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=10,
        accumulate_grad_batches=cfg.get('accumulate_grad_batches', 1),
        #num_sanity_val_steps=0
    )

    trainer.fit(lightning_model, datamodule=datamodule)

if __name__ == '__main__':
    main()
