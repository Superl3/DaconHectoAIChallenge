import os
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import load_config
from seed_utils import seed_everything
from model import TResNetBackbone, ClassificationLightningModule
from datamodule import CarDataModule

torch.set_float32_matmul_precision('medium')  # Tensor Core 최적화

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
    train_transform, val_transform = get_transforms(cfg)
    datamodule = CarDataModule(cfg, train_transform, val_transform)
    datamodule.setup()
    num_classes = len(datamodule.class_names)
    model = TResNetBackbone(num_classes=num_classes, weights_path=cfg['pretrained_weights'])
    model = model.to(memory_format=torch.channels_last)
    lightning_model = ClassificationLightningModule(model, learning_rate=cfg['learning_rate'])
    #lightning_model = torch.compile(lightning_model)
    precision_mode = 'bf16-mixed' if torch.cuda.is_bf16_supported() else 16

    # --- 콜백 및 부가기능 설정 ---
    callbacks = []
    wandb_logger = WandbLogger(entity=cfg['wandb']['entity'], project=cfg['wandb']['project'], name=cfg['wandb']['name'], config=cfg)
    checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename='best_model')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks += [checkpoint, early_stop, lr_monitor]

    # SWA (Stochastic Weight Averaging) 옵션
    if cfg.get('use_swa', False):
        from pytorch_lightning.callbacks import StochasticWeightAveraging
        swa_lr = cfg.get('swa_lr', 1e-4)
        if isinstance(swa_lr, str):
            try:
                swa_lr = float(swa_lr)
            except Exception:
                raise ValueError(f"swa_lr config value '{swa_lr}' could not be converted to float.")
        callbacks.append(StochasticWeightAveraging(swa_lrs=swa_lr))
    # EMA (Exponential Moving Average) 옵션
    if cfg.get('use_ema', False):
        from pytorch_lightning.callbacks.ema import EMA
        callbacks.append(EMA(
            ema_decay=cfg.get('ema_decay', 0.999),
            validate_original_weights=False
        ))

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=wandb_logger,
        callbacks=callbacks,
        precision=precision_mode,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=10,
        accumulate_grad_batches=cfg.get('accumulate_grad_batches', 1),
    )

    # Automatic Batch Size Finder
    auto_scale_bs = cfg.get('auto_scale_batch_size', False)

    if auto_scale_bs:
        trainer.tune(lightning_model, datamodule=datamodule)
    trainer.fit(lightning_model, datamodule=datamodule)

if __name__ == '__main__':
    main()
