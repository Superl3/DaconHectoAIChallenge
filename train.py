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
from pytorch_lightning.tuner import Tuner

import cv2

torch.set_float32_matmul_precision('medium')  # Tensor Core 최적화

def clahe_lab(image, **kwargs):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return out.astype(image.dtype)

def repeat3channel(x, **kwargs):
    if x.ndim == 2:
        out = np.repeat(x[:, :, None], 3, axis=2)
    elif x.ndim == 3 and x.shape[2] == 1:
        out = np.repeat(x, 3, axis=2)
    elif x.ndim == 3 and x.shape[2] == 3:
        grey = 0.114 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.299 * x[:, :, 2]
        out = np.repeat(grey[:, :, None], 3, axis=2)
    else:
        raise ValueError(f"Unexpected image shape: {x.shape}")
    return out.astype(x.dtype)

def get_transforms(cfg):
    import PIL
    import cv2

    greyscale = cfg.get('greyscale', False)
    greyscale_p = float(cfg.get('greyscale_p', 0.3))
    clahe = cfg.get('clahe', False)
    clahe_p = float(cfg.get('clahe_p', 0.3))

    train_tfms = [
        A.LongestMaxSize(max_size=cfg['img_size'], interpolation=PIL.Image.BILINEAR),
        A.PadIfNeeded(
            min_height=cfg['crop_size'], min_width=cfg['crop_size'],
            border_mode=cv2.BORDER_CONSTANT
        ),
        A.RandomResizedCrop(size=(cfg['crop_size'], cfg['crop_size']),
                            scale=(0.85, 1.0), ratio=(1.0, 1.0), interpolation=1),
        A.Resize(cfg['crop_size'], cfg['crop_size'], interpolation=PIL.Image.BILINEAR),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            rotate=(-7, 7),
            shear=(-3, 3),
            fit_output=False,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.Perspective(scale=(0.02, 0.05), keep_size=True, fit_output=False, border_mode=cv2.BORDER_CONSTANT, p=0.1),
    ]
    if greyscale:
        train_tfms.append(A.Lambda(image=repeat3channel, p=greyscale_p))
    train_tfms.append(
        A.OneOf([
            A.Lambda(image=clahe_lab, p=clahe_p) if clahe else A.NoOp(),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.07, p=0.5),
            A.NoOp()
        ], p=0.5)
    )
    train_tfms += [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ]
    train_transform = A.Compose(train_tfms)

    val_transform = A.Compose([
        A.LongestMaxSize(max_size=cfg['crop_size'], interpolation=PIL.Image.BILINEAR),
        A.PadIfNeeded(
            min_height=cfg['crop_size'], min_width=cfg['crop_size'],
            border_mode=cv2.BORDER_CONSTANT
        ),
        #A.CenterCrop(cfg['crop_size'], cfg['crop_size']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ])
    return train_transform, val_transform

def save_augmented_samples(dataset, transform, out_dir, n_samples=100):
    import cv2
    import numpy as np
    import os

    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for i in range(len(dataset)):
        img, label = dataset[i]
        # img: torch.Tensor (C, H, W) or np.ndarray (H, W, C)
        # 원본 이미지를 transform 없이 저장
        if hasattr(dataset, 'imgs'):
            img_path = dataset.imgs[i][0]
            orig = cv2.imread(img_path)
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        else:
            orig = img
        # 증강 적용 (normalize, totensor 제외)
        aug = transform(image=orig)
        aug_img = aug['image']
        # ToTensorV2 이전이므로 np.uint8 (H, W, C)
        if isinstance(aug_img, (np.ndarray,)):
            save_img = aug_img
        else:
            save_img = aug_img.permute(1, 2, 0).cpu().numpy()
        save_img = np.clip(save_img, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f'sample_{i:04d}.png'), cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
        count += 1
        if count >= n_samples:
            break
    print(f"[INFO] Saved {count} augmented samples to {out_dir}")

def main(config_path='config.yaml'):
    cfg = load_config(config_path)
    seed_everything(cfg['seed'])
    train_transform, val_transform = get_transforms(cfg)
    datamodule = CarDataModule(cfg, train_transform, val_transform)
    datamodule.setup()
    num_classes = len(datamodule.class_names)

    lightning_model = get_lightning_model_from_config(cfg, datamodule.class_names)
    #lightning_model = torch.compile(lightning_model)
    precision_mode = '16-mixed'
    
    print("[DEBUG] Model summary:")
    # print(lightning_model)
    try:
        from torchinfo import summary
        summary(lightning_model.model, input_size=(1, 3, cfg['crop_size'], cfg['crop_size']))
    except Exception as e:
        print(f"[DEBUG] torchinfo.summary error: {e}")

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
        num_sanity_val_steps=0,
        #profiler="simple",
        gradient_clip_val=cfg.get('gradient_clip_val', 1.0),
    )
    
    # if cfg.get('auto_scale_batch_size', False):
    #     tuner = Tuner(trainer)
    #     # mode는 'power'(기본, 지수적으로 증가) 또는 'binsearch'(이진탐색) 중 선택
    #     tuner.scale_batch_size(lightning_model, datamodule=datamodule, mode='power')
    #     lr_finder = tuner.lr_find(lightning_model, datamodule=datamodule)
    #     # 추천 learning rate
    #     print(f"[LR Finder] Suggested learning rate: {lr_finder.suggestion()}")
    #     # 시각화 및 저장
    #     fig = lr_finder.plot(suggest=True)
    #     fig.savefig('lr_finder_plot.png')  # 이미지로 저장
    #     print("[LR Finder] Plot saved as lr_finder_plot.png")
    #     # 모델에 적용
    #     lightning_model.hparams.learning_rate = lr_finder.suggestion()

    # 증강 샘플 저장 (normalize/ToTensorV2 이전)
    from dataset import CustomImageDataset
    import albumentations as A
    import PIL
    import numpy as np
    # 증강 파이프라인에서 Normalize, ToTensorV2 제외
    aug_tfms = [t for t in train_transform.transforms if not isinstance(t, (A.Normalize, A.pytorch.transforms.ToTensorV2))]
    aug_transform = A.Compose(aug_tfms)
    train_dataset = CustomImageDataset(cfg['train_root'], transform=None)
    save_augmented_samples(train_dataset, aug_transform, out_dir='samples', n_samples=100)

    try:
        trainer.fit(lightning_model, datamodule=datamodule,
                    ckpt_path=cfg.get('checkpoint_path', None))
        if trainer.state.finished:
            print("[INFO] Training finished. Running inference with infer_config.yaml...")
            from utils import load_config as load_infer_config
            import sys
            infer_config_path = 'infer_config.yaml'
            if len(sys.argv) > 2:
                infer_config_path = sys.argv[2]
            infer_cfg = load_infer_config(infer_config_path)
            cfg = load_config(config_path)
            from infer import infer_and_submit
            infer_and_submit(infer_cfg)
        else:
            print("[INFO] Training did not finish (possibly interrupted). Skipping inference.")
    except KeyboardInterrupt:
        print("[INFO] Training interrupted by user. Skipping inference.")

if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    main(config_path)