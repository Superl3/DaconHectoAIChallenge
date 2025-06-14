import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import CustomImageDataset
from utils import load_config
from model import get_lightning_model_from_config, ClassificationLightningModule
from seed_utils import seed_everything
from tqdm import tqdm

torch.set_float32_matmul_precision('medium')  # Tensor Core 최적화

def tta_predict(model, images, tta_cfg, device):
    preds = []
    scales = tta_cfg.get('multiscale', [images.shape[-1]])
    do_flip = tta_cfg.get('flip', False)
    for scale in scales:
        # Resize if needed
        if images.shape[-1] != scale:
            images_resized = torch.nn.functional.interpolate(images, size=(scale, scale), mode='bilinear', align_corners=False)
        else:
            images_resized = images
        # Original
        preds.append(model(images_resized.to(device)))
        # Flip
        if do_flip:
            flipped = torch.flip(images_resized, dims=[3])
            preds.append(model(flipped.to(device)))
    return torch.mean(torch.stack(preds), dim=0)

def prepare_datasets(cfg):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import PIL
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=cfg['crop_size'], interpolation=PIL.Image.BILINEAR),
        A.PadIfNeeded(min_height=cfg['crop_size'], min_width=cfg['crop_size'], border_mode=0, fill=(0,0,0)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ])
    train_dataset = CustomImageDataset(cfg['train_root'], transform=None)
    class_names = train_dataset.classes
    num_classes = len(class_names)
    test_dataset = CustomImageDataset(cfg['test_root'], transform=val_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg.get('num_workers', 4), pin_memory=True)
    return class_names, num_classes, test_loader

def load_model_for_inference(cfg, num_classes, device):
    import importlib
    backbone_name = cfg.get('backbone', 'tresnet')
    weights_path = cfg['checkpoint_path']
    ext = os.path.splitext(weights_path)[-1].lower()
    model = None
    backbone = None
    if ext == '.ckpt':
        try:
            lightning_model = get_lightning_model_from_config(cfg, num_classes=num_classes)
            lightning_model = ClassificationLightningModule.load_from_checkpoint(weights_path, model=lightning_model.model, cfg=cfg)
            model = lightning_model.model.to(device)
            print(f"[INFO] Loaded LightningModule from {weights_path} (ckpt)")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load LightningModule from .ckpt: {e}")
    elif ext in ['.pth', '.bin']:
        try:
            lightning_model = get_lightning_model_from_config(cfg, num_classes=num_classes)
            state = torch.load(weights_path, map_location=device)
            if 'state_dict' in state:
                state = state['state_dict']
            lightning_model.load_state_dict(state, strict=False)
            model = lightning_model.model.to(device)
            print(f"[INFO] Loaded weights into LightningModule from {weights_path} (pth/bin)")
        except Exception as e:
            print(f"[WARN] LightningModule load failed: {e}\nTrying to load backbone only...")
            try:
                backbone_module = importlib.import_module(f"models.{backbone_name}")
                class_candidates = [attr for attr in dir(backbone_module) if attr.lower().startswith(backbone_name.lower()) and attr.lower().endswith('backbone')]
                if not class_candidates:
                    raise ValueError(f"models/{backbone_name}.py에 '*Backbone' 클래스를 정의해야 합니다.")
                backbone_class = getattr(backbone_module, class_candidates[0])
                backbone = backbone_class(num_classes=num_classes, weights_path=None)
                backbone = backbone.to(device)
                state = torch.load(weights_path, map_location=device)
                if 'state_dict' in state:
                    state = {k.replace('model.', ''): v for k, v in state['state_dict'].items() if k.startswith('model.')}
                backbone.load_state_dict(state, strict=False)
                model = backbone
                print(f"[INFO] Loaded weights into backbone only from {weights_path} (pth/bin)")
            except Exception as e2:
                raise RuntimeError(f"[ERROR] Failed to load both LightningModule and backbone from {weights_path}: {e2}")
    else:
        raise ValueError(f"Unknown checkpoint extension: {ext}")
    model.eval()
    return model

def run_inference(model, test_loader, class_names, tta_cfg, device):
    import torch.nn.functional as F
    print("[INFO] Inference started...")
    results = []
    with torch.inference_mode():
        for images in tqdm(test_loader, desc="[Inference] Batches", unit="batch"):
            images = images.to(device)
            outputs = tta_predict(model, images, tta_cfg, device) if tta_cfg else model(images)
            probs = F.softmax(outputs, dim=1)
            for prob in probs.cpu():
                result = {class_names[i]: prob[i].item() for i in range(len(class_names))}
                results.append(result)
    print(f"[INFO] Inference completed. Total samples: {len(results)}")
    return results

def save_submission(results, cfg):
    import pandas as pd
    submission = pd.read_csv(cfg['sample_submission'], encoding='utf-8-sig')
    class_columns = submission.columns[1:]
    pred = pd.DataFrame(results)
    pred = pred[class_columns]
    submission[class_columns] = pred.values
    submission.to_csv('baseline_submission.csv', index=False, encoding='utf-8-sig')
    print('Submission file saved as baseline_submission.csv')
    return submission

def infer_and_submit(cfg):
    print("[STEP 1] Setting seed...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(cfg['seed'])
    print("[STEP 2] Preparing datasets...")
    class_names, num_classes, test_loader = prepare_datasets(cfg)
    print(f"[INFO] Number of classes: {num_classes}, Test batches: {len(test_loader)}")
    print("[STEP 3] Loading model...")
    model = load_model_for_inference(cfg, num_classes, device)
    print("[STEP 4] Running inference...")
    tta_cfg = cfg.get('tta', {})
    results = run_inference(model, test_loader, class_names, tta_cfg, device)
    print("[STEP 5] Saving submission...")
    submission = save_submission(results, cfg)
    print("[ALL DONE] Inference and submission complete.")
    return submission

if __name__ == '__main__':
    import sys
    from utils import load_config
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'infer_config.yaml'
    infer_cfg = load_config(config_path)
    infer_and_submit(infer_cfg)
