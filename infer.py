import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import CustomImageDataset
from utils import load_config
from model import get_lightning_model_from_config
from seed_utils import seed_everything

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

def infer_and_submit():
    cfg = load_config('infer_config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(cfg['seed'])
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
    lightning_model = get_lightning_model_from_config(cfg, num_classes)
    checkpoint_path = cfg['checkpoint_path']
    tta_cfg = cfg.get('tta', {})
    lightning_model.load_state_dict(torch.load(checkpoint_path, map_location=device)['state_dict'], strict=False)
    lightning_model.to(device)
    lightning_model.eval()
    results = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = tta_predict(lightning_model, images, tta_cfg, device) if tta_cfg else lightning_model(images)
            probs = F.softmax(outputs, dim=1)
            for prob in probs.cpu():
                result = {class_names[i]: prob[i].item() for i in range(len(class_names))}
                results.append(result)
    pred = pd.DataFrame(results)
    # --- Submission 생성 ---
    submission = pd.read_csv(cfg['sample_submission'], encoding='utf-8-sig')
    class_columns = submission.columns[1:]
    pred = pred[class_columns]
    submission[class_columns] = pred.values
    submission.to_csv('baseline_submission.csv', index=False, encoding='utf-8-sig')
    print('Submission file saved as baseline_submission.csv')
    return submission

if __name__ == '__main__':
    infer_and_submit()
