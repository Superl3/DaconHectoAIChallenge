import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import TResNet
from dataset import CustomImageDataset
from utils import load_config

def inference():
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_transform = None
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import PIL
        val_transform = A.Compose([
            A.LongestMaxSize(max_size=cfg['img_size'], interpolation=PIL.Image.BILINEAR),
            A.PadIfNeeded(min_height=cfg['img_size'], min_width=cfg['img_size'], border_mode=0, fill=(0,0,0)),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2()
        ])
    except ImportError:
        raise ImportError('albumentations and pillow are required for transforms')

    # Load class names from train set
    train_dataset = CustomImageDataset(cfg['train_root'], transform=None)
    class_names = train_dataset.classes

    test_dataset = CustomImageDataset(cfg['test_root'], transform=val_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False)

    model = TResNet(num_classes=cfg['num_classes'])
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            for prob in probs.cpu():
                result = {class_names[i]: prob[i].item() for i in range(len(class_names))}
                results.append(result)
    pred = pd.DataFrame(results)
    return pred, class_names

if __name__ == '__main__':
    pred, class_names = inference()
    print(pred.head())
