import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "TResNet"))
from src.models.tresnet_v2.tresnet_v2 import TResnetL_V2 as TResnetL368

# 모델 백본 정의 (교체 가능)
class TResNetBackbone(nn.Module):
    def __init__(self, num_classes, weights_path=None):
        super().__init__()
        model_params = {'num_classes': 196}
        self.backbone = TResnetL368(model_params)
        if weights_path:
            pretrained_weights = torch.load(weights_path, map_location='cpu')
            self.backbone.load_state_dict(pretrained_weights['model'])
        self.feature_dim = self.backbone.num_features
        self.backbone.head = nn.Identity()
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

# LightningModule: 백본을 주입받아 사용
class ClassificationLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
