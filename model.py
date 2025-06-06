import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import importlib

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

def get_lightning_model_from_config(cfg, num_classes=None):
    """
    cfg['backbone']에 해당하는 백본을 models/ 폴더에서 import하여
    ClassificationLightningModule에 주입해 반환합니다.
    num_classes가 None이면 cfg['num_classes'] 사용.
    """
    backbone_name = cfg.get('backbone', 'tresnet')
    if num_classes is None:
        num_classes = cfg.get('num_classes', 1000)
    weights_path = cfg.get('pretrained_weights', None)
    # models/ 폴더에서 해당 백본 import
    try:
        backbone_module = importlib.import_module(f"models.{backbone_name}")
    except ImportError as e:
        raise ImportError(f"models/{backbone_name}.py 파일이 존재해야 합니다: {e}")
    # 백본 클래스명 규칙: {BackboneName}Backbone (예: TResNetBackbone)
    class_candidates = [attr for attr in dir(backbone_module) if attr.lower().startswith(backbone_name.lower()) and attr.lower().endswith('backbone')]
    if not class_candidates:
        raise ValueError(f"models/{backbone_name}.py에 '*Backbone' 클래스를 정의해야 합니다.")
    backbone_class = getattr(backbone_module, class_candidates[0])
    # 백본 인스턴스 생성
    backbone = backbone_class(num_classes=num_classes, weights_path=weights_path)
    backbone = backbone.to(memory_format=torch.channels_last)
    # LightningModule import
    
    lightning_model = ClassificationLightningModule(backbone, learning_rate=cfg.get('learning_rate', 1e-4))
    return lightning_model
