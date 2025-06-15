import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import importlib
import os
import utils
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm




# LightningModule: 백본을 주입받아 사용
class ClassificationLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, class_names=None, cfg=None):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])
        self.cfg = cfg or {}
        self.automatic_optimization = True
        self.label_smoothing = float(self.cfg.get('label_smoothing', 0.05))
        self.criterion = self.CrossEntropyLoss
        self.class_names = class_names if class_names is not None else []
    def forward(self, x):
        return self.model(x)

    def CrossEntropyLoss(self, x, y, reduction='mean'):
        # x: (B, num_classes), y: (B,)
        # reduction 파라미터를 추가하여, 평균 loss와 샘플별 loss를 모두 계산할 수 있도록 합니다.
        return F.cross_entropy(x, y, label_smoothing=self.label_smoothing, reduction=reduction)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        if self.cfg.get('Img_Mix', False):
            mixmethod_name = self.cfg.get('mixmethod', 'snapmix')
            
            # getattr을 사용하여 utils 모듈에서 문자열 이름으로 함수를 동적으로 가져옵니다.
            mix_fn = getattr(utils, mixmethod_name)
            images, label_a, label_b, lam_a, lam_b = mix_fn(images, labels, self.cfg, self.model)
            outputs = self(images)
            loss_a = self.criterion(outputs, label_a, reduction='none')
            loss_b = self.criterion(outputs, label_b, reduction='none')
            loss = torch.mean(loss_a* lam_a + loss_b* lam_b)
        else:
            outputs = self(images)
            loss = self.criterion(outputs, labels)
        # NaN/Inf 체크 및 상세 로깅
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[NaN/Inf DETECTED] batch_idx={batch_idx}")
            print(f"images.shape: {images.shape}, labels.shape: {labels.shape}")
            print(f"images.min: {images.min().item()}, images.max: {images.max().item()}")
            print(f"labels.min: {labels.min().item()}, labels.max: {labels.max().item()}")
            print(f"outputs.min: {outputs.min().item()}, outputs.max: {outputs.max().item()}")
            print(f"loss: {loss}")
            raise ValueError('NaN/Inf detected in loss!')
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        #print(f"[Train] Epoch={self.current_epoch} Batch={batch_idx} Loss={loss.item():.6f} Acc={acc.item():.4f}")
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels).mean()
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[NaN/Inf DETECTED - VAL] batch_idx={batch_idx}")
            print(f"logits.min: {logits.min().item()}, logits.max: {logits.max().item()}")
            print(f"loss: {loss}")
            raise ValueError('NaN/Inf detected in val loss!')
        acc = (logits.argmax(dim=1) == labels).float().mean()
        probs = torch.softmax(logits, dim=1)
        # outputs를 인스턴스 변수에 저장 (Lightning 2.x 권장)
        if not hasattr(self, '_val_outputs'):
            self._val_outputs = []
        self._val_outputs.append({'probs': probs.detach().cpu(), 'labels': labels.detach().cpu()})
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        # Lightning 2.x: outputs를 인스턴스 변수에서 꺼내서 사용
        if not hasattr(self, '_val_outputs') or not self._val_outputs:
            return
        all_probs = torch.cat([o['probs'] for o in self._val_outputs], dim=0).numpy()
        all_labels = torch.cat([o['labels'] for o in self._val_outputs], dim=0).numpy()
        class_list = self.class_names if hasattr(self, 'class_names') else [str(i) for i in range(all_probs.shape[1])]
        import pandas as pd
        import numpy as np
        from sklearn.metrics import log_loss
        submission_df = pd.DataFrame(all_probs, columns=class_list)
        submission_df['ID'] = np.arange(len(submission_df))
        answer_df = pd.DataFrame({'ID': np.arange(len(all_labels)), 'label': [class_list[l] for l in all_labels]})
        try:
            probs = submission_df[class_list].values
            probs = probs / probs.sum(axis=1, keepdims=True)
            y_pred = np.clip(probs, 1e-15, 1 - 1e-15)
            true_labels = answer_df['label'].tolist()
            true_idx = [class_list.index(lbl) for lbl in true_labels]
            logloss = log_loss(true_idx, y_pred, labels=list(range(len(class_list))))
        except Exception as e:
            print(f"[WARN] logloss 계산 실패: {e}")
            logloss = float('nan')
        self.log('val_logloss', logloss, prog_bar=True, logger=True)
        # outputs 초기화
        self._val_outputs = []

    def configure_optimizers(self):
        cfg = self.cfg if self.cfg is not None else {}
        optimizer_name = cfg.get('optimizer', 'adamw').lower()
        scheduler_name = cfg.get('scheduler', 'cosine').lower()
        lr = cfg.get('learning_rate', 1e-4)
        wd = float(cfg.get('weight_decay', 1e-4))
        # Optimizer 선택
        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        # Scheduler 선택
        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
            scheduler_dict = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        elif scheduler_name == 'step':
            step_size = int(cfg.get('step_size', 5))
            gamma = float(cfg.get('gamma', 0.2))
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            scheduler_dict = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        elif scheduler_name == 'reduce_on_plateau':
            patience = int(cfg.get('plateau_patience', 3))
            factor = float(cfg.get('plateau_factor', 0.2))
            min_lr = float(cfg.get('plateau_min_lr', 1e-7))
            mode = cfg.get('plateau_mode', 'min')
            monitor = cfg.get('plateau_monitor', 'val_loss')
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                verbose=True,
            )
            scheduler_dict = {
                'scheduler': scheduler,
                'monitor': monitor,
                'interval': 'epoch',
                'frequency': 1
            }
        elif scheduler_name == 'linear_warmup_cosine':
            warmup_epochs = int(cfg.get('warmup_epochs', 2))
            max_epochs = self.trainer.max_epochs
            eta_min = float(cfg.get('eta_min', 5e-7))
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs, max_epochs, eta_min=eta_min)
            scheduler_dict = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        elif scheduler_name == 'none':
            scheduler = None
            scheduler_dict = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        if scheduler_dict is not None:
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler_dict
            }
        else:
            return {'optimizer': optimizer}

    # def on_train_epoch_end(self):
    #     # 매 epoch마다 수동으로 체크포인트 저장
    #     if hasattr(self, 'trainer') and self.trainer is not None:
    #         save_dir = 'checkpoints/manual_epoch_ckpt'
    #         os.makedirs(save_dir, exist_ok=True)
    #         ckpt_path = os.path.join(save_dir, f'epoch_{self.current_epoch:03d}.ckpt')
    #         self.trainer.save_checkpoint(ckpt_path)
    #         print(f"[Checkpoint] Saved manual checkpoint: {ckpt_path}")

import numpy as np
def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)

def get_lightning_model_from_config(cfg, class_names=None):
    """
    cfg['backbone']에 해당하는 백본을 models/ 폴더에서 import하여
    ClassificationLightningModule에 주입해 반환합니다.
    num_classes가 None이면 cfg['num_classes'] 사용.
    """
    backbone_name = cfg.get('backbone', 'tresnet')
    num_classes = len(class_names)
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
    #backbone = backbone.to(memory_format=torch.channels_last)
    # LightningModule import
    
    lightning_model = ClassificationLightningModule(backbone, class_names, cfg=cfg)
    return lightning_model

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)



from torch.optim.lr_scheduler import _LRScheduler
import math

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * float(self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            return [
                self.eta_min + (base_lr - self.eta_min)
                * 0.5
                * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
                for base_lr in self.base_lrs
            ]
