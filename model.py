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
        self.criterion = self.CELoss
        #nn.CrossEntropyLoss()
        self.automatic_optimization = False  # <-- manual optimization 모드

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # images, labels = batch
        # logits = self(images)
        # loss = self.criterion(logits, labels).mean()
        # acc = (logits.argmax(dim=1) == labels).float().mean()
        # self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        # return loss

        images, labels = batch
        optimizer = self.optimizers()
        # SAM step 1
        outputs = self(images)
        loss = self.criterion(outputs, labels).mean()
        self.manual_backward(loss)
        optimizer.first_step(zero_grad=True)

        # SAM step 2
        outputs2 = self(images)
        loss2 = self.criterion(outputs2, labels).mean()
        self.manual_backward(loss2)
        optimizer.second_step(zero_grad=True)

        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels).mean()
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        optimizer = SAM(self.model.parameters(),torch.optim.SGD,lr=self.hparams.learning_rate,adaptive=False,momentum=0.9,weight_decay=5e-4)
        #torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def smooth_crossentropy(self, pred, gold, smoothing=0.1):
        n_class = pred.size(1)

        one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
        one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
        log_prob = F.log_softmax(pred, dim=1)

        return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

    def CELoss(self, x, y):
        return self.smooth_crossentropy(x, y, smoothing=0.1)

# SAM
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

import numpy as np
def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)

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
