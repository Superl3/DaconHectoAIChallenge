import timm
import torch
import torch.nn as nn

class vitBackbone(nn.Module):
    def __init__(self, num_classes=1000, weights_path=None, model_name="vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k", **kwargs):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes if num_classes > 0 else 0,
            **kwargs
        )
        # if weights_path is not None:
        #     state = torch.load(weights_path, map_location='cpu')
        #     self.model.load_state_dict(state, strict=False)
    def forward(self, x):
        return self.model(x)