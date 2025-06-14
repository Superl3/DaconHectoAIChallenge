import timm
import torch
import torch.nn as nn

class RDNet_largeBackbone(nn.Module):
    def __init__(self, num_classes=1000, weights_path=None, model_name="rdnet_large.nv_in1k_ft_in1k_384", **kwargs):
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