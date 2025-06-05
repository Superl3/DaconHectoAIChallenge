import torch
import torch.nn as nn

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "TResNet"))

from src.models.tresnet_v2.tresnet_v2 import TResnetL_V2 as TResnetL368

class TResNet(nn.Module):
    def __init__(self, num_classes, weights_path=None):
        super(TResNet, self).__init__()
        model_params = {'num_classes': 196}
        self.backbone = TResnetL368(model_params)
        if weights_path:
            pretrained_weights = torch.load(weights_path)
            self.backbone.load_state_dict(pretrained_weights['model'])
        self.feature_dim = self.backbone.num_features
        self.backbone.head = nn.Identity()
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
