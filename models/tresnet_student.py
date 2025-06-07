import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../TResNet"))
from src.models.tresnet_v2.tresnet_v2 import TResnetL_V2 as TResnetL368

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class TResNet_StudentBackbone(nn.Module):
    def __init__(self, num_classes, weights_path=None):
        super().__init__()
        model_params = {'num_classes' : 196}
        self.backbone = TResnetL368(model_params)
        net_layers = list(self.backbone.children())
        classifier = net_layers[1:3]
        net_layers = net_layers[0]
        net_layers = list(net_layers.children())

        # Network_Wrapper 생성
        self.model = Student_Wrapper(net_layers, classifier)
        pretrained_weights = torch.load(weights_path)
        self.model.load_state_dict(pretrained_weights)
        self.feature_dim = self.model.classifier_initial[0].fc.in_features
        self.model.classifier_initial = nn.Identity()  # feature extractor로만 사용
        self.head = nn.Linear(self.feature_dim, num_classes)  # 분류기

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x
    
from torch.nn.modules.batchnorm import _BatchNorm
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Features(nn.Module):
    def __init__(self, net_layers_FeatureHead):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(net_layers_FeatureHead[0])
        self.net_layer_1 = nn.Sequential(*net_layers_FeatureHead[1])
        self.net_layer_2 = nn.Sequential(*net_layers_FeatureHead[2])
        self.net_layer_3 = nn.Sequential(*net_layers_FeatureHead[3])
        self.net_layer_4 = nn.Sequential(*net_layers_FeatureHead[4])
        self.net_layer_5 = nn.Sequential(*net_layers_FeatureHead[5])

    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x1 = self.net_layer_3(x)
        x2 = self.net_layer_4(x1)
        x3 = self.net_layer_5(x2)
        return x1, x2, x3


class Network_Wrapper(nn.Module):
    def __init__(self, net_layers, num_classes, classifier):
        super().__init__()
        self.Features = Features(net_layers)
        self.classifier_pool = nn.Sequential(classifier[0])
        
        # classifier_initial을 num_classes에 맞게 수정
        self.classifier_initial = nn.Linear(2048, num_classes)  # 기존 196을 num_classes로 변경
        
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.max_pool1 = nn.MaxPool2d(kernel_size=46, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=23, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=12, stride=1)

        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        _, _, x3 = self.Features(x) # , x2, x3
        # map1 = x1.clone()
        # map2 = x2.clone()
        # map3 = x3.clone()

        classifiers = self.classifier_pool(x3).view(x3.size(0), -1)
        classifiers = self.classifier_initial(classifiers)  # 이제 num_classes 출력

        # x1_ = self.conv_block1(x1)
        # x1_ = self.max_pool1(x1_)
        # x1_f = x1_.view(x1_.size(0), -1)

        # x1_c = self.classifier1(x1_f)

        # x2_ = self.conv_block2(x2)
        # x2_ = self.max_pool2(x2_)
        # x2_f = x2_.view(x2_.size(0), -1)
        # x2_c = self.classifier2(x2_f)

        # x3_ = self.conv_block3(x3)
        # x3_ = self.max_pool3(x3_)
        # x3_f = x3_.view(x3_.size(0), -1)
        # x3_c = self.classifier3(x3_f)

        return classifiers #x1_c , x2_c, x3_c , map1, map2, map3


class Anti_Noise_Decoder(nn.Module):
    def __init__(self, scale, in_channel):
        super(Anti_Noise_Decoder, self).__init__()
        self.Sigmoid = nn.Sigmoid()

        in_channel = in_channel // (scale * scale)

        self.skip = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)

        )

        self.process = nn.Sequential(
            nn.PixelShuffle(scale),
            nn.Conv2d(in_channel, 256, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 3, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x, map):
        x_ = self.process(map)
        if not (x.size() == x_.size()):
            x_ = F.interpolate(x, (x.size(2),x.size(3)), mode='bilinear')
        return self.skip(x) + x_


def img_add_noise(x, transformation_seq):
    x = x.permute(0, 2, 3, 1)
    x = x.cpu().numpy()
    x = transformation_seq(images=x)
    x = torch.from_numpy(x.astype(np.float32))
    x = x.permute(0, 3, 1, 2)
    return x

def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

def CELoss(x, y):
    return smooth_crossentropy(x, y, smoothing=0.1)

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
    



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


class Student_Wrapper(nn.Module):
    def __init__(self, net_layers, classifier):
        super(Student_Wrapper, self).__init__()
        self.net_layer_0 = nn.Sequential(net_layers[0])
        self.net_layer_1 = nn.Sequential(*net_layers[1])
        self.net_layer_2 = nn.Sequential(*net_layers[2])
        self.net_layer_3 = nn.Sequential(*net_layers[3])
        self.net_layer_4 = nn.Sequential(*net_layers[4])
        self.net_layer_5 = nn.Sequential(*net_layers[5])

        self.classifier_pool = nn.Sequential(classifier[0])
        self.classifier_initial = nn.Sequential(classifier[1])

    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x1 = self.net_layer_3(x)
        x2 = self.net_layer_4(x1)
        x3 = self.net_layer_5(x2)


        classifiers = self.classifier_pool(x3).view(x3.size(0), -1)
        out = self.classifier_initial(classifiers)

        return out#, x1, x2, x3