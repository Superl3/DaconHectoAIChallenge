import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../TResNet"))
from src.models.tresnet_v2.tresnet_v2 import TResnetL_V2 as TResnetL368

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.tresnet import TResNetBackbone
class TResNet_CMALBackbone(nn.Module):
    def __init__(self, num_classes, weights_path=None):
        super().__init__()
        model_params = {'num_classes': 196}
        self.backbone = TResnetL368(model_params)
        if weights_path:
            pretrained_weights = torch.load(weights_path, map_location='cpu')
            self.backbone.load_state_dict(pretrained_weights['model'])

        net_layers = list(self.backbone.children())
        classifier = net_layers[1:3]
        net_layers = net_layers[0]
        net_layers = list(net_layers.children())

        # Network_Wrapper 생성
        self.model = Network_Wrapper(net_layers, num_classes)


    def forward(self, x):
        x = self.model(x)
        return x
    
from torch.nn.modules.batchnorm import _BatchNorm
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
    def __init__(self, net_layers,num_classes):
        super().__init__()
        self.Features = Features(net_layers)

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

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )



    def forward(self, x):
        x1, x2, x3 = self.Features(x)

        x1_ = self.conv_block1(x1)
        map1 = x1_.clone().detach()
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)

        x1_c = self.classifier1(x1_f)

        x2_ = self.conv_block2(x2)
        map2 = x2_.clone().detach()
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)

        x3_ = self.conv_block3(x3)
        map3 = x3_.clone().detach()
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        x_c_all = torch.cat((x1_f, x2_f, x3_f), -1)
        x_c_all = self.classifier_concat(x_c_all)

        return x1_c, x2_c, x3_c, x_c_all, map1, map2, map3
    
def map_generate(attention_map, pred, p1, p2):
    batches, feaC, feaH, feaW = attention_map.size()

    out_map=torch.zeros_like(attention_map.mean(1))

    for batch_index in range(batches):
        map_tpm = attention_map[batch_index]
        map_tpm = map_tpm.reshape(feaC, feaH*feaW)
        map_tpm = map_tpm.permute([1, 0])
        p1_tmp = p1.permute([1, 0])
        map_tpm = torch.mm(map_tpm, p1_tmp)
        map_tpm = map_tpm.permute([1, 0])
        map_tpm = map_tpm.reshape(map_tpm.size(0), feaH, feaW)

        pred_tmp = pred[batch_index]
        pred_ind = pred_tmp.argmax()
        p2_tmp = p2[pred_ind].unsqueeze(1)

        map_tpm = map_tpm.reshape(map_tpm.size(0), feaH * feaW)
        map_tpm = map_tpm.permute([1, 0])
        map_tpm = torch.mm(map_tpm, p2_tmp)
        out_map[batch_index] = map_tpm.reshape(feaH, feaW)

    return out_map

def attention_im(images, attention_map, theta=0.5, padding_ratio=0.1):
    images = images.clone()
    attention_map = attention_map.clone().detach()
    batches, _, imgH, imgW = images.size()

    for batch_index in range(batches):
        image_tmp = images[batch_index]
        map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm = torch.nn.functional.upsample_bilinear(map_tpm, size=(imgH, imgW)).squeeze()
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)
        map_tpm = map_tpm >= theta
        nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)
        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

        image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
        image_tmp = torch.nn.functional.upsample_bilinear(image_tmp, size=(imgH, imgW)).squeeze()

        images[batch_index] = image_tmp

    return images



def highlight_im(images, attention_map, attention_map2, attention_map3, theta=0.5, padding_ratio=0.1):
    images = images.clone()
    attention_map = attention_map.clone().detach()
    attention_map2 = attention_map2.clone().detach()
    attention_map3 = attention_map3.clone().detach()

    batches, _, imgH, imgW = images.size()

    for batch_index in range(batches):
        image_tmp = images[batch_index]
        map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm = torch.nn.functional.upsample_bilinear(map_tpm, size=(imgH, imgW)).squeeze()
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)


        map_tpm2 = attention_map2[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm2 = torch.nn.functional.upsample_bilinear(map_tpm2, size=(imgH, imgW)).squeeze()
        map_tpm2 = (map_tpm2 - map_tpm2.min()) / (map_tpm2.max() - map_tpm2.min() + 1e-6)

        map_tpm3 = attention_map3[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm3 = torch.nn.functional.upsample_bilinear(map_tpm3, size=(imgH, imgW)).squeeze()
        map_tpm3 = (map_tpm3 - map_tpm3.min()) / (map_tpm3.max() - map_tpm3.min() + 1e-6)

        map_tpm = (map_tpm + map_tpm2 + map_tpm3)
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)
        map_tpm = map_tpm >= theta

        nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)
        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

        image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
        image_tmp = torch.nn.functional.upsample_bilinear(image_tmp, size=(imgH, imgW)).squeeze()

        images[batch_index] = image_tmp

    return images



def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)

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