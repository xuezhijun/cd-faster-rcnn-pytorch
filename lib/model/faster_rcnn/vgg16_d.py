from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.models as models

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn_d import _fasterRCNN


# (8,512,38,75)-->(8,512,1,1)
class ChannelAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# (8,512,38,75)-->(8,1,38,75)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class vgg16(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False,
                 fc=False, cv=False):  ####

        self.model_path = cfg.VGG16_PATH
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        self.fc = fc  ####
        self.cv = cv  ####

        _fasterRCNN.__init__(self, classes, class_agnostic,
                             self.fc, self.cv)  ####

    def _init_modules(self):
        vgg = models.vgg16()

        if self.pretrained:
            print("Loading pretrained weights from %s" % self.model_path)
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        ########
        if self.fc:
            self.CoralFCLayer = nn.Linear(4096, 64)
        if self.cv:
            self.CA = ChannelAttention(512, 16)
            self.SA = SpatialAttention(3)
            self.CoralCVLayer = nn.Linear(38*75, 64)
        ########

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.RCNN_base[layer].parameters():
                p.requires_grad = False

        self.RCNN_top = vgg.classifier

        feat_d = 4096
        self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)

        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)

    def _head_to_tail(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)
        return fc7
