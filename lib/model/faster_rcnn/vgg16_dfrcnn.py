from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.models as models

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn_dfrcnn import _fasterRCNN


def flatten(x):
    N = list(x.size())[0]
    #print('dim 0', N, 1024*19*37)
    return x.view(N, -1)


class netD_img(nn.Module):
    def __init__(self, beta=1, ch_in=1024, ch_out=1024, W=38, H=75, stride_1=1, padding_1=1, kernel=3):
        super(netD_img, self).__init__()

        self.ch_out = ch_out
        self.conv_image = nn.Conv2d(ch_in, ch_out, stride=stride_1, padding=padding_1, kernel_size=kernel)
        self.relu = nn.ReLU(inplace=True)
        self.bn_image = nn.BatchNorm2d(ch_out)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.bn_2 = nn.BatchNorm2d(ch_out)
        self.fc_1_image = nn.Linear(1, 2)
        #self.softmax = nn.Softmax()
        #self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv_image(x)
        x = self.relu(x)
        x = self.bn_image(x)
        x = self.maxpool(x)
        x = self.bn_2(x)
        # convert to 1024*W*H x 1.
        x = flatten(x)
        x = torch.transpose(x, 0, 1)
        x = self.fc_1_image(x)
        # 1 x n vector
        #y = self.softmax(x)
        #x = self.logsoftmax(x)
        #return x, y
        return x


# pool_feat dim: N x 2048, where N may be 300.
class netD_inst(nn.Module):
    def __init__(self, beta=1, fc_size=2048):
        super(netD_inst, self).__init__()

        self.fc_1_inst = nn.Linear(fc_size, 100)
        self.fc_2_inst = nn.Linear(100, 2)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(2)
        #self.softmax = nn.Softmax()
        #self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.relu(self.fc_1_inst(x))
        x = self.relu(self.bn(self.fc_2_inst(x)))
        #y = self.softmax(x)
        #x = self.logsoftmax(x)
        #return x, y
        return x


class vgg16(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False):

        self.model_path = cfg.VGG16_PATH
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _fasterRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        vgg = models.vgg16()

        if self.pretrained:
            print("Loading pretrained weights from %s" % self.model_path)
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        ########
        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        self.netD_img = netD_img(ch_in=512, ch_out=512)
        feat_d = 4096
        self.netD_inst = netD_inst(fc_size=feat_d)

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.RCNN_base[layer].parameters():
                p.requires_grad = False

        self.RCNN_top = vgg.classifier

        self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)

        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)
        ########

    def _head_to_tail(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)
        return fc7
