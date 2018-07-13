from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
from model.utils.layer_utils import conv_1x1_bn, conv_bn, InvertedResidual

__all__ = ['mobilenetv2']


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(input_size/32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x




class mobilenetv2(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False, lighthead=False):
        self.model_path = 'data/pretrained_model/mobilenet_v2.pth.tar'
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.lighthead = lighthead
        self.dout_base_model = 320
        if self.lighthead:
            self.dout_lh_base_model = 1280

        _fasterRCNN.__init__(self, classes, class_agnostic, lighthead, compact_mode=False)

    def _init_modules(self):
        mobilenet = MobileNetV2()

        if self.pretrained == True:
            print("Loading pretrained weights from %s" %(self.model_path))
            if torch.cuda.is_available():
                state_dict = torch.load(self.model_path)
            else:
                state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage)

            mobilenet.load_state_dict({k:v for k,v in state_dict.items() if k in mobilenet.state_dict()})

        mobilenet.classifier = nn.Sequential(*list(mobilenet.features._modules.values())[-2:-1])
        
        # Build mobilenet.
        self.RCNN_base = nn.Sequential(*list(mobilenet.features._modules.values())[:-2])

        # Fix Layers
        if self.pretrained:
            for layer in range(len(self.RCNN_base)):
                for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        if self.lighthead:
            self.lighthead_base = mobilenet.classifier
            self.RCNN_top = nn.Sequential(nn.Linear(490 * 7 * 7, 2048), nn.ReLU(inplace=True))
        else:
            self.RCNN_top = mobilenet.classifier

        c_in = 2048 if self.lighthead else 1280*7*7

        self.RCNN_cls_score = nn.Linear(c_in, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(c_in, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(c_in, 4 * self.n_classes)

    def _head_to_tail(self, pool5):
        if self.lighthead:
            pool5_flat = pool5.view(pool5.size(0), -1)
            fc7 = self.RCNN_top(pool5_flat)    # or two large fully-connected layers
        else:
            print(pool5.shape)
            fc7 = self.RCNN_top(pool5)
            fc7 = fc7.view(fc7.size(0), -1)
        return fc7
