""" 
Creates an Xception-like Model as defined in:
Zeming Li, Chao Peng, Gang Yu, Xiangyu Zhang, Yangdong Deng, Jian Sun
Light-Head R-CNN: In Defense of Two-Stage Object Detector
https://arxiv.org/pdf/1711.07264.pdf
REMEMBER to set your image size to 3x224x224 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 224x224
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.utils.layer_utils import _Block
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch

__all__ = ['xception']


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False)      # 224 x 224 -> 112 x 112
        self.bn1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)     # -> 56 x 56

        # Stage 2
        self.block1 = _Block(24, 144, 1+3, 2, start_with_relu=False, grow_first=True)     # -> 28 x 28

        # Stage 3
        self.block2 = _Block(144, 288, 1+7, 2, start_with_relu=True, grow_first=True)     # -> 14 x 14

        # Stage 4
        self.block3 = _Block(288, 576, 1+3, 2, start_with_relu=True, grow_first=True)     # -> 7 x 7

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(576, num_classes)



        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------





    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class xception(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False, lighthead=True):
        self.dout_base_model = 576      # Output channel at Stage4
        self.dout_lh_base_model = 576
        self.class_agnostic = class_agnostic
        self.pretrained = pretrained

        _fasterRCNN.__init__(self, classes, class_agnostic, lighthead, compact_mode=True)

    def _init_modules(self):
        xception = Xception()

        # Check pretrained
        if self.pretrained == True:
            print("Loading pretrained weights from %s" %(self.model_path))
            if torch.cuda.is_available():
                state_dict = torch.load(self.model_path)
            else:
                state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage)
            xception.load_state_dict({k:v for k,v in state_dict.items() if k in xception.state_dict()})

        # Build xception-like network.
        self.RCNN_base = nn.Sequential(xception.conv1, xception.bn1,xception.relu, xception.maxpool,    # Conv1
            xception.block1,xception.block2,xception.block3)

        self.RCNN_top = nn.Sequential(nn.Linear(490 * 7 * 7, 2048), nn.ReLU(inplace=True))

        self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(2048, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

        # Fix blocks
        if self.pretrained:
            for layer in range(len(self.RCNN_base)):
                for p in self.RCNN_base[layer].parameters(): p.requires_grad = False
            
    def _head_to_tail(self, pool5):
        pool5 = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5)   # or two large fully-connected layers
        
        return fc7