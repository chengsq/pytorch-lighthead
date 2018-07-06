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
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch

__all__ = ['xception']

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()
        
        # Do not use pre-activation design (no identity mappings!)
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters

        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=strides,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=strides,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        # Scaling already applied by stride
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)
        
        return x



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
        self.block1=Block(24,144,1 + 3,2,start_with_relu=False,grow_first=True)     # -> 28 x 28

        # Stage 3
        self.block2=Block(144,288,1 + 7,2,start_with_relu=True,grow_first=True)     # -> 14 x 14

        # Stage 4
        self.block3=Block(288,576,1 + 3,2,start_with_relu=True,grow_first=True)     # -> 7 x 7

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
    def __init__(self, classes, class_agnostic=False, lighthead=True):
        self.dout_base_model = 576      # Output channel at Stage4
        self.class_agnostic = class_agnostic

        _fasterRCNN.__init__(self, classes, class_agnostic, lighthead, setting='S')

    def _init_modules(self):
        xception = Xception()

        # Build resnet.
        self.RCNN_base = nn.Sequential(xception.conv1, xception.bn1,xception.relu,
            xception.maxpool,xception.block1,xception.block2,xception.block3)

        self.RCNN_top = nn.Sequential(nn.Linear(576 * 7 * 7, 2048), nn.ReLU(inplace=True))

        self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(2048, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

        # Fix blocks
        for p in self.RCNN_base[0].parameters(): p.requires_grad=False
        for p in self.RCNN_base[1].parameters(): p.requires_grad=False

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS >= 3:
            for p in self.RCNN_base[6].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 2:
            for p in self.RCNN_base[5].parameters(): p.requires_grad=False
        if cfg.RESNET.FIXED_BLOCKS >= 1:
            for p in self.RCNN_base[4].parameters(): p.requires_grad=False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False

        self.RCNN_base.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
        # Set fixed blocks to be in eval mode
            self.RCNN_base.eval()
            self.RCNN_base[5].train()
            self.RCNN_base[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base.apply(set_bn_eval)
            self.RCNN_top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        pool5 = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5)   # or two large fully-connected layers
        
        return fc7