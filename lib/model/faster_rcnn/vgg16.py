# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False, lighthead=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.lighthead = lighthead
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.dout_base_model = 512
    if self.lighthead:
      self.dout_lh_base_model = 512

    _fasterRCNN.__init__(self, classes, class_agnostic, lighthead, compact_mode=True)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        if torch.cuda.is_available():
          state_dict = torch.load(self.model_path)
        else:
          state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    if self.pretrained:
      for layer in range(len(self.RCNN_base)):
        for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)
    if self.lighthead:
      self.RCNN_top = nn.Sequential(nn.Linear(490 * 7 * 7, 2048), nn.ReLU(inplace=True))    # 490 channels input into FC layer
    else:
      self.RCNN_top = vgg.classifier

    # Prediction
    d_in = 2048 if self.lighthead else 4096
    self.RCNN_cls_score = nn.Linear(d_in, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(d_in, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(d_in, 4 * self.n_classes)

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

