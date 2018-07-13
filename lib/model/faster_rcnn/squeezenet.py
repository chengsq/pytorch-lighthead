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

class squeezenet(_fasterRCNN):
  def __init__(self, classes, version, pretrained=False, class_agnostic=False, lighthead=False):
    self.model_path = 'data/pretrained_model/squeezenet{}.pth'.format(version)
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.lighthead = lighthead
    self.version = version
    self.dout_base_model = 256
    if self.lighthead:
      self.dout_lh_base_model = 512
    self.clip = None

    _fasterRCNN.__init__(self, classes, class_agnostic, lighthead, compact_mode=True)

  def _init_modules(self):
    if self.version == '1_0':
        squeezenet = models.squeezenet1_0()
        self.clip = -2
    elif self.version == '1_1':
        squeezenet = models.squeezenet1_1()
        self.clip = -5
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        if torch.cuda.is_available():
          state_dict = torch.load(self.model_path)
        else:
          state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        squeezenet.load_state_dict({k:v for k,v in state_dict.items() if k in squeezenet.state_dict()})

    squeezenet.classifier = nn.Sequential(*list(squeezenet.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    if self.lighthead:
      self.RCNN_base = nn.Sequential(*list(squeezenet.features._modules.values())[:self.clip])
    else:
      self.RCNN_base = nn.Sequential(*list(squeezenet.features._modules.values()))

    # Fix Layers
    for layer in range(len(self.RCNN_base)):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)
    if self.lighthead:
      self.lighthead_base = nn.Sequential(*list(squeezenet.features._modules.values())[self.clip+1:])
      self.RCNN_top = nn.Sequential(nn.Linear(490 * 7 * 7, 2048), nn.ReLU(inplace=True))
    else:
      self.RCNN_top = squeezenet.classifier

    d_in = 2048 if self.lighthead else 512

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(d_in, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(d_in, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(d_in, 4 * self.n_classes)

  def _head_to_tail(self, pool5):
    if self.lighthead:
      pool5_flat = pool5.view(pool5.size(0), -1)
      fc7 = self.RCNN_top(pool5_flat)
    else:
      fc7 = self.RCNN_top(pool5)
      fc7 = fc7.view(fc7.size(0), -1)
    return fc7