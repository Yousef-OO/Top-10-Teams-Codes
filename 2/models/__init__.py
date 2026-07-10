#!/usr/bin/python
# -*- coding:utf-8 -*-
from models.Resnet1d import resnet18 as resnet18_1d
from models.Resnet1d import resnet34 as resnet34_1d
from models.CNN_LSTM import CNNLSTM as cnnlstm
from models.resnet_NG import resnet34 as resnet34_NG
from models.SEResnet1d import resnet18 as seresnet18_1d
from models.SEResnetLSTM import resnet18 as seresnetlstm18_1d
from models.SEResnet1dMore import resnet18 as seresnet18_2_1d
from models.SEResnetDilation1d import resnet18 as seresnet18_dilation_1d
from models.SCSEResnet1d import resnet18 as scseresnet18_1d
from models.SEResnet1dag import resnet18 as seresnet18_1d_ag
