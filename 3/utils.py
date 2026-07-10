# -*- coding: utf-8 -*-

import torch
import numpy as np
#olderr = np.seterr(all='ignore')
import time,os
from sklearn.metrics import f1_score
from torch import nn
import torch.nn.functional as F


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

#计算F1score
def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
#    print(y_pre.view(-1).cpu().detach().numpy())
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)

#打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

#多标签使用类别权重（方式2）
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()
#方式1（所有类别权重相等）
class WeightedMultilabel_1(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel_1, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
#        return (loss * self.weights).mean()
        return loss.mean()
#方式3（类别数量倒数*metric权重加和）
class WeightedMultilabel_3(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel_3, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights
        self.weight1 = torch.tensor([12.25  , 12.75  , 12.225 , 12.225 , 12.575 , 10.675 , 12.6125,
       11.475 , 10.675 , 10.675 , 11.1   ,  8.175 , 10.675 , 11.625 ,
       11.8   , 11.625 , 12.6125, 11.625 , 11.625 , 10.375 , 12.25  ,
       12.75  , 11.625 , 12.225 , 12.225 , 11.625 , 12.575 ]).cuda()
    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights * self.weight1).mean()
class signLoss(nn.Module):
    def __init__(self):
        super(signLoss, self).__init__()
        self.criterion = nn.BCELoss(reduction='none')
        self.sig = nn.Sigmoid()
        
    def forward(self, output, y_true):
        '''
        output: raw output from model (batch_size * num_classes)
        target: true label (batch_size * num_classes)
        device: cuda device
        '''
        y_pred  = self.sig(output)
        abs_sub = torch.abs(y_true-y_pred)

        return torch.mean(torch.sum(F.relu(torch.sign(abs_sub-0.5))*self.criterion(y_pred, y_true)+
                          F.relu(torch.sign(0.5-abs_sub))*((y_true*torch.pow((1-y_pred),2)+(1-y_true)*torch.pow(y_pred,2)))*self.criterion(y_pred, y_true), dim=1)) 
class labelSmoothing(nn.Module):
    '''
    implement of label smoothing.
    eps: smooth coefficient, usually be 0.05 or 0.1
    '''
    def __init__(self, use_weights, weights, eps):
 
        super(labelSmoothing, self).__init__()
 
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.use_weights = use_weights
        self.weights = weights
        self.eps = eps 
 
 
    def forward(self, output, target, device):
        """
        output: raw output from model
        target: true label
        """
        new_label = target.clone()
        new_label.to(device)
                     
        num_classes=target.size(1)
        lb_pos, lb_neg = 1. - self.eps, self.eps / num_classes
        
        new_label = new_label * lb_pos + lb_neg
        
        loo = self.criterion(output, new_label)
        if self.use_weights:
            loss = (loo * self.weights).mean()
            
        else:
            loss = loo.mean()
        
        return loss 
