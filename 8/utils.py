# -*- coding: utf-8 -*-
'''
@time: 2019/9/12 15:16

@ author: javis
'''
import torch
import numpy as np
import time, os
from torch import nn
import torch
import torch.nn.functional as F


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# 多标签使用类别权重
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)

        return (loss * self.weights).mean()


# fbeta loss
class FbetaLoss(nn.Module):
    def __init__(self, beta=2):
        super(FbetaLoss, self).__init__()
        self.small_value = 1e-6
        self.beta = beta

    def forward(self, logits, labels):
        beta = self.beta
        batch_size = logits.size()[0]
        sigmoid = torch.nn.Sigmoid()
        p = sigmoid(logits)
        l = labels
        num_pos = torch.sum(p, 1) + self.small_value
        num_pos_hat = torch.sum(l, 1) + self.small_value
        tp = torch.sum(l * p, 1)
        precise = tp / num_pos
        recall = tp / num_pos_hat
        fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + self.small_value)
        loss = fs.sum() / batch_size
        return 1 - loss

class gbetaLoss(nn.Module):
    def __init__(self, beta=2):
        super(gbetaLoss, self).__init__()
        self.small_value = 1e-6
        self.beta = beta

    def forward(self, logits, labels):
        beta = self.beta
        batch_size = logits.size()[0]
        sigmoid = torch.nn.Sigmoid()
        p = sigmoid(logits)
        l = labels
        num_pos = torch.sum(p, 1) + self.small_value
        num_pos_hat = torch.sum(l, 1) + self.small_value
        tp = torch.sum(l * p, 1)
        precise = tp / num_pos
        recall = tp / num_pos_hat
        fs = precise * recall / ( beta * precise + recall - beta * precise * recall  + self.small_value)
        loss = fs.sum() / batch_size
        return 1 - loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()


class CombineLoss(nn.Module):
    def __init__(self, weight):
        super(CombineLoss, self).__init__()
        self.fbeta_loss = FbetaLoss(beta=2)
        self.bceloss = WeightedMultilabel(weight)
        # self.gbeta_loss = gbetaLoss(beta=2)

    def forward(self, logits, labels):
        loss_beta = self.fbeta_loss(logits, labels)
        loss_bce = self.bceloss(logits, labels)
        # loss_gbeta = self.gbeta_loss(logits, labels)
        return 0.4 * loss_beta + 0.6 * loss_bce


def compute_beta_score(labels, output, beta, num_classes, threshold=0.5,  check_errors=True):
    # labels = labels.view(-1).cpu().detach().numpy().astype(np.int)
    # output = output.view(-1).cpu().detach().numpy() > threshold
    labels = labels.cpu().detach().numpy().astype(np.int)
    output = output.cpu().detach().numpy() > threshold
    # Check inputs for errors.
    if check_errors:
        if len(output) != len(labels):
            raise Exception('Numbers of outputs and labels must be the same.')

    # Populate contingency table.
    num_recordings = len(labels)

    fbeta_l = np.zeros(num_classes)
    gbeta_l = np.zeros(num_classes)
    fmeasure_l = np.zeros(num_classes)
    accuracy_l = np.zeros(num_classes)

    f_beta = 0
    g_beta = 0
    f_measure = 0
    accuracy = 0

    # Weight function
    C_l = np.ones(num_classes);

    for j in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for i in range(num_recordings):

            num_labels = np.sum(labels[i])

            if labels[i][j] == 1 and output[i][j]:
                tp += 1 / num_labels
            elif not labels[i][j] == 1 and output[i][j]:
                fp += 1 / num_labels
            elif labels[i][j] == 1 and not output[i][j]:
                fn += 1 / num_labels
            elif not labels[i][j] == 1 and not output[i][j]:
                tn += 1 / num_labels

        # Summarize contingency table.
        if ((1 + beta ** 2) * tp + (fn * beta ** 2) + fp):
            fbeta_l[j] = float((1 + beta ** 2) * tp) / float(((1 + beta ** 2) * tp) + (fn * beta ** 2) + fp)
        else:
            fbeta_l[j] = 1.0

        if (tp + fp + beta * fn):
            gbeta_l[j] = float(tp) / float(tp + fp + beta * fn)
        else:
            gbeta_l[j] = 1.0

        if tp + fp + fn + tn:
            accuracy_l[j] = float(tp + tn) / float(tp + fp + fn + tn)
        else:
            accuracy_l[j] = 1.0

        if 2 * tp + fp + fn:
            fmeasure_l[j] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            fmeasure_l[j] = 1.0

    for i in range(num_classes):
        f_beta += fbeta_l[i] * C_l[i]
        g_beta += gbeta_l[i] * C_l[i]
        f_measure += fmeasure_l[i] * C_l[i]
        accuracy += accuracy_l[i] * C_l[i]

    f_beta = float(f_beta) / float(num_classes)
    g_beta = float(g_beta) / float(num_classes)
    f_measure = float(f_measure) / float(num_classes)
    accuracy = float(accuracy) / float(num_classes)

    return accuracy, f_measure, f_beta, g_beta
