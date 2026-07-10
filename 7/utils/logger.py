import io
import re
import os
import numpy as np
import torch
import torch.nn as nn
import torchnet as tnt
from torchnet.engine import Engine
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import scikitplot as skplt
  
class Logger():
  """
  """
  def __init__(self, classes=None, num_labels=2, custom_metrics=None, fold=None):
    self.classes = classes
    self.num_labels = num_labels
    self.loss_meter = tnt.meter.AverageValueMeter()
    self.add_custom_metrics(custom_metrics)

  def add_custom_metrics(self, data):
    self.custom_metrics = []
    if data:
      for metric in data:
        title, ylabel, func = metric
        self.custom_metrics.append((func, title))
  
  def loss(self, value):
    self.loss_meter.add(value)

  def log_train(self, niter):
    loss_meter_value = self.loss_meter.value()[0]
    self.reset()
    return loss_meter_value
    
  def log_eval(self, epoch, report):
    loss_meter_value = self.loss_meter.value()[0]
    self.reset()
    return loss_meter_value
  
  def log_custom_metrics(self, epoch, y_true, y_pred, y_scores, y_probs):
    report = {}
    if not self.custom_metrics:
      return report
    for idx in range(len(self.custom_metrics)):
      value = self.custom_metrics[idx][0](y_true, y_pred, y_scores, y_probs)

      report[self.custom_metrics[idx][1]] = value
    return report

  def average_logger(self, func, epoch, report, key, factor=None):
    if isinstance(report, dict):
      func.log(epoch, np.nan_to_num(report[key]))
      return
    if factor is None:
      factor = np.ones(len(report))
    values = [np.nan_to_num(class_report[key]) for class_report in report]
    average = (factor * np.array(values)).mean()
    func.log(epoch, average)

  def reset(self):
    self.loss_meter.reset()