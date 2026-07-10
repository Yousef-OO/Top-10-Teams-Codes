#!/usr/bin/env python

import os
import torch
import numpy as np
import torchvision.transforms as transforms
import shutil
from scipy import interpolate

from network.causal_cnn_old import CausalCNNEncoder
from utils.cinc_utils import *


# def run_12ECG_classifier(data, header_data, classes, models):
#   transform = transforms.Compose([cinc_utils.ApplyGain(), cinc_utils.ToTensor()])
#   y_probs = torch.empty((0)).float()
#   for model in models:
#     model.eval()
#     with torch.no_grad():
#       # only extract first 5000 sample points, since this is what was trained on
#       data = data[:, 0:5000]
#       # transform the data
#       data_transformed = transform(data)
#       # prepend dimension to create a batch of the data with shape 1x12xL
#       waveforms = data_transformed.unsqueeze(dim=0)
#       y_score = model(waveforms)
#       y_prob = torch.sigmoid(y_score)
#       y_probs = torch.cat((y_probs, y_prob.data.cpu()), 0)
#   avg_y_prob = y_probs.mean(dim=0)
#   y_pred = torch.round(avg_y_prob)
#   y_pred = y_pred.data.cpu().numpy().astype(int)
#   y_prob = avg_y_prob.data.cpu().numpy()
#   return y_pred, y_prob


def add_equivalent_classes(m, classes, classes_adj, equivalent_classes):
  # it is important that we append the copies in the same order as in
  # `equivalent_classes` to be consistent to the load_weights function
  tmp = np.zeros((len(classes_adj)))
  tmp[:len(classes)] = m
  for eq in equivalent_classes:
    k = classes.index(eq[0])
    i = classes_adj.index(eq[1])
    # add a copy for the equivalent class label
    tmp[i] = m[k]
  return tmp


def run_12ECG_classifier(data, header_data, model_data):
  models = model_data['models']
  classes = model_data['classes']
  classes_adj = model_data['classes_adj']
  eq_classes = model_data['eq_classes']
  device = model_data['device']

  transform = transforms.Compose([ApplyGainPredict(), ToTensorPredict()])
  y_probs = torch.empty((0)).float()
  for model in models:
    model.eval()
    with torch.no_grad():
      # resample if necessary
      length = data.shape[1]
      sample_fs = int(header_data[0].split(' ')[2])
      if sample_fs != 500:
          print('Resampling signal to 500Hz')
          x = np.linspace(0, length / sample_fs, num = length)
          f = interpolate.interp1d(x, data, axis = 1)

          xnew = np.linspace(0, length / sample_fs, 
                               num = (length / sample_fs) * 500)
          data = f(xnew)   # use interpolation function returned by `interp1d`
      
      # only extract first 5000 sample points, since this is what was trained on
      data = data[:, 0:5000]
      # transform the data
      data_transformed = transform(data)
      # prepend dimension to create a batch of the data with shape 1x12xL
      waveforms = data_transformed.unsqueeze(dim=0).to(device)
      y_score = model(waveforms)
      y_prob = torch.sigmoid(y_score)
      y_probs = torch.cat((y_probs, y_prob.data.cpu()), 0)
  avg_y_prob = y_probs.mean(dim=0)
  y_pred = torch.round(avg_y_prob)
  y_pred = y_pred.data.cpu().numpy().astype(int)
  y_prob = avg_y_prob.data.cpu().numpy()
  # add equivalent classes
  y_prob = add_equivalent_classes(y_prob, classes, classes_adj, eq_classes)
  y_pred = add_equivalent_classes(y_pred, classes, classes_adj, eq_classes)
  return y_pred.astype(int), y_prob, classes_adj


def load_12ECG_model(input_directory):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('DEVICE:', device)
  if os.path.isdir(os.path.join(input_directory, "cloud")):
    input_directory = os.path.join(input_directory, "cloud")
    print("Using cloud trained model...")
  elif os.path.isdir(os.path.join(input_directory, "local")):
    input_directory = os.path.join(input_directory, "local")
    print("Using locally trained model...")
  elif os.path.isdir("local"):
    shutil.copytree('local', os.path.join(input_directory, "local"))
    print("Copying local model files to directory...")
    input_directory = os.path.join(input_directory, "local")
    print("Using locally trained model...")
  else:
    print("No model found, either local or cloud")
  models = []
  for fold_idx in range(10):
    path = os.path.join(input_directory, 'final_model_{}'.format(fold_idx))
    folder = [f for f in os.listdir(path) if f.endswith('.pt')]
    epochs = []
    for filename in folder:
      epochs.append(int(filename.split('_')[2].split('.')[0]))    
    best_epoch = os.path.join(path, 'seed_124129434.epoch_{}.pt'.format(max(epochs)))
    print('Best epoch for fold {}: {}'.format(fold_idx, max(epochs)))
    checkpoint = torch.load(best_epoch, map_location=device)
    model = CausalCNNEncoder(**checkpoint['network_params'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    models.append(model)
  classes = ['270492004', '164889003', '164890007', '426627000', '713427006', '713426002', '445118002', '39732003', '164909002', '251146004', '698252002', '10370003', '284470004', '427172004', '164947007', '111975006', '164917005', '47665007', '427393009', '426177001', '426783006', '427084000', '164934002', '59931005']
  eq_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
  classes_adj = classes + [eq[1] for eq in eq_classes]
  model_data = {
    'models': models,
    'classes': classes,
    'classes_adj': classes_adj,
    'eq_classes': eq_classes,
    'device': device,
  }
  return model_data


# def load_12ECG_model():
#   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#   filenames = [
#     'fold-0_epoch_66.pt', 'fold-1_epoch_89.pt', 'fold-2_epoch_129.pt',
#     'fold-3_epoch_127.pt', 'fold-4_epoch_65.pt', 'fold-5_epoch_76.pt',
#     'fold-6_epoch_120.pt', 'fold-7_epoch_106.pt', 'fold-8_epoch_85.pt',
#     'fold-9_epoch_95.pt',
#   ]
#   models = []
#   for filename in filenames:
#     checkpoint = torch.load(filename, map_location=device)
#     model = GTCN(**checkpoint['network_params'])
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device)
#     model.eval()
#     models.append(model)
#   return models
