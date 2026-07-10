import sys

import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skmultilearn.model_selection import IterativeStratification

import utils.cinc_utils as cinc_utils
import utils.evaluate_12ECG_score as cinc_eval
from utils.dataloader import CINC2020Dataset
from utils.loss import BinaryFocalLoss
from learn.trainer import Trainer


def calc_pos_weights(dataset):
  freq = np.sum(dataset.y, axis=0)
  freq = np.where(freq==0, freq.max(), freq)
  return torch.Tensor(np.around(freq.max()/freq, decimals=1))

def get_datasets(root_dir, X_train, y_train, X_test, y_test, classes):
  transform = transforms.Compose([cinc_utils.ApplyGain(umc=False), cinc_utils.ToTensor()])
  trainset = CINC2020Dataset(X_train, y_train, classes, root_dir, transform=transform, num_leads=12, max_sample_length=ARGS.max_sample_length)
  validset = CINC2020Dataset(X_test, y_test, classes, root_dir, transform=transform, num_leads=12, max_sample_length=ARGS.max_sample_length)
  return trainset, validset

def get_dataloader(dataset):
  return DataLoader(dataset, batch_size=ARGS.batch_size, shuffle=True, num_workers=0)

def main():
  cross_validate(ARGS.n_folds)

def cross_validate(n_folds=10):
  root_dir = ARGS.input_dir
  mapping = pd.read_csv('pretraining/label_mapping_ecgnet_eq.csv', delimiter=';')
  class_mapping = mapping[['SNOMED CT Code', 'Training Code']]
  X, y, classes = cinc_utils.get_xy(
    root_dir, max_sample_length=ARGS.max_sample_length, cut_off=ARGS.cut_off_samples,
    class_mapping=class_mapping)
  # convert classes to string types
  classes = [str(c) for c in classes]
#   random_state = 821385989 # DO NOT CHANGE THIS VALUE
#   stratifier = IterativeStratification(
#     n_splits=n_folds, order=ARGS.cv_order, random_state=random_state)
  stratifier = IterativeStratification(
    n_splits=n_folds, order=ARGS.cv_order)
  
  best_epochs = []
  for k, (train_indexes, test_indexes) in enumerate(stratifier.split(X, y)):
    # get X and y for this fold
    X_train, y_train = X[train_indexes, :], y[train_indexes, :]
    X_test, y_test = X[test_indexes, :], y[test_indexes, :]
    # get datasets
    trainset, validset = get_datasets(root_dir, X_train, y_train, X_test, y_test, classes)
    # get dataloaders
    trainloader = get_dataloader(trainset)
    validloader = get_dataloader(validset)
    # run training procedure
    trainer = get_trainer(classes, fold_idx=k, pos_weights=((calc_pos_weights(trainset) - 1) * .5) + 1)
    best_epoch = trainer.train(ARGS.epochs, trainloader, validloader)
    best_epochs.append((k, best_epoch))
  print('Best epochs:', str(best_epochs).strip('[]'))


def add_equivalent_classes(m, classes, classes_adj, equivalent_classes):
  # it is important that we append the copies in the same order as in
  # `equivalent_classes` to be consistent to the load_weights function
  tmp = np.zeros((m.shape[0], len(classes_adj)))
  tmp[:, :len(classes)] = m
  for eq in equivalent_classes:
    k = classes.index(eq[0])
    i = classes_adj.index(eq[1])
    # add a copy for the equivalent class label
    tmp[:, i] = m[:, k]
  return tmp


def load_weights(classes, equivalent_classes):
  classes_adj = classes + [eq[1] for eq in equivalent_classes]
  weights_file = 'pretraining/weights.csv'
  weights = cinc_eval.load_weights(weights_file, classes_adj)
  return weights, classes_adj


def get_trainer(classes, fold_idx=None, pos_weights=None):
  equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
  num_classes = len(classes)
  weights, classes_adj = load_weights(classes, equivalent_classes)
  
  def loss_hook(model, criterion, output, y_true):
    y_score = output
    loss = criterion(y_score, y_true, training=model.training)
    y_prob = torch.sigmoid(y_score)
    y_pred = torch.round(y_prob)
    return loss, y_pred, y_true, y_score, y_prob

  def fbeta_metric(y_true, y_pred, y_scores, y_probs):
    fbeta, _ = cinc_eval.compute_beta_measures(y_true, y_pred, beta=2)
    return fbeta
  
  def gbeta_metric(y_true, y_pred, y_scores, y_probs):
    _, gbeta = cinc_eval.compute_beta_measures(y_true, y_pred, beta=2)
    return gbeta
  
  def auc_metric(y_true, y_pred, y_scores, y_probs):
    auroc, _ = cinc_eval.compute_auc(y_true, y_probs)
    return auroc
  
  def challenge_metric(y_true, y_pred, y_scores, y_probs):
    # add copies of the equivalent class labels
    y_true_adj = add_equivalent_classes(y_true, classes, classes_adj, equivalent_classes)
    y_pred_adj = add_equivalent_classes(y_pred, classes, classes_adj, equivalent_classes)
    normal_class = '426783006'
    challenge_metric = cinc_eval.compute_challenge_metric(weights, y_true_adj, y_pred_adj, classes_adj, normal_class)
    return challenge_metric
  
  def winning_criteria(loss_value, reports, custom_report, data):
    winning, updated_data = False, data
    if 'cinc_challenge_metric' not in data or custom_report['cinc_challenge_metric'] >= data['cinc_challenge_metric']:
      winning = True
      updated_data['cinc_challenge_metric'] = custom_report['cinc_challenge_metric']
      updated_data['save_model'] = True
    else:
      updated_data['save_model'] = False
    return winning, updated_data
  
  network_params = {
    'in_channels': 12,
    'channels': 108,
    'depth': 6,
    'reduced_size': 216,
    'out_channels': num_classes,
    'kernel_size': 3,
  }
  
  if ARGS.local:
    save_location = os.path.join(ARGS.output_dir, "local")
  else:
    save_location = os.path.join(ARGS.output_dir, "cloud")
  
  load_path = None
  if ARGS.resume:
    load_path = 'pretraining/classes24_epoch_6.pt'
    print('Resuming training after', load_path)
  
  freeze_modules = None
  if ARGS.freeze:
    freeze_modules = ['network\.0\.network\.[01234].*']
    print('Freezing modules:', freeze_modules)
  
  return Trainer(
    trainer_id='final_model_{}'.format(fold_idx),
    network_name='gtcn',
    network_params=network_params,
    criterion=BinaryFocalLoss(
      gamma=2,
      pos_weight=pos_weights, 
    ),
    save_location=save_location,
    lr=ARGS.lr,
    min_num_epochs=ARGS.min_num_epochs,
    patience=ARGS.patience,
    loss_hook=loss_hook,
    classes=classes,
    custom_metrics=[
      ('cinc_f2', 'Score', fbeta_metric),
      ('cinc_g2', 'Score', gbeta_metric),
      ('cinc_auc', 'Score', auc_metric),
      ('cinc_challenge_metric', 'Score', challenge_metric),
    ],
    trim_padded_batch=ARGS.trim_padded_batch,
    winning_criteria=winning_criteria,
    seed=ARGS.seed,
    eval_logging=False,
    load_path=load_path,
    freeze_modules=freeze_modules,
  )


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Physionet challenge')
  parser.add_argument('--epochs', default=20, type=int,
                      help='max number of epochs')
  parser.add_argument('--batch_size', default=128, type=int,
                      help='number of samples per batch')
  parser.add_argument('--lr', default=0.001, type=float,
                      help='learning rate')
  parser.add_argument('--min_num_epochs', default=5, type=int,
                      help='minimum number of epochs')
  parser.add_argument('--patience', default=10, type=int,
                      help='number of early stopping epochs')
  parser.add_argument('--trim_padded_batch', default=True, type=bool,
                      help='trim padded batch to length of longest sequence in batch')
  parser.add_argument('--max_sample_length', default=5000, type=int,
                      help='maximum length of a time serie')
  parser.add_argument('--cut_off_samples', default=True, type=bool,
                      help='whether to cut off samples at max_sample_length or only keep' \
                      'samples with length less than max_sample_length')
  parser.add_argument('--n_folds', default=10, type=int,
                      help='number of folds in k-fold cross validation')
  parser.add_argument('--cv_order', default=2, type=int,
                      help='order of cross validation iterative stratification method')
  parser.add_argument('--freeze', default=True, dest='freeze', action='store_true',
                      help='freeze layers')
  parser.add_argument('--seed', type=int, default=124129434, help='random seed value')
  parser.add_argument('--resume', default=True, dest='resume', action='store_true',
                      help='resume a previous training')
  parser.add_argument('--input_dir', default="input_training_directory",
                      help='input directory', type=str)
  parser.add_argument('--output_dir', default="output_training_directory",
                      help='output directory', type=str)
  parser.add_argument('--local', default=False, dest='local', action='store_true',
                      help='local or cloud training')

  ARGS = parser.parse_args()
  np.random.seed(ARGS.seed)
  torch.manual_seed(ARGS.seed)
  main()
