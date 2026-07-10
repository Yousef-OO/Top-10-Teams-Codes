import sys
sys.path.append('..')

import os
import re
from tqdm import tqdm
import numpy as np
import torch
from utils.loss import BinaryFocalLoss
from network.main import get_network
import utils.evaluation as eval_utils
from utils.logger import Logger


class Trainer():
  def __init__(
    self, network_name, trainer_id, criterion, classes, seed, network_params=None,
    save_location='/media/data_ssd_1t/ECGnet/data_students/rarediseases/binary_classification/models',
    log_interval=10, loss_hook=None, lr=0.001, min_num_epochs=0, patience=20,
    env_prefix='', load_path=None, exclude_modules=None, freeze_modules=None,
    checkpoint_info=None, custom_metrics=None, trim_padded_batch=False,
    winning_criteria=None, eval_logging=True,
  ):
    self.network_name = network_name
    self.network_params = network_params or {}
    self.criterion = criterion
    self.seed = seed
    self.save_location = save_location
    self.log_interval = log_interval
    self.loss_hook = loss_hook
    self.trim_padded_batch = trim_padded_batch
    self.winning_criteria = winning_criteria
    self.winning_criteria_data = {}
    self.device = torch.device('cuda')
    torch.cuda.empty_cache()
    self.min_num_epochs = min_num_epochs
    self.patience = patience
    self.checkpoint_info = checkpoint_info or {}
    self.load_path = load_path
    self.model = get_network(network_name)(**network_params)
    self.load_model(exclude_modules, freeze_modules)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    self.model.to(self.device)
    self.env = trainer_id
    self.logger = Logger(classes=classes, custom_metrics=custom_metrics)
    self.save_path = os.path.join(self.save_location, self.env)
    os.makedirs(self.save_path, exist_ok=True)
    self.exists_check()
    self.n_epochs_not_winning = 0
    
  def exists_check(self):
    models = os.listdir(self.save_path)
    if any([re.search(f'seed_{str(self.seed)}.epoch_(.*).pt', m) for m in models]):
      raise ValueError(f'Saved models for seed {self.seed} already exist.')
    
  def train(self, num_epochs, trainloader, testloader):
    """
    Returns: the epoch based on a specified winning criteria
    """
    winning_epoch = None
    for epoch in tqdm(range(1, num_epochs+1)):
      self.model.train()
      self.run_epoch(trainloader, epoch)
      self.logger.reset()
      self.model.eval()
      with torch.no_grad():
        stop, winning = self.run_epoch(testloader, epoch)
      if winning:
        winning_epoch = epoch
      if stop:
        print('Early stopping and keeping best model...')
        break
    return winning_epoch

  def run_epoch(self, dataloader, epoch):
    if not self.model.training:
      y_scores = torch.empty((0)).float()
      y_probs = torch.empty((0)).float()
      y_preds = torch.empty((0)).float()
      y_trues = torch.empty((0)).float()
    for batch_idx, batch in enumerate(tqdm(dataloader)):
      waveforms = batch['waveform'].to(self.device)
      y_true = batch['label'].to(self.device)
      if self.model.training:
        self.optimizer.zero_grad()
      # trim the batch size to the length of the longest sequence
      if self.trim_padded_batch:
        waveforms = self.narrow_batch(waveforms, batch['length'])
      # perform forward propagation
      output = self.model(waveforms)
      loss, y_pred, y_true, y_score, y_prob = self.loss_fn(output, y_true)
      # perform backward propagation during training
      if self.model.training:
        loss.backward()
        self.optimizer.step()
      # add values for the loss and confusion matrix to the logger
      self.logger.loss(loss.item())
      if not self.model.training:
        # concatenate the model outputs, predicted labels and true labels
        y_scores = torch.cat((y_scores, y_score.data.cpu()), 0)
        y_probs = torch.cat((y_probs, y_prob.data.cpu()), 0)
        y_preds = torch.cat((y_preds, y_pred.data.cpu()), 0)
        y_trues = torch.cat((y_trues, y_true.data.cpu()), 0)
    # log the training process at a specified interval
    if self.model.training:
      loss_value = self.logger.log_train(epoch)
    if not self.model.training:
      report = eval_utils.classification_report(
        y_trues.numpy(), y_preds.numpy(), y_scores.numpy(), y_probs.numpy())
      loss_value = self.logger.log_eval(epoch, report)
      custom_report = self.logger.log_custom_metrics(
        epoch, y_trues.numpy(), y_preds.numpy(), y_scores.numpy(), y_probs.numpy())
      print(custom_report)
      return self.early_stopping_check(loss_value, epoch, report, custom_report)
    return False, False
  
  def narrow_batch(self, waveforms, lengths):
    max_length = max(lengths)
    return waveforms.narrow_copy(-1, 0, max_length)
  
  def early_stopping_check(self, loss_value, epoch, report, custom_report):
    """
    Use ROC AUC as the early stopping check, because the evaluation set is imbalanced.
    The evaluation loss therefore will decrease if high specificity and low sensitivity,
    which is undesirable. The goal is to get a higher ROC AUC not a lower evaluation loss.
    """
    if epoch == self.min_num_epochs:
      print('Minimum number of epochs reached...')
    # Save model if it checks the specified winning criteria
    checked_winning_criteria, self.winning_criteria_data = self.winning_criteria(
      loss_value, report, custom_report, self.winning_criteria_data)
    if self.winning_criteria_data['save_model'] == True:
      print('Saving model...')
      self.save_model(epoch, report, custom_report)
    stop = False
    if checked_winning_criteria or epoch == self.min_num_epochs:
      self.n_epochs_not_winning = 0
      return stop, checked_winning_criteria
    self.n_epochs_not_winning += 1
    print(f'Early stopping counter: {self.n_epochs_not_winning} ' \
          f'out of {self.patience}')
    if epoch > self.min_num_epochs and self.n_epochs_not_winning >= self.patience:
      stop = True
    return stop, checked_winning_criteria

  def loss_fn(self, output, y_true):
    if self.loss_hook:
      return self.loss_hook(self.model, self.criterion, output, y_true)
    y_score = output
    loss = self.criterion(y_score, y_true, training=self.model.training)
    y_prob = torch.sigmoid(y_score)
    y_pred = torch.round(y_score)
    return loss, y_pred, y_true, y_score, y_prob

  def load_model(self, exclude_modules, freeze_modules):
    if self.load_path:
      checkpoint = torch.load(self.load_path)
      checkpoint_dict = checkpoint['model_state_dict']
      model_state_dict = self.model.state_dict()
      # filter out unnecessary keys
      if exclude_modules:
        checkpoint_dict = {
          k: v for k, v in checkpoint_dict.items()
          if k in model_state_dict and
            not any(re.compile(p).match(k) for p in exclude_modules)
        }
      # overwrite entries in the existing state dict
      model_state_dict.update(checkpoint_dict)
      # load the new state dict into the model
      self.model.load_state_dict(model_state_dict)
    # freeze the network's model weights of the module names
    # provided
    if not freeze_modules:
      return
    for k, param in self.model.named_parameters():
      if any(re.compile(p).match(k) for p in freeze_modules):
        param.requires_grad = False
  
  def save_model(self, epoch, report, custom_report):
    torch.save(
      {
        'epoch': epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'network_name': self.network_name,
        'network_params': self.network_params,
        'report': report,
        'custom_report': custom_report,
        **self.checkpoint_info,
      },
      os.path.join(self.save_path, f'seed_{self.seed}.epoch_{epoch}.pt'),
    )

  def keep_best_model(self):
    eval_utils.keep_best_model(self.save_path, self.seed)


