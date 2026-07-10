#!/usr/bin/env python

from data_process import write_classes, cut_data_folds, train_27, name2index
from config import config
from torch.utils.data import DataLoader
from dataset import ECGDataset
from utils import WeightedMultilabel, adjust_learning_rate, compute_beta_score
from torch import nn, optim
import os
import torch
import models
import utils
import numpy as np
from scipy.io import loadmat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False


def train_12ECG_classifier(input_directory, output_directory):
    print("process data...")
    process_data(input_directory, output_directory)

    print("training...")
    model = getattr(models, config.model_name)()
    model = model.to(device)

    train_dataset = ECGDataset(output_directory, data_path=os.path.join(output_directory, 'train.pth'), train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_dataset = ECGDataset(output_directory, data_path=os.path.join(output_directory, 'train.pth'), train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
    criterion = WeightedMultilabel(w)

    model_save_dir = output_directory
    best_f1 = -1
    best_f_beta = -1
    best_g_beta = -1
    min_loss = 100
    lr = config.lr
    start_epoch = 1
    stage = 1
    patience = 5

    for epoch in range(start_epoch, config.max_epoch + 1):
        # since = time.time()
        train_loss, train_acc, train_f1, train_f_beta, train_g_beta = train_epoch(model, optimizer, criterion,
                                                                                  train_dataloader, show_interval=100)
        val_loss, val_acc, val_f1, val_f_beta, val_g_beta = val_epoch(model, criterion, val_dataloader)

        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'acc': val_acc,
                 'f_beta': val_f_beta, 'g_beta': val_g_beta, 'lr': lr,
                 'stage': stage}

        if val_loss > min_loss:
            patience -= 1
        else:
            save_ckpt(state, val_loss < min_loss, model_save_dir)
            patience = 5
        if patience == 0:
            print("Early stopping.")
            exit(0)
        min_loss = min(min_loss, val_loss)
        if epoch in config.stage_epoch:
            stage += 1
            # lr /= config.lr_decay
            lr = config.stage_lr[stage - 2]
            best_w = os.path.join(model_save_dir, config.best_w)
            model.load_state_dict(torch.load(best_w)['state_dict'])
            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            adjust_learning_rate(optimizer, lr)


# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header


# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)


def process_data(input_directory, output_directory):
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
                'mat'):
            input_files.append(f)
    write_classes(input_directory, output_directory, input_files)
    cut_data_folds(input_directory, output_directory, 5)
    name2idx = name2index(os.path.join(output_directory, 'arrythmia.txt'))
    idx2name = {idx: name for name, idx in name2idx.items()}
    train_27(name2idx, idx2name, 1, input_directory, output_directory)


def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=10):
    model.train()
    # f1_meter, loss_meter, it_count = 0, 0, 0
    accuracy_meter, f_measure_meter, f_beta_meter, g_beta_meter, loss_meter, it_count = 0, 0, 0, 0, 0, 0
    for inputs, target in train_dataloader:
        inputs = inputs.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        # f1 = utils.calc_f1(target, torch.sigmoid(output))
        # f1_meter += f1
        accuracy, f_measure, f_beta, g_beta = utils.compute_beta_score(target, output, beta=2, num_classes=config.num_classes)
        accuracy_meter += accuracy
        f_measure_meter += f_measure
        f_beta_meter += f_beta
        g_beta_meter += g_beta
    return loss_meter / it_count, accuracy_meter / it_count, f_measure_meter / it_count, f_beta_meter / it_count, g_beta_meter / it_count


def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    # f1_meter, loss_meter, it_count = 0, 0, 0
    accuracy_meter, f_measure_meter, f_beta_meter, g_beta_meter, loss_meter, it_count = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for inputs, target in val_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            # f1 = utils.calc_f1(target, output, threshold)
            # f1_meter += f1
            accuracy, f_measure, f_beta, g_beta = compute_beta_score(target, output, beta=2, num_classes=config.num_classes)
            accuracy_meter += accuracy
            f_measure_meter += f_measure
            f_beta_meter += f_beta
            g_beta_meter += g_beta

    return loss_meter / it_count, accuracy_meter / it_count, f_measure_meter / it_count, f_beta_meter / it_count, g_beta_meter / it_count


def save_ckpt(state, is_best, model_save_dir):
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, best_w)


