#!/usr/bin/env python

import numpy as np
import os
import torch
from torch import nn
import warnings
import models
from scipy.signal import resample
import math
import pandas as pd
from get_12ECG_features import get_12ECG_features

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if torch.cuda.is_available():
    device = torch.device("cuda")
    device_count = torch.cuda.device_count()
else:
    warnings.warn("gpu is not available")
    device = torch.device("cpu")

def Resample(input_signal, src_fs, tar_fs):
    '''
    :param input_signal:输入信号
    :param src_fs:输入信号采样率
    :param tar_fs:输出信号采样率
    :return:输出信号
    '''
    dtype = input_signal.dtype
    audio_len = input_signal.shape[1]
    audio_time_max = 1.0 * (audio_len) / src_fs
    src_time = 1.0 * np.linspace(0, audio_len, audio_len) / src_fs
    tar_time = 1.0 * np.linspace(0, np.int(audio_time_max * tar_fs), np.int(audio_time_max * tar_fs)) / tar_fs
    for i in range(input_signal.shape[0]):
        if i == 0:
            output_signal = np.interp(tar_time, src_time, input_signal[i, :]).astype(dtype)
            output_signal = output_signal.reshape(1, len(output_signal))
        else:
            tmp = np.interp(tar_time, src_time, input_signal[i, :]).astype(dtype)
            tmp = tmp.reshape(1, len(tmp))
            output_signal = np.vstack((output_signal, tmp))
    return output_signal

def processing_data(data, win_length, src_fs, tar_fs):

    """
    Add any preprocessing at here
    """
    data = Resample(data, src_fs, tar_fs)
    num = data.shape[1]
    if num < win_length:
        zeros_padding = np.zeros(shape=(data.shape[0], win_length - num), dtype=np.float32)
        data = np.hstack((data, zeros_padding))
    data = data.astype(np.float32)
    data = torch.from_numpy(data)
    data = torch.unsqueeze(data, 0)

    return data

def prepare_data(age, gender):
    data = np.zeros(5,)
    if age >= 0:
        data[0] = age / 100
    if 'F' in gender:
        data[2] = 1
        data[4] = 1
    elif gender == 'Unknown':
        data[4] = 0
    elif 'f' in gender:
        data[2] = 1
        data[4] = 1
    else:
        data[3] = 1
        data[4] = 1
    return data
    
def read_ag(header_data):
    for lines in header_data:
        if lines.startswith('#Age'):
            tmp = lines.split(': ')[1].strip()
            if tmp == 'NaN':
                age = -1
            else:
                age = int(tmp)
        if lines.startswith('#Sex'):
            tmp = lines.split(': ')[1].strip()
            if tmp == 'NaN':
                gender = 'Unknown'
            else:
                gender = tmp
    data = prepare_data(age, gender)
    data = torch.from_numpy(data).float()
    data = torch.unsqueeze(data, 0)
    return data

def run_12ECG_classifier(data, header_data, model):
    A = np.load('./magic_weight.npz')
    threshold = A['arr_0']
    num_classes = 24
    tar_fs = 257
    src_fs = int(header_data[0].split(' ')[2].strip())
    ag = read_ag(header_data)
    ag = ag.to(device)
    
    
    win_length = 4096
    current_label = np.zeros(num_classes, dtype=int)

    m = nn.Sigmoid()
    data = processing_data(data, win_length, src_fs, tar_fs)
    inputs = data.to(device)
    # Use your classifier here to obtain a label and score for each class.


    val_length = inputs.shape[2]
    overlap = 256
    patch_number = math.ceil(abs(val_length - win_length) / (win_length - overlap)) + 1
    if patch_number > 1:
        start = int((val_length - win_length) / (patch_number - 1))
    prob = 0
    logits_prob = 0
    for j in range(len(model)):
        model_one = model[j]
        for i in range(patch_number):
            if i == 0:
                logit = model_one(inputs[:, :, 0: val_length], ag)
                logits_prob = m(logit)
            elif i == patch_number - 1:
                logit = model_one(inputs[:, :, val_length - win_length: val_length], ag)
                logits_prob_tmp = m(logit)
                logits_prob = (logits_prob + logits_prob_tmp) / patch_number
            else:
                logit = model_one(inputs[:, :, i * start:i * start + win_length], ag)
                logits_prob_tmp = m(logit)
                logits_prob = logits_prob + logits_prob_tmp


        prob = prob + logits_prob

    logits_prob = prob / len(model)


    _, y_pre_label = torch.max(logits_prob, 1)
    y_pre_label = y_pre_label.cpu().detach().numpy()
    current_label[y_pre_label] = 1

    score = logits_prob.cpu().detach().numpy()

    y_pre = (score - threshold) >= 0
    current_label = current_label + y_pre
    current_label[current_label > 1.1] = 1

    current_label = np.squeeze(current_label)
    current_score = np.squeeze(score)

    # Get the label
    label_file_dir = './utils/dx_mapping_scored.csv'
    label_file = pd.read_csv(label_file_dir)
    equivalent_classes = ['59118001', '63593006', '17338001']
    classes = sorted(list(set([str(name) for name in label_file['SNOMED CT Code']]) - set(equivalent_classes)))

    return current_label, current_score, classes


def load_12ECG_model(model_input):
    # load the model from disk
    model_list = lsdir(rootdir=model_input, suffix=".pth")
    accuracy = np.array([float(i.split('-')[-2]) for i in model_list])
    resumes = [model_list[int(np.argmax(accuracy))]]
    model_all = []

    for resume in resumes:
        model = getattr(models, 'seresnet18_1d_ag')(in_channel=12, out_channel=24)
        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            if device_count > 1:
                model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(resume))
        else:
            model.load_state_dict(torch.load(resume, map_location=device))

        model.to(device)
        model.eval()
        model_all.append(model)

    return model_all


def lsdir(rootdir="", suffix=".png"):
    file_list = []
    assert os.path.exists(rootdir)
    for r, y, names in os.walk(rootdir):
        for name in names:
            if str(name).endswith(suffix):
                file_list.append(os.path.join(r, name))
    return file_list
