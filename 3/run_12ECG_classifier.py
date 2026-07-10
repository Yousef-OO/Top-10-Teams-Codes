#!/usr/bin/env python

import numpy as np, os, sys
from get_12ECG_features import get_12ECG_features

import models_A, models_B
import torch
from config import config
from scipy.io import loadmat
from keras.preprocessing import sequence
import rule_model_bradycardia
import PVC_predict
import brady_qrslow
from scipy.signal import resample
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(41)
torch.cuda.manual_seed(41)

dd = torch.load(config.train_data)
label_classes = list(dd['idx2name'].values())


def run_12ECG_classifier(data, header_data, loaded_model):

    model_A, model_B = loaded_model
    weight_A = config.weight_A
    weight_B = config.weight_B

    ecgdata = data

    # if config.lead_nums_flag:
    #    criterion = nn.BCEWithLogitsLoss()
    model_A.eval()
    model_B.eval()

    filename, lead_nums, fs = header_data[0].strip().split()[:3]
    gain = header_data[1].strip().split()[2].rstrip('./mV')

    # 判断采样率
    fs = int(fs)

    new_ecgdata = np.zeros((ecgdata.shape[0], int(ecgdata.shape[1] / fs * 500)))
    if fs != 500 and fs>0:
        for ind_ in range(ecgdata.shape[0]):
            new_ecgdata[ind_, :] = resample(ecgdata[ind_, :], int(ecgdata.shape[1] / fs * 500))
        ecgdata = new_ecgdata
    ecgdata1 = ecgdata[config.lead_index]


    ECG1 = sequence.pad_sequences(ecgdata1, 15000, dtype='float32', truncating='post', padding='post')
    #    print(ECG1.shape)
    ECG1 = ECG1[np.newaxis, :, :]
    #    print(ECG1.shape)
    ECG1 = torch.from_numpy(ECG1)
    ECG1 = ECG1.to(device)

    ECG_B = sequence.pad_sequences(ecgdata1, maxlen=config.seq_length, dtype='float32', truncating='post', padding='post')
    #    print(ECG1.shape)
    ECG_B = ECG_B[np.newaxis, :, :]
    #    print(ECG1.shape)
    ECG_B = torch.from_numpy(ECG_B)
    ECG_B = ECG_B.to(device)


    y_pred_A = model_A(ECG1)
    y_pred_A = torch.sigmoid(y_pred_A)
    y_pred_A = y_pred_A.cpu().detach().numpy()
    prob_out_A = y_pred_A[0].tolist()
    prob_out_A = np.asarray(prob_out_A, dtype=np.float64)

    y_pred_B = model_B(ECG_B)
    y_pred_B = torch.sigmoid(y_pred_B)
    y_pred_B = y_pred_B.cpu().detach().numpy()
    prob_out_B = y_pred_B[0].tolist()
    prob_out_B = np.asarray(prob_out_B, dtype=np.float64)

    prob_out = weight_A*prob_out_A + weight_B*prob_out_B

    thresh_H_list = config.thresh_H_list
    thresh_L_list = config.thresh_L_list
    threshold_H = config.threshold_H
    threshold_L = config.threshold_L

    label = []
    threshold = 0.36
    if config.two_threshold:
        for i in range(config.num_classes):
            if i in thresh_H_list:
                threshold = threshold_H
            elif i in thresh_L_list:
                threshold = threshold_L
            if np.average(prob_out[i]) >= threshold:
                label.append(1)
            else:
                label.append(0)
    else:
        for i in range(config.num_classes):
            if np.average(prob_out[i]) >= threshold:
                label.append(1)
            else:
                label.append(0)
    label = np.array(label, dtype=np.int32)

    # 检测心动过缓
    if config.brady_rule:
        try:
            predict_bradycardia = rule_model_bradycardia.judge_bradycardia(data, header_data)
        except Exception as  e:
            print(filename)
            print(e)
            predict_bradycardia = 'null'
        # 心动过缓 index = 18(从0起算)
        if label[18] == 1:
            if predict_bradycardia == 0:
                label[18] = 0
                prob_out[18] = 0
    #unet检测PVC  PVC index = 21 (从0起算)
    if config.unet:
        predict_pvc = PVC_predict.judge_PVC(data, header_data)
        if label[21] == 1:
            if predict_pvc == 0:
                label[21] = 0
                prob_out[21] = 0
        if label[21] == 0:
            if predict_pvc == 1:
                label[21] = 1
                prob_out[21] = 1

    #qrs低电压 index = 14(从0起算)
    if config.qrs_rule:
        try:
            predict_qrs = brady_qrslow.judge_qrslowvol(data, header_data)
        except:
            predict_qrs = 'null'
        if label[14] == 1:
            if predict_qrs == 0:
                label[14] = 0
                prob_out[14] = 0


    if config.prob_revise:
        if np.sum(label) == 0:
            label[19] = 1
            prob_out[19] = 1


    label_dict = np.asarray(label, dtype=np.int32)
    prob_out = np.asarray(prob_out, dtype=np.float64)
    return label_dict, prob_out, label_classes

    # 以下是官网的return
    # return current_label, current_score, classes

def load_12ECG_model(input_directory):
    # load the model from disk
    model_A = getattr(models_A, config.model_name)()
    weights_A_path = config.best_w_15K_8L  # 模型权重路径
    model_A.load_state_dict(torch.load(weights_A_path, map_location='cpu')['state_dict'])
    model_A = model_A.to(device)

    model_B = getattr(models_B, config.model_name)()        # 加载模型文件
    weights_B_path = os.path.join(input_directory, config.best_w)        # 模型权重路径
    model_B.load_state_dict(torch.load(weights_B_path, map_location='cpu')['state_dict'])
    model_B = model_B.to(device)

    model = [model_A, model_B]

    return model
