#!/usr/bin/env python

import numpy as np
import models
from config import config
import torch
import os
from dataset import transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False


def load_12ECG_model(model_input):
    # load the model from disk
    model = getattr(models, config.model_name)()
    model.load_state_dict(
        torch.load(os.path.join(model_input, 'best_w.pth'), map_location='cpu')['state_dict'])
    model = model.to(device)
    model.eval()

    return model


def run_12ECG_classifier(data, header_data, model):

    # num_classes = len(classes)
    classes = ['10370003','111975006','164889003','164890007','164909002','164917005','164934002',
            '164947007','17338001','251146004','270492004','284470004','39732003','426177001','426627000',
            '426783006','427084000','427172004','427393009','445118002','47665007',
            '59118001','59931005','63593006','698252002','713426002','713427006']

    # Use your classifier here to obtain a label and score for each class.
    with torch.no_grad():
        data = data.transpose()
        ecg_len = data.shape[0]
        cut_pos = int(ecg_len / 5000)
        if cut_pos == 0:
            cut_pos = 1
        elif ecg_len % 5000 != 0 :
            cut_pos += 1
        X = np.ones(shape=(1, 5000, 12)) * np.nan
        for i in range(cut_pos):
            cut_start = 5000 * i
            cut_end = cut_start + config.fs * config.ecg_time
            ECG_temp = np.zeros((config.fs * config.ecg_time, 12))
            if cut_end > ecg_len and ecg_len < 5000:
                ECG_temp[0: ecg_len - cut_start, :] = data[cut_start: ecg_len, :]
            elif cut_end > ecg_len and ecg_len > 5000:
                ECG_temp = data[ecg_len - 5000: ecg_len, :]
            else:
                ECG_temp = data[cut_start: cut_end, :]
            X = ECG_temp
            x = transform(X).unsqueeze(0).to(device)
            output = torch.sigmoid(model(x)).squeeze().cpu().numpy()
            if i == 0:
                result = output
            else:
                for j, out in enumerate(output):
                    result[j] = max(result[j], out)
        max_pro = max(result)
        current_score = result
        if i == cut_pos - 1:
            current_label = [1 if out > max_pro / 5.0 else 0 for out in current_score]
        return current_label, current_score, classes