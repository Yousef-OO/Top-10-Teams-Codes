from scipy.io import loadmat
import numpy as np
from biosppy.signals import ecg
import os
from pytorch_unet import UNet
import torch
from config import config

def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()


    return data, header_data

def judge_PVC(data,header_data):
    
    # data, header_data = load_challenge_data(input_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_class = 1
    model = UNet(num_class).to(device)
    model.load_state_dict(torch.load(config.best_cp_unet))

    maxlen = config.seq_length

    with torch.no_grad():
        ecg = np.zeros((maxlen,))
        ecg[:min(maxlen,data[1].shape[0])] = data[1][:min(maxlen,data[1].shape[0])]
        input = torch.Tensor(ecg).to(device).unsqueeze(0).unsqueeze(0)
        preds = model(input).squeeze(1)
        preds[preds < 0] = 0

    if preds.sum().item() > 0:
        return 1
    else:
        return 0
    