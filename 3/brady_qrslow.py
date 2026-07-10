from scipy.io import loadmat
import numpy as np
from biosppy.signals import ecg
import os

def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()


    return data, header_data
    
def get_RR_interval(data, header_data):
    
    tmp_hea = header_data[0].split(' ')
    tmp_hea1 = header_data[1].split(' ')
    sample_Fs = int(tmp_hea[2])
    gain = int(tmp_hea1[2].split('/')[0])
    
    out = ecg.ecg(data[0], sampling_rate=sample_Fs, show=False)
    
    idx = out['rpeaks']
    
    RR = np.array(np.ediff1d(idx))

    return RR

def get_R_amplitudes(data, header_data):
    
    tmp_hea = header_data[0].split(' ')
    tmp_hea1 = header_data[1].split(' ')
    sample_Fs = int(tmp_hea[2])
    
    leads = []
    
    for i in range(12):
        
        out = ecg.ecg(data[i], sampling_rate=sample_Fs, show=False)
        try:
            ramps = out['filtered'][out['rpeaks']]
        except:
            ramps = None
        leads.append(ramps)
    
    return leads

def judge_bradycardia(input_file):
    data, header_data = load_challenge_data(input_file)
    RR = get_RR_interval(data, header_data)
    bin_RR = ((RR> 500)&(RR< 800))+0
    mean_RR = 0 if len(bin_RR)<6 else bin_RR.mean()

    if mean_RR > 0.5:
        return 1
    elif mean_RR < 0.5:
        return 0
    else:
        return 'error'
    
def judge_qrslowvol(data, header_data):
    # data, header_data = load_challenge_data(input_file)
    leads = get_R_amplitudes(data, header_data)
    
    low_v_count = 0
    
    for ramps in leads[:6]:
        if ramps is None:
            continue
        if ramps.mean() <= 500:
            low_v_count += 1
            
    for ramps in leads[6:]:
        if ramps is None:
            continue
        if ramps.mean() <= 1000:
            low_v_count += 1
            
    if low_v_count >= 6:
        return 1
    else:
        return 0
    
#input_file =  '/wd-nas-0/ephwha/ephwha/physionet2020/Data/test/Q0060.mat'
#predict_bradycardia = judge_bradycardia(input_file) 
#print(predict_bradycardia)