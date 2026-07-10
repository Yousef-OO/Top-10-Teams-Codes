
import os, sys
import torch
import numpy as np
from config import config
from torch.utils.data import Dataset
from scipy.io import loadmat
from keras.preprocessing import sequence
import pywt

# Load challenge data.
def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header

def twel_lead_ecg_filter(data,sampling_rate=500):
    # data=data.T
#    print(data.shape)
#    print(data)
    for i in range(12):
#        print(data[i])
#        print(data[i].shape)
#        print(wavelet_denoising(data[i]).shape)
        try:
            data[i]=wavelet_denoising(data[i])
        except:
            print(data[i])
            print(data[i].shape)
            aws=wavelet_denoising(data[i])
            print(aws)
            print(aws.shape)
            data[i]=data[i]
            print('有一个转换出错')
    return data

# 小波滤噪
def wavelet_denoising(data):
    # 小波函数取db4
    # bior = pywt.Wavelet('bior2.6')
    bior = pywt.Wavelet('db5')
    level=8
    # 分解
    coeffs = pywt.wavedec(data, bior,level)
#    print('coeffs:::::::::::::::::::::::',len(coeffs))
    # 高频系数置零
    coeffs[len(coeffs)-1] *= 0
    coeffs[len(coeffs)-2] *= 0
    coeffs[len(coeffs)-8] *= 1
    # 重构
    meta = pywt.waverec(coeffs, bior)
    mintime = 0
    maxtime = mintime + data.shape[0]
    return meta[mintime:maxtime]


class ECGDataset(Dataset):

    def __init__(self, data_path, data_info_pth_path, train=True):
        super(ECGDataset, self).__init__()
        dd = torch.load(data_info_pth_path)
        self.train = train
        self.data = dd['train'] if train else dd['val']
        self.idx2name = dd['idx2name']
        self.file2idx = dd['file2idx']
        self.wc = 1. / np.log(dd['wc'])
        self.data_path = data_path
#        if dd['wc'] == 0:
#            self.wc=1e-6
#        else:
#            self.wc = 1. / np.log(dd['wc'])

    def __getitem__(self, index):
        fid = self.data[index]

        tmp_input_file = os.path.join(self.data_path, fid + '.hea')
        ecgdata, header_data = load_challenge_data(tmp_input_file)
        filename, lead_nums, fs = header_data[0].strip().split()[:3]

        if config.lead_nums_flag:
            ecgdata = ecgdata[config.lead_index]

        # 判断采样率
        if int(fs) == 1000:
            ecgdata = ecgdata[:, ::2]

        # 是否进行小波去噪
        if config.preprocess:
            ecgdata = self.wavelet_denoising(ecgdata)

        ECG1 = sequence.pad_sequences(ecgdata, maxlen=config.seq_length, dtype='float32',truncating='post',padding='post')
#        print('padding后维度：',ECG1.shape)
        mat = np.transpose(ECG1)
#        print('转置后维度：',mat.shape)
        
        target = np.zeros(config.num_classes)
#        print(target)
        #注意index要减1，从0起算
        cur=[]
        for each in self.file2idx[fid]:
            if each != 'null':
                cur.append(each)
        target[cur] = 1
        target = torch.tensor(target, dtype=torch.float32)
        return ECG1, target

    def __len__(self):
        return len(self.data)

    # 小波滤噪
    def wavelet_denoising(self,data):
        # 小波函数取db4
        bior = pywt.Wavelet('bior2.6')
        level=8;
        # 分解
        coeffs = pywt.wavedec(data, bior,level)
    #    print('coeffs:::::::::::::::::::::::',len(coeffs))
        # 高频系数置零
        coeffs[len(coeffs)-1] *= 0
        coeffs[len(coeffs)-2] *= 0
        coeffs[len(coeffs)-8] *= 1
        # 重构
        meta = pywt.waverec(coeffs, bior)
        mintime = 0
        maxtime = mintime + data.shape[0]
        return meta[mintime:maxtime]