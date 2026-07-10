# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 19:47

@ author: javis
'''
import os
import copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from scipy import signal


def resample(sig, target_point_num=None):
    '''
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    '''
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig


def transform(sig, train=False):
    # 前置不可或缺的步骤
    sig = resample(sig, config.target_point_num)
    # 数据增强
   
    # 后置不可或缺的步骤
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, output_directory, data_path, train=True):
        super(ECGDataset, self).__init__()
        # dd = torch.load(config.train_data)
        dd = torch.load(data_path)
        self.train = train
        self.data = dd['train'] if train else dd['val']
        self.idx2name = dd['idx2name']
        self.file2idx = dd['file2idx']
        self.wc = 1
        self.output_directory = output_directory

    def __getitem__(self, index):
        fid = self.data[index]
        file_path = os.path.join(self.output_directory, fid)
        df = pd.read_csv(file_path, sep=' ').values
        x = transform(df, self.train)
        target = np.zeros(config.num_classes)
        target[self.file2idx[fid]] = 1
        target = torch.tensor(target, dtype=torch.float32)
        return x, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # d = ECGDataset(config.train_data)
    # print(d[0])
    # dd = torch.load(config.train_data)
    # print(dd['val'])
    # a = np.concatenate((np.zeros(shape=(2000, 12)), np.zeros(shape=(2000, 12))))
    # print(a.shape)
    import torchvision
    print(torchvision.__version__)


