import torch
import pandas as pd
import numpy as np
import scipy
import scipy.io
import os
import sys
from scipy import misc, interpolate
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import fnmatch

from .cinc_utils import parse_header, split_dataset
      
class CINC2020Dataset(Dataset):
    '''Computing in Cardiology 2020 challenge dataset for pretraining'''
    def __init__(self, X, y, classes, root_dir, transform=None, num_leads=12, max_sample_length=3000):
        '''
        Args:
        root_dir (string): Directory with all the datapoints.
        transform (callable, optional): Optional transform to be applied
        on a sample.
        '''
        self.X = X
        self.y = y
        self.classes = classes
        self.root_dir = root_dir
        self.transform = transform
        self.num_leads = num_leads
        self.max_sample_length = max_sample_length
        print('CINC2020Dataset initialized\nNumber of samples: {}\nUnique classes: {}'.format(self.__len__(), self.classes))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        waveform = scipy.io.loadmat(self.generate_path(idx, 'waveform'))['val']
        length = waveform.shape[1]

        with open(self.generate_path(idx, 'header'), 'r') as f:
            header = parse_header(f.readlines())
       
        if header['sample_Fs'] != 500:
            print('Resampling signal to 500Hz')
            x = np.linspace(0, length / header['sample_Fs'], num = length)
            f = interpolate.interp1d(x, waveform, axis = 1)

            xnew = np.linspace(0, length / header['sample_Fs'], 
                               num = (length / header['sample_Fs']) * 500)
            waveform = f(xnew)   # use interpolation function returned by `interp1d`
        
        if self.max_sample_length:
            length = np.min([waveform.shape[1], self.max_sample_length])
            waveform_padded = np.zeros((waveform.shape[0], self.max_sample_length))
            waveform_padded[:, 0:length] = waveform[:, 0:length]
        
        labels = self.y[idx]
        sample = {
          'waveform': waveform_padded if self.max_sample_length else waveform,
          'header': header,
          'label': labels,
          'length': length,
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    # Generate the path to the waveform or header file
    def generate_path(self, idx, type):
        ext = 'mat' if type == 'waveform' else 'hea'
        fname = self.X[idx][0]
        return os.path.join(self.root_dir, f'{fname}.{ext}')

    # Plots the waveform of the specified lead indices.
    def plot_waveform(self, sample, lead_idx=None, interval=(0, 12500)):
        if lead_idx != None:
            fig, axs = plt.subplots(len(lead_idx), 1)
            for i, lead_ix in enumerate(lead_idx):
                axs[i].plot(sample['waveform'][lead_ix-1][interval[0]:interval[1]])
                plt.xlabel('Samples @ {} hz'.format(sample['header']['sample_Fs']))
                axs[i].set_ylabel('Lead {}'.format(lead_ix))
        else:
            fig, axs = plt.subplots(sample['waveform'].shape[0], 1)
            for i, ax in enumerate(axs):
                ax.plot(sample['waveform'][i][interval[0]:interval[1]])
                plt.xlabel('Samples @ {} hz'.format(sample['header']['sample_Fs']))
                ax.set_ylabel('Lead {}'.format(i + 1))
        fig.suptitle('ECG: {}, Label: {}'.format(sample['header']['ptID'], sample['header']['label']),fontsize=12)
        plt.show()

        
if __name__ == '__main__':

    # Initialize dataset
    root_dir = '/media/data_ssd_1t/ECGnet/data_students/cinc2020/Training_WFDB'
    X_train, y_train, X_test, y_test, classes = split_dataset(root_dir, test_size=0.1)
    trainset = CINC2020Dataset(X_train, y_train, classes, root_dir, num_leads=12)
    validset = CINC2020Dataset(X_test, y_test, classes, root_dir, num_leads=12)
    
    # Example fo how te inspect a single item in the dataset
    sample = dataset.__getitem__(0)
    dataset.plot_waveform(sample, lead_idx=[1,2,3,4,5,6], interval=(0,1000))
    
    # Example of how to create a dataloader with the dataset
    dataloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)    
    for i_batch, sample_batched in enumerate(dataloader):
        print('Sample size: {}'.format(sample_batched['waveform'].shape))
        sys.exit()

