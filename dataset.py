import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wfdb
import os
from utils import split_records_to_intervals
import matplotlib.pyplot as plt 
import numpy as np

class AF_dataset(Dataset):
    def __init__(self, folder_path, record_names, sample_length, channel, overlap = False, transform = False):
        super(AF_dataset).__init__()
        self.transform = transform
        # load all records from files:
        X = []
        y = []
        for name in record_names:
            file_name = os.path.join(folder_path, name)
            record = wfdb.rdrecord(file_name)
            annotation = wfdb.rdann(file_name, 'atr')
            intervals, annots = split_records_to_intervals(record, 
                                                           annotation, 
                                                           sample_length = sample_length, #in seconds!
                                                           channel = channel, # lead
                                                           overlap = overlap)
            X.append(intervals)
            y.append(annots)

        X = torch.vstack(X)
        y = torch.hstack(y)
        print(f'created dataset with {X.shape[0]} intervals , each with {X.shape[1]} samples')
        self.X = X
        self.y = y
        self.fs = record.fs

    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        signal = self.X[index,:]
        label = self.y[index]
        if self.transform:
            signal = self.transform(signal)

        return signal, label
     
if __name__ == '__main__':
    folder_path = 'C:/Users/nogak/Desktop/MyMaster/YoachimsCourse/files/'
    record_names = []
    for file in os.listdir(folder_path):
        if file.endswith('.hea'):  # we find only the .hea files.
            record_names.append(file[:-4])  # we remove the extensions, keeping only the number itself.
    
    ds = AF_dataset(folder_path, record_names, sample_length=10, channel=0, overlap=False)
    fs = ds.fs
    for i, idx in enumerate(np.random.randint(0, len(ds) , 6)):
        signal, label = ds[idx]
        plt.subplot(3, 2, i + 1)
        t = np.arange(0, len(signal)/fs, 1/fs)
        plt.plot(t , signal)
        # plt.xlabel('time[sec]')
        plt.title(f'Label = {label}')

    plt.show()

