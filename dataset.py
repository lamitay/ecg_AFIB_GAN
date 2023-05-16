import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wfdb
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from utils import create_dataset
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

class AF_dataset(Dataset):
    def __init__(self, dataset_folder_path, record_names, transform = False, data_quality_thr = None):
        super().__init__()
        self.transform = transform
        # Load csv files from dataset folder:
        X = pd.read_hdf(os.path.join(dataset_folder_path,'data.h5'))
        y = pd.read_csv(os.path.join(dataset_folder_path,'labels.csv'))
        meta_data = pd.read_csv(os.path.join(dataset_folder_path,'meta_data.csv')) 
        meta_data['record_file_name'] = meta_data['record_file_name'].str[:-4]#remove ".dat" from the record names

        self.X = X[meta_data['record_file_name'].isin(record_names)]
        self.y = y[meta_data['record_file_name'].isin(record_names)]
        self.meta_data = meta_data[meta_data['record_file_name'].isin(record_names)]
        print(f'created dataset with {self.X.shape[0]} intervals , each with {self.X.shape[1]} samples')
        # if data quality threshold is provided, remove signals that has a bsqi below threshold
        if isinstance(data_quality_thr, float):
            self.X = X[meta_data['bsqi_scores'] > data_quality_thr]
            self.y = y[meta_data['bsqi_scores'] > data_quality_thr]
            self.meta_data = meta_data[meta_data['bsqi_scores'] > data_quality_thr]
            print(f'Using bsqi scores to filter dataset with threshold of {data_quality_thr}')
            print(f"Number of samples that has been filtered are {(meta_data['bsqi_scores'] < data_quality_thr).sum()}") 
            assert len(self.X) > 0 ,'The bsqi filtering filtered all the samples, please choose a lower threshold and run again'

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        signal = torch.tensor(self.X.iloc[index])
        label = self.y.iloc[index][1]

        if self.transform:
            signal = self.transform(signal)

        return signal, label

if __name__ == '__main__':
    folder_path = 'C:/Users/nogak/Desktop/MyMaster/YoachimsCourse/dataset_30_10_0/'
    record_names = []
    for file in os.listdir('C:/Users/nogak/Desktop/MyMaster/YoachimsCourse/files'):
        if file.endswith('.hea'):  # we find only the .hea files.
            record_names.append(file[:-4])  # we remove the extensions, keeping only the number itself.
    
    ds = AF_dataset(folder_path, record_names[0:10], sample_length=30, channel=0, overlap=10, data_quality_thr=0.8)
    fs = 250
    for i, idx in enumerate(np.random.randint(0, len(ds) , 6)):
        signal, label = ds[idx]
        plt.subplot(3, 2, i + 1)
        t = np.arange(0, len(signal)/fs, 1/fs)
        plt.plot(t , signal)
        # plt.xlabel('time[sec]')
        plt.title(f'Label = {label}')

    plt.show()

