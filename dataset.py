import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wfdb
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from clearml import Logger
from utils import create_dataset, drop_unnamed_columns, report_df_to_clearml
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

class AF_dataset(Dataset):
    def __init__(self, dataset_folder_path, exp_dir, clearml_task, record_names, transform = False, config=None, d_type='No data type specified'):
        super().__init__()

        self.transform = transform
        # Load csv files from dataset folder:
        X = pd.read_hdf(os.path.join(dataset_folder_path,'data.h5'))
        y = pd.read_csv(os.path.join(dataset_folder_path,'labels.csv'))
        y = drop_unnamed_columns(y)
        meta_data = pd.read_csv(os.path.join(dataset_folder_path,'meta_data.csv')) 
        meta_data = drop_unnamed_columns(meta_data)
        meta_data['record_file_name'] = meta_data['record_file_name'].str[:-4]#remove ".dat" from the record names
        
        self.X = X[meta_data['record_file_name'].isin(record_names)]
        self.y = y[meta_data['record_file_name'].isin(record_names)]
        self.meta_data = meta_data[meta_data['record_file_name'].isin(record_names)]
        print(f'created {d_type} dataset with {self.X.shape[0]} intervals , each with {self.X.shape[1]} samples')

        if config is not None:
            # if data quality threshold is provided, remove signals that has a bsqi below threshold
            if isinstance(config['bsqi_th'], float):
                data_quality_thr = config['bsqi_th']
                pre_bsqi_size = self.X.shape[0]
                filter_series = pd.Series(meta_data['bsqi_scores'] > data_quality_thr)
                self.X = self.X[filter_series]
                self.y = self.y[filter_series]
                self.meta_data = meta_data[filter_series]
                print(f'Using bsqi scores to filter {d_type} dataset with threshold of {data_quality_thr}')
                # print(f"Number of samples that has been filtered are {(meta_data['bsqi_scores'] < data_quality_thr).sum()}") 
                print(f"bsqi filtered {d_type} from {pre_bsqi_size} to {self.X.shape[0]}") 
                assert len(self.X) > 0 ,'The bsqi filtering filtered all the samples, please choose a lower threshold and run again'

            if config['debug']:
                orig_size = self.X.shape[0]
                debug_size = int(config['debug_ratio'] * orig_size)
                self.X = self.X[:debug_size]
                self.y = self.y[:debug_size]
                self.meta_data = self.meta_data[:debug_size]
                print(f'debug mode, squeeze {d_type} data from {orig_size} to {debug_size}') 

            self.meta_data.to_csv(os.path.join(exp_dir,'dataframes', d_type+'_df.csv'), index=False)
            if clearml_task:
                report_df_to_clearml(self.meta_data, clearml_task, d_type)



    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        signal = torch.tensor(self.X.iloc[index])
        label = torch.tensor(self.y.iloc[index])

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

