import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wfdb
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from clearml import Logger
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from Classifier.utils import *
import random


class AF_dataset(Dataset):
    def __init__(self, dataset_folder_path, record_names, clearml_task = False,exp_dir = None, transform = False, config=None, d_type='No data type specified', GAN_label=-1):
        super().__init__()

        self.transform = transform
        self.folder_path = dataset_folder_path
        # Load meta data csv file from dataset folder:
        meta_data = pd.read_csv(os.path.join(dataset_folder_path,'meta_data.csv')) 
        meta_data = drop_unnamed_columns(meta_data)
        record_names = [name+'.dat' for name in record_names]
        self.meta_data = meta_data[meta_data['record_file_name'].isin(record_names)]
        print('--------------------------------------------------------------')
        print(f'created {d_type} dataset with {len(self.meta_data)} intervals')

        if GAN_label >= 0:
            self.meta_data = self.meta_data[self.meta_data['label'] == GAN_label]
            print(f'GAN {d_type} dataloader with labels {GAN_label} size is: {len(self.meta_data)}')

        if config is not None:
            # if data quality threshold is provided, remove signals that has a bsqi below threshold
            if isinstance(config['bsqi_th'], float):
                data_quality_thr = config['bsqi_th']
                pre_bsqi_size = len(self.meta_data)
                filter_series = pd.Series(self.meta_data['bsqi_score'] > data_quality_thr)
                self.meta_data = self.meta_data[filter_series] 
                print(f'Using bsqi scores to filter {d_type} dataset with threshold of {data_quality_thr}')
                print(f"bsqi filtered {d_type} from {pre_bsqi_size} to {len(self.meta_data)}") 
                assert len(self.meta_data) > 0 ,'The bsqi filtering filtered all the samples, please choose a lower threshold and run again'


            if config['debug']:
                orig_size = len(self.meta_data)
                debug_size = int(config['debug_ratio'] * orig_size)
                self.meta_data = self.meta_data.sample(n=debug_size)
                print(f'debug mode, squeeze {d_type} data from {orig_size} to {debug_size}')
            if exp_dir:
                self.meta_data.to_csv(os.path.join(exp_dir,'dataframes', d_type+'_df.csv'), index=False)
            if clearml_task:
                report_df_to_clearml(self.meta_data, clearml_task, d_type)
            print('--------------------------------------------------------------')

    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, index):
        signal_path = self.meta_data.iloc[index]['interval_path']
        signal = np.load(os.path.join(self.folder_path,'intervals',signal_path))
        label = self.meta_data.iloc[index]['label']
        meta_data = self.meta_data.iloc[index]    
        signal = signal.reshape((1, len(signal)))
        if self.transform:
            signal = self.transform(signal)

        return (signal, label), meta_data.to_dict()


if __name__ == '__main__':
    folder_path = 'C:/Users/nogak/Desktop/MyMaster/YoachimsCourse/dataset_len30_overlab5_chan0/'
    record_names = []
    for file in os.listdir('C:/Users/nogak/Desktop/MyMaster/YoachimsCourse/files'):
        if file.endswith('.hea'):  # we find only the .hea files.
            record_names.append(file[:-4])  # we remove the extensions, keeping only the number itself.
    config = load_config('config.yaml')
    ds = AF_dataset(folder_path, record_names, config=config)
    dataset_meta_data = ds.meta_data
    fs = 250
    for i, idx in enumerate(np.random.randint(0, len(ds) , 6)):
        (signal, label), meta_data = ds[idx]
        plt.subplot(3, 2, i + 1)
        t = np.arange(0, signal.shape[-1]/fs, 1/fs)
        plt.plot(t , signal.T)
        # plt.xlabel('time[sec]')
        plt.title(f'Label = {label}')

    plt.show()

