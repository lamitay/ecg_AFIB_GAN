import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from clearml import Task
from utils import *
from trainer import Trainer
from model import EcgResNet34
from dataset import *
from transform import Normalize
import random

def main(config):
    # Load the dataset:
    exp_name = f"{config['user']}_{config['experiment_name']}"
    exp_base_dir = '/tcmldrive/NogaK/ECG_classification/experiments/'
    exp_dir = build_exp_dirs(exp_base_dir, exp_name)
    records_folder_path = '/tcmldrive/NogaK/ECG_classification/files/'
    data_folder_path = '/tcmldrive/NogaK/ECG_classification/data/physioNet_challenge_dataset_len6_overlab0_chan0/'
    meta_data = pd.read_csv(os.path.join(data_folder_path, 'meta_data.csv'))
    records_names = pd.unique(meta_data['record_file_name']).tolist()
    dataset = AF_dataset(dataset_folder_path = data_folder_path, 
                        exp_dir = exp_dir, 
                        clearml_task = False,
                        record_names = records_names, 
                        transform = transforms.Compose([transforms.ToTensor(), Normalize()]), 
                        config  =config, 
                        d_type = 'Train')
    labels = dataset.meta_data['label']
    print(f'label 0: {len(labels[labels==False])}   |   Prec: {"{:.2f}".format(len(labels[labels==False])/len(labels))}%') 
    print(f'label 1: {len(labels[labels==True])}   |   Prec: {"{:.2f}".format(len(labels[labels==True])/len(labels))}%')
    loader = DataLoader(dataset, 
                             batch_size=config['batch_size'], 
                             shuffle=False,  
                             num_workers=4)
    model = EcgResNet34(num_classes=1, layers=(1, 1, 1, 1))
    if config['user'] == 'Noga' or config['user'] == 'tcml':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    criterion = F.binary_cross_entropy
    trainer = Trainer(model,
                      exp_dir, 
                      train_loader = None, 
                      validation_loader = None, 
                      test_loader = loader, 
                      optimizer = None, 
                      criterion = criterion, 
                      scheduler = None, 
                      device = device, 
                      config = config, 
                      clearml_task = False, 
                      dataset_path = data_folder_path)
    
    trainer.evaluate(data_type='test', different_exp_dir = '/tcmldrive/NogaK/ECG_classification/experiments/tcml_classifier_real_data_training_balanced_classes_20230630_18_28_45')#/models/epoch_16_model.pth')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='External Dataset Evaluation')
    parser.add_argument('--config', type=str, default='Classifier/classifier_config.yaml', help='Path to the configuration file')

    args = parser.parse_args()

    config = load_config(args.config)
    main(config)