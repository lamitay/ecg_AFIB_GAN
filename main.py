import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from clearml import Task
from utils import *
from trainer import Trainer
from model import EcgResNet34
from dataset import *


def main(config):
    
    # Define the experiment directories
    if config['user'] == 'Noga':
        data_folder_path = 'C:/Users/nogak/Desktop/MyMaster/YoachimsCourse/files/'
        exp_base_dir = 'C:/Users/nogak/Desktop/MyMaster/YoachimsCourse/exp/'
    else:
        data_folder_path = '/Users/amitaylev/Desktop/Amitay/Msc/4th semester/ML_physiological_time_series_analysis/Project/dataset/files.nosync/'
        exp_base_dir = '/Users/amitaylev/Desktop/Amitay/Msc/4th semester/ML_physiological_time_series_analysis/Project/experiments/'
    
    exp_name = f"{config['user']}_{config['experiment_name']}"
    exp_dir = build_exp_dirs(exp_base_dir, exp_name)
    
    if config['clearml']:
        clearml_task = Task.init(project_name="ecg_AFIB_GAN", task_name=exp_name)
    else:
        clearml_task=0

    torch.manual_seed(config['seed'])

    # Data
    record_names = get_record_names_from_folder(data_folder_path)
    train_records_names, validation_records_names, test_records_names = split_records_train_val_test(record_names, config['train_prec'])

    # Datasets and Dataloaders 
    train_dataset = AF_dataset(data_folder_path, train_records_names, sample_length=10, channel=0, overlap=False)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_dataset = AF_dataset(data_folder_path, validation_records_names, sample_length=10, channel=0, overlap=False)
    validation_loader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=False)
    test_dataset = AF_dataset(data_folder_path, test_records_names, sample_length=10, channel=0, overlap=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Model and optimizers config
    model = EcgResNet34(num_classes=1)
    print_model_summary(model, config['batch_size'])

    if config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    if config['loss'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    if config['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
    
    if config['user'] == 'Noga':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)


    # Training
    trainer = Trainer(model, exp_dir, train_loader, validation_loader, test_loader, optimizer, criterion, scheduler, device, config, clearml_task)
    trainer.train()

    #TODO: Add evaluation for the test set + Metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier Trainer')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')

    args = parser.parse_args()

    config = load_config(args.config)
    main(config)

