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


def main(config, exp_name=None, train_fake_perc=10):
    
    # Define the experiment directories
    if config['user'] == 'Noga':
        records_folder_path = 'C:/Users/nogak/Desktop/MyMaster/YoachimsCourse/files/'
        exp_base_dir = 'C:/Users/nogak/Desktop/MyMaster/YoachimsCourse/exp/'
        data_folder_path = 'C:/Users/nogak/Desktop/MyMaster/YoachimsCourse/dataset_len30_overlab5_chan0/'
    elif config['user'] == 'Amitay':
        records_folder_path = '/Users/amitaylev/Desktop/Amitay/Msc/4th semester/ML_physiological_time_series_analysis/Project/dataset/files.nosync/'
        exp_base_dir = '/Users/amitaylev/Desktop/Amitay/Msc/4th semester/ML_physiological_time_series_analysis/Project/experiments/'
        data_folder_path = '/Users/amitaylev/Desktop/Amitay/Msc/4th semester/ML_physiological_time_series_analysis/Project/dataset_processed.nosync/dataset_30_10_0/'
    elif config['user'] == 'tcml':
        exp_base_dir = '/tcmldrive/NogaK/ECG_classification/experiments/'
        records_folder_path = '/tcmldrive/NogaK/ECG_classification/files/'
        data_folder_path = '/tcmldrive/NogaK/ECG_classification/data/dataset_len6_overlab0_chan0/'
    
    if exp_name is None:
        exp_name = f"{config['user']}_{config['experiment_name']}_{train_fake_perc}_fake_percent"
    exp_dir = build_exp_dirs(exp_base_dir, exp_name)
    
    if config['clearml']:
        clearml_task = Task.init(project_name="ecg_AFIB_GAN", task_name=exp_name)
        clearml_task.connect_configuration(config)
    else:
        clearml_task=0

    torch.manual_seed(config['seed'])
    random.seed(config['seed'])

    # Datasets and Dataloaders
    num_workers = 4
    train_dataset = AF_mixed_dataset(real_data_folder_path=config['real_data_path'],
                                    fake_data_folder_path=config['fake_data_path'],
                                    exp_dir= exp_dir, 
                                    clearml_task= clearml_task,
                                    transform = transforms.Compose([transforms.ToTensor(), Normalize()]), 
                                    config=config, 
                                    d_type='Train',
                                    train_fake_perc=train_fake_perc)
    print('class distribution for the train_dataset')
    print_dataset_distribution(train_dataset)
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['batch_size'], 
                              shuffle=True, 
                              num_workers=num_workers)
    
    validation_dataset = AF_mixed_dataset(real_data_folder_path=config['real_data_path'],
                                        fake_data_folder_path=config['fake_data_path'],
                                        exp_dir= exp_dir, 
                                        clearml_task= clearml_task, 
                                        transform = transforms.Compose([transforms.ToTensor(), Normalize()]), 
                                        config=config, 
                                        d_type='Validation')
    print('class distribution for the validation_dataset')
    print_dataset_distribution(validation_dataset)
    validation_loader = DataLoader(validation_dataset, 
                                   batch_size=config['batch_size'], 
                                   shuffle=False, 
                                   num_workers=num_workers)
    
    test_dataset = AF_mixed_dataset(real_data_folder_path=config['real_data_path'],
                                    fake_data_folder_path=config['fake_data_path'],   
                                    exp_dir= exp_dir, 
                                    clearml_task= clearml_task, 
                                    transform = transforms.Compose([transforms.ToTensor(), Normalize()]), 
                                    config=config, 
                                    d_type='Test')
    print('class distribution for the test_dataset')
    print_dataset_distribution(test_dataset)
    test_loader = DataLoader(test_dataset, 
                             batch_size=config['batch_size'], 
                             shuffle=False,  
                             num_workers=num_workers)

    # Model and optimizers config
    model = EcgResNet34(num_classes=1, layers=(1, 1, 1, 1))
    if config['user'] == 'Noga' or config['user'] == 'tcml':
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print_model_summary(model, config['batch_size'], device='cpu')
    model.to(device)
    if config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    if config['loss'] == 'binary_cross_entropy':    
        criterion = F.binary_cross_entropy
    if config['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
    
    # Training
    trainer = Trainer(model, exp_dir, train_loader, validation_loader, test_loader, optimizer, criterion, scheduler, device, config, clearml_task, data_folder_path)
    print('Started training!')
    trainer.train()
    print('Finished training, Started test set evaluation!')
    trainer.evaluate(data_type='test')
    print('Finished experiment!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier Trainer')
    parser.add_argument('--config', type=str, default='Classifier/classifier_config.yaml', help='Path to the configuration file')
    parser.add_argument('--train_fake_perc', type=int, default=10, help='Perecentage of fake data in the training set')
    args = parser.parse_args()
    config = load_config(args.config)
    train_fake_perc = args.train_fake_perc
    # exp_name = 'Classifier_mixed_data_gen2'
    exp_name = None
    main(config, exp_name, train_fake_perc)

