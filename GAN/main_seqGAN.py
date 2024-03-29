import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from clearml import Task
import sys
import matplotlib
matplotlib.use('Agg') 

sys.path.append('.')
from Classifier.utils import *
from Classifier.model import EcgResNet34
from Classifier.dataset import *
from Classifier.transform import Normalize
from GAN.models import DCDiscriminator
import random
import GAN.seq_models as seqGAN
from GAN.trainer import *

def main(config):
    
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
        exp_base_dir = '/tcmldrive/NogaK/ECG_classification/fixed_dataset_experiments/'
        records_folder_path = '/tcmldrive/NogaK/ECG_classification/files/'
        data_folder_path = '/tcmldrive/NogaK/ECG_classification/fixed_datasets/dataset_len20_overlab0_chan0'
    
    exp_name = f"{config['user']}_{config['experiment_name']}"
    exp_dir = build_exp_dirs(exp_base_dir, exp_name)
    
    if config['clearml']:
        clearml_task = Task.init(project_name="ecg_AFIB_GAN", task_name=exp_name)
    else:
        clearml_task=0

    torch.manual_seed(config['seed'])
    random.seed(config['seed'])

    # Set the seed for the dataloader workers
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)


    # Data
    record_names = get_record_names_from_folder(records_folder_path)
    train_records_names, validation_records_names, test_records_names = split_records_train_val_test(record_names, config['train_prec'])

    # Datasets and Dataloaders
    num_workers = 4
    train_dataset = AF_dataset(dataset_folder_path= data_folder_path, 
                               exp_dir= exp_dir, 
                               clearml_task= clearml_task,
                               record_names= train_records_names, 
                               transform = transforms.Compose([transforms.ToTensor(), Normalize()]), 
                               config=config, 
                               d_type='Train',
                               GAN_label=config['GAN_label'])
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['GAN_batch_size'], 
                              shuffle=True, 
                              num_workers=4, 
                              worker_init_fn=seed_worker, 
                              generator=g)
    
    validation_dataset = AF_dataset(dataset_folder_path= data_folder_path, 
                                    exp_dir= exp_dir, 
                                    clearml_task= clearml_task, 
                                    record_names= validation_records_names, 
                                    transform = transforms.Compose([transforms.ToTensor(), Normalize()]), 
                                    config=config, 
                                    d_type='Validation',
                                    GAN_label=config['GAN_label'])
    validation_loader = DataLoader(validation_dataset, 
                                   batch_size=config['GAN_batch_size'], 
                                   shuffle=False, 
                                   num_workers=4,  
                                   worker_init_fn=seed_worker, 
                                   generator=g)
    
    test_dataset = AF_dataset(dataset_folder_path= data_folder_path, 
                              exp_dir= exp_dir, 
                              clearml_task= clearml_task, 
                              record_names= test_records_names, 
                              transform = transforms.Compose([transforms.ToTensor(), Normalize()]), 
                              config=config, 
                              d_type='Test',
                              GAN_label=config['GAN_label'])
    
    test_loader = DataLoader(test_dataset, 
                             batch_size=config['GAN_batch_size'], 
                             shuffle=False,  
                             num_workers=4, 
                             worker_init_fn=seed_worker, 
                             generator=g)

    if config['user'] == 'Noga' or config['user'] == 'tcml':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    signal_length = config['sample_length'] * config['fs']
    print(f'pytorch is using {device}')

    g = seqGAN.Generator(signal_length, hidden_dim =  config['lstm_hid_dim'], n_features=config['noise_size'], tanh_output = True, num_layers = config['lstm_num_layers'])
    d = DCDiscriminator()
    
    GAN_trainer = GAN_Trainer(
        generator=g,
        discriminator=d,
        batch_size=config['GAN_batch_size'],
        num_epochs=config['GAN_epochs'],
        data_loader=train_loader,
        noise_length=config['noise_size'],
        device=device,
        label=config['GAN_label'],
        discriminator_lr = config['discriminator_lr'],
        generator_lr = config['generator_lr'],
        clearml=config['clearml'],
        exp_dir=exp_dir,
        noise_std=config['noise_std_for_real'],
        seq_model=True,
        early_stopping = config['early_stopping'],
        early_stopping_patience = config['early_stopping_patience']
    )
    
    GAN_trainer.run()

    # # Training
    # trainer = Trainer(model, exp_dir, train_loader, validation_loader, test_loader, optimizer, criterion, scheduler, device, config, clearml_task, data_folder_path)
    # print('Started training!')
    # trainer.train()
    # print('Finished training, Started test set evaluation!')
    # trainer.evaluate(data_type='test')
    # print('Finished experiment!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAN Trainer')
    parser.add_argument('--config', type=str, default='GAN/config.yaml', help='Path to the configuration file')

    args = parser.parse_args()

    config = load_config(args.config)
    main(config)

