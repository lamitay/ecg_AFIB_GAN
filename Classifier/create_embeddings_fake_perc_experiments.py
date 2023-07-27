import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from utils import *
from model import EcgResNet34
from torchvision import transforms
from transform import Normalize
from dataset import *
from torch.utils.data import DataLoader

def main(args):
    experiments_dir = args.experiments_dir
    real_data_folder_path = args.real_data_folder_path
    fake_data_folder_path = args.fake_data_folder_path
    batch_size = args.batch_size

    experiments = [exp for exp in os.listdir(experiments_dir) if exp.startswith('tcml_Classifier_mixed_data_gen2_')]
    experiments.sort(key=extract_percentage)

    for exp in tqdm(experiments, desc="Processing experiments"):
        # For each experiment, we want to draw embeddings of training and test sets for the current baseline
        fake_percentage = int(exp.split('_')[5])
        curr_exp_path = os.path.join(experiments_dir, exp)
        # Create directory to save results
        embeddings_dir = os.path.join(curr_exp_path, 'embedding_results')
        os.makedirs(embeddings_dir, exist_ok=True)

        # Get the training and test sets dataframes
        train_df = pd.read_csv(os.path.join(curr_exp_path, 'dataframes', 'Train_df.csv'))
        train_df = drop_unnamed_columns(train_df)
        test_df = pd.read_csv(os.path.join(curr_exp_path, 'dataframes', 'Test_df.csv'))
        test_df = drop_unnamed_columns(test_df)

        num_workers = 4
        train_dataset = AF_mixed_dataset_from_df(meta_data_df=train_df,
                                                real_data_folder_path=real_data_folder_path,
                                                fake_data_folder_path=fake_data_folder_path,
                                                exp_dir=curr_exp_path, 
                                                clearml_task=False,
                                                transform = transforms.Compose([transforms.ToTensor(), Normalize()]), 
                                                d_type='Train')
        train_loader = DataLoader(train_dataset, 
                                batch_size=batch_size, 
                                shuffle=False,   #Shuffle is false because we only infer the training data for embeddings
                                num_workers=num_workers)
        test_dataset = AF_mixed_dataset_from_df(meta_data_df=test_df,
                                                real_data_folder_path=real_data_folder_path,
                                                fake_data_folder_path=fake_data_folder_path,
                                                exp_dir=curr_exp_path, 
                                                clearml_task=False,
                                                transform = transforms.Compose([transforms.ToTensor(), Normalize()]), 
                                                d_type='Test')
        test_loader = DataLoader(test_dataset, 
                                batch_size=batch_size, 
                                shuffle=False,   
                                num_workers=num_workers)
        
        
        # Model
        model = EcgResNet34(num_classes=1, layers=(1, 1, 1, 1))
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load the saved model
        model = load_latest_model(model, curr_exp_path)
        
        # Get embeddings
        create_and_save_embeddings(model, train_loader, embeddings_dir, 'Train', device)
        create_and_save_embeddings(model, test_loader, embeddings_dir, 'Test', device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake experiments embddings')
    parser.add_argument('--experiments_dir', type=str, default='/tcmldrive/NogaK/ECG_classification/experiments/', help='Path to the experiments directory')    
    parser.add_argument('--real_data_folder_path', type=str, default='/tcmldrive/NogaK/ECG_classification/data/dataset_len6_overlab0_chan0/', help='Path to the real data directory')    
    parser.add_argument('--fake_data_folder_path', type=str, default='/tcmldrive/NogaK/ECG_classification/data/fake_data_6_secs_50000_samples_gen2/', help='Path to the fake generated data directory')    
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    args = parser.parse_args()
    main(args)
