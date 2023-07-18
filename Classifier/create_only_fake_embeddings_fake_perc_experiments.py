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
    summary_dir = args.summary_dir

    output_path = os.path.join(experiments_dir, summary_dir, 'embeddings')
    pca_output_path = os.path.join(os.path.join(output_path, 'PCA'))
    t_sne_output_path = os.path.join(os.path.join(output_path, 't-SNE'))
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(pca_output_path, exist_ok=True)
    os.makedirs(t_sne_output_path, exist_ok=True)

    experiments = [exp for exp in os.listdir(experiments_dir) if exp.startswith('tcml_Classifier_mixed_data_gen2_')]
    experiments.sort(key=extract_percentage)

    pca_files = 'Train_embeddings_reduced_PCA.csv'
    t_sne_files = 'Train_embeddings_reduced_tSNE.csv'

    pca_dfs = get_file_paths(experiments_dir, experiments, pca_files)
    t_sne_dfs = get_file_paths(experiments_dir, experiments, t_sne_files)

    pca_output_dirs = [pca_output_path for _ in experiments]
    t_sne_output_dirs = [t_sne_output_path for _ in experiments]
    d_types = ['Train' for _ in experiments]
    fake_percs = [int(exp.split('_')[5]) for exp in experiments]
    only_AFs = [True for _ in experiments]
    
    fig_axes = []

    # Call the functions and store the returned figure and axes in fig_axes
    for df, output_dir, d_type, fake_perc, only_AF in zip(pca_dfs, pca_output_dirs, d_types, fake_percs, only_AFs):
        fig, ax = plot_pca_from_df_mpl(df, output_dir, d_type, fake_perc, only_AF)
        fig_axes.append((fig, ax))

    # Create the subplot grid
    create_subplot_grid(fig_axes, 4, 3, output_dir, reduce_met='PCA')

    # Close all the figure windows
    plt.close('all')

    # Empty the fig_axes list to hold the t-SNE figures and axes
    fig_axes = []

    # Call the functions and store the returned figure and axes in fig_axes
    for df, output_dir, d_type, fake_perc, only_AF in zip(t_sne_dfs, t_sne_output_dirs, d_types, fake_percs, only_AFs):
        fig, ax = plot_tsne_from_df_mpl(df, output_dir, d_type, fake_perc, only_AF)
        fig_axes.append((fig, ax))

    # Create the subplot grid
    create_subplot_grid(fig_axes, 4, 3, output_dir, reduce_met='t-SNE')
    
    # Close all the figure windows
    plt.close('all')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake experiments embddings')
    parser.add_argument('--experiments_dir', type=str, default='/tcmldrive/NogaK/ECG_classification/experiments/', help='Path to the experiments directory')    
    parser.add_argument('--summary_dir', type=str, default='fake_training_data_summary', help='Path to the experiments directory')    
    parser.add_argument('--real_data_folder_path', type=str, default='/tcmldrive/NogaK/ECG_classification/data/dataset_len6_overlab0_chan0/', help='Path to the real data directory')    
    parser.add_argument('--fake_data_folder_path', type=str, default='/tcmldrive/NogaK/ECG_classification/data/fake_data_6_secs_50000_samples_gen2/', help='Path to the fake generated data directory')    
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    args = parser.parse_args()
    main(args)
