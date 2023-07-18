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
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from scipy.linalg import sqrtm
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def main(args):
    experiments_dir = args.experiments_dir
    summary_dir = args.summary_dir

    output_path = os.path.join(experiments_dir, summary_dir, 'embeddings', 'distances')
    os.makedirs(output_path, exist_ok=True)
    
    experiments = [exp for exp in os.listdir(experiments_dir) if exp.startswith('tcml_Classifier_mixed_data_gen2_')]
    experiments.sort(key=extract_percentage)

    embedding_file_name = 'Train_embeddings.npy'
    embedding_df_file_name = 'Train_embeddings_reduced_PCA.csv'    
    embedding_files = get_file_paths(experiments_dir, experiments, embedding_file_name)
    embedding_df_files = get_file_paths(experiments_dir, experiments, embedding_df_file_name)

    # Define a dataframe to hold  distance metrics
    metrics_df = pd.DataFrame(columns=['Experiment', 'Frechet_Distance', 'Cosine_Distance', 'Euclidean_Distance'])

    # Iterate over experiments
    for i, exp in enumerate(experiments):
        fake_perc = extract_percentage(exp)

        if fake_perc == 0 or fake_perc == 100:
            frechet_distance = 0
            mean_cosine = 0
            mean_euclidean = 0
        
        else:
            # Load embeddings DataFrame
            embeddings_df = pd.read_csv(embedding_df_files[i])
            
            # Filter DataFrame for real and fake
            real_df = embeddings_df[(embeddings_df['label'] == 1) & (embeddings_df['fake'] == 0)]
            fake_df = embeddings_df[(embeddings_df['label'] == 1) & (embeddings_df['fake'] == 1)]
            
            # Get the indices
            real_indices = real_df.index.values
            fake_indices = fake_df.index.values
            
            # Load the embeddings
            embeddings = np.load(embedding_files[i])
            
            # Use the indices to get the real and fake embeddings
            real_embeddings = embeddings[real_indices]
            fake_embeddings = embeddings[fake_indices]
            
            # # # convert tensor to numpy array
            # real_embeddings = asarray(real_embeddings)
            # fake_embeddings = asarray(fake_embeddings)

            # calculate mean and covariance statistics
            mu1, sigma1 = real_embeddings.mean(axis=0), cov(real_embeddings, rowvar=False)
            mu2, sigma2 = fake_embeddings.mean(axis=0), cov(fake_embeddings, rowvar=False)

            # calculate frechet distance
            frechet_distance = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            
            # Calculate cosine and euclidean distances
            cosine_dist = pairwise_distances(real_embeddings, fake_embeddings, metric='cosine')
            euclidean_dist = pairwise_distances(real_embeddings, fake_embeddings, metric='euclidean')
            
            # Mean distances across all pairs
            mean_cosine = np.mean(cosine_dist)
            mean_euclidean = np.mean(euclidean_dist)
        
        exp_name = f'{fake_perc}%_fake_AF_class_training_set'
        
        # Concat metrics to the dataframe  
        new_row = pd.DataFrame({'Experiment': [exp_name],
                        'Frechet_Distance': [frechet_distance],
                        'Cosine_Distance': [mean_cosine],
                        'Euclidean_Distance': [mean_euclidean]})
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    
    metrics_df.to_csv(os.path.join(output_path, 'fake_percentage_embedding_distances.csv'))
    # Print the metrics DataFrame
    print(metrics_df)
 




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake experiments embddings')
    parser.add_argument('--experiments_dir', type=str, default='/tcmldrive/NogaK/ECG_classification/experiments/', help='Path to the experiments directory')    
    parser.add_argument('--summary_dir', type=str, default='fake_training_data_summary', help='Path to the experiments directory')    
    parser.add_argument('--real_data_folder_path', type=str, default='/tcmldrive/NogaK/ECG_classification/data/dataset_len6_overlab0_chan0/', help='Path to the real data directory')    
    parser.add_argument('--fake_data_folder_path', type=str, default='/tcmldrive/NogaK/ECG_classification/data/fake_data_6_secs_50000_samples_gen2/', help='Path to the fake generated data directory')    
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    args = parser.parse_args()
    main(args)
