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
    output_dir = args.output_dir
    real_data_folder_path = args.real_data_folder_path
    fake_data_folder_path = args.fake_data_folder_path
    batch_size = args.batch_size

    output_dir_path = os.path.join(experiments_dir, output_dir)

    os.makedirs(output_dir_path, exist_ok=True)

    experiments = [exp for exp in os.listdir(experiments_dir) if exp.startswith('tcml_Classifier_mixed_data_gen2_')]

    for exp in experiments:
        # For each experiment, we want to draw embeddings of training and test sets for the current baseline
        fake_percentage = int(exp.split('_')[5])
        curr_exp_path = os.path.join(experiments_dir, exp)
        # Create directory to save results
        embeddings_dir = os.path.join(curr_exp_path, 'embedding_results')
        os.makedirs(embeddings_dir, exist_ok=True)

        # Get the training and test sets dataframes
        train_df = pd.read_csv(os.path.join(curr_exp_path, 'dataframes', 'Train'))
        train_df = drop_unnamed_columns(train_df)
        test_df = pd.read_csv(os.path.join(curr_exp_path, 'dataframes', 'Test'))
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load the saved model
        model = load_latest_model(model, curr_exp_path)
        
        # Get embeddings
        create_and_save_embeddings(model, train_loader, embeddings_dir, 'Train', curr_exp_path)
        create_and_save_embeddings(model, test_loader, embeddings_dir, 'Test', curr_exp_path)
        
        # embeddings = []
        # labels = []
        # preds = []
        # fakes = []
        # intervals = []

        # with torch.no_grad():
        #     for (inputs, targets), meta_data in tqdm(train_loader):
        #         inputs = inputs.to(device).squeeze(1)
        #         targets = targets.to(device).float()

        #         embedding = model(inputs, return_embedding=True)
        #         outputs = model(inputs).squeeze(1)
                
        #         # Threshold the predictions
        #         thresholded_predictions = np.where(outputs.cpu().numpy() >= config['classifier_th'], 1, 0)        

        #         embeddings.append(embedding.cpu().numpy())
        #         labels.append(targets.cpu().numpy())
        #         preds.append(thresholded_predictions.astype(int))
        #         fakes.append(meta_data['fake'])
        #         intervals.append(meta_data['interval_path'])

        # embeddings = np.concatenate(embeddings)
        # labels = np.concatenate(labels)
        # preds = np.concatenate(preds)
        # fakes = np.concatenate(fakes)
        # intervals = np.concatenate(intervals)

        # # Save the embeddings
        # np.save(os.path.join(embeddings_dir, 'embeddings.npy'), embeddings)

        # # Reduce dimensionality
        # pca = PCA(n_components=2)
        # embeddings_reduced = pca.fit_transform(embeddings)

        # # Save the reduced embeddings
        # np.save(os.path.join(embeddings_dir, 'embeddings_reduced.npy'), embeddings_reduced)

        # # Create a DataFrame
        # df = pd.DataFrame(embeddings_reduced, columns=['component1', 'component2'])
        # df['label'] = labels
        # df['prediction'] = preds
        # df['fake'] = fakes
        # df['interval_path'] = intervals

        # # Save the DataFrame
        # df.to_csv(os.path.join(embeddings_dir, 'embeddings_plotly.csv'), index=False)

        # # Plot
        # fig1 = px.scatter(df, x='component1', y='component2',
        #                 symbol=df['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
        #                 color='label',  # Coloring according to the label values
        #                 hover_data=['label', 'prediction', 'fake', 'interval_path'])

        # # Save the figure as an HTML file
        # pio.write_html(fig1, os.path.join(embeddings_dir, 'label_embeddings_pca_2d.html'))
        
        # # report the plotly figure
        # clearml_task.get_logger().report_plotly(
        # title="Classifier Embeddings - Labels", series="Labels", iteration=0, figure=fig1
        # )

        # # fig1.show()
        
        # fig2 = px.scatter(df, x='component1', y='component2',
        #                 symbol=df['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
        #                 color='fake',  # Coloring according to the label values
        #                 hover_data=['label', 'prediction', 'fake', 'interval_path'])

        # # Save the figure as an HTML file
        # pio.write_html(fig2, os.path.join(embeddings_dir, 'fake_embeddings_pca_2d.html'))
        
        # # report the plotly figure
        # clearml_task.get_logger().report_plotly(
        # title="Classifier Embeddings - Fake", series="Fake vs. Real", iteration=0, figure=fig2
        # )

        # # fig2.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier Trainer')
    parser.add_argument('--config', type=str, default='Classifier/classifier_config.yaml', help='Path to the configuration file')    
    args = parser.parse_args()
    config = load_config(args.config)
    exp_base_dir = '/tcmldrive/NogaK/ECG_classification/experiments/'
    experiments = [exp for exp in os.listdir(exp_base_dir) if exp.startswith('tcml_Classifier_mixed_data_gen2_')]
    main(config, experiments)
