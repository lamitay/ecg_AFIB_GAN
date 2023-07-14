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
from utils import load_config, build_exp_dirs
from model import EcgResNet34
from torchvision import transforms
from dataset import AF_mixed_dataset
from torch.utils.data import DataLoader

def main(config, experiments):
    fig, axs = plt.subplots(3, 4, figsize=(20,15))
    fig.suptitle('Embedding Space', fontsize=20)

    for i, exp_name in enumerate(tqdm(experiments, desc="Processing Experiments")):
        exp_dir = build_exp_dirs(config['base_dir'], exp_name)

        # Model
        model = EcgResNet34(num_classes=1, layers=(1, 1, 1, 1))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load the saved model
        ckpt = os.path.join(exp_dir, 'model_state.pth')  # Specify your checkpoint file path
        model.load_state_dict(torch.load(ckpt))

        # Dataset and Dataloader
        transform = transforms.Compose([transforms.ToTensor(), Normalize()]) 
        dataset = AF_mixed_dataset(
            real_data_folder_path=config['real_data_path'],
            fake_data_folder_path=config['fake_data_path'],
            exp_dir=exp_dir,
            transform=transform,
            config=config,
            d_type='Train'
        )
        dataloader = DataLoader(dataset, batch_size=config['batch_size'])

        # Get embeddings
        embeddings = []
        labels = []
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Getting Embeddings"):
                inputs = inputs.to(device).squeeze(1)
                targets = targets.to(device).float()

                embedding = model(inputs, return_embedding=True)
                embeddings.append(embedding.cpu().numpy())
                labels.append(targets.cpu().numpy())

        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels)

        # Perform t-SNE
        embeddings_reduced = TSNE(n_components=2).fit_transform(embeddings)

        # Plot the embeddings using plotly
        df = pd.DataFrame(embeddings_reduced, columns=['component1', 'component2'])
        df['label'] = labels
        fig1 = px.scatter(df, x='component1', y='component2', color='label')

        # Save the figure as an HTML file
        pio.write_html(fig1, os.path.join(exp_dir, 'tsne.html'))

        # Save the figure as a PNG file
        fig1.write_image(os.path.join(exp_dir, 'tsne.png'))

        # Plot on the subplot
        axs[i//4, i%4].scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1], s=5, c=labels)
        axs[i//4, i%4].set_title(f'{exp_name} - {100*sum(labels)/len(labels):.2f}% Fake Training')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('all_experiments.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier Trainer')
    parser.add_argument('--config', type=str, default='Classifier/classifier_config.yaml', help='Path to the configuration file')    
    args = parser.parse_args()
    config = load_config(args.config)
    exp_base_dir = '/tcmldrive/NogaK/ECG_classification/experiments/'
    experiments = [exp for exp in os.listdir(exp_base_dir) if exp.startswith('tcml_Classifier_mixed_data_gen2_')]
    main(config, experiments)
