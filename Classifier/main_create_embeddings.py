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
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm


def main(config, exp_name=None):
    
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
        exp_name = f"{config['user']}_{config['experiment_name']}"
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
                                    d_type='Train')
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model.to(device)

    # Create directory to save results
    embeddings_dir = os.path.join(exp_dir, 'embedding_results')
    os.makedirs(embeddings_dir, exist_ok=True)

    ckpt = config['ckpt_for_inference']
    model.load_state_dict(torch.load(ckpt))

    # Get embeddings
    embeddings = []
    labels = []
    preds = []
    fakes = []
    intervals = []

    with torch.no_grad():
        for (inputs, targets), meta_data in tqdm(train_loader):
            inputs = inputs.to(device).squeeze(1)
            targets = targets.to(device).float()

            embedding = model(inputs, return_embedding=True)
            outputs = model(inputs).squeeze(1)
            
            # Threshold the predictions
            thresholded_predictions = np.where(outputs.cpu().numpy() >= config['classifier_th'], 1, 0)        

            embeddings.append(embedding.cpu().numpy())
            labels.append(targets.cpu().numpy())
            preds.append(thresholded_predictions.astype(int))
            fakes.append(meta_data['fake'])
            intervals.append(meta_data['interval_path'])

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)
    fakes = np.concatenate(fakes)
    intervals = np.concatenate(intervals)

    # Save the embeddings
    np.save(os.path.join(embeddings_dir, 'embeddings.npy'), embeddings)

    # Reduce dimensionality
    pca = PCA(n_components=2)
    embeddings_reduced = pca.fit_transform(embeddings)

    # Save the reduced embeddings
    np.save(os.path.join(embeddings_dir, 'embeddings_reduced.npy'), embeddings_reduced)

    # Create a DataFrame
    df = pd.DataFrame(embeddings_reduced, columns=['component1', 'component2'])
    df['label'] = labels
    df['prediction'] = preds
    df['fake'] = fakes
    df['interval_path'] = intervals

    # Save the DataFrame
    df.to_csv(os.path.join(embeddings_dir, 'embeddings_plotly.csv'), index=False)

    # Plot
    fig1 = px.scatter(df, x='component1', y='component2',
                    symbol=df['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
                    color='label',  # Coloring according to the label values
                    hover_data=['label', 'prediction', 'fake', 'interval_path'])

    # Save the figure as an HTML file
    pio.write_html(fig1, os.path.join(embeddings_dir, 'label_embeddings_pca_2d.html'))
    
    # report the plotly figure
    clearml_task.get_logger().report_plotly(
    title="Classifier Embeddings - Labels", series="Labels", iteration=0, figure=fig1
    )

    # fig1.show()
    
    fig2 = px.scatter(df, x='component1', y='component2',
                    symbol=df['fake'].map({0: "cross", 1: "circle"}),  # Different symbols for 'fake' status
                    color='fake',  # Coloring according to the label values
                    hover_data=['label', 'prediction', 'fake', 'interval_path'])

    # Save the figure as an HTML file
    pio.write_html(fig2, os.path.join(embeddings_dir, 'fake_embeddings_pca_2d.html'))
    
    # report the plotly figure
    clearml_task.get_logger().report_plotly(
    title="Classifier Embeddings - Fake", series="Fake vs. Real", iteration=0, figure=fig2
    )

    # fig2.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier Trainer')
    parser.add_argument('--config', type=str, default='Classifier/classifier_config.yaml', help='Path to the configuration file')    
    args = parser.parse_args()
    config = load_config(args.config)
    exp_name = 'baseline_classifier_training_embeddings_10k_fake_samples_gen2'
    # exp_name = None
    main(config, exp_name)

