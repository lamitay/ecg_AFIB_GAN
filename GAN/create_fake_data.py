import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam
from models import Generator, Discriminator, DCGenerator, DCDiscriminator
import seq_models as seqGAN
import argparse
from tqdm import tqdm as tqdm

def generate_and_save_data(gan_model='seq', model_path=None, base_output_dir=None, fake_amount=10):
    
    device = 'cuda'
    signal_length = 1500
    batch_size = 1
    
    # Create directories
    exp_name = f'fake_data_6_secs_{fake_amount}_samples'
    output_dir = os.path.join(base_output_dir, exp_name)
    intervals_dir = os.path.join(output_dir, 'intervals')
    images_dir = os.path.join(output_dir, 'images')
    
    dirs = [output_dir, intervals_dir, images_dir]
    for curr_dir in dirs:
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)
    
    # Load model
    if gan_model == 'seq':
        gen = seqGAN.Generator(signal_length, hidden_dim=512, n_features=100, tanh_output=True, num_layers=3)
    else:
        gen = DCGenerator(nz=100)

    gen.load_state_dict(torch.load(model_path))
    gen.to(device)

    df_list = []
    
    for index in tqdm(range(fake_amount), desc='Generating fake data'):
        fixed_noise = torch.randn(batch_size, 1, 100, device=device)

        if gan_model == 'seq':
            hid = gen.init_hidden(batch_size)
            fake = gen(fixed_noise, hidden=hid)
        else:
            fake = gen(fixed_noise)

        sample_name = f'fake_{index}_label_1'
        
        # Save .npy file
        interval_path = os.path.join(intervals_dir, sample_name + '.npy')
        np.save(interval_path, fake.detach().cpu().squeeze(1).numpy()[:])

        # Save plot as png
        image_path = os.path.join(images_dir, sample_name + '.png')
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        axs[0].plot(fixed_noise.detach().cpu().squeeze(1).numpy()[:].transpose())
        axs[0].set_title('Input noise')
        axs[1].plot(fake.detach().cpu().squeeze(1).numpy()[:].transpose())
        axs[1].set_title('Generated sample')
        plt.suptitle('Time series data of Input Noise and Generated Sample', fontsize=14)
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close(fig)

        # Append row to DataFrame
        df_list.append({
            'record_file_name': f'fake_{index}',
            'interval_path': sample_name + '.npy',
            'image_path': sample_name + '.png',
            'num_of_bits': -1,
            'bsqi_score': -1,
            'label': 1
        })

    # Create DataFrame and save as .df
    df = pd.DataFrame(df_list)
    df.to_csv(os.path.join(output_dir, 'meta_data.csv'), index=False)
    print('Successfully generated and saved fake data.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate fake data using a GAN model.')
    parser.add_argument('--gan_model', type=str, default='seq', help='The GAN model to use for generating data.')
    parser.add_argument('--model_path', type=str, default='/tcmldrive/NogaK/ECG_classification/experiments/tcml_seqGAN_3layersLSTM_1000epochs_more_dropout_in_dis_20230628_15_33_53/models/epoch_999_generator_model.pth', help='The directory where the GAN model is stored.')
    parser.add_argument('--base_output_dir', type=str, default='/tcmldrive/NogaK/ECG_classification/data/', help='The base directory to output the generated data.')
    parser.add_argument('--fake_amount', type=int, default=100, help='The amount of fake data to generate.')
    args = parser.parse_args()
    
    generate_and_save_data(args.gan_model, args.model_path, args.base_output_dir, args.fake_amount)
