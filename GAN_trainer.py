import os
import time

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam
from clearml import Logger

from GAN_models import Generator, Discriminator
# from .dataset_git import ECGDataset, get_dataloader
# from .config_git import Config

# from ..dataset import AF_dataset


class GAN_Trainer:
    def __init__(
        self,
        generator,
        discriminator,
        batch_size,
        num_epochs,
        data_loader,
        signal_length,
        device,
        label,
        clearml,
        exp_dir
    ):
        
        self.clearml = clearml
        self.exp_dir = exp_dir
        self.results_dir = os.path.join(exp_dir, 'results')
        
        self.device = device
        self.netG = generator.to(self.device)
        self.netD = discriminator.to(self.device)
        
        self.optimizerD = Adam(self.netD.parameters(), lr=1e-5)
        self.optimizerG = Adam(self.netG.parameters(), lr=1e-5)
        self.criterion = nn.BCELoss()
        
        self.batch_size = batch_size
        self.signal_dim = [self.batch_size, 1, signal_length]
        self.num_epochs = num_epochs
        self.dataloader = data_loader
        
        self.fixed_noise = torch.randn(self.batch_size, 1, signal_length, device=self.device)
        self.g_errors = []
        self.d_errors = []
        
    def _one_epoch(self):
        real_label = 1
        fake_label = 0
        
        # for i, data in enumerate(self.dataloader, 0):
        for (inputs, _), meta_data in self.dataloader:
            inputs = inputs.to(self.device).squeeze(1)
            # targets = targets.to(self.device).float()
            ##### Update Discriminator: maximize log(D(x)) + log(1 - D(G(z))) #####
            ## train with real data
            self.netD.zero_grad()
            real_data = inputs
            # dim for noise
            batch_size = real_data.size(0)
            self.signal_dim[0] = batch_size
            
            label = torch.full((batch_size,), real_label,
                           dtype=real_data.dtype, device=self.device)
            
            output = self.netD(real_data)
            output = output.view(-1)
       
            errD_real = self.criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            ## train with fake data
            noise = torch.randn(self.signal_dim, device=self.device)
            fake = self.netG(noise)
            label.fill_(fake_label)
            
            output = self.netD(fake.detach())
            output = output.view(-1)
            
            errD_fake = self.criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake 
            self.optimizerD.step()
            
            ##### Update Generator: maximaze log(D(G(z)))  
            self.netG.zero_grad()
            label.fill_(real_label) 
            output = self.netD(fake)
            output = output.view(-1)
            
            errG = self.criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            self.optimizerG.step()
            
        return errD.item(), errG.item()
        
    def run(self):
        for epoch in tqdm(range(self.num_epochs)):
            errD_, errG_ = self._one_epoch()
            self.d_errors.append(errD_)
            self.g_errors.append(errG_)
            
            # ClearML logging
            if self.clearml:
                Logger.current_logger().report_scalar(title="Epoch Loss", series="Generator Loss", value=errG_, iteration=epoch)
                Logger.current_logger().report_scalar(title="Epoch Loss", series="Discriminator Loss", value=errD_, iteration=epoch)
            
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Loss_D: {errD_} | Loss_G: {errG_} | Time: {time.strftime('%H:%M:%S')}")
                
                fake = self.netG(self.fixed_noise)
                
                plt.figure()
                plt.plot(fake.detach().cpu().squeeze(1).numpy()[:].transpose())
                plt.title(f'generated_samples_epoch_{epoch}')
                plt.savefig(os.path.join(self.results_dir,f'generated_samples_epoch_{epoch}.png'))
                plt.close()

        # Save best models
        gen_name = f"epoch_{epoch}_generator_model.pth"
        disc_name = f"epoch_{epoch}_discriminator_model.pth"
        gen_path = os.path.join(self.exp_dir, 'models', gen_name)
        disc_path = os.path.join(self.exp_dir, 'models', disc_name)
        torch.save(self.netG.state_dict(), gen_path)
        torch.save(self.netD.state_dict(), disc_path)


if __name__ == '__main__':
    # config = Config()
    g = Generator()
    d = Discriminator()

    GAN_trainer = GAN_Trainer(
    generator=g,
    discriminator=d,
    batch_size=96,
    num_epochs=3000,
    label='Fusion of ventricular and normal'
    )
    GAN_trainer.run()