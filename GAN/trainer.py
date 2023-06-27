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
from torch.optim import AdamW, Adam, SGD
from clearml import Logger

from GAN.models import Generator, Discriminator
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
        noise_length,
        device,
        label,
        discriminator_lr,
        generator_lr,
        clearml,
        exp_dir,
        noise_std = 0.1,
        seq_model = False,
        wgan_gp = False,
        wgan_gp_lambda = 0.3,
        early_stopping = True,
        early_stopping_patience = 100 
        ):
        
        self.clearml = clearml
        self.exp_dir = exp_dir
        self.results_dir = os.path.join(exp_dir, 'results')
        
        self.device = device
        self.netG = generator.to(self.device)
        self.netD = discriminator.to(self.device)
        
        self.optimizerD = Adam(self.netD.parameters(), lr=discriminator_lr)
        self.optimizerG = Adam(self.netG.parameters(), lr=generator_lr)
        self.criterion = nn.BCELoss()
        
        self.batch_size = batch_size
        self.signal_dim = [self.batch_size, 1, noise_length]
        self.num_epochs = num_epochs
        self.dataloader = data_loader
        self.noise_std = noise_std
        self.seq_model = seq_model

        
        self.wgan_gp = wgan_gp
        self.wgan_gp_lambda = wgan_gp_lambda
            
    
        self.fixed_noise = torch.randn(self.batch_size, 1, noise_length, device=self.device)
        self.g_errors = []
        self.d_errors = []

        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(real_data.size(0), 1, 1).to(self.device)
        alpha = alpha.expand_as(real_data)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.requires_grad_(True)

        disc_interpolates = netD(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones_like(disc_interpolates),
                                create_graph=True, retain_graph=True)[0]
                                
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
        
    def _one_epoch(self):
        real_label = 1
        fake_label = 0
        
        err_Ds = []
        err_Gs = []
        out_real = []
        out_fake = [] 
        # for i, data in enumerate(self.dataloader, 0):
        for (inputs, _), meta_data in self.dataloader:
            inputs = inputs.to(self.device).squeeze(1)
            if inputs.shape[0] != self.batch_size and self.seq_model:
                break
            # targets = targets.to(self.device).float()
            ##### Update Discriminator: maximize log(D(x)) + log(1 - D(G(z))) #####
            ## train with real data
            self.netD.zero_grad()
            real_data = inputs
            
            if self.noise_std != 0:
                # Add the noise to the real data
                real_data += torch.randn_like(real_data) * self.noise_std 

            # dim for noise
            batch_size = real_data.size(0)
            self.signal_dim[0] = batch_size
            
            label = torch.full((batch_size,), real_label,
                           dtype=real_data.dtype, device=self.device)
            
            output = self.netD(real_data)
            output = output.view(-1)
       
            errD_real = self.criterion(output, label)
            errD_real.backward()
            D_real = output.mean().item()
            
            ## train with fake data
            noise = torch.randn(self.signal_dim, device=self.device)
            if self.seq_model:
                hid = self.netG.init_hidden(batch_size = batch_size)
                fake = self.netG(noise, hidden = hid)
            else:
                fake = self.netG(noise)

            label.fill_(fake_label)
            
            output = self.netD(fake.detach())
            output = output.view(-1)
            
            errD_fake = self.criterion(output, label)

            if self.wgan_gp:
                gradient_penalty = self.calc_gradient_penalty(self.netD, real_data, fake)
                errD_fake += self.wgan_gp_lambda * gradient_penalty
                
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
            D_fake = output.mean().item()
            self.optimizerG.step()
            
            err_Ds.append(errD.item())
            err_Gs.append(errG.item())
            out_fake.append(D_fake)
            out_real.append(D_real)

        return errD.item(), errG.item(), D_fake, D_real
        
    def run(self):
        best_loss = float('inf')
        for epoch in tqdm(range(self.num_epochs)):
            errD_, errG_, D_fake_, D_real_ = self._one_epoch()
            self.d_errors.append(errD_)
            self.g_errors.append(errG_)
            
            # ClearML logging
            if self.clearml:
                Logger.current_logger().report_scalar(title="Epoch Loss", series="Generator Loss", value=errG_, iteration=epoch)
                Logger.current_logger().report_scalar(title="Epoch Loss", series="Discriminator Loss", value=errD_, iteration=epoch)
                Logger.current_logger().report_scalar(title="Epoch Discriminator Mean Output", series=" Discriminator Mean Output - Real", value=D_real_, iteration=epoch)
                Logger.current_logger().report_scalar(title="Epoch Discriminator Mean Output", series=" Discriminator Mean Output - Fake", value=D_fake_, iteration=epoch)
                Logger.current_logger().report_scalar(title="Noise STD", series="Noise STD", value=self.noise_std, iteration=epoch)
                

            if epoch % 100 == 0:  
                if epoch != 0:
                    self.noise_std = self.noise_std*0.5 # reduce the variance of the noise that being added to the real data          
                print(f"Epoch: {epoch} | Loss_D: {errD_} | Loss_G: {errG_} | Mean_D_fake: {D_fake_} | Mean_D_real: {D_real_} | Time: {time.strftime('%H:%M:%S')}")
                if self.seq_model:
                    hid = self.netG.init_hidden(self.batch_size)
                    fake = self.netG(self.fixed_noise, hidden = hid)
                else:
                    fake = self.netG(self.fixed_noise)
                # Sample 10 generated signals to plot:
                samples_idx = torch.randint(low=0, high=self.batch_size, size=(10,))
                samples = fake[samples_idx,...]
                plt.figure()
                plt.plot(samples.detach().cpu().squeeze(1).numpy()[:].transpose())
                plt.title(f'generated_samples_epoch_{epoch}')
                plt.savefig(os.path.join(self.results_dir,f'generated_samples_epoch_{epoch}.png'))
                plt.close()

            # early stopping on the generator loss:
            if errG_ < best_loss:
                best_loss = errG_
                epochs_without_improvement = 0
                # self.save_models(epoch=epoch)

            else:
                epochs_without_improvement += 1
            
            if self.early_stopping:
                if epochs_without_improvement >= self.early_stopping_patience:
                    print(f"Generator error keep increasing... Early stopping criterion met at Epoch {epoch}. Training stopped.")
                    break

        # Save last models
        self.save_models(epoch=epoch)

        # Save loss curves
        self.save_loss_curves()

    def save_loss_curves(self):
        plt.figure()
        plt.plot(self.d_errors)
        plt.plot(self.g_errors)
        plt.title('Loss Curve')
        plt.savefig(os.path.join(self.results_dir,'loss_curve.png'))
        plt.xlabel('# Epoch')
        plt.close()

    def save_models(self, epoch):
        gen_name = f"epoch_{epoch}_generator_model.pth"
        disc_name = f"epoch_{epoch}_discriminator_model.pth"
        gen_path = os.path.join(self.exp_dir, 'models', gen_name)
        disc_path = os.path.join(self.exp_dir, 'models', disc_name)
        torch.save(self.netG.state_dict(), gen_path)
        torch.save(self.netD.state_dict(), disc_path)
        print(f"Saved generator model from Epoch: {epoch}' at {gen_path}")
        print(f"Saved discriminator model from Epoch: {epoch}' at {disc_path}")
        


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