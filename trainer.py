import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from clearml import Logger
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, exp_dir, train_loader, validation_loader, test_loader, optimizer, criterion, scheduler, device, config, clearml_task):
        self.model = model
        self.exp_dir = exp_dir
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scheduler = scheduler
        if self.config['clearml']:
            self.logger = Logger(clearml_task)


    def save_model(self, epoch):
        model_name = f"epoch_{epoch}_model.pth"
        model_path = os.path.join(self.exp_dir, 'models', model_name)
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved model from Epoch: {epoch}' at {model_path}")


    def train(self):
        best_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in tqdm(range(self.config['epochs'])):
            self.model.train()
            total_train_loss = 0.0
            num_train_examples = 0

            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_train_loss += loss.item() * inputs.size(0)
                num_train_examples += inputs.size(0)

                loss.backward()
                self.optimizer.step()

            train_loss = total_train_loss / num_train_examples
            val_loss = self.evaluate('validation')

            # ClearML logging
            if self.config['clearml']:
                self.logger.report_scalar(title="Epoch Loss", series="Training Loss", value=train_loss, iteration=epoch)
                self.logger.report_scalar(title="Epoch Loss", series="Validation Loss", value=val_loss, iteration=epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            self.scheduler.step(val_loss)

            if epochs_without_improvement >= self.config['early_stopping_patience']:
                # self.logger.report_text("Early stopping criterion met. Training stopped.")
                print(f"Early stopping criterion met at Epoch {epoch}. Training stopped.")
                break
        
        # TODO: Make sure this happens for early stopping aswell
        self.save_model(epoch=epoch)



    def evaluate(self, data_type):
        if data_type == 'validation':
            loader = self.validation_loader
        elif data_type == 'test':
            loader = self.test_loader
        
        self.model.eval()
        total_eval_loss = 0.0
        num_examples = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_eval_loss += loss.item() * inputs.size(0)
                num_examples += inputs.size(0)

        return total_eval_loss / num_examples
