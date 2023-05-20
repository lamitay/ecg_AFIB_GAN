import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from clearml import Logger
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from metrics import Metrics

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
        
        if self.config['debug']:
            self.epochs = self.config['debug_epochs']
        else:
            self.epochs = self.config['epochs']
        
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

        for epoch in tqdm(range(self.epochs)):            
            self.model.train()
            total_train_loss = 0.0
            num_train_examples = 0  
            for inputs, targets in tqdm(self.train_loader):
                inputs = inputs.to(self.device).unsqueeze(1)
                targets = targets.to(self.device).float()

                self.optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, targets.squeeze())

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
        true_labels = []
        predicted_labels = []
        predicted_probas = []

        with torch.no_grad():
            for inputs, targets in tqdm(loader):
                inputs = inputs.to(self.device).unsqueeze(1)
                targets = targets.to(self.device).float()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_eval_loss += loss.item() * inputs.size(0)
                num_examples += inputs.size(0)

                # Threshold the predictions
                thresholded_predictions = np.where(predictions >= self.config['classifier_th'], 1, 0)        

                # Convert tensors to numpy arrays
                true_labels.extend(targets.cpu().numpy())
                predicted_labels.extend(thresholded_predictions.cpu().numpy())
                predicted_probas.extend(outputs.cpu().numpy())

        # Calculate Loss
        eval_loss = total_eval_loss / num_examples
        
        # Convert lists to numpy arrays
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        predicted_probas = np.array(predicted_probas)
        
        # Calculate metrics
        accuracy, confusion_mat, f1_score, precision, recall, probas = Metrics.calculate_metrics(true_labels, predicted_labels, class_labels)

        # Plot and log confusion matrix
        Metrics.plot_and_log_confusion_matrix(confusion_mat, class_labels, self.logger)

        # Plot ROC curve and log it to ClearML
        Metrics.plot_roc_curve(true_labels, probas, self.logger)

        # return eval_loss, accuracy, f1_score, precision, recall, probas
        return eval_loss, accuracy, f1_score, precision, recall, probas
        
        # # Calculate metrics
        # predictions = np.array(predictions)
        # np_labels = np.array(np_labels)
        # accuracy = accuracy_score(np_labels, thresholded_predictions)
        # precision = precision_score(np_labels, thresholded_predictions)
        # recall = recall_score(np_labels, thresholded_predictions)
        # f1 = f1_score(np_labels, thresholded_predictions)

        # # Log metrics (you can modify this part based on your logging preference)
        # print(f'{data_type.capitalize()} Loss: {eval_loss:.4f}')
        # print(f'{data_type.capitalize()} Accuracy: {accuracy:.4f}')
        # print(f'{data_type.capitalize()} Precision: {precision:.4f}')
        # print(f'{data_type.capitalize()} Recall: {recall:.4f}')
        # print(f'{data_type.capitalize()} F1-Score: {f1:.4f}')

