import pandas as pd
import os
from tabulate import tabulate

d_type = ['Train', 'Validation', 'Test']
dataset_folder_path = '/tcmldrive/NogaK/ECG_classification/best_experiments/tcml_classifier_with_6_seconds_signals_smaller_arch_20230614_15_04_42/dataframes'

for curr_set in d_type:
    if curr_set == 'Train':
        train_df = pd.read_csv(os.path.join(dataset_folder_path, curr_set + '_df.csv'))
    elif curr_set == 'Validation':
        val_df = pd.read_csv(os.path.join(dataset_folder_path, curr_set + '_df.csv'))
    else:
        test_df = pd.read_csv(os.path.join(dataset_folder_path, curr_set + '_df.csv'))
    
    curr_df = pd.read_csv(os.path.join(dataset_folder_path, curr_set + '_df.csv'))

    label_counts = curr_df['label'].value_counts().reset_index()
    label_counts.columns = ['Label', 'Count']
    label_counts['Count'] = label_counts['Count'].astype(int)

    label_prec = curr_df['label'].value_counts(normalize=True).reset_index()
    label_prec.columns = ['Label', 'Percentage']
    label_prec['Percentage'] = label_prec['Percentage'].round(4) * 100

    label_stat = pd.merge(label_counts, label_prec, on='Label')

    label_0 = curr_df.label.value_counts()[0]
    label_1 = curr_df.label.value_counts()[1]
    label_tot = label_0 + label_1
    print(f'{curr_set} df Real data:')
    print(f'label 0: {label_0}   |   Prec: {100*(label_0/label_tot):.2f}%')
    print(f'label 1: {label_1}   |   Prec: {100*(label_1/label_tot):.2f}%')
    




import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Get embeddings
embeddings = []
labels = []
preds = []
fakes = []

model.to(device)
for i, (inputs, targets, is_fake) in enumerate(dataloader):  # assuming dataloader returns (inputs, targets, is_fake)
    inputs = inputs.to(device)
    targets = targets.to(device)
    with torch.no_grad():
        embedding = model(inputs, return_embedding=True)
        output = model(inputs)
    embeddings.append(embedding.cpu().numpy())
    labels.append(targets.cpu().numpy())
    preds.append((output.cpu().numpy() > 0.75).astype(int))
    fakes.append(is_fake)

embeddings = np.concatenate(embeddings)
labels = np.concatenate(labels)
preds = np.concatenate(preds)
fakes = np.concatenate(fakes)

# Reduce dimensionality
pca = PCA(n_components=2)
embeddings_reduced = pca.fit_transform(embeddings)

# Create a DataFrame
df = pd.DataFrame(embeddings_reduced, columns=['component1', 'component2'])
df['label'] = labels
df['prediction'] = preds
df['fake'] = fakes

# Plot
fig = px.scatter(df, x='component1', y='component2', color='label', 
                 hover_data=['label', 'prediction', 'fake'])
fig.show()







def evaluate(self, data_type, epoch=0, ckpt=None, different_exp_dir = None):
    if data_type == 'validation':
        loader = self.validation_loader
    elif data_type == 'test':
        loader = self.test_loader
        self.load_model(ckpt, different_exp_dir)

    results_dir = os.path.join(self.exp_dir, 'results')

    self.model.eval()
    total_eval_loss = 0.0
    num_examples = 0
    true_labels = []
    predicted_labels = []
    predicted_probas = []
    meta_data_list = []
    embeddings = []

    with torch.no_grad():
        for (inputs, targets), meta_data in tqdm(loader):
            inputs = inputs.to(self.device).squeeze(1)
            targets = targets.to(self.device).float()

            outputs = self.model(inputs).squeeze(1)
            embeddings.extend(self.model(inputs, return_embedding=True).cpu().numpy())  # Save embeddings
            
            loss = self.criterion(outputs, targets)
            total_eval_loss += loss.item() * inputs.size(0)
            num_examples += inputs.size(0)

            # Threshold the predictions
            thresholded_predictions = np.where(outputs.cpu().numpy() >= self.config['classifier_th'], 1, 0)        

            # Convert tensors to numpy arrays
            true_labels.extend(targets.cpu().numpy())
            predicted_labels.extend(thresholded_predictions)
            predicted_probas.extend(outputs.cpu().numpy())
            meta_data_list.append(pd.DataFrame(meta_data))

    # Calculate Loss
    eval_loss = total_eval_loss / num_examples
    
    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    predicted_probas = np.array(predicted_probas)
    meta_data_df = pd.concat(meta_data_list, axis=0, ignore_index=True)
    
    # Call the plot_embeddings function from utils
    from utils import plot_embeddings
    plot_embeddings(embeddings, true_labels, predicted_labels, meta_data_df['fake'].values)

    # Calculate metrics
    accuracy, confusion_mat, f1_score, precision, recall, auroc, avg_prec = Metrics.calculate_metrics(data_type, epoch, true_labels, predicted_labels, predicted_probas, self.config['clearml'], results_dir)

    # rest of the code ...
