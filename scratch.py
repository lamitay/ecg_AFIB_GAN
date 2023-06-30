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
    




