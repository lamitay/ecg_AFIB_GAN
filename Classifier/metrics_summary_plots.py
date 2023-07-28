import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os
import numpy as np


def main(args):
    """
    This script takes an output directory as input. 
    It creates a DataFrame with fake percentages and their corresponding metrics (TP, TN, FP, FN), 
    saves the DataFrame to a CSV file, and generates bar plots for each metric.
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # # List of fake percentages
    # fake_percentages = list(range(0, 110, 10))

    # # Metrics for each fake percentage taken from confusion matrices
    # TN = [10701, 11249, 10772, 11167, 10974, 12349, 12074, 11254, 11973, 12239, 18729]
    # FP = [8053, 7505, 7982, 7587, 7780, 6405, 6680, 7500, 6781, 6515, 25]
    # FN = [291, 316, 465, 446, 487, 797, 879, 606, 874, 670, 8354]
    # TP = [8072, 8047, 7898, 7917, 7876, 7566, 7484, 7757, 7489, 7693, 9]

    # # Create a DataFrame
    # metrics_df = pd.DataFrame({'fake_percentage': fake_percentages,
    #                 'TP': TP,
    #                 'TN': TN,
    #                 'FP': FP,
    #                 'FN': FN})

    # # Save DataFrame as a CSV file
    # metrics_df.to_csv(os.path.join(output_dir, 'conf_metrics_summary.csv'), index=False)

    # Read the DF
    metrics_df = pd.read_csv(os.path.join(output_dir, 'conf_metrics_summary.csv'), index_col=False)

    # Comment this line to get also the expeiment with 100% fake AF class training set data
    metrics_df = metrics_df[metrics_df['fake_percentage'] < 100]

    # List of metrics
    # metrics = ['TP', 'TN', 'FP', 'FN']
    metrics = ['TN', 'FP', 'FN', 'TP']
    
    num_metrics = len(metrics)
    num_experiments = len(metrics_df['fake_percentage'])
    colors = cm.rainbow(np.linspace(0, 1, num_experiments))  # Generate a rainbow color palette
    
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Flatten the 2D array of axes to make it iterable
    axs = axs.ravel()

    for i, metric in enumerate(metrics):
        axs[i].bar(metrics_df['fake_percentage'], metrics_df[metric], color=colors, width=7) # Change width to make bars thinner
        axs[i].set_title(f'{metric} Bar Plot')
        axs[i].set_xlabel('Fake AF Class Training Set Percentage')
        axs[i].set_ylabel('Counts')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'conf_metrics_summary_bar_plot_2x2_no_100%_exp.png'))



    # # Same plots but different arrangement of subplots
    # fig, axs = plt.subplots(num_metrics, figsize=(10, 8))

    # for i, metric in enumerate(metrics):
    #     axs[i].bar(metrics_df['fake_percentage'], metrics_df[metric], color=colors, width=7) # Change width to make bars thinner
    #     axs[i].set_title(f'{metric} Bar Plot')
    #     axs[i].set_xlabel('Fake AF Class Training Set Percentage')
    #     axs[i].set_ylabel('Counts')

    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, 'conf_metrics_summary_bar_plot.png'))
    # # plt.show()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate metrics data and plot it.')
    parser.add_argument('--output_dir', type=str, default='/tcmldrive/NogaK/ECG_classification/experiments/fake_training_data_summary/conf_metrics', help='Directory to save output CSV file')
    args = parser.parse_args()

    main(args)
