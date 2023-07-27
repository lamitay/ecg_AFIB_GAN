import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def extract_metrics(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Remove leading/trailing whitespace and split on last space
    split_lines = [line.strip().rsplit(maxsplit=1) for line in lines[1:]]  # [1:] to skip the header

    # Split each line into Metric and Value
    metrics = {line[0].split(maxsplit=1)[1]: float(line[1]) for line in split_lines if len(line) == 2}

    # Convert dictionary to a pandas Series
    metrics_series = pd.Series(metrics)

    return metrics_series[['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUROC', 'Average Precision']]


def main(args):
    experiments_dir = args.experiments_dir
    output_dir = args.output_dir
    output_dir_path = os.path.join(experiments_dir, output_dir)

    os.makedirs(output_dir_path, exist_ok=True)

    experiments = [exp for exp in os.listdir(experiments_dir) if exp.startswith('tcml_Classifier_mixed_data_gen2_')]

    results = []
    for exp in experiments:
        fake_percentage = int(exp.split('_')[5])
        results_file_path = os.path.join(experiments_dir, exp, 'results', 'metrics_results.txt')
        metrics = extract_metrics(results_file_path)
        metrics['fake_percentage'] = fake_percentage
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('fake_percentage')

    results_df.to_csv(os.path.join(output_dir_path, 'fake_training_data_summary_results.csv'), index=False)
    
    plt.figure(figsize=(10, 8))
    plt.plot(results_df['fake_percentage'], results_df['Accuracy'], label='Accuracy', marker='o')
    plt.plot(results_df['fake_percentage'], results_df['F1 Score'], label='F1 Score', marker='o')
    plt.plot(results_df['fake_percentage'], results_df['Precision'], label='Precision', marker='o')
    plt.plot(results_df['fake_percentage'], results_df['Recall'], label='Recall', marker='o')
    plt.plot(results_df['fake_percentage'], results_df['AUROC'], label='AUROC', marker='o')
    plt.plot(results_df['fake_percentage'], results_df['Average Precision'], label='Average Precision', marker='o')

    plt.xlabel('Fake Training Percentage')
    plt.ylabel('Metric Values')
    plt.title('Metrics for different Fake Percentages')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_dir_path, 'full_summary_plot.png'))
    # plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(results_df['fake_percentage'], results_df['Accuracy'], label='Accuracy', marker='o')
    plt.plot(results_df['fake_percentage'], results_df['F1 Score'], label='F1 Score', marker='o')
    # plt.plot(results_df['fake_percentage'], results_df['Precision'], label='Precision', marker='o')
    # plt.plot(results_df['fake_percentage'], results_df['Recall'], label='Recall', marker='o')
    plt.plot(results_df['fake_percentage'], results_df['AUROC'], label='AUROC', marker='o')
    plt.plot(results_df['fake_percentage'], results_df['Average Precision'], label='Average Precision', marker='o')

    plt.xlabel('Fake Training Percentage')
    plt.ylabel('Metric Values')
    plt.title('Metrics for different Fake Percentages')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_dir_path, 'top_summary_plot.png'))
    # plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot experiment results.')
    parser.add_argument('--experiments_dir', type=str, default='/tcmldrive/NogaK/ECG_classification/experiments/',
                        help='The path to the directory where the experiments results are stored.')
    parser.add_argument('--output_dir', type=str, default='fake_training_data_summary',
                        help='The path to the output directory where the experiments summary will be stored.')
    args = parser.parse_args()
    main(args)
