import numpy as np
import matplotlib.pyplot as plt
from clearml import Task, Logger
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay
import os
import seaborn as sns
import pandas as pd


class Metrics:
    @staticmethod
    def calculate_metrics(d_type, epoch, true_labels, predicted_labels, probas, clearml=False, results_dir=None):
        # Calculate metrics
        accuracy = np.mean(true_labels == predicted_labels)
        confusion_mat = confusion_matrix(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        auroc = roc_auc_score(true_labels, probas)
        avg_prec = average_precision_score(true_labels, probas)

        # Log metrics 
        Metrics.log_metric(d_type, epoch, 'Accuracy', accuracy, clearml, results_dir)
        Metrics.log_metric(d_type, epoch, 'F1 Score', f1, clearml, results_dir)
        Metrics.log_metric(d_type, epoch, 'Precision', precision, clearml, results_dir)
        Metrics.log_metric(d_type, epoch, 'Recall', recall, clearml, results_dir)
        Metrics.log_metric(d_type, epoch, 'AUROC', auroc, clearml, results_dir)
        Metrics.log_metric(d_type, epoch, 'Average Precision', avg_prec, clearml, results_dir)
        
        if d_type == 'test':
            # Create metrics table
            metrics_table = pd.DataFrame([
                ['Accuracy', accuracy],
                ['F1 Score', f1],
                ['Precision', precision],
                ['Recall', recall],
                ['AUROC', auroc],
                ['Average Precision', avg_prec]
                ], 
                columns=['Metric', 'Value'])
            Metrics.log_test_results(metrics_table, clearml, results_dir)

        return accuracy, confusion_mat, f1, precision, recall, auroc, avg_prec

    @staticmethod
    def log_test_results(metrics_table, log_to_clearml=False, results_dir=None):
        # Print and save metrics table
        print('Test set results:')
        print(metrics_table)

        with open(os.path.join(results_dir, 'metrics_results.txt'), 'a') as file:
            file.write(str(metrics_table) + '\n')

        if log_to_clearml:
            Logger.current_logger().report_table("Test set results", "Metrics", iteration=0, table_plot=metrics_table)

    @staticmethod
    def log_metric(d_type, epoch, metric_name, metric_value, log_to_clearml=False, results_dir=None):
        metric_name = f'{d_type}_{metric_name}'
        if log_to_clearml:
            Logger.current_logger().report_scalar(title=metric_name, series=metric_name, value=metric_value, iteration=epoch)

    @staticmethod
    def plot_and_log_confusion_matrix(confusion_mat, class_labels, task, log_to_clearml=False, results_dir=None):
        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))        
        
        # Plot the confusion matrix using seaborn
        svm=sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)

        # Add title and labels
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Save confusion matrix as PNG
        confusion_matrix_file = os.path.join(results_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_file, dpi=400)

        # Show the plot
        plt.show()

        # Close the plot
        plt.close()

    @staticmethod
    def plot_roc_curve(true_labels, probas, task, log_to_clearml=False, results_dir=None):
        # Calculate false positive rate, true positive rate, and thresholds
        fpr, tpr, thresholds = roc_curve(true_labels, probas)

        # Calculate area under the ROC curve
        roc_auc = auc(fpr, tpr)
        
        # Create the ROC curve display
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

        # Plot the ROC curve
        roc_display.plot()

        # Set the plot title
        plt.title('Receiver Operating Characteristic (AUC = {:.2f})'.format(roc_auc))

        # Save ROC curve as PNG
        roc_curve_file = os.path.join(results_dir, 'roc_curve.png')
        plt.savefig(roc_curve_file, dpi=400)

        # Show the plot
        plt.show()
        
        # Close the plot
        plt.close()

    @staticmethod
    def plot_pr_curve(true_labels, probas, task, log_to_clearml=False, results_dir=None):
        precision, recall, _ = precision_recall_curve(true_labels, probas)
        auprc = average_precision_score(true_labels, probas)
        
        # Create the Precision-Recall display
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=auprc)

        # Plot the Precision-Recall curve
        pr_display.plot()

        # Set the plot title
        plt.title('Precision-Recall Curve (AUPRC = {:.2f})'.format(auprc))

        # Save PR curve as PNG
        pr_curve_file = os.path.join(results_dir, 'pr_curve.png')
        plt.savefig(pr_curve_file, dpi=400)

        # Show the plot
        plt.show()
        
        # Close the plot
        plt.close()
