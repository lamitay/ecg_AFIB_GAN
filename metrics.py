import numpy as np
import matplotlib.pyplot as plt
from clearml import Task, Logger
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import os
import seaborn as sns


class Metrics:
    @staticmethod
    def calculate_metrics(d_type, epoch, true_labels, predicted_labels, class_labels, clearml=False, results_dir=None):
        # Calculate metrics
        accuracy = np.mean(true_labels == predicted_labels)
        confusion_mat = confusion_matrix(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')


        # Log metrics to ClearML
        for label in class_labels:
            Metrics.log_metric(d_type, epoch, label, 'Accuracy', accuracy, clearml, results_dir)
            Metrics.log_metric(d_type, epoch, label, 'F1 Score', f1, clearml, results_dir)
            Metrics.log_metric(d_type, epoch, label, 'Precision', precision, clearml, results_dir)
            Metrics.log_metric(d_type, epoch, label, 'Recall', recall, clearml, results_dir)
        
        return accuracy, confusion_mat, f1, precision, recall

    @staticmethod
    def log_metric(d_type, epoch, class_label, metric_name, metric_value, log_to_clearml=False, results_dir=None):
        metric_name = f'{d_type}_{class_label}_{metric_name}'
        # print(f'{metric_name}: {metric_value}')
        
        # Log metric to file
        if d_type=='test':
            with open(os.path.join(results_dir, 'metrics_results.txt'), 'a') as file:
                file.write(f'{metric_name}: {metric_value}\n')

        if log_to_clearml:
            Logger.current_logger().report_scalar(title=metric_name, series=metric_name, value=metric_value, iteration=epoch)

    @staticmethod
    def plot_and_log_confusion_matrix(confusion_mat, class_labels, task, log_to_clearml=False, results_dir=None):
        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        # disp = ConfusionMatrixDisplay(confusion_mat, display_labels=class_labels)
        # disp.plot()
        # Plot the confusion matrix using seaborn
        svm=sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)

        # Add title and labels
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        # # Show the plot
        # plt.show()
        
        # # Log confusion matrix to ClearML
        # if log_to_clearml:
        #     Logger.current_logger().report_matrix(
        #         title='Confusion Matrix',
        #         series='Confusion Matrix',
        #         matrix=confusion_mat,
        #         xaxis='Predicted',
        #         yaxis='True'
        #     )

        # Save confusion matrix as PNG
        confusion_matrix_file = os.path.join(results_dir, 'confusion_matrix.png')
        
            
        figure = svm.get_figure()    
        figure.savefig(confusion_matrix_file, dpi=400)
        # plt.savefig(confusion_matrix_file)
        # plt.close()


    @staticmethod
    def plot_roc_curve(true_labels, probas, task, log_to_clearml=False, results_dir=None):
        # Calculate false positive rate, true positive rate, and thresholds
        fpr, tpr, thresholds = roc_curve(true_labels, probas)

        # Calculate area under the ROC curve
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        # Show the plot
        plt.show()
        
        # Save ROC curve as PNG
        roc_curve_file = os.path.join(results_dir, 'roc_curve.png')
        plt.savefig(roc_curve_file)

        # # Log ROC curve to ClearML
        # if log_to_clearml:
        #     Logger.current_logger().report_image(title='ROC Curve', series='ROC Curve', image=plt)

        plt.close()

        