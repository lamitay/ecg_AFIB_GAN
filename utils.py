import torch
import wfdb
from datetime import datetime
import os
import numpy as np
from pecg import Preprocessing as Pre
from wfdb import processing
import yaml
from torchsummary import summary
from fvcore.nn import flop_count, FlopCountAnalysis, flop_count_table


def load_config(config_path):
    assert os.path.exists(config_path), f"Invalid config path: {config_path}"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_exp_dirs(exp_base_path, exp_name):
    assert os.path.exists(exp_base_path), f"Invalid experiments base path: {exp_base_path}"

    timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    exp_dir = os.path.join(exp_base_path, f"{exp_name}_{timestamp}")

    assert not os.path.exists(exp_dir), f"Experiment directory already exists: {exp_dir}"

    os.makedirs(exp_dir)
    # os.makedirs(os.path.join(exp_dir, "data"))
    os.makedirs(os.path.join(exp_dir, "models"))
    os.makedirs(os.path.join(exp_dir, "results"))
    os.makedirs(os.path.join(exp_dir, "logs"))

    return exp_dir


def get_record_names_from_folder(folder_path):
    assert os.path.exists(folder_path), f"Invalid folder path: {folder_path}"
    record_names = []
    for file in os.listdir(folder_path):
        if file.endswith('.hea'):  # we find only the .hea files.
            if file[:-4] == '00735' or file[:-4] == '03665':
                continue
            record_names.append(file[:-4])  # we remove the extensions, keeping only the number itself.
    return record_names


def split_records_train_val_test(record_names, train_prec=80):
    #split records to training and test to 80% - 20%
    num_of_train_val_records = round(len(record_names)*(train_prec/100))
    train_val_records_names = record_names[:num_of_train_val_records]
    test_records_names = record_names[num_of_train_val_records:]
    # Split train-validation to 80-20 % of the train_val
    num_of_train_records = round(len(train_val_records_names)*(train_prec/100))
    train_records_names = train_val_records_names[:num_of_train_records]
    val_records_names = train_val_records_names[num_of_train_records:]
    print(f"Training set: {len(train_records_names)} records\nValidation set: {len(val_records_names)} records\nTest set: {len(test_records_names)} records")
    
    return train_records_names, val_records_names, test_records_names


def split_records_to_intervals(record, annotation, sample_length, channel, overlap):
    """split each of the records to intervals and the annotation to a binar 1D vector 

    Args:
        record (wfdb.record): the record object of wfdb
        annotation (wfdb.annotation): the annotation object of wfdb
        sample_length (float/int): the length of the required intervals in seconds 
        channel (int): if 0, take the first lead, if 1 take the second lead
        overlap (bool or float/int): if False, split without overlap. Else splitting the records with overlap (value in seconds!)

    Raises:
        NotImplementedError: needs to implement the overlap option

    Returns:
        torch.Tensors: intervals, labels
    """
    signal = torch.Tensor(record.p_signal[:, channel]) #start the signal from the index we got annotation for 
    fs = record.fs

    #create annotation signal with the length of records:
    annots_signal = torch.zeros_like(signal)
    for i, idx in enumerate(annotation.sample):
        if annotation.aux_note[i] == '(N':
            continue #label stays zero
        else:
            if i == len(annotation.sample)-1: #the last iteration in the loop:
                 annots_signal[idx:] = 1
            else:
                annots_signal[idx:annotation.sample[i+1]] = 1

    num_of_samples_in_interval = fs*sample_length
    if overlap == False: # split the record without overlap
        intervals, labels = segment_and_label(signal, annots_signal, num_of_samples_in_interval, 0)
    else:
        overlap_in_samples = overlap*fs
        intervals, labels = segment_and_label(signal, annots_signal, num_of_samples_in_interval,  overlap_in_samples)
    return intervals, labels

def segment_and_label(x, y, m, w):
    """
    Segments the time series x into segments of length m with overlap w,
    and creates a label vector for each segment based on the labels in y.

    Args:
    - x: a 1D PyTorch tensor representing the time series
    - y: a 1D PyTorch tensor representing the label vector
    - m: an integer representing the length of each segment
    - w: an integer representing the overlap between segments

    Returns:
    - segments: a 2D PyTorch tensor of shape (num_segments, m),
      where num_segments is the number of segments created from x
    - labels: a 1D PyTorch tensor of shape (num_segments,) representing
      the label vector for each segment
    """
    
    assert x.shape[0] == y.shape[0], 'length of time series should be equal to annotations signal'

    num_segments = (x.shape[0] - m) // (m - w) + 1  # compute number of segments
    segments = torch.zeros(num_segments, m)  # initialize segments tensor
    labels = torch.zeros(num_segments, dtype=torch.int64)  # initialize labels tensor

    for i in range(num_segments):
        start = i * (m - w)
        end = start + m
        segment = x[start:end]
        label = y[start:end]
        if torch.all(label == 0):
            labels[i] = 0
            segments[i] = segment
        elif torch.all(label == 1):
            labels[i] = 1
            segments[i] = segment

    # remove any segments that do not have a valid label
    final_segments = segments[segments.sum(axis=1)!=0, :]
    final_labels = labels[segments.sum(axis=1)!=0]

    return final_segments, final_labels

def bsqi(signal , fs):
    xqrs_inds = processing.xqrs_detect(signal, fs, verbose=False)
    gqrs_inds = processing.gqrs_detect(signal, fs)
    pre_pecg = Pre.Preprocessing(signal, fs)
    bsqi = pre_pecg.bsqi(xqrs_inds, gqrs_inds)   
    return bsqi

if __name__ == '__main__':
    folder_path = 'C:/Users/nogak/Desktop/MyMaster/YoachimsCourse/files/'
    name = '04015'
    file_name = os.path.join(folder_path, name)
    record = wfdb.rdrecord(file_name)
    annotation = wfdb.rdann(file_name, 'atr')
    intervals, final_annots = split_records_to_intervals(record, annotation, sample_length = 10, channel = 0, overlap = 3)

    ## TESTS :
    # test1 verify that no mixed labels intervals are being created:
    x = torch.rand(100000)
    y = torch.zeros(100000)
    y[::2] = 1
    interval, labels = segment_and_label(x, y, 100, 0)
    assert len(interval) == 0, 'In this tests all intervals should have mixed labels so no intervals should be created'

    # test2 verify that if there are no mixed labels, all the intervals are being created:
    x = torch.rand(100000)
    y = torch.zeros(100000)
    interval, labels = segment_and_label(x, y, 100, 0)
    assert len(interval) == 100000/100, 'In this tests all intervals that possible needs to be created'

def print_model_summary(model, batch_size, num_ch=1, samp_per_record=2500):
    """
    Prints the model summary including the model architecture, number of parameters,
    and the total number of FLOPs (Floating Point Operations) required for a given input size.

    Args:
        model (torch.nn.Module): The PyTorch model to summarize.
        batch_size (int): The batch size used for computing FLOPs.
        num_ch (int, optional): The number of input channels. Defaults to 1.
        samp_per_record (int, optional): The number of samples per record. Defaults to 2500.

    Example:
        >>> model = MyModel()
        >>> batch_size = 64
        >>> num_ch = 1
        >>> samp_per_record = 2500
        >>> print_model_summary(model, batch_size, num_ch, samp_per_record)
        Output: Prints the model summary, parameter count, and total number of FLOPs.
    """
    summary(model, input_size=(num_ch, samp_per_record))
    # Create a sample input tensor
    input_size = (batch_size, num_ch, samp_per_record)
    rand_inputs = torch.randn(*input_size)
    # Compute FLOPs
    flops = FlopCountAnalysis(model, rand_inputs)
    print(flop_count_table(flops))
    print(f"Total number of FLOPs: {humanize_number(flops.total())} Flops")

def humanize_number(number):
    """
    Converts a large number into a human-readable format with appropriate unit suffixes.

    Args:
        number (float or int): The number to be formatted.

    Returns:
        str: The formatted number with unit suffix.

    Example:
        >>> number = 1512015806464
        >>> formatted_number = humanize_number(number)
        >>> print(formatted_number)
        Output: '1.51T'
    """

    units = ['', 'K', 'M', 'B', 'T']
    unit_index = 0
    while abs(number) >= 1000 and unit_index < len(units) - 1:
        number /= 1000.0
        unit_index += 1

    formatted_number = '{:.2f}{}'.format(number, units[unit_index])
    return formatted_number
