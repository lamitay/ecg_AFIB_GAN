import torch
import wfdb
import os
import numpy as np

def split_records_to_intervals(record, annotation, sample_length, channel, overlap):
    """split each of the records to intervals and the annotation to a binar 1D vector 

    Args:
        record (wfdb.record): the record object of wfbd
        annotation (wfbd.annotation): the annotation object of wfbd
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
        num_of_intervals = int(np.floor(len(signal)/num_of_samples_in_interval))
        intervals = signal[:num_of_intervals*num_of_samples_in_interval].reshape(num_of_intervals, num_of_samples_in_interval)
        annot_intervals = annots_signal[:num_of_intervals*num_of_samples_in_interval].reshape(num_of_intervals, num_of_samples_in_interval)

    else:
        raise NotImplementedError('need to implement the overlap case!')

    #remove all interval that has mixed annotations:
    sum_of_rows = torch.sum(annot_intervals, axis=1)
    only_N_intervals_idx = torch.where(sum_of_rows == 0)
    only_AF_intervals_idx = torch.where(sum_of_rows == num_of_samples_in_interval)
    final_annots_indices = torch.hstack([only_AF_intervals_idx[0],only_N_intervals_idx[0]])
    final_annots = sum_of_rows[final_annots_indices]/num_of_samples_in_interval #to get a binar signal

    #get the final intervals after filtering the mixed annotation:
    intervals = intervals[final_annots_indices, :]

    return intervals, final_annots








if __name__ == '__main__':
    folder_path = 'C:/Users/nogak/Desktop/MyMaster/YoachimsCourse/files/'
    name = '04015'
    file_name = os.path.join(folder_path, name)
    record = wfdb.rdrecord(file_name)
    annotation = wfdb.rdann(file_name, 'atr')
    split_records_to_intervals(record, annotation, sample_length = 10, channel = 0, overlap = True)







