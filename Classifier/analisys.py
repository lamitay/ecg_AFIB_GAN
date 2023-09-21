import antropy as ant
import numpy as np
from pecg import Preprocessing as Pre
from pecg.ecg import FiducialPoints as Fp
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, chisquare
from pyentrp import entropy as ent

""" Calculate COSEn for the real and fake intervals and plot the histograms
"""

def calc_mean_rr(x, fs):
    fp = Fp.FiducialPoints(x, fs)
    xqrs_inds = fp.jqrs()
    RR_intervals = np.diff(xqrs_inds)##/fs #in seconds
    return RR_intervals, RR_intervals.mean()


def COSEn(x, fs, debug = False):
    RR_intervals, mean_RR = calc_mean_rr(x, fs)
    r =  0.1*np.std(x)
    sample_entropy = ent.sample_entropy(RR_intervals, 2, r)
    cosen = sample_entropy[0] - np.log(2*r) - np.log(mean_RR)
    if debug:
        return cosen, sample_entropy[0]
    else:
        return cosen

def calc_cosen_for_dataset(dataset_folder_path, fs, num_of_samples = None, calc_normal_class = False):
    meta_data = pd.read_csv(os.path.join(dataset_folder_path,'meta_data.csv')) 
    normal_indices = np.where(meta_data['label']==0.)[0]
    AF_indices = np.where(meta_data['label']==1.)[0]
    if num_of_samples:
        if calc_normal_class:
            normal_indices = random.sample(normal_indices.tolist(), num_of_samples)
        AF_indices = random.sample(AF_indices.tolist(), num_of_samples)

    af_cosen = []
    af_mean_RR = []
    normal_cosen = []
    normal_mean_RR = []
    for i in range(num_of_samples):
        if calc_normal_class:
            norm_signal_path = meta_data.iloc[normal_indices[i]]['interval_path']
            norm_signal = np.load(os.path.join(dataset_folder_path,'intervals',norm_signal_path)).flatten()

            pre = Pre.Preprocessing(norm_signal, fs)
            norm_signal = pre.bpfilt()
            cosen, mean_RR = COSEn(norm_signal, fs = fs, debug = True)
            normal_cosen.append(cosen)
            normal_mean_RR.append(mean_RR)

        af_signal_path = meta_data.iloc[AF_indices[i]]['interval_path']
        af_signal = np.load(os.path.join(dataset_folder_path,'intervals',af_signal_path)).flatten()

        pre = Pre.Preprocessing(af_signal, fs)
        af_signal = pre.bpfilt()
        cosen, mean_RR = COSEn(af_signal, fs = fs, debug = True)

        af_cosen.append(cosen)
        af_mean_RR.append(mean_RR)
    
    af_cosen = np.array(af_cosen)[~np.isinf(np.array(af_cosen))]
    af_cosen = af_cosen[~np.isnan(np.array(af_cosen))]
    if calc_normal_class:
        normal_cosen = np.array(normal_cosen)[~np.isinf(np.array(normal_cosen))]
        normal_cosen = normal_cosen[~np.isnan(np.array(normal_cosen))]
        return af_cosen, normal_cosen
    else:
        return af_cosen

if __name__ == "__main__":
    real_dataset_folder_path = '/tcmldrive/NogaK/ECG_classification/fixed_datasets/dataset_len6_overlab0_chan0'
    real_af_cosen, normal_cosen = calc_cosen_for_dataset(real_dataset_folder_path, fs = 250, num_of_samples = 1000, calc_normal_class=True)
    # fake_dataset_folder_path = '/tcmldrive/NogaK/ECG_classification/data/fake_data_6_secs_50000_samples_gen2/'
    # fake_af_cosen = calc_cosen_for_dataset(fake_dataset_folder_path, fs = 250, num_of_samples = 10000, calc_normal_class=False)
    plt.figure()
    plt.hist(real_af_cosen, bins=30, alpha=0.5)
    plt.hist(normal_cosen, bins=30, alpha=0.5)
    # plt.hist(fake_af_cosen, bins=30, alpha=0.5)
    plt.legend(['AF - real', 'Normal - real'])#, 'AF - fake'])
    # plt.savefig('realNfake_af_normal.png')
    plt.title('COSEn of AF and normal ECG - 6 seconds intervals')
    plt.show()
    # plt.close()







