#!/bin/bash

export SLURM_NTASKS=4

#SBATCH --job-name=train_GAN
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mem=50GB 
#SBATCH --gres=gpu:2
    
hostname
pwd

source /tcmldrive/lib/miniconda3/etc/profile.d/conda.sh
conda activate Noga_ECG2

conda info

python -u /tcmldrive/NogaK/ECG_classification/ecg_AFIB_GAN/GAN/main_seqGAN.py
 
echo "END"