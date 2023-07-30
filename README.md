## Leveraging Generative Adversarial Networks for Atrial Fibrillation Data Augmentation in Single Lead Electrocardiogram Classification
This repository contains the code for a final project in a graduate course in the Technion, Israel Institute of Technology (IIT). 
The project focuses on the application of Generative Adversarial Networks (GANs) to Electrocardiogram (ECG) data, specifically generating synthetic 6 seconds Atrial Fibrillation (AF) intervals.

## Dataset

The dataset used in this project is a subset of the MIT-BIH Atrial Fibrillation Database. The data has been filtered using the Beat-to-Beat Signal Quality Index (BSQI) and split into 6-second intervals.

## Structure

The repository is divided into three main sections:

# Classifier: This directory contains the scripts for a classifier model. The files include:
analisys.py: Script for data analysis.
classifier_config.yaml: Configurations file for experiments.
dataset.py: Script for dataset preparation and processing.
main.py: Main script to run the classifier.
metrics.py: Script for defining metrics.
model.py: Script for defining the classifier model.
trainer.py: Script for training the classifier model.
transform.py: Script for data transformations.
utils.py: Utility functions.

More summarization scripts are also part of this directory and can be run as standalone.
# GAN: This directory contains the scripts for the GAN model. The files include:
GAN_exploration.ipynb: Jupyter notebook for GAN products exploration.
config.yaml: Configurations file for experiments.
GAN_models_LSTM.py: Script for defining the GAN models with LSTM.
models.py: Script for defining the GAN models.
seq_models.py: Script for defining sequence models.
trainer.py: Script for training the GAN models.
main_<GAN_type>.py: Main script to run different GAN models.
# other_GAN: This directory contains scripts for an alternative GAN model that was not used eventually. The files include:
dataset_git.py: Script for dataset preparation and processing.
gan.py: Script for defining the GAN model.
train.py: Script for training the GAN model.
# run_multiple.sh: Bash script for running multiple experiments

## Setup

To set up the environment to run the code, use the provided environment.yml file. This will create a conda environment with all the necessary dependencies.

bash
Copy code
conda env create -f environment.yml

## Usage

To run the classifier:

bash
Copy code
cd 
python main.py
To run the GAN model:

bash
Copy code
cd GAN
python trainer.py

## Note
This project is part of a graduate course and is intended for educational purposes. The models and code provided should not be used for medical diagnosis or treatment without further validation.

All work in this project was done equally by Amitay Lev and Noga Kertes on the same server, therefore all commits are from the same user.
