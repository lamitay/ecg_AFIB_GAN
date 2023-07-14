#!/bin/bash

# Python script to be run
SCRIPT="Classifier/main_mixed_data_training.py"

# Array of fake percentage values
PERCENTAGES=(0 10 20 30 40 50 60 70 80 90 100)

# Activate conda env
conda activate Noga_ECG2

# Loop over the array and run the python script with each percentage value
for PERC in "${PERCENTAGES[@]}"; do
    # Run python script in background with nohup and save output to a log file
    nohup python $SCRIPT --train_fake_perc $PERC > logs/log_${PERC}_fake_perc.out &
done
