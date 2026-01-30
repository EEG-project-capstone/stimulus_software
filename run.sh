#!/bin/bash

# Setup instructions (run once):
# conda create -n eeg python=3.12
# conda activate eeg
# conda install tk
# conda install pip
# pip install -r requirements.txt

# Create logs directory if it doesn't exist
# mkdir -p logs

# Run the application and capture terminal output
# Appends to a single log file while also showing in terminal
python main.py 2>&1 | tee -a logs/terminal_output.log