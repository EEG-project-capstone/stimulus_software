#!/bin/bash

# Setup instructions (run once):
# conda create -n eeg python=3.12
# conda activate eeg
# conda install tk
# conda install pip
# pip install -r requirements.txt

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eeg

# Run the application and capture terminal output
python main.py 2>&1 | tee -a logs/terminal_output.log