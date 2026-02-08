#!/bin/bash

# Run the Stimulus Software application

set -e

ENV_NAME="eeg"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the application inside the conda environment, capturing output
conda run -n "$ENV_NAME" python main.py 2>&1 | tee -a logs/terminal_output.log
