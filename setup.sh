#!/bin/bash

# One-time environment setup for Stimulus Software
# Requires: Anaconda or Miniconda

set -e

ENV_NAME="eeg"
PYTHON_VERSION="3.12"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Install Anaconda or Miniconda first."
    exit 1
fi

# Create environment if it doesn't exist
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists, updating..."
else
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

# Install dependencies
echo "Installing dependencies..."
conda install -y -n "$ENV_NAME" tk pip
conda run -n "$ENV_NAME" pip install -r requirements.txt

echo ""
echo "Setup complete. Run ./run.sh to start the application."
