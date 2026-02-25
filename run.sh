#!/bin/bash

# Run the Stimulus Software application

set -e

VENV_DIR=".venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtual environment not found. Run ./setup.sh first."
  exit 1
fi

mkdir -p logs

"$VENV_DIR/bin/python" main.py 2>&1 | tee -a logs/terminal_output.log
