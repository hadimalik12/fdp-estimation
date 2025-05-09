#!/bin/bash

set -e  # Exit on error

# Set working directory
cd /tmp

# Check if fdp-env already exists
if [ -d "fdp-env" ]; then
    echo "Virtual environment 'fdp-env' already exists in /tmp."
else
    echo "Creating virtual environment 'fdp-env' in /tmp..."
    python3.11 -m venv fdp-env
fi

# Activate the virtual environment
source fdp-env/bin/activate

# Upgrade pip and install required packages
pip install --upgrade pip
pip install jupyterlab ipykernel

# Register the kernel
python -m ipykernel install --user --name=fdp-env --display-name "Python (fdp-env)"

echo "Setup complete! To start Jupyter Lab, run:"
echo "source /tmp/fdp-env/bin/activate && jupyter lab"

# Optionally, uncomment the next line to launch Jupyter Lab automatically:
# jupyter lab