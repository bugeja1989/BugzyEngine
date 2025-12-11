#!/bin/bash

# This script sets up the environment for BugzyEngine on macOS.

# Check for Homebrew
if ! command -v brew &> /dev/null
then
    echo "Homebrew not found. Please install it from https://brew.sh/"
    exit 1
fi

# Install Python 3.11
brew install python@3.11

# Create a virtual environment
python3.11 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the required packages
pip install -r requirements.txt

# Check for MPS (Apple Silicon GPU) availability
python -c "import torch; print(\"MPS available: \", torch.backends.mps.is_available())"

echo "Installation complete. Run a new terminal or 'source venv/bin/activate' to use the environment."
