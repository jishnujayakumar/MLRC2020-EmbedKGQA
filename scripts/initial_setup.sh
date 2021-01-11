#!/bin/bash

# Install required packages
pip install -r requirements.txt

# Create required directories
mkdir -p checkpoints/ KGQA/RoBERTa/results/

echo "Initial setup complete."