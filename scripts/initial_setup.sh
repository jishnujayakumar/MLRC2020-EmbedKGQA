#!/bin/bash

# Install required packages
pip install -r requirements.txt

# Create required directories
mkdir -p $EMBED_KGQA_DIR/checkpoints/ $EMBED_KGQA_DIR/KGQA/RoBERTa/results/

echo "Initial setup complete."