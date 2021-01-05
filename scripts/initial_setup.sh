#!/bin/bash

# Install required packages
pip install -r requirements.txt

# Create required directories
mkdir -p checkpoints/ MLRC2020-EmbedKGQA/KGQA/RoBERTa/results/

# Change script permissions
chmod -R 700 scripts/

# Set a new env variable called EMBED_KGQA_DIR with MLRC2020-EmbedKGQA/ directory's absolute path as value
# If using bash shell, use 
echo 'export EMBED_KGQA_DIR=`pwd`' >> ~/.bash_profile && source ~/.bash_profile

echo "Initial setup complete."