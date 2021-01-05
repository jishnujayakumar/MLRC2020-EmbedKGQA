#!/bin/bash

# Retriev and install LibKGE in development mode
cd train_embeddings/
git clone https://github.com/uma-pi1/kge.git && 
pip install -e .
