#!/bin/bash

# Retrieve and install LibKGE in development mode
cd train_embeddings/
git clone https://github.com/uma-pi1/kge.git && cd kge
pip install -e .

cp $EMBED_KGQA_DIR/scripts/preprocess_webqsp_dataset.sh data
cp -R $EMBED_KGQA_DIR/data/fbwq_* data

#Run kge training

