#!/bin/bash

# Retrieve and install LibKGE in development mode
cd train_embeddings/
git clone https://github.com/uma-pi1/kge.git && cd kge
pip install -e .

cp $EMBED_KGQA_DIR/scripts/preprocess_webqsp_dataset.sh data
cp -R $EMBED_KGQA_DIR/data/fbwq_* data

# To train WebQSP dataset using pretrained ComplEx kge model, fbwq_full_new is required
cp -R data/fbwq_full/ data/fbwq_full_new/ 

echo "LibKGE setup complete."
