#!/bin/bash

kg_embedding_model=$1
kg_type=$2
n_hops=$3
batch_size=$4
gpu_id=$5
relation_dim=$6

cd $EMBED_KGQA_DIR/KGQA/LSTM;
python main.py  --mode train 
                --nb_epochs 100
                --relation_dim $relation_dim
                --hidden_dim 256
                --gpu $gpu_id #GPU-ID
                --freeze 0 
                --batch_size $batch_size
                --validate_every 4 
                --hops $n_hops #n-hops
                --lr 0.0005 
                --entdrop 0.1 
                --reldrop 0.2  
                --scoredrop 0.2
                --decay 1.0
                --model $kg_embedding_model #KGE models
                --patience 10 
                --ls 0.0 
                --use_cuda True #Enable CUDA
                --kg_type $kg_type


#Usage: $EMBED_KGQA_DIR/scripts/train_metaQA.sh <ComplEX/TuckER> <half/full> <1/2/3> <batch_size> <gpu_id> <relation_dim>