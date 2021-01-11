#!/bin/bash

kg_embedding_model=$1
que_embedding_model=$2
kg_type=$3
batch_size=$4
gpu_id=$5
relation_dim=$6
cd $EMBED_KGQA_DIR/$que_embedding_model
python main.py --mode train --relation_dim $relation_dim --que_embedding_model $que_embedding_model --do_batch_norm 0 \
--gpu $gpu_id --freeze 1 --batch_size $batch_size --validate_every 1 --hops webqsp_$kg_type --lr 0.00002 --entdrop 0.0 --reldrop 0.0 --scoredrop 0.0 \
--decay 1.0 --model $kg_embedding_model --patience 10 --ls 0.0 --l3_reg 0.001 --nb_epochs 200 --outfile webqsp_$kg_type_$que_embedding_model

# Usage: $EMBED_KGQA_DIR/scripts/train_webqsp.sh <ComplEx/TuckER> <RoBERTa/XLNet/...> <half/full> <batch_size> <gpu_id> <relation_dim>