#!/bin/sh

BASEDIR=`$EMBED_KGQA_DIR/data`

#fbwq_half dataset
if [ ! -d "$BASEDIR/fbwq_half" ]; then
    echo "\"fbwq_half\" dataset not found. Kindly download it by following steps mentioned in getting-started section [README]."
else
    echo "\"fbwq_half\ is present."

if [ ! -f "$BASEDIR/fbwq_half/dataset.yaml" ]; then
    python preprocess/preprocess_default.py data/fbwq_half/
else
    echo fbwq_half already prepared
fi

#fbwq_full dataset
if [ ! -d "$BASEDIR/fbwq_full" ]; then
    echo "\"fbwq_full\" dataset not found. Kindly download it by following steps mentioned in getting-started section [README]."
else
    echo "\"fbwq_full\ is present."

if [ ! -f "$BASEDIR/fbwq_full/dataset.yaml" ]; then
    python preprocess/preprocess_default.py data/fbwq_full/
else
    echo fbwq_full already prepared
fi