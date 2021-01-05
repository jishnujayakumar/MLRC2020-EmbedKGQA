#!/bin/bash

BASEDIR=`$EMBED_KGQA_DIR/data`

for webqsp_dataset in {'fbwq_half', 'fbwq_full'}
do
    if [ ! -d "$BASEDIR/$webqsp_dataset" ]; then
        echo "\"$webqsp_dataset\" dataset not found. Kindly download it by following steps mentioned in getting-started section [README]."
    else
        echo "\"$webqsp_dataset\" is present."

    if [ ! -f "$BASEDIR/$webqsp_dataset/dataset.yaml" ]; then
        python preprocess/preprocess_default.py data/$webqsp_dataset/
    else
        echo "\"$webqsp_dataset\" already prepared."
    fi
done