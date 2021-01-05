# Knowledge Graph Embedding training

### **NOTE:**
- First, make sure to complete all the steps in [getting-started](https://github.com/jishnujayakumar/MLRC2020-EmbedKGQA#get-started) section.
- Make sure to be in **MLRC2020-EmbedKGQA/train_embeddings** directory.
  - ```bash 
    cd $EMBED_KGQA_DIR/train_embeddings/
    ```
- Please tune the hyperparameters according to your need. If needed, add more parameters from [here](https://github.com/jishnujayakumar/MLRC2020-EmbedKGQA/blob/main/train_embeddings/main.py#L312).
- Tests have been performed on the following models
    - ComplEx: Pretrained-model have been taken from [EmbedKGQA](https://github.com/malllabiisc/EmbedKGQA#metaqa).
    - TuckER: Training has been performed.
- Other supported types include: 
    - DistMult
    - SimplE
    - RESCAL  

### For MetaQA Dataset

```bash
python main.py  --model TuckER \
                --cuda True \ # CUDA enabled
                --outfile output_file_name \
                --valid_steps 1 \
                --dataset <fbwq_half or fbwq_full> \ #for kg_type:{half, full} use dataset:{fbwq_half, fbwq_full}
                --num_iterations 5 \
                --batch_size 256 \
                --l3_reg .00001
```

### For WebQSP Dataset

```bash
To be written
```
