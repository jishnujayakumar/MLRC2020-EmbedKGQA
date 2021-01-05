# Knowledge Graph Embedding training

### **NOTE:**
- First, make sure to be in **MLRC2020-EmbedKGQA/train_embeddings** directory after following the steps in [getting-started](https://github.com/jishnujayakumar/MLRC2020-EmbedKGQA#get-started) section.
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
#kg_type: half, gpu-mode:enabled
python main.py  --model TuckER \
                --cuda True \
                --outfile TuckER.fbwq_half \
                --valid_steps 1 \
                --dataset fbwq_half --num_iterations 5 \
                --batch_size 256 \
                --l3_reg .00001
```
```bash
#kg_type: full
python main.py  --model TuckER \
                --cuda True \
                --outfile TuckER.fbwq_full \
                --valid_steps 1 \
                --dataset fbwq_full --num_iterations 5 \
                --batch_size 256 \
                --l3_reg .00001
```

### For WebQSP Dataset

```bash
To be written
```
