# Steps to train Knowledge Graph Embedding Models 

### **NOTE:**
- First, make sure to complete all the steps mentioned in [getting-started](https://github.com/jishnujayakumar/MLRC2020-EmbedKGQA#get-started) section.
- Make sure to be in **MLRC2020-EmbedKGQA/train_embeddings** directory.
  - ```bash 
    cd $EMBED_KGQA_DIR/train_embeddings/
    ```
- Please tune the hyperparameters according to your need. If needed, add more parameters from [here](https://github.com/jishnujayakumar/MLRC2020-EmbedKGQA/blob/main/train_embeddings/main.py).
- Tests have been performed on the following models
    - ComplEx: A pretrained-model has been taken from [EmbedKGQA](https://github.com/malllabiisc/EmbedKGQA#metaqa)[1].
    - TuckER: For this, training has been performed.
- Other supported types include: 
    - DistMult
    - SimplE
    - RESCAL  

- ToDo: After training respective QA dataset set the outputs to `$EMBED_KGQA_DIR/pretrained_models/embeddings/<model>_<dataset_name>/`

### Train MetaQA KG

```bash
python main.py  --model TuckER \
                --cuda True \ # CUDA enabled
                --outfile <output_file_name> \
                --valid_steps 1 \
                --dataset <fbwq_half or fbwq_full> \ #for kg_type:{half, full} use dataset:{fbwq_half, fbwq_full}
                --num_iterations 5 \
                --batch_size 256 \
                --l3_reg .00001
```
- **Output path**: `$EMBED_KGQA_DIR/checkpoints/MetaQA/....`
### Train WebQuestionsSP KG

```bash
#for kg_type:{half, full} use config_suffix:{half, full}
kge start $EMBED_KGQA_DIR/config/relational_tucker3-train-webqsp-<half or full>.yaml \
--job.device cuda:<gpu-id>
```
- **Output path**: `$EMBED_KGQA_DIR/train_embeddings/kge/local/....`
- This scheme is used as suggested by [1]'s author. View [here](https://github.com/malllabiisc/EmbedKGQA#webquestionssp).
- Feel free to try out different parameters mentioned in config/*.yaml as per your need.
- After training, get the checkpoint `(*.pt)` and `entity.del` files to be used for training WebQSP QA dataset. 
- See [LibKGE](https://github.com/uma-pi1/kge) for more details regarding the `kge` tool.

