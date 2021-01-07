# Steps to train Knowledge Graph Embedding Models 

### **NOTE:**
- First, make sure to complete all the steps mentioned in [getting-started](https://github.com/jishnujayakumar/MLRC2020-EmbedKGQA#get-started) section.
- Make sure to be in **MLRC2020-EmbedKGQA/train_embeddings** directory.
  - ```bash 
    cd $EMBED_KGQA_DIR/train_embeddings/
    ```
- Please tune the hyperparameters according to your need. If needed, add more parameters from [here](https://github.com/jishnujayakumar/MLRC2020-EmbedKGQA/blob/main/train_embeddings/main.py).
- Tests have been performed on the following models(`<model>`)
    - ComplEx: A pretrained-model has been taken from [EmbedKGQA](https://github.com/malllabiisc/EmbedKGQA#metaqa)[1].
    - TuckER: For this, training has been performed.
- Other supported types include: 
    - DistMult
    - SimplE
    - RESCAL  

### Train MetaQA KG

```bash
python main.py  --model TuckER \
                --cuda True \ # CUDA enabled
                --outfile <output_file_name> \
                --valid_steps 1 \
                --dataset <MetaQA_half or MetaQA> \ #for kg_type:{half, full} use dataset:{MetaQA_half, MetaQA}
                --num_iterations 5 \
                --batch_size 256 \
                --l3_reg .00001
```
- **Output path**: `$EMBED_KGQA_DIR/kg_embeddings/MetaQA/....`

- After training the respective MetaQA KG dataset, place the output to predefined path to train the QA dataset
    -   ```bash
        # For MetaQA_half dataset
        # dataset-name-1=MetaQA_half, dataset-name-2=MetaQA_half

        # For MetaQA dataset
        # dataset-name-1=MetaQA_full, dataset-name-2=MetaQA
        
        # Predefined path to store the necessary KGE training output files
        # to be used for training QA dataset
        destination_path="EMBED_KGQA_DIR/pretrained_models/embeddings/<model>_<dataset-name-1>/"
        
        # Make the required directory
        mkdir -p $destination_path
        
        # Copy to predefined location
        cp -R $EMBED_KGQA_DIR/kg_embeddings/<model>/<dataset-name-2>/* $destination_path
        ```
### Train WebQuestionsSP KG

```bash
#for kg_type:{half, full} use config_suffix:{half, full}
kge start $EMBED_KGQA_DIR/config/relational_tucker3-train-webqsp-<half or full>.yaml \
--job.device cuda:<gpu-id>
```
- **Output path**: `$EMBED_KGQA_DIR/train_embeddings/kge/local/experiments/....`

- After training the respective WebQSP KG dataset, place the output to predefined path to train the QA dataset
    -   ```bash
        # NOTE: For dataset-name:{fbwq_half, fbwq_full}, kg-type:{half, full}

        # Predefined path to store the necessary KGE training output files
        # to be used for training QA dataset
        destination_path=`$EMBED_KGQA_DIR/pretrained_models/embeddings/<model>_<dataset-name>/`

        # Make the required directory
        mkdir -p $destination_path

        # Find the best checkpoint and copy to predefined path for training QA dataset

        # NOTE: Don't remove '*' prior to <kg-type>

        find \
        $EMBED_KGQA_DIR/train_embeddings/kge/local/experiments/*<kg-type> \
        -type f -name 'checkpoint_best.pt' -print0 | xargs -0 -r cp -t $destination_path

        # Copy entity_ids.del to predefined path for training QA dataset
        cp $EMBED_KGQA_DIR/data/<dataset-name>/entity_ids.del $destination_path
        ```

- This scheme is used as suggested by [1]'s author. View [here](https://github.com/malllabiisc/EmbedKGQA#webquestionssp).
- Feel free to try out different parameters mentioned in config/*.yaml as per your need.
- See [LibKGE](https://github.com/uma-pi1/kge) for more details regarding the `kge` tool.

