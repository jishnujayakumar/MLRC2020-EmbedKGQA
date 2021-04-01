# EmbedKGQA: Reproduction and Extended Study 
- This is the code for the [MLRC2020 challenge](https://paperswithcode.com/rc2020) w.r.t. the [ACL 2020](https://acl2020.org/) paper [Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings](https://malllabiisc.github.io/publications/papers/final_embedkgqa.pdf)[1]
- The code is build upon [1]:[5d8fdbd4](https://github.com/malllabiisc/EmbedKGQA/tree/5d8fdbd4be77fdcb2e67a0dc8a7115844606175a)
- Minor modifications have been made to [5d8fdbd4](https://github.com/malllabiisc/EmbedKGQA/tree/5d8fdbd4be77fdcb2e67a0dc8a7115844606175a) in order to perform the ablation study. In case of any query relating to the original code[1], please contact [Apoorva](https://apoorvumang.github.io/).
# Additional Experiments
- Knowledge Graph Embedding model
     - [TuckER](https://arxiv.org/abs/1901.09590)
     - Tested on {MetaQA_full, MetaQA_half} datasets 
- Question embedding models
    - [ALBERT](https://arxiv.org/abs/1909.11942)
    - [XLNet](https://arxiv.org/abs/1906.08237)
    - [Longformer](https://arxiv.org/abs/2004.05150)
    - [SentenceBERT](https://arxiv.org/abs/1908.10084) (SentenceTransformer)
    - Tested on {fbwq_full, fbwq_half} datasets 

# Requirements
- Python >= 3.7.5, pip
- zip, unzip
- Docker (Recommended)
- Pytorch version [1.3.0a0+24ae9b5](https://github.com/pytorch/pytorch/tree/24ae9b504094937fbc7c24012fbe5c601e024bcd). For more info, visit [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_19-10.html).

# Helpful pointers
- Docker Image: [Cuda-Python[2]](https://hub.docker.com/r/qts8n/cuda-python/) can be used. Use the `runtime` tag.
    - ```bash
      docker run -itd --rm --runtime=nvidia -v /raid/kgdnn/:/raid/kgdnn/ --name embedkgqa__4567 -e NVIDIA_VISIBLE_DEVICES=4,5,6,7  -p 7777:7777 qts8n/cuda-python:runtime
      ```
- Alternatively, Docker Image: [Embed_KGQA[3]](https://hub.docker.com/r/jishnup/embed_kgqa) can be used as well. It's build upon [2] and contains all the packages for conducting the experiments. 
    - Use `env` tag for image without models.
    - Use `env-models` tag for image with models.
    - ```bash
      docker run -itd --rm --runtime=nvidia -v /raid/kgdnn/:/raid/kgdnn/ --name embedkgqa__4567 -e NVIDIA_VISIBLE_DEVICES=4,5,6,7  -p 7777:7777 jishnup/embed_kgqa:env
      ```
    - All the required packages and models (from the extended study with better performance) are readily available in [3].
        - Model location within the docker container: `/raid/mlrc2020models/`
            - `/raid/mlrc2020models/embeddings/` contain the KG embedding models.
            - `/raid/mlrc2020models/qa_models/` contain the QA models.
- The experiments have been done using [2]. The requirements.txt packages' version have been set accordingly. This may vary w.r.t. [1].
- `KGQA/LSTM` and `KGQA/RoBERTa` directory nomenclature hasn't been changed to avoid unnecessary confusion w.r.t. the original codebase[1].

- `fbwq_full` and `fbwq_full_new` are the same but independent existence is required because
    - Pretrained `ComplEx` model uses `fbwq_full_new` as the dataset name
    - Trained `SimplE` model uses `fbwq_full` as the dataset name
- No `fbwq_full_new` dataset was found in the data shared by the author[1], so went ahead with this setting.

- Also, pretrained qa_models were absent in the data shared. The reproduction results are based on training scheme used by us.

- For training QA datasets, use ```batch_size >= 2```. 

# Get started
```bash
# Clone the repo
git clone https://github.com/jishnujayakumar/MLRC2020-EmbedKGQA && cd "$_"

# Set a new env variable called EMBED_KGQA_DIR with MLRC2020-EmbedKGQA/ directory's absolute path as value
# If using bash shell, run 
echo 'export EMBED_KGQA_DIR=`pwd`' >> ~/.bash_profile && source ~/.bash_profile

# Change script permissions
chmod -R 700 scripts/

# Initial setup
./scripts/initial_setup.sh

# Download and unzip, data and pretrained_models from the original EmbedKGQA paper
./scripts/download_artifacts.sh

# Install LibKGE
./scripts/install_libkge.sh
```

# Train KG Embeddings
- [Steps](https://github.com/jishnujayakumar/MLRC2020-EmbedKGQA/tree/main/train_embeddings#steps-to-train-knowledge-graph-embedding-models) to train KG embeddings.

# Train QA Datasets
Hyperparameters in the following commands are set w.r.t. [[1]](https://github.com/malllabiisc/EmbedKGQA#metaqa).
### MetaQA
```bash
# Method: 1
cd $EMBED_KGQA_DIR/KGQA/LSTM;
python main.py  --mode train \
            --nb_epochs 100 \
            --relation_dim 200 \
            --hidden_dim 256 \
            --gpu 0 \ #GPU-ID
            --freeze 0 \
            --batch_size 64 \
            --validate_every 4 \
            --hops <1/2/3> \ #n-hops
            --lr 0.0005 \
            --entdrop 0.1 \ 
            --reldrop 0.2 \
            --scoredrop 0.2 \
            --decay 1.0 \
            --model <ComplEx/TuckER> \ #KGE models
            --patience 10 \
            --ls 0.0 \
            --use_cuda True \ #Enable CUDA
            --kg_type <half/full>

        
# Method: 2
# Modify the hyperparameters in the script file w.r.t. your usecase
$EMBED_KGQA_DIR/scripts/train_metaQA.sh \
    <ComplEX/TuckER> \
    <half/full> \
    <1/2/3> \
    <batch_size> \
    <gpu_id> \
    <relation_dim>
```

### WebQuestionsSP
```bash
# Method: 1
cd $EMBED_KGQA_DIR/KGQA/RoBERTa;
python main.py  --mode train \
                --relation_dim 200 \
                --que_embedding_model RoBERTa \
                --do_batch_norm 0 \
                --gpu 0 \
                --freeze 1 \
                --batch_size 16 \
                --validate_every 10 \
                --hops webqsp_half \
                --lr 0.00002 \
                --entdrop 0.0 
                --reldrop 0.0 \
                --scoredrop 0.0 \
                --decay 1.0 \
                --model ComplEx \
                --patience 20 \
                --ls 0.0 \
                --l3_reg 0.001 \
                --nb_epochs 200 \
                --outfile delete

# Method: 2
# Modify the hyperparameters in the script file w.r.t. your usecase
$EMBED_KGQA_DIR/scripts/train_webqsp.sh \
    <ComplEx/SimplE> \
    <RoBERTa/ALBERT/XLNet/Longformer/SentenceTransformer> \
    <half/full> \
    <batch_size> \
    <gpu_id> \
    <relation_dim>
```

# Test QA Datasets
Set the mode parameter as `test` (keep the other hyperparameters same as used in training)

# Helpful links
- [Details](https://github.com/malllabiisc/EmbedKGQA#instructions) about data and pretrained weights.
- [Details](https://github.com/malllabiisc/EmbedKGQA#dataset-creation) about dataset creation.
- [Presentation](https://slideslive.com/38929421/improving-multihop-question-answering-over-knowledge-graphs-using-knowledge-base-embeddings) for [1] by [Apoorv](https://apoorvumang.github.io/).


### Citation:
Please cite the following if you incorporate our work.

```bibtex
@inproceedings{
p2021embedkgqa,
title={Embed{KGQA}: Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings},
author={Jishnu Jaykumar P and Ashish Sardana},
booktitle={ML Reproducibility Challenge 2020},
year={2021},
url={https://openreview.net/forum?id=VFAwCMdWY7}
}

#Placeholder for ReScience C BibTex. To be updated upon acceptance.
```

Following 3 options are available for any clarification, comments or suggestions
- Join the [discussion forum](https://github.com/jishnujayakumar/MLRC2020-EmbedKGQA/discussions/).
- Create an [issue](https://github.com/jishnujayakumar/MLRC2020-EmbedKGQA/issues).
- Contact [Jishnu](https://jishnujayakumar.github.io/) or [Ashish](mailto:asardana@nvidia.com).
