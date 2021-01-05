# EmbedKGQA: Reproduction and Ablation Study 
This is the code for the [MLRC2020 challenge](https://paperswithcode.com/rc2020) for the [ACL 2020](https://acl2020.org/) paper [Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings](https://malllabiisc.github.io/publications/papers/final_embedkgqa.pdf)[1]

# Requirements
- Python >= 3.7.5
- zip
- unzip
- Pytorch version [1.3.0a0+24ae9b5](https://github.com/pytorch/pytorch/tree/24ae9b504094937fbc7c24012fbe5c601e024bcd). For more info, visit [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_19-10.html).
- Huggingface == 4.1.1 [For new transformers like Reformer]


# Helpful pointers
- Docker Image: [Cuda-Python[2]](https://hub.docker.com/r/qts8n/cuda-python/) can be used. Use the `runtime` tag.
- The experiments have been done using [2]. The requirements.txt packages' version have been set accordingly. This may vary w.r.t. [1].
- `KGQA/LSTM` and `KGQA/RoBERTa` directory nomenclature hasn't been changed to avoid unnecessary confusion w.r.t. the original codebase[1].

# Get started
```bash
# Clone the repo
git clone https://github.com/jishnujayakumar/MLRC2020-EmbedKGQA
cd MLRC2020-EmbedKGQA/ && pip install -r requirements.txt
mkdir -p checkpoints/ MLRC2020-EmbedKGQA/KGQA/RoBERTa/results/

# Change script permissions
chmod -R 700 scripts/

# Set a new env variable called EMBED_KGQA_DIR with MLRC2020-EmbedKGQA/ directory's absolute path as value
# If using bash shell, use 
echo 'export EMBED_KGQA_DIR=`pwd`' >> ~/.bash_profile && source ~/.bash_profile

# Download and unzip data and pretrained_models
# Google frive folder: https://drive.google.com/drive/folders/1RlqGBMo45lTmWz9MUPTq-0KcjSd3ujxc
# gdown requires anyone with the link id; right click on each file and get it 
gdown --id 1uWaavrpKKllVSQ73TTuLWPc4aqVvrkpx && unzip data.zip
gdown --id 1Ly_3RR1CsYDafdvdfTG35NPIG-FLH-tz && unzip pretrained_models.zip
```

# KG Embedding Training
- [Steps](https://github.com/jishnujayakumar/MLRC2020-EmbedKGQA/blob/main/train_embeddings/README.md) to train KG embeddings.

# QA Dataset Training
## For MetaQA
```bash
cd $EMBED_KGQA_DIR/KGQA/LSTM

python main.py  --mode train 
                --nb_epochs 100
                --relation_dim 200
                --hidden_dim 256
                --gpu 0 #GPU-ID
                --freeze 0 
                --batch_size 64
                --validate_every 4 
                --hops <1/2/3> #n-hops
                --lr 0.0005 
                --entdrop 0.1 
                --reldrop 0.2  
                --scoredrop 0.2
                --decay 1.0
                --model <Complex/TuckER> #KGE models
                --patience 10 
                --ls 0.0 
                --use_cuda True #Enable CUDA
                --kg_type <half/full>
```

## For WebQuestionsSP
```bash
cd $EMBED_KGQA_DIR/KGQA/RoBERTa

python main.py  --mode train 
                --relation_dim 200
                --que_embedding_model <RoBERTa/ALBERT/XLNet/Reformer/SentenceTransformer>
                --do_batch_norm 0
                --gpu 0 #GPU-ID
                --freeze 1 
                --batch_size 16 --validate_every 10 
                --hops <webqsp_half/webqsp_full>
                --lr 0.00002 
                --entdrop 0.0 
                --reldrop 0.0 
                --scoredrop 0.0
                --decay 1.0 
                --model <Complex/TuckER> 
                --patience 20 
                --ls 0.0 
                --l3_reg 0.001 
                --nb_epochs 200 
                --outfile <output_file_name>
```

# Helpful links
- [Details](https://github.com/malllabiisc/EmbedKGQA#instructions) about data and pretrained weights.
- [Details](https://github.com/malllabiisc/EmbedKGQA#dataset-creation) about dataset creation.
- [Presentation](https://slideslive.com/38929421/improving-multihop-question-answering-over-knowledge-graphs-using-knowledge-base-embeddings) for [1] by Apoorva Saxena.


### Citation:
Please cite the following paper if you use this code in your work.

```bibtex
Placeholder for ReScience C BibTex
```

For any clarification, comments, or suggestions please create an issue or contact [Jishnu](https://jishnujayakumar.github.io/) or [Ashish](mailto:asardana@nvidia.com).
