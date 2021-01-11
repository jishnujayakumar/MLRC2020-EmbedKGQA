#For batch_size >=2
def custom_collate_fn(batch):
    # print(len(batch))
    # exit(0)
    question_tokenized = batch[0]
    attention_mask = batch[1]
    head_id = batch[2]
    tail_onehot = batch[3]
    question_tokenized = torch.stack(question_tokenized, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    return question_tokenized, attention_mask, head_id, tail_onehot 

# SentenceTransformer Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_pretrained_model_name(transformer_name):
        if transformer_name == 'RoBERTa':
            return 'roberta-base'
        elif transformer_name == 'XLNet':
            return 'xlnet-base-cased'
        elif transformer_name == 'ALBERT':
            return 'albert-base-v2'
        elif transformer_name == 'SentenceTransformer':
            return 'sentence-transformers/bert-base-nli-mean-tokens'
        elif transformer_name == 'Longformer':
            return 'allenai/longformer-base-4096'
        else:
            print('Incorrect pretrained model name specified:', transformer_name)
            exit(0)