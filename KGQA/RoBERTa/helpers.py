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