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