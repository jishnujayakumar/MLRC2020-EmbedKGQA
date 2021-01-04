'''
This document contains helper functions
'''

def get_tokenizer(transformer_name):
    if transformer_name == 'RoBERTa':
        return RobertaTokenizer.from_pretrained('roberta-base')
    elif transformer_name == 'XLNet':
        return XLNetTokenizer.from_pretrained('xlnet-base-cased')
    elif transformer_name == 'ALBERT':
        return AlbertTokenizer.from_pretrained('albert-base-v2')
    elif transformer_name == 'SentenceTransformer':
        return AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
    elif transformer_name == 'Reformer':
        return ReformerTokenizer.from_pretrained('google/reformer-crime-and-punishment')
    else:
        print('Incorrect transformer specified:', transformer_name)
        exit(0)

def tokenize_question(question, tokenizer):
    question = f"<s>{question}</s>"
    encoded_question = tokenizer.encode_plus(
                            question, # Question to encode
                            add_special_tokens = False, # Add '[CLS]' and '[SEP]', as per original paper
                            max_length = 64,           # Pad & truncate all sentences.
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt'     # Return pytorch tensors.
                        )
    return encoded_question
