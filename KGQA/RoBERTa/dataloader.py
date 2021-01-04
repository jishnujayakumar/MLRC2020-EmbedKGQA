import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import unicodedata
import re
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from transformers import *


class DatasetWebQSP(Dataset):
    def __init__(self, data, entities, entity2idx, transformer_name):
        self.data = data
        self.entities = entities
        self.entity2idx = entity2idx
        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)
        self.index_array = list(self.entities.keys())
        self.tokenizer = self.get_tokenizer(transformer_name)

    def get_tokenizer(self, transformer_name):
        if transformer_name == 'RoBERTa':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif transformer_name == 'XLNet':
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        elif transformer_name == 'ALBERT':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        elif transformer_name == 'SentenceTransformer':
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        elif transformer_name == 'Reformer':
            self.tokenizer = ReformerTokenizer.from_pretrained('google/reformer-crime-and-punishment')
        else:
            print('Incorrect transformer specified:', transformer_name)
            exit(0)

    def __len__(self):
        return len(self.data)
    
    def pad_sequence(self, arr, max_len=128):
        num_to_add = max_len - len(arr)
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        batch_size = len(indices)
        vec_len = len(self.entity2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        # one_hot = -torch.ones(vec_len, dtype=torch.float32)
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[1]
        question_tokenized, attention_mask = self.tokenize_question(question_text)
        head_id = self.entity2idx[data_point[0].strip()]
        tail_ids = []
        for tail_name in data_point[2]:
            tail_name = tail_name.strip()
            #TODO: dunno if this is right way of doing things
            if tail_name in self.entity2idx:
                tail_ids.append(self.entity2idx[tail_name])
        tail_onehot = self.toOneHot(tail_ids)
        return question_tokenized, attention_mask, head_id, tail_onehot 

    def tokenize_question(self, question):
        question = "<s>"+question+"</s>"
        print("Question: ", question)
        encoded_question = self.tokenizer(
            question, # Question to encode,
            add_special_tokens = False, # Add '[CLS]' and '[SEP]', as per original paper
            max_length = 64,           # Pad & truncate all sentences.
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt'     # Return pytorch tensors.
        )
        print("Complete")
        return encoded_question.input_ids, encoded_question.attention_mask

# def _collate_fn(batch):
#     print(len(batch))
#     exit(0)
#     question_tokenized = batch[0]
#     attention_mask = batch[1]
#     head_id = batch[2]
#     tail_onehot = batch[3]
#     question_tokenized = torch.stack(question_tokenized, dim=0)
#     attention_mask = torch.stack(attention_mask, dim=0)
#     return question_tokenized, attention_mask, head_id, tail_onehot 

class DataLoaderWebQSP(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderWebQSP, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

