import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

import spacy
import string
import pymorphy3
import re
import os
from pymorphy3 import MorphAnalyzer


class TextDataset(Dataset):
    def __init__(self, text, sinonims, regs, spacy_load, encoding="utf-8"):
        self.text = text
        self.sinonims_dict = {key:[val_2 for val_2 in val_1.values() if pd.notna(val_2)] for key, val_1 in sinonims.to_dict('index').items()}
        self.reg_dict = regs.to_dict(orient='split', index=False)    
        self.nlp = spacy_load
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        sentence = self.text[item]
        output = self.data_preprocessor(sentence)
        output = self.replace_reg(output)
        output = self.replace_sinonim(output)
        output = self.text_lematizer_spacy(output)
        return output

    def data_preprocessor(self, text):
        text = re.sub(r'(?<=[0-9]),(?=[0-9])', ".", text)                       #замена запятых в числах на точки
        text = text.replace('_', ' ')
        text = text.lower()
        
        punctuation = set(string.punctuation)
        exclude=set(['%', '-', '.'])
        punctuation.difference_update(exclude)
        punctuation.update(['’', '—', ',', '”', '..', '...', '…', '. .', '. . .', '- -', '- - -', '--', '---'])
        for elem in punctuation:
          text = text.replace(elem, ' ')
        
        text = re.sub(r'(\s\s+)', ' ', text)
        text = text.strip()
#        text = text.astype(str)
        
        return text

    def replace_reg(self, text):
        text = re.sub(r'(\s\s+)', ' ', text)
        text = text.strip()
        
        for line in self.reg_dict['data']:
            text = re.sub(rf'{line[0]}', line[1], text)
        
        text = re.sub(r'(\s\s+)', ' ', text)
        text = text.strip()
        return text
   
    def replace_sinonim(self, text):
        for key in self.sinonims_dict.keys():
            for val in self.sinonims_dict[key]:
                text = text.replace(str(val)+' ', str(key)+' ')
        
        text = re.sub(r'(\s\s+)', ' ', text)
        text = text.strip()
        
        return text

    def text_lematizer_spacy(self, text):
        morph = MorphAnalyzer()
        doc = self.nlp(text)
        sentence_new = ' '.join([morph.parse(i.lemma_)[0].normal_form for i in doc])
        
        return sentence_new


class WordDataset(Dataset):

  def __init__(self, texts, tokenizer):
    self.texts = texts
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = str(self.texts[idx])

    encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                return_token_type_ids=False,
                return_attention_mask=True,
                truncation=True
    )

    return {
      'text': text,
      'input_ids': encoding['input_ids'],
      'attention_mask': encoding['attention_mask']
    }


def length_to_mask(length, max_len=512, dtype=None, device='cpu'):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    if max_len != None and length.max().item() > max_len:
        max_len = max_len 
    else:
        max_len = length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype)
    if device is not None:
        mask = torch.as_tensor(mask, device=device)
    return mask


def collate_fn_with_padding(
    input_batch: dict, pad_id=0, max_len=512) -> torch.Tensor:
    
    seq_lens = [len(x['input_ids']) for x in input_batch]
    max_seq_len = min(max(seq_lens), max_len)

    sent_input, attention_mask = [], []
    for item in input_batch:
        item['input_ids'] = item['input_ids'][:max_seq_len]
        item['attention_mask'] = item['attention_mask'][:max_seq_len]
        for _ in range(max_seq_len - len(item['input_ids'])):
            item['input_ids'].append(pad_id)
            item['attention_mask'].append(pad_id)
        sent_input.append(item['input_ids'])
        attention_mask.append(item['attention_mask'])

    new_batch = {
        'input_ids': sent_input,
        'attention_mask': attention_mask
        }

    return {key: torch.tensor(value) for key, value in new_batch.items()}

def preprocessing(data, tokenizer):
    
    nlp = spacy.load('ru_core_news_lg')
    sinonims = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'changes.xlsx'), sheet_name = 'sinonims', index_col='change_name')
    regs = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'changes.xlsx'), sheet_name = 'reg', usecols = ['regexp', 'change_name'])

    #text_data = pd.DataFrame([text]).to_numpy()
    text_data = data
    text_dataset = TextDataset(text_data, sinonims, regs, spacy_load=nlp, encoding="utf-8")
    word_dataset = WordDataset(texts=text_dataset, tokenizer=tokenizer)
    test_dataloader = DataLoader(word_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn_with_padding, drop_last=False)
    
    return test_dataloader

