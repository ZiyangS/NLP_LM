#! -*- coding: utf-8 -*-
# test: feature extraction
import torch
from models.BERT import *
from layers.tokenizers import Tokenizer
from models import build_transformer_model
import pandas as pd
import pytorch_lightning as pl

# fixed seed
pl.seed_everything(3407)

# reading csv files
root_model_path = "./"
data = pd.read_csv(root_model_path + 'sentiment/sentiment.train.data', names=['seq', 'label'], delimiter="\t", on_bad_lines='skip')
# load model
vocab_path = root_model_path + "bert-base-cased/vocab.txt"
config_path = root_model_path + "bert-base-cased/config.json"
checkpoint_path = root_model_path + 'bert-base-cased/pytorch_model.bin'
# setup tokenizer
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
# encoding sequence
sentence = u'language model'
token_ids, segment_ids = tokenizer.encode(sentence)
tokens_ids_tensor, segment_ids_tensor = torch.tensor([token_ids]), torch.tensor([segment_ids])
# setup model by loading weights
model = build_transformer_model(config_path, checkpoint_path)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

print('\n ===== predicting =====\n')
model.eval()
with torch.no_grad():
  output = model(tokens_ids_tensor, segment_ids_tensor)[0]
  print(output)