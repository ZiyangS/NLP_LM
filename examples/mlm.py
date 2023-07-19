#! -*- coding: utf-8 -*-
# test: mlm prediction
import time
import torch
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
vocab_path = root_model_path + "bert-base-uncased/vocab.txt"
config_path = root_model_path + "bert-base-uncased/config.json"
checkpoint_path = root_model_path + 'bert-base-uncased/pytorch_model.bin'
# setup tokenizer
tokenizer = Tokenizer(vocab_path, do_lower_case=True)

# mlm
sentence = "New York has a very famous statue called [MASK] of liberty."
token_ids, segment_ids = tokenizer.encode(sentence)
mask_position = token_ids.index(103)
tokens_ids_tensor, segment_ids_tensor = torch.tensor([token_ids]), torch.tensor([segment_ids])
# setup model by loading weights, set with_mlm=True
model = build_transformer_model(config_path, checkpoint_path, with_mlm=True)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")


print('\n ===== predicting =====\n')
model.eval()
with torch.no_grad():
  output = model(tokens_ids_tensor, segment_ids_tensor)[0]
  result = torch.argmax(output[mask_position, :]).item()
  print(tokenizer.convert_ids_to_tokens([result])[0])
