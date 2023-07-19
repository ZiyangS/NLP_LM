import json
import time

from models.BERT import *
from models.base_transformer import *

def build_transformer_model(
    config_path=None,
    checkpoint_path=None,
    model='bert',
    application='encoder',
    **kwargs
    ):
    """
    Build a transformer model based on the provided configuration. Loading weight from checkpoint optionally
    """
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings', 512)
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)
    models = {
        'bert': BERT,
        'fnet': FNet
    }

    my_model = models[model]
    transformer = my_model(**configs)
    if checkpoint_path is not None:
        transformer.load_weights_from_pytorch_checkpoint(checkpoint_path)
    return transformer