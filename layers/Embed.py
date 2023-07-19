import torch
import torch.nn as nn
import math
from layers.BERT_layers import LayerNorm


class PositionalEmbedding(nn.Module):
    '''
    Pos embedding is used for age
    '''
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    '''
    Value Embedding, convolutional layer
    '''
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    '''
    The fixed embedding
    '''
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    '''
    difference between temporal embedding and time feature embedding
    '''
    def __init__(self, d_model, embed_type='fixed'):
        super(TemporalEmbedding, self).__init__()

        day_size = 32
        month_size = 13
        year_size = 120

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        self.year_embed = Embed(year_size, d_model)
        self.month_embed = Embed(month_size, d_model)
        self.day_embed = Embed(day_size, d_model)

    def forward(self, x):
        x = x.long()

        year_x = self.year_embed(x[:, :, 0])
        month_x = self.month_embed(x[:, :, 1])
        day_x = self.day_embed(x[:, :, 2])

        return day_x + month_x + year_x


class DataEmbedding(nn.Module):
    '''
    value embedding + temporal embedding +  postional embedding (FedFormer, Informer)
    '''
    def __init__(self, c_in, d_model, embed_type='fixed', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    '''
    value embedding + temporal embedding (AutoFormer)
    '''
    def __init__(self, c_in, d_model, embed_type='fixed', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class BertEmbeddings(nn.Module):
    """
    BERT embeddings: generate word, position and token_type embeddings.
    vocab_size: the total number of unique tokens
    hidden_size: the size of embeddings
    max_position: the maximum position or sequence length that BERT can handle
    segment_vocab_size: the number of distinct segment in the input sequence
    """
    def __init__(self, vocab_size, hidden_size, max_position, segment_vocab_size, drop_rate):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.segment_embeddings = nn.Embedding(segment_vocab_size, hidden_size)
        self.layerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, token_ids, segment_ids=None):
        seq_length = token_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)

        words_embeddings = self.word_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)

        embeddings = words_embeddings + position_embeddings + segment_embeddings
        embeddings = self.layerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

