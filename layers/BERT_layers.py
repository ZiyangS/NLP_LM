import torch
import torch.nn as nn
from torch.nn import functional as F
import math


def gelu(x):
    """
    Gaussian Error Linear Units (GELUs) activation function, https://arxiv.org/abs/1606.08415
    In GPT, it uses approximation 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


activations = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    '''
    This normalization helps to stabilize the hidden states in a neural network.
    The optional conditional normalization is derived from https://spaces.ac.cn/archives/7124,
    '''
    def __init__(self, hidden_size, eps=1e-12, conditional=False):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        self.conditional = conditional
        if conditional:
            # Conditional layer normalization (2 linear layers) for conditional text generation task
            # Input is concated of input vector and conditional vector; Initialized as 0
            self.dense1 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.dense1.weight.data.uniform_(0, 0)
            self.dense2 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.dense2.weight.data.uniform_(0, 0)

    def forward(self, x):
        if self.conditional:
            inputs = x[0]
            cond = x[1]
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)
            u = inputs.mean(-1, keepdim=True)
            s = (inputs - u).pow(2).mean(-1, keepdim=True)
            x = (inputs - u) / torch.sqrt(s + self.eps)
            # weight and bias are modifiedrespectively by the outputs of dense1 and dense2 (with respect to the condition vector cond)
            return (self.weight + self.dense1(cond)) * x + (self.bias + self.dense2(cond))
        else:
            # Compute the mean and std along the last axis (hidden_size)
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps) # normalize by subtracting the mean and dividing by std
            return self.weight * x + self.bias # the normalized x is scaled by the weight and shifted by the bias

class MultiHeadAttentionLayer(nn.Module):
    '''
    multi-head attention mechanism for capturing semantic dependencies in an input sequence.
    '''
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 dropout_rate,
                 attention_scale=True, # ?
                 return_attention_scores=False # ?
                 ):
        super(MultiHeadAttentionLayer, self).__init__()

        assert hidden_size % num_attention_heads == 0 # hidden states need to be divided into multiple heads

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores

        # make q,k,v with same size
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)

        self.o = nn.Linear(hidden_size, hidden_size) # input is the concatenated outputs of all heads, and output final embedding

        self.dropout = nn.Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        '''
        rearranges the last two dimensions of the input tensor to fit the required shape for multi-head attention calculation.
        '''
        # reshape to (batch_size, sequence_length, num_attention_heads, attention_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # return (batch_size, num_attention_heads, sequence_length, attention_head_size)

    def forward(self, query, key, value, attention_mask=None):
        mixed_query_layer = self.q(query) # (batch_size, query_len, hidden_size)
        mixed_key_layer = self.k(key) # (batch_size, key_len, hidden_size)
        mixed_value_layer = self.v(value) # (batch_size, value_len, hidden_size)

        query_layer = self.transpose_for_scores(mixed_query_layer) # (batch_size, num_attention_heads, query_len, attention_head_size)
        key_layer = self.transpose_for_scores(mixed_key_layer) # (batch_size, num_attention_heads, key_len, attention_head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer) # (batch_size, num_attention_heads, value_len, attention_head_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # (batch_size, num_attention_heads, query_len, key_len)

        # 是否进行attention scale
        if self.attention_scale:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Given binary attention_mask, ignore the elements with 0 by assigning a large negative value (zeroing out after applying softmax)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask

        # todo: question, did not divide by \sqrt(d_k)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # (batch_size, num_attention_heads, query_len, attention_head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (batch_size, query_len, num_attention_heads, attention_head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,) # (batch_size, query_len, hidden_size)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.return_attention_scores:
            return self.o(context_layer), attention_scores # attention_scores without softmax
        else:
            return self.o(context_layer) # the final embedding is computed from the output layer using concatenated output of multiple heads


class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed-forward network (FFN) consists of two linear layers
    '''
    def __init__(self, hidden_size, intermediate_size, dropout_rate=0.5, hidden_act='gelu', is_dropout=True):
        super(PositionWiseFeedForward, self).__init__()

        self.is_dropout = is_dropout
        self.intermediate_act_fn = activations[hidden_act]
        self.intermediateDense = nn.Linear(hidden_size, intermediate_size)
        self.outputDense = nn.Linear(intermediate_size, hidden_size)
        if self.is_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.is_dropout:
            x = self.dropout(self.intermediate_act_fn(self.intermediateDense(x))) # (batch size, seq len, intermediate_size)
        else:
            x = self.intermediate_act_fn(self.intermediateDense(x))
        return self.outputDense(x)
