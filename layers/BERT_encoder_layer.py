import torch.nn as nn
from layers.BERT_layers import LayerNorm, MultiHeadAttentionLayer, PositionWiseFeedForward


class BertLayer(nn.Module):
    """
    Transformer encoder layer
    Architecture: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm
    Feed forward network cosnsists of two linear layer, hidden_size to intermediate_size to hidden_size
    """
    def __init__(self, hidden_size, num_attention_heads, dropout_rate, intermediate_size, hidden_act, is_dropout=False):
        super(BertLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttentionLayer(hidden_size, num_attention_heads, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layerNorm1 = LayerNorm(hidden_size, eps=1e-12)
        self.feedForward = PositionWiseFeedForward(hidden_size, intermediate_size, hidden_act, is_dropout=is_dropout)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layerNorm2 = LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask):
        # todo: check whether it should transform to Q, K, V?
        self_attn_output = self.multiHeadAttention(hidden_states, hidden_states, hidden_states, attention_mask)
        hidden_states = hidden_states + self.dropout1(self_attn_output) # residual connections and dropout
        hidden_states = self.layerNorm1(hidden_states)
        self_attn_output2 = self.feedForward(hidden_states)
        hidden_states = hidden_states + self.dropout2(self_attn_output2)
        hidden_states = self.layerNorm2(hidden_states)
        return hidden_states

