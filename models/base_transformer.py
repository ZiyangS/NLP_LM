import torch
import torch.nn as nn
import copy
import json

class Transformer(nn.Module):
    """
    Base model
    """
    def __init__(
            self,
            vocab_size,  # number of unique tokens
            hidden_size,  # dim of encoding vector
            num_hidden_layers,  # number of Transformer layers
            num_attention_heads,  # number of multi-attention heads
            intermediate_size,  # dimensionality of  feed-forward hidden layer
            hidden_act,  # activation function for feed-forward hidden layer
            dropout_rate,  # dropout rate
            embedding_size=None,  # embedding_size. If not specified, uses config setting
            attention_head_size=None,  # Attention V head_size
            #  If not specified, it defaults to hidden_size / num_attention_heads
            attention_key_size=None,  # Attention Q,K head_size
            sequence_length=None,  # Indicates whether the sequence length is fixed
            keep_tokens=None,  # Indicates whether keep token IDs in the vocabulary
            compound_tokens=None,  #  An additional set of tokens to be added to the vocabulary
            residual_attention_scores=False,  # Specifies whether to include residual connections in the attention matrix
            ignore_invalid_weights=False,  # Specifies whether to ignore invalid weights during model loading
            **kwargs
    ):
        super(Transformer, self).__init__()
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            vocab_size += len(compound_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size # ? difference of  attention_head_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size or self.hidden_size // self.num_attention_heads # Attention V head_size
        #  If not specified, it defaults to hidden_size / num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.residual_attention_scores = residual_attention_scores
        self.ignore_invalid_weights = ignore_invalid_weights

    def init_model_weights(self, module):
        raise NotImplementedError

    def variable_mapping(self):
        """
        构建pytorch层与checkpoint的变量名之间的映射表
        defining variable mappings between pytorch and checkpoints
        """
        return {}

    def load_weights_from_pytorch_checkpoint(self, checkpoint, mapping=None):
        """
        根据mapping从checkpoint加载权重
        loading weights from a PyTorch checkpoint file
        """
        # model = self
        state_dict = torch.load(checkpoint, map_location='cpu')
        mapping = mapping or self.variable_mapping()

        for new_key, old_key in mapping.items():
            if old_key in state_dict.keys():
                state_dict[new_key] = state_dict.pop(old_key)
        self.load_state_dict(state_dict, strict=self.ignore_invalid_weights)

def lm_mask(segment_ids):
    """
    creates a lower triangular attention mask for language model
    segment_ids is a tensor representing the segment or sentence IDs of each token in the input sequence.
    The shape of segment_ids is assumed to be (batch_size, sequence_length).
    """
    idxs = torch.arange(0, segment_ids.shape[1])
    mask = (idxs.unsqueeze(0) <= idxs.unsqueeze(1)).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    return mask

def unilm_mask(tokens_ids, segment_ids):
    """
    define UniLM (Unified Language Model) Attention Mask, which is used for models (UniLM: https://arxiv.org/abs/1905.03197)
    token_ids: shape is (batch_size, seq_length)
    segment_ids: shape is (batch_size, seq_length), divide source and target segment
    """
    # initially ignore the padding regions (set 1) and later set the mask values for those regions to 0.
    ids = segment_ids + (tokens_ids <= 0).long()
    # cumulatively sums up the values in each row of ids (seq dim)
    idxs = torch.cumsum(ids, dim=1)
    # create mask matrix using tokens_ids, shape is (batch_size, 1, seq_length, 1)
    extended_mask = tokens_ids.unsqueeze(1).unsqueeze(3)
    # create unilm mask matrix，shape is (batch_size, num_heads, from_seq_length, to_seq_length)
    mask = (idxs.unsqueeze(1) <= idxs.unsqueeze(2)).unsqueeze(1).to(dtype=torch.float32)
    # sets the attention weights to 0 for padding tokens
    mask *= extended_mask
    return mask

# token_ids = torch.tensor([[1, 2, 3, 4, 0], [1, 2, 0, 0, 0]])
# segment_ids = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]])
# print(unilm_mask(token_ids,segment_ids))