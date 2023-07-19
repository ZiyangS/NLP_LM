import torch
import torch.nn as nn
import copy
import json
from layers.BERT_layers import LayerNorm, MultiHeadAttentionLayer, PositionWiseFeedForward, activations
from models.base_transformer import Transformer
from layers.utils import truncated_normal_
from layers.Embed import BertEmbeddings
from layers.BERT_encoder_layer import BertLayer

class BERT(Transformer):
    """
    Built BERT model with Bidirectional Encoder Representations from Encoder-only Transformer model.
    BERT pre-trains deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context.
    This class creates and initializes the BERT model, including its architecture and various functionalities like
    Next Sentence Prediction (NSP) and Masked Language Model (MLM).
    """
    def __init__(
            self,
            max_position,  # the maximum length of the sequence
            segment_vocab_size=2, # the number of distinct segment in the input sequence
            # It is default as 2 (corresponding to the two segments for tasks like NSP)
            initializer_range=0.02, # the standard deviation of the truncated normal distribution for weight initialization
            with_pool=False,  # specifies whether to include a pooling layer at the end of the model
            with_nsp=False,  # specifies whether to include NSP
            with_mlm=False,  # specifies whether to include MLM
            # It also initializes an additional bias parameter mlmBias, which is explicitly set as the bias of the MLM decoder. It also adds a dense transformation (mlmDense) and a layer normalization (mlmLayerNorm) to the MLM head.
            hierarchical_position=None,  # 是否层次分解位置编码 ?
            custom_position_ids=False,  # 是否自行传入位置id whether to use custom position id for the position encoding
            **kwargs
    ):
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.initializer_range = initializer_range
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.hierarchical_position = hierarchical_position
        self.custom_position_ids = custom_position_ids
        if self.with_nsp and not self.with_pool: # NSP can only be used if with_pool=True
            self.with_pool = True

        super(BERT, self).__init__(**kwargs)

        self.embeddings = BertEmbeddings(self.vocab_size, self.hidden_size, self.max_position, self.segment_vocab_size, self.dropout_rate)
        layer = BertLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.intermediate_size, self.hidden_act, is_dropout=False)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)]) #  the stack of transformer layers
        if self.with_pool:
            # create a Linear layer and a Tanh activation function as the pooler
            # extract the [CLS] vector after all transformer layers
            self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
            self.pooler_activation = nn.Tanh()
            # todo: check the shape of output of each step in the pooler, the tanh outputs a vector or a value between -1 and 1
            if self.with_nsp:
                # the input of NSP is pooled_output
                self.nsp = nn.Linear(self.hidden_size, 2)
        else:
            self.pooler = None
            self.pooler_activation = None
        if self.with_mlm:
            # A linear docoder layer predicts the masked word
            self.mlmDecoder = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            # 需不需要这一操作，有待验证
            # self.mlmDecoder.weight = self.embeddings.word_embeddings.weight
            self.mlmBias = nn.Parameter(torch.zeros(self.vocab_size))
            self.mlmDecoder.bias = self.mlmBias
            self.mlmDense = nn.Linear(self.hidden_size, self.hidden_size) #?
            self.transform_act_fn = activations[self.hidden_act] #?
            self.mlmLayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.apply(self.init_model_weights)

    def init_model_weights(self, module):
        """
        Initialize weights, biases of all the linear, embedding, and layer normalization layers
        """
        if isinstance(module, (nn.Linear, nn.Embedding)): # initialize linear embedding by (truncated)normal
            # module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            module.weight.data = truncated_normal_(module.weight.data, mean=0.0, std=self.initializer_range)
            # print(np.mean(module.weight.data.detach().numpy()), np.std(module.weight.data.detach().numpy()))
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0) #  initializing weights to 1 maintains the input's variance as close as possible to its original value
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, token_ids, segment_ids=None, attention_mask=None, output_all_encoded_layers=False):
        """
        token_ids: a sequence of tokens where each token is represented by its corresponding index in the vocabulary

        segment_ids： the segment IDs for each token in the input. (e.g., 0 for the first sentence and 1 for the second sentence).
        If not provided, it only has one segment.

        attention_mask: indicates which tokens in the input sequence should be attended (1) to attention module or be ignored (0).

        the shape of these 3 variables is: (batch_size, sequence_length)
        """
        if attention_mask is None:
            # If not provided, create a mask based on non-zero token IDs (i.e., not padding)
            # Unsqueeze the mask to (batch_size, 1, 1, to_seq_length) in order to be broadcastable as
            # (batch_size, num_heads, from_seq_length, to_seq_length) for the multi-head attention mechanism
            attention_mask = (token_ids != 0).long().unsqueeze(1).unsqueeze(2)
        if segment_ids is None:
            # If not provided, create a tensor of zeros (same segment)
            segment_ids = torch.zeros_like(token_ids)

        # match the data type of the model parameters (compatible with mixed precision fp16)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        # Given binary attention_mask, ignore the elements with 0 by assigning a large negative value (zeroing out after applying softmax)
        # attention_mask = (1.0 - attention_mask) * -10000.0

        # initialize input embedding
        hidden_states = self.embeddings(token_ids, segment_ids)
        # Apply each layer in the encoder model to hidden states
        encoded_layers = [hidden_states] # 添加embedding的输出
        for layer_module in self.encoderLayer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers: # If True, store the outputs by each layer
                encoded_layers.append(hidden_states)
        # todo: test the last transformer encoder layer?
        if not output_all_encoded_layers: # Only store the output of the final layer
            encoded_layers.append(hidden_states)

        # obtain the output of the final layer
        sequence_output = encoded_layers[-1]
        # whether obtain the final layer's output? do we need it?
        # todo: check whether we need this
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # apply pooling to the first token [CLS] of the final layer's output
        if self.with_pool:
            # todo: check dim of sequence_output and sequence_output[:, 0]
            pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        else:
            pooled_output = None
        # apply linear layer to the pooled_output to produce NSP prediction score
        if self.with_pool and self.with_nsp:
            nsp_scores = self.nsp(pooled_output)
        else:
            nsp_scores = None
        # Apply to the final layer's output to produce MLM prediction scores
        if self.with_mlm:
            mlm_hidden_state = self.mlmDense(sequence_output)
            mlm_hidden_state = self.transform_act_fn(mlm_hidden_state)
            mlm_hidden_state = self.mlmLayerNorm(mlm_hidden_state)
            mlm_scores = self.mlmDecoder(mlm_hidden_state)
        else:
            mlm_scores = None
        # return values according to tasks
        if mlm_scores is None and nsp_scores is None:
            return encoded_layers, pooled_output
        elif mlm_scores is not None and nsp_scores is not None:
            return mlm_scores, nsp_scores
        elif mlm_scores is not None:
            return mlm_scores
        else:
            return nsp_scores

    def variable_mapping(self):
        '''
        creates a dictionary that maps the parameters of a local model (defined by myself)
        to the parameters of a pre-trained BERT model (provided by a library such as Hugging Face's Transformers).
        With this mapping, we can load the pre-trained weights into your local model correctly even with different parameter names
        '''
        mapping = {
            'embeddings.word_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
            'embeddings.position_embeddings.weight': 'bert.embeddings.position_embeddings.weight',
            'embeddings.segment_embeddings.weight': 'bert.embeddings.token_type_embeddings.weight',
            'embeddings.layerNorm.weight': 'bert.embeddings.LayerNorm.weight',
            'embeddings.layerNorm.bias': 'bert.embeddings.LayerNorm.bias',
            'pooler.weight': 'bert.pooler.dense.weight',
            'pooler.bias': 'bert.pooler.dense.bias',
            'nsp.weight': 'cls.seq_relationship.weight',
            'nsp.bias': 'cls.seq_relationship.bias',
            'mlmDense.weight': 'cls.predictions.transform.dense.weight',
            'mlmDense.bias': 'cls.predictions.transform.dense.bias',
            'mlmLayerNorm.weight': 'cls.predictions.transform.LayerNorm.weight',
            'mlmLayerNorm.bias': 'cls.predictions.transform.LayerNorm.bias',
            'mlmBias': 'cls.predictions.bias',
            'mlmDecoder.weight': 'cls.predictions.decoder.weight'

        }
        for i in range(self.num_hidden_layers): # Create mapping in a loop due to multiple encoder layer
            prefix = 'bert.encoder.layer.%d.' % i
            mapping.update({'encoderLayer.%d.multiHeadAttention.q.weight' % i: prefix + 'attention.self.query.weight',
                            'encoderLayer.%d.multiHeadAttention.q.bias' % i: prefix + 'attention.self.query.bias',
                            'encoderLayer.%d.multiHeadAttention.k.weight' % i: prefix + 'attention.self.key.weight',
                            'encoderLayer.%d.multiHeadAttention.k.bias' % i: prefix + 'attention.self.key.bias',
                            'encoderLayer.%d.multiHeadAttention.v.weight' % i: prefix + 'attention.self.value.weight',
                            'encoderLayer.%d.multiHeadAttention.v.bias' % i: prefix + 'attention.self.value.bias',
                            'encoderLayer.%d.multiHeadAttention.o.weight' % i: prefix + 'attention.output.dense.weight',
                            'encoderLayer.%d.multiHeadAttention.o.bias' % i: prefix + 'attention.output.dense.bias',
                            'encoderLayer.%d.layerNorm1.weight' % i: prefix + 'attention.output.LayerNorm.weight',
                            'encoderLayer.%d.layerNorm1.bias' % i: prefix + 'attention.output.LayerNorm.bias',
                            'encoderLayer.%d.feedForward.intermediateDense.weight' % i: prefix + 'intermediate.dense.weight',
                            'encoderLayer.%d.feedForward.intermediateDense.bias' % i: prefix + 'intermediate.dense.bias',
                            'encoderLayer.%d.feedForward.outputDense.weight' % i: prefix + 'output.dense.weight',
                            'encoderLayer.%d.feedForward.outputDense.bias' % i: prefix + 'output.dense.bias',
                            'encoderLayer.%d.layerNorm2.weight' % i: prefix + 'output.LayerNorm.weight',
                            'encoderLayer.%d.layerNorm2.bias' % i: prefix + 'output.LayerNorm.bias'
                            })
        return mapping

