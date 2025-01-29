# Explanation: this file contains classes that are used to make the LSTM and transformer model variations.
#---------------------------------------------------------------------------------Import-----------------------------------------------------------------------

import torch 
from torch import nn
import math
# importing typing for writing function input types
from typing import List, Callable

#-----------------------------------------------------------------Functions for building transformer-----------------------------------------------------------
#normal positional encoding
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, seq_len):
    super().__init__()

    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0)/d_model))
    pe[:, 0::2] = torch.sin(position*div_term)
    pe[:, 1::2] = torch.cos(position*div_term)
    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    return x + self.pe[:, :x.shape[1]]

#multihead attention layer
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super().__init__()
    assert d_model % num_heads == 0, "d_model should be divisible by num_heads"
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)

  def scaled_dot_product_attention(self, Q, K, V):
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    attn_probs = torch.softmax(attn_scores, dim=1)
    output = torch.matmul(attn_probs, V)
    return output

  def split_heads(self, x):
    batch_size, seq_len, d_model = x.shape
    return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

  def combine_heads(self, x):
    batch_size, num_heads, seq_len, d_k = x.shape
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

  def forward(self, Q, K, V):
    Q = self.split_heads(self.w_q(Q))
    K = self.split_heads(self.w_k(K))
    V = self.split_heads(self.w_v(V))

    attn_output = self.scaled_dot_product_attention(Q, K, V)
    output = self.w_o(self.combine_heads(attn_output))
    return output

# feed forward layer
class PositionWiseFeedForward(nn.Module):
  def __init__(self, d_model, d_ff):
    super().__init__()
    self.fc1 = nn.Linear(d_model, d_ff)
    self.fc2 = nn.Linear(d_ff, d_model)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.fc2(self.relu(self.fc1(x)))

# encoder
class EncoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout):
    super().__init__()
    self.self_attn = MultiHeadAttention(d_model, num_heads)
    self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    attn_output = self.self_attn(x, x, x)
    x = self.norm1(x + self.dropout(attn_output))
    ff_output = self.feed_forward(x)
    x = self.norm2(x + self.dropout(ff_output))
    return x

#------------------------------------------------------------------Transformer Models---------------------------------------------------------------------------
# encoder based transformer model for classification (This parent class has no positional encoding)
# PE is added to the inherited classes so the code is more clean and clear to read 
class Transformer(nn.Module):
    def __init__(self, class_names: List[str], seq_len: int, d_model: int, nhead: int, d_ff: int = 2048, num_layers: int = 2, dropout: float = 0.1):
        """
        Transformer model for sign language classification
        Parameters:
            class_names : list of all the classes in the dataset.
            seq_len : length of input sequences-> corresponds to frame numbers in a video sample.
            d_model : dimention of the model inputs (number of features).
            nhead : the number of attention heads in the multi-head attention layer.
            d_ff : the dimension of the feedforward network.
            num_layers: the number of layers in the Transformer encoder. Default is 2.
            dropout : the dropout probability.
        """
        super().__init__()
        self.model_type = 'transformer' # this is used in the training to save some of the resutls in the correct directory for the model
        self.class_names = class_names
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff, dropout) for i in range(num_layers)])
        self.classifier = nn.Linear(in_features=d_model, out_features=len(self.class_names))
        
    def forward(self, src: torch.Tensor):
        output = src
        for encoder_layer in self.encoder_layers:
            output = encoder_layer(output)
            
        output = torch.mean(output, dim=1)
        output = self.classifier(output)
        return output

# encoder based transformer model for classification, with positional encoding
class PETransformer(Transformer):
    def __init__(self, class_names: List[str], seq_len: int, d_model: int, nhead: int, d_ff: int = 2048, num_layers: int = 2, dropout: float = 0.1):
        super().__init__(class_names, seq_len, d_model, nhead, d_ff, num_layers, dropout)
        self.model_type = 'PEtransformer'
        self.positional_encoding = PositionalEncoding(d_model, seq_len)

    def forward(self, src: torch.Tensor):
        output = self.positional_encoding(src)
        for encoder_layer in self.encoder_layers:
            output = encoder_layer(output)
        output = torch.mean(output, dim=1)
        output = self.classifier(output)
        return output

# encoder based transformer model for classification, with a learnable parameter for positional encoding
class ParamTransformer(Transformer):
    def __init__(self, class_names: List[str], seq_len: int, d_model: int, nhead: int, d_ff: int = 2048, num_layers: int = 2, dropout: float = 0.1):
        """
        Transformer model with learnable parameter as encoding
        """
        super().__init__(class_names, seq_len, d_model, nhead, d_ff, num_layers, dropout)
        self.model_type = 'paramtransformer'
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, src: torch.Tensor):
        output = src + self.positional_encoding
        for encoder_layer in self.encoder_layers:
            output = encoder_layer(output)
        output = torch.mean(output, dim=1)
        output = self.classifier(output)
        return output
#----------------------------------------------------------------------------LSTM Model------------------------------------------------------------------------
class LstmModel(nn.Module):
    def __init__(self, class_names: List[str], input_size: int, hidden_size: int, num_layers: int= 1, activition: Callable= nn.ReLU()):
        super().__init__()
        self.model_type= 'lstm'
        self.num_layers = num_layers
        self.class_names= class_names
        self.lstm_layers= nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True))
        
        for i in range(1, num_layers):
            self.lstm_layers.append(nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True))
        
        self.fc = nn.Linear(in_features= hidden_size, out_features= len(self.class_names))
        self.activition = activition

    def forward(self, src):
        output = src
        for lstm in self.lstm_layers:
            output, final_states = lstm(output)
            output = self.activition(output)

        output= self.fc(output[:,-1,:])
        return output