# importing libraries necessary for creating the models

import torch
from torch import nn
from typing import List, Callable

# Transformer classes

class ParamTransformer(nn.Module):
    def __init__(self,
                 class_names: List[str],
                 seq_len: int= 40, 
                 d_model: int= 1662,
                 nhead: int= 6,
                 d_ff: int=2048,
                 num_layers: int= 2):

        super().__init__()
        self.model_type = 'paramtransformer'
        self.class_names= class_names
        self.positional_encoding= nn.Parameter(torch.randn(1, seq_len, d_model))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model= d_model, nhead= nhead, dim_feedforward= d_ff, batch_first= True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer= self.encoder_layer, num_layers= num_layers)
        self.classifier = nn.Linear(in_features= d_model, out_features= len(self.class_names))


    def forward(self, src: torch.Tensor):
        """
        Transformer model with learable parameter for positional encoding
        Args:
            src: Tensor of shape ``[batch_size, seq_len, input_shape]``
        Returns:
            output: Tensor of shape ``[batch_size, len(class_names)]``
        """
        output = src+ self.positional_encoding
        output = torch.mean(output, dim=1)        
        output = self.classifier(output)

        return output


class LinearParamTransformer(nn.Module):
    def __init__(self,
                 class_names: List[str],
                 seq_len: int= 40, 
                 d_model: int= 128,
                 nhead: int= 4,
                 d_ff: int=2048,
                 num_layers: int= 2,
                 input_shape: int= 1662):
 
        super().__init__()
        self.model_type = 'linearparamtransformer'
        
        self.class_names= class_names
        self.positional_encoding= nn.Parameter(torch.randn(1, seq_len, d_model))
        self.linear= nn.Linear(in_features= input_shape, out_features= d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model= d_model, nhead= nhead, dim_feedforward= d_ff, batch_first= True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer= self.encoder_layer, num_layers= num_layers)
        self.classifier = nn.Linear(in_features= d_model, out_features= len(self.class_names))


    def forward(self, src: torch.Tensor):
        """
        Transformer model with linear layer and learnable parameter for positional encoding (only encoder)
        Args:
            src: Tensor of shape ``[batch_size, seq_len, input_shape]``
        Returns:
            output: Tensor of shape ``[batch_size, len(class_names)]``
        """
        output = self.linear(src)
        output= output+ self.positional_encoding
        output = self.transformer_encoder(output)
        output = torch.mean(output, dim=1)        
        output = self.classifier(output)

        return output


class ConvoTransformer(nn.Module):
    def __init__(self,
                 class_names: List[str],
                 seq_len: int= 40, 
                 d_model: int= 128,
                 nhead: int= 4,
                 d_ff: int=2048,
                 num_layers: int= 2,
                 input_shape: int= 1662,
                 kernel_size: int= 1):

        super().__init__()
        self.model_type = 'convotransformer'
        self.class_names= class_names
        self.positional_encoding= nn.Conv1d(in_channels= input_shape, out_channels= d_model, kernel_size= kernel_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model= d_model, nhead= nhead, dim_feedforward= d_ff, batch_first= True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer= self.encoder_layer, num_layers= num_layers)
        self.classifier = nn.Linear(in_features= d_model, out_features= len(self.class_names))


    def forward(self, src: torch.Tensor):
        """
        Transformer model with convolutional layer for positional encoding  (only encoder)
        Args:
            src: Tensor of shape ``[batch_size, seq_len, input_shape]``
        Returns:
            output: Tensor of shape ``[batch_size, len(class_names)]``
        """
        src = src.permute(0, 2, 1)
        output = self.positional_encoding(src)
        output = output.permute(0, 2, 1)
        output = self.transformer_encoder(output)
        
        # Use the mean of all timesteps instead of just the last one
        output = torch.mean(output, dim=1)        
        output = self.classifier(output)
        return output


class Transformer(nn.Module):
    def __init__(self,
                 class_names: List[str],
                 seq_len: int= 40, 
                 d_model: int= 1662,
                 nhead: int= 6,
                 d_ff: int=2048,
                 num_layers: int= 2):

        super().__init__()
        self.model_type = 'transformer'
        self.class_names= class_names
        self.encoder_layer = nn.TransformerEncoderLayer(d_model= d_model, nhead= nhead, dim_feedforward= d_ff, batch_first= True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer= self.encoder_layer, num_layers= num_layers)
        self.classifier = nn.Linear(in_features= d_model, out_features= len(self.class_names))


    def forward(self, src: torch.Tensor):
        """
        Transformer model without any positional encoding (only encoder)
        Args:
            src: Tensor of shape ``[batch_size, seq_len, input_shape]``
        Returns:
            output: Tensor of shape ``[batch_size, len(class_names)]``
        """
        output = self.transformer_encoder(src)
        output = torch.mean(output, dim=1)        
        output = self.classifier(output)

        return output


#Lstm classes
class LstmModel(nn.Module):
    def __init__(self,
                 class_names: List[str],
                 input_size: int= 1662,
                 hidden_size: int= 277,
                 num_layers: int= 1,
                 activition: Callable= nn.ReLU()):
        super().__init__()
        self.model_type= 'lstm'
        self.num_layers = num_layers
        self.class_names= class_names
        self.lstm_layers= nn.ModuleList()

        self.lstm_layers.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True))
        for _ in range(1, num_layers):
            self.lstm_layers.append(nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True))
        
        self.fc = nn.Linear(in_features= hidden_size, out_features= len(self.class_names))
        self.activition = activition

    def forward(self, src):
        """
        Args:
            src: Tensor of shape ``[batch_size, seq_len, input_shape]``
            
        Returns:
            output: Tensor of shape ``[batch_size, len(class_names)]``
        """
        output = src
        for lstm in self.lstm_layers:
            output, _ = lstm(output)
            output = self.activition(output)

        output= self.fc(output[:,-1,:])
        return output


def reset_model_parameters(model):
    """
    Resets the model parameters
    """
    for name, module in model.named_children():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
