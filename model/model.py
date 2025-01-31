import torch
from torch import nn
import math

class InputEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size, d_model) 
        self.d_model = d_model

    def forward(self, x):
        return super().forward(x) * math.sqrt(self.d_model)  

class PositionalEncodding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout()
        self.positional_encoding = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size , dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.positional_encoding = self.positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', self.positional_encoding)
    def forward(self, x):
        return self.dropout(x + (self.positional_encoding[:, :x.size(1), :]).requires_grad_(False))