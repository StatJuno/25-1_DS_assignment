import torch
import torch.nn as nn
import math
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionEmbedding, self).__init__()
        #TODO
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        #TODO one line!
        return self.pe[:, :x.size(1)]