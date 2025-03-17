# transformer.py
import torch.nn as nn
from .embeddings import TokenEmbedding, PositionEmbedding
from .encoder import TransformerEncoderLayer
from .decoder import TransformerDecoderLayer

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, d_ff, num_encoder_layers, num_decoder_layers, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.decoder_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.position_embedding = PositionEmbedding(d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(num_decoder_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder
        src_emb = self.encoder_embedding(src) + self.position_embedding(src)
        src_emb = self.dropout(src_emb)
        
        for layer in self.encoder_layers:
            src_emb = layer(src_emb)
        
        # Decoder
        tgt_emb = self.decoder_embedding(tgt) + self.position_embedding(tgt)
        tgt_emb = self.dropout(tgt_emb)
        
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, src_emb, src_mask, tgt_mask)
        
        output = self.fc_out(tgt_emb)
        return output
